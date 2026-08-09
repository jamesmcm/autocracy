# Driving the real Democracy 3 simulation externally via gdb.

This directory contains a working prototype of driving the *actual* Democracy 3
game binary's simulation from outside the process, using gdb under Xvfb.

The game (`Democracy3.bin.x86_64`, v1.30.2) is **not stripped and has
debug_info** (function-level symbols), so every simulation entry point is
reachable:

| symbol | address | purpose |
|--------|---------|---------|
| `SIM_GetSimulation()` | 0x5ab1a0 | the simulation singleton (instance at 0xa38360) |
| `SIM_Simulation::NextTurn()` | 0x60e120 | the complete turn step |
| `SIM_Simulation::GetNeuronByName(std::string)` | 0x60c140 | read any neuron |
| `SIM_Simulation::Initialise()` | 0x60f560 | initialise a country |
| `SIM_Policy::ForceSlider(float)` | 0x5f5ba0 | set a policy slider |
| `SIM_LoadGame::OpenSavedFile(std::string)` | 0x5d4a90 | open a save XML |
| `SIM_LoadGame::LoadGameData()` | 0x5dcba0 | load the full state |
| `SIM_LoadGame::ProcessGameLoad()` | 0x5dcee0 | the whole load flow |
| `SIM_SaveGame::Save*()` | - | serialize the state to XML |

`SIM_Simulation::NextTurn()` calls the manager NextTurns (EventManager,
FinanceManager, GlobalEconomy, MinisterManager, PolicyManager,
PressureGroupManager, SituationManager) and is exactly the step the Python
simulator reimplements.

## Static follow-up for injection

The installed binary was inspected with `nm`, `readelf`, and `objdump`; the
game was not launched during this audit. The load/save singleton storage is
now pinned down for a future version-specific injector:

| object | address | guard | constructor |
|--------|---------|-------|-------------|
| `SIM_LoadGame` | `0xa3aa00` | `0xa3a9f8` | `0x5d39c0` |
| `SIM_SaveGame` | `0xa3b5c0` | `0xa3a9f0` | `0x602940` |
| `SIM_FinanceManager` | returned by `SIM_GetFinanceManager()` (`0x5b9790`) | `0xa3a6b0` | `0x5ca880` |

The static turn order is `IssueManager`, `MinisterManager`, `EventManager`,
`PressureGroupManager`, `PolicyManager`, `GlobalEconomy`,
`SituationManager`, `FinanceManager::NextTurn`, gameplay date update, all
`NeuralEffect::NextTurn` calls, neuron value/history updates, then dilemma,
party, voter, poll, grudge, and the final finance total/history update.
`SIM_Policy::NextTurn` keeps 20 float samples for cost history at policy
offset `0x408` and income history at `0x458`, with the newest sample at index
zero. This is the native behavior mirrored by the Python policy-history
fields.

`SIM_FinanceManager::ApplyInterestRateCalculations` uses
`INTEREST_RATE_MIN + (INTEREST_RATE_MAX - INTEREST_RATE_MIN) *
(_global_interest_rates_ - 0.5 + min((credit_rating / 9)^2, 1))`; the global
neuron is `SIM_Simulation`'s value at the finance-manager call site. These
offsets are tied to Democracy 3 v1.30.2 and must be revalidated before any
`LD_PRELOAD` or gdb command invokes a constructor or member function.

The country-load portion is pinned as well. `SIM_LoadGame::LoadMission()`
calls `SIM_MissionManager::LoadMissions()`, `GetByName()`, and `SetCurrent()`;
then it calls `SIM_Simulation::ApplyMissionSpecificData(false)`, initializes
`SIM_Names`, and loads mission options. That sequence must precede
`ProcessGameLoad()`/`NextTurn()`; it is not safe to jump directly to
`OpenSavedFile()` from the GUI breakpoint.

Run `preflight.py` before preparing a version-specific injector. It only reads
the ELF symbol table and never launches the game:

```
PYTHONPATH=. uv run python gamedrive/preflight.py
```

The load-flow audit also pinned the `SIM_LoadGame` filename `std::string` at
`+0x838`. `ProcessGameLoad()` opens that file, calls gameplay release/preload,
loads the ordered game-data sections, calls gameplay postload, and frees the
temporary load buffer. `LoadEffects()` restores input-effect throttles and raw
33-slot histories, but does not serialize the desired output throttle for every
outgoing effect; this is the runtime state behind the remaining targeted ring
residuals.

The same save/load boundary applies to voters: `LoadVoters()` restores the
individual ideology inputs, sympathy fields, party pointer, and organization
list before `SIM_VoterManager::PostLoad()` rebuilds manager-owned membership
lists. The XML therefore contains enough per-voter inputs to seed a model, but
not the manager's live party/group lists; the Python save bridge now preserves
the serialized inputs instead of silently discarding them. It also retains the
top-level party definitions and their member/activist history rings; those
history values are serialized, while the live member lists remain native
manager state.

## What the prototype does

`gdb_drive.py` starts an Xvfb display, launches the game under gdb with
`harness.gdb`, and reports what is reachable:

1. The game reaches `mainLoop()` (the GUI event loop) without the graphics
   resize crash that otherwise kills it under Xvfb.
2. `SIM_GetSimulation()` returns the live simulation singleton.
3. `SIM_Simulation::NextTurn()` is callable and reachable, but requires a
   country to have been loaded first (calling it here faults in
   `SIM_Names::GetRandomFullName` — a division by an uninitialized value).

## Usage

```
PYTHONPATH=. python gamedrive/gdb_drive.py          # default, reports the probe
PYTHONPATH=. python gamedrive/gdb_drive.py --verbose
```

## The blocker to a full round-trip

Advancing a real game requires a country to be loaded first.  The load flow is
driven by the GUI (`SIM_LoadGame::LoadGame` -> `ProcessGameLoad` ->
`SIM_LoadGame::OpenSavedFile` + `LoadGameData` + `SIM_Gameplay::PreLoadGame`/
`PostLoadGame`).  Driving those C++ calls from gdb is blocked by:

* the `SIM_GetLoadGame()`/`SIM_GetSaveGame()` getters are **static inline**
  (no symbol), and their singletons are lazily constructed (vtable is zero at
  `mainLoop`), so the instances must be constructed manually;
* gdb has **no usable C++ object layout/type info** beyond the member accesses
  visible in disassembly. The filename offset is pinned, but the remaining
  load/runtime objects still need to be reconstructed;
* the `LoadGame` flow starts threads and a loading screen, and direct calls
  into `OpenSavedFile`/`LoadGameData` from the breakpoint hang (no progress);
* `SIM_Simulation::NextTurn()` without a loaded country SIGFPEs.

The mission selection order above is no longer a blocker; the remaining work
is reconstructing enough lazy singleton/object layout to drive that sequence
through the game's thread and loading-screen control path.

A complete harness therefore needs the class layouts pinned down (from
`objdump`/`readelf` on the `.data.rel.ro` vtables and the constructors) and a
`SIM_Simulation::Initialise` + `SIM_Mission::Load` sequence driven in the right
order, after which `NextTurn()` and the
`SIM_SaveGame::Save*` serializers give exact ground truth that round-trips
through the existing `autocracy/savegame.py` XML parser.
