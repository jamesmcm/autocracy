# Driving the real Democracy 3 simulation externally via gdb.

This directory contains a working prototype of driving the *actual* Democracy 3
game binary's simulation from outside the process, using gdb, a version-pinned
`LD_PRELOAD` probe, and Xvfb for a headless display.

The game (`Democracy3.bin.x86_64`, v1.30.2) is **not stripped and has
debug_info** (function-level symbols), so every simulation entry point is
reachable:

| symbol | address | purpose |
|--------|---------|---------|
| `SIM_GetSimulation()` | 0x5ab1a0 | the simulation singleton (instance at 0xa38360) |
| `SIM_Simulation::NextTurn()` | 0x60e120 | the complete turn step |
| `SIM_Gameplay::NextTurn()` | 0x5d0f80 | GUI-facing asynchronous turn launcher |
| `NextTurnThread(void*)` | 0x5cff30 | native full turn worker entrypoint |
| `SIM_Simulation::GetNeuronByName(std::string)` | 0x60c140 | read any neuron |
| `SIM_Simulation::Initialise()` | 0x60f560 | initialise a country |
| `SIM_GlobalEconomy::BackProjectHistory()` | 0x5d1370 | rebuild hidden global history |
| `SIM_GlobalEconomy::Calculate()` | 0x5d14e0 | calculate the global cycle/random factor |
| `SIM_GlobalEconomy::NextTurn()` | 0x5d1670 | advance the global cycle |
| `SIM_Voter::UpdateIncome()` | 0x61e620 | rebuild per-voter income groups |
| `SIM_Policy::ForceSlider(float)` | 0x5f5ba0 | set a policy slider |
| `SIM_LoadGame::OpenSavedFile(std::string)` | 0x5d4a90 | open a save XML |
| `SIM_LoadGame::LoadGameData()` | 0x5dcba0 | load the full state |
| `SIM_LoadGame::ProcessGameLoad()` | 0x5dcee0 | the whole load flow |
| `SIM_SaveGame::Save*()` | - | serialize the state to XML |

`SIM_Simulation::NextTurn()` calls the manager NextTurns (EventManager,
FinanceManager, GlobalEconomy, MinisterManager, PolicyManager,
PressureGroupManager, SituationManager) and is exactly the step the Python
simulator reimplements.

## Static and live injection path

The installed v1.30.2 binary was inspected with `nm`, `readelf`, `objdump`, and
bounded gdb runs. The load/save singleton storage is pinned down for this
version-specific injector:

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

The voter/party entry points are pinned for follow-up native manager work:
`SIM_Voter::CalculateApproval()` (0x61b880),
`SIM_Voter::ConsiderPartyMembership(int)` (0x61d000),
`SIM_Voter::UpdateIncome()` (0x61e620),
`SIM_VoterManager::PreJoinParties()` (0x6208e0),
`SIM_VoterType::CalculatePercentage()` (0x6229c0),
`SIM_VoterType::ForceVoter(SIM_Voter*, float)` (0x623350),
`SIM_Party::NextTurn()` (0x5f4650),
`SIM_Party::CalculateActivists()` (0x5f4290),
`SIM_PartyManager::NextTurn()` (0x5f4920),
`SIM_PartyManager::CalculateActivists()` (0x5f49a0),
`SIM_PollsManager::CalculateVoteRate()` (0x5fb450),
`SIM_PollsManager::NextTurn()` (0x5fb750),
`SIM_Complacency::NextTurn()` (0x5bfc10),
`SIM_Voter::GetEffectiveApproval()` (0x61c5a0), and
`SIM_Voter::WillVoteForPlayer()` (0x61db50), and
`SIM_SaveGame::SaveParties()` (0x604d40). The RNG hooks
`GRandom::Init(int, int)` (0x6354a0), `GRandom::RandUnitFloat()` (0x635310),
and `GRandom::RandReal(float, float)` (0x6356f0) are pinned as well.
`preflight.py` verifies these alongside the load/turn, neuron, situation,
voter-manager, and finance entry points without starting the installed game.

The global-economy audit also pinned `BackProjectHistory()` (0x5d1370),
`Calculate()` (0x5d14e0), and `NextTurn()` (0x5d1670). `Calculate()` calls
`GRandom::RandReal(0.9, 1.1)` after the sinusoidal cycle term. The save stores
the resulting hidden value and its 33-slot history, but not the random-array
cursor, so the Python bridge preserves the observed history without inventing
a deterministic multiplier sequence.

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
`SIM_Voter::UpdateIncome()` additionally rebuilds the runtime income-group
links from the per-voter income fields. The simulator now preserves each
serialized nested VoterType `<income>` value and evaluates its direct graph
inputs; the manager-added per-voter host links remain a documented runtime
boundary because they are not serialized.

The static audit also covers `SIM_VoterManager::PreSimulatedNextTurn()`
(0x620510), `SIM_VoterType::CalculateValue()` (0x622390),
`SIM_VoterType::AddMostSuitable()` (0x6226d0), and
`SIM_Voter::CalcEffectOfNeuron()` (0x61a1d0). These entry points make the
remaining ordering and host-link boundaries explicit without pretending that
the live lists or approval modifiers can be reconstructed from XML.

## What the prototype does

`gdb_drive.py` remains the small reachability probe. `inject_drive.py` uses
`harness_inject.gdb` and the compiled `libd3probe.so` to:

1. Start one native asynchronous load from the first `mainLoop()` stop.
2. Wait for the game's loading-complete flag before calling into live objects.
3. Save the loaded native state through `SIM_SaveGame`.
4. Optionally write a named neuron's live float slot (`SIM_Neuron + 0x38`) and
   save that edited state under a separate name.
5. Run the game's `NextTurnThread` entrypoint synchronously and save the
   resulting native turn, or use the experimental asynchronous launcher.

The staged breakpoint only stops `mainLoop()` at load/turn boundaries. The
expensive native worker runs without a breakpoint on every GUI frame.

## Usage

```
PYTHONPATH=. python gamedrive/gdb_drive.py          # default, reports the probe
PYTHONPATH=. python gamedrive/gdb_drive.py --verbose

# Build and validate the version-pinned injector
make -C gamedrive
PYTHONPATH=. uv run python gamedrive/preflight.py

# Load d3_probe_turn0.xml, run the full native worker, and save its output
PYTHONPATH=. uv run python gamedrive/inject_drive.py \
  --sync-gameplay-turn --timeout 120

# Load only, then edit one live neuron and persist a separate native save
PYTHONPATH=. uv run python gamedrive/inject_drive.py --skip-turn \
  --edit-node GDP --edit-value 0.123
```

The wrapper reads and writes the installed user's save directory:
`/home/gopostal/.local/share/democracy3/savegames`. Copy a source capture there
under a new name first; do not use an existing save name. The wrapper checks
that required output files exist and returns nonzero on a timeout or missing
output. The game process is terminated by gdb, and the source capture is not
modified. Memory edits are intentionally limited to `--skip-turn`; to run a
turn from an edited state, use the resulting XML as the input to a fresh probe.

## Ground-truth results

Using `parity_cases/dem3saves/turn0_initial.xml` as input, the native load
save (`d3_probe_loaded*.xml`) matched the parser's complete extracted snapshot:
40 ordinary simvalues, 36 situations, voter aggregates, policy/runtime maps,
finance fields, party history, and effect-memory records.

The native `NextTurnThread` run produced turn 1. Compared with the Python
simulator processing the same no-order input:

| section | result |
|---|---|
| finance | total income and expenditure exact to serialized float32 values |
| policies/effect throttles | exact; 123 policies and 61 effect throttles |
| ordinary simvalues | 14/40 differ; max delta 0.021644, mean delta 0.002241 |
| situations | 16/36 differ; max delta 0.030829 |
| voter values | 20/21 differ; max delta 0.111084 |
| voter percentages | 19/21 differ; max delta 0.496000 |
| voter frequencies/incomes | frequencies differ below 1e-6; incomes exact |

This is a same-input comparison. `turn1_initial.xml` follows the captured
`turn0_orders.xml`, so it must not be compared with a native run loaded from
`turn0_initial.xml` without applying those orders. Loading the captured
`turn0_orders.xml` directly through this headless path currently exceeds the
bounded loader timeout; the next step is to drive the native order/slider
entrypoints rather than treat an orders save as a completed-turn input.
Probe-produced output XML is still a valuable ground-truth artifact for the
parser, but reloading an edited/probe-produced save through this loader path
also currently exceeds the bounded timeout. Treat each probe run as a fresh
source-capture-to-output operation until that save round-trip is repaired.

## Current limitations and safety boundaries

The load flow is driven by the game's own asynchronous path
(`SIM_LoadGame::LoadGame` -> `ProcessGameLoad` -> `OpenSavedFile` +
`LoadGameData` + `PreLoadGame`/`PostLoadGame`). Direct calls to
`OpenSavedFile()`/`LoadGameData()` from a breakpoint still hang; the probe uses
the native `LoadGame()` launcher and waits on the completion flag instead.

The `--gameplay-turn` mode is retained as an experiment, but the game's
`SIM_Gameplay::NextTurn()` thread orchestration can stall under ptrace. Use
`--sync-gameplay-turn`: it invokes the same `NextTurnThread(void*)` worker and
manager order synchronously, producing the reliable ground-truth boundary.

The probe is tightly version-pinned:

* the `SIM_GetLoadGame()`/`SIM_GetSaveGame()` getters are **static inline**
  (no symbol), and their singletons are lazily constructed (vtable is zero at
  `mainLoop`), so the instances must be constructed manually;
* gdb has **no usable C++ object layout/type info** beyond the member accesses
  visible in disassembly. The filename offset is pinned, but the remaining
  load/runtime objects still need to be reconstructed;
* `SIM_Simulation::NextTurn()` without a loaded country SIGFPEs.

The library performs no constructor-time game calls. All native calls happen
after an explicit gdb breakpoint, and memory edits are opt-in. Since gdb kills
the inferior after each bounded run, an edit is process-local; nevertheless,
always write new save names and retain the original source capture.

A future order-driving harness still needs the native slider/order entrypoints
and the manager-owned party/income links. Those are the remaining pieces for
an apples-to-apples replay of the captured 12-turn action sequence.
