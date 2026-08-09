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
* gdb has **no C++ type info** (the debug_info is function-level only), so the
  class member layouts (e.g. where `SIM_LoadGame` stores the filename) must be
  reverse-engineered from the disassembly;
* the `LoadGame` flow starts threads and a loading screen, and direct calls
  into `OpenSavedFile`/`LoadGameData` from the breakpoint hang (no progress);
* `SIM_Simulation::NextTurn()` without a loaded country SIGFPEs.

A complete harness therefore needs the class layouts pinned down (from
`objdump`/`readelf` on the `.data.rel.ro` vtables and the constructors) and a
`SIM_Simulation::Initialise` + `SIM_Mission::Load` + `ApplyMissionSpecificData`
sequence driven in the right order, after which `NextTurn()` and the
`SIM_SaveGame::Save*` serializers give exact ground truth that round-trips
through the existing `autocracy/savegame.py` XML parser.
