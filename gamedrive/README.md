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
| `SIM_PolicyManager::GetPolicy(std::string)` | 0x5fa090 | resolve a live policy |
| `SIM_Policy::SetSlider(float)` | 0x5f5b70 | apply a native order target |
| `SIM_Policy::Implement()` / `Cancel()` | 0x5f68f0 / 0x5f66f0 | native policy state changes |
| `SIM_PoliticalCapital::SpendPoints(int)` | 0x5fadc0 | charge native order cost |
| `SIM_VoterManager::PreJoinParties()` | 0x6208e0 | rebuild live voter-party links |
| `SIM_VoterManager::PreCalculateIncome()` | 0x620920 | rebuild live income hosts |
| `SIM_PartyManager::CalculateActivists()` | 0x5f49a0 | refresh party activist counts |
| `SIM_PollsManager::CalculateVoteRate()` | 0x5fb450 | refresh the live poll rate |
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

The voter/party entry points are pinned for the opt-in native manager audit:
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
manager state. The injector can now run the native refresh methods in-process,
write a manager census (party member/activist fields, poll rate, party links,
and per-voter income-host link counts), and optionally save the refreshed
native XML under a separate name.
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
4. Translate each pre-turn `_orders` save into native `Implement`, `Cancel`,
   `SetSlider`, and political-capital calls; an omitted orders file is a
   deliberate no-op turn.
5. Run one or a bounded number of native `NextTurnThread` workers
   synchronously and save every completed turn under fresh names.
6. Optionally refresh the native voter/party/poll managers, write their live
   census, or edit a named neuron's float slot (`SIM_Neuron + 0x38`).

The staged breakpoint only stops `mainLoop()` at load/turn boundaries. The
expensive native worker runs without a breakpoint on every GUI frame.

## Usage

```
PYTHONPATH=. python gamedrive/gdb_drive.py          # default, reports the probe
PYTHONPATH=. python gamedrive/gdb_drive.py --verbose

# Build and validate the version-pinned injector
make -C gamedrive
PYTHONPATH=. uv run python gamedrive/preflight.py

# Load a copied source, run one synchronous native worker, and save fresh output
PYTHONPATH=. uv run python gamedrive/inject_drive.py \
  --load-name d3_probe_turn0_copy \
  --turn-mode sync --timeout 120

# Replay the captured pre-turn orders through twelve native turns
PYTHONPATH=. uv run python gamedrive/inject_drive.py \
  --load-name d3_probe_turn0_copy \
  --initial-file parity_cases/dem3saves/turn0_initial.xml \
  --orders-dir parity_cases/dem3saves \
  --capture-prefix d3_probe_replay_run42 \
  --timeout 120

# Compare those native captures with the same twelve-turn simulator replay
PYTHONPATH=. uv run python gamedrive/capture.py \
  --initial-file parity_cases/dem3saves/turn0_initial.xml \
  --orders-dir parity_cases/dem3saves \
  --native-dir /home/gopostal/.local/share/democracy3/savegames \
  --native-prefix d3_probe_replay_run42 --turns 12

# Generate a UK electoral term plus two extra years (24 no-order turns)
PYTHONPATH=. uv run python gamedrive/term_capture.py \
  --country uk --load-name turn0_initial \
  --initial-file parity_cases/dem3saves/turn0_initial.xml \
  --capture-prefix autocracy_uk_term_noorders_run42 --timeout 180

# Generate the captured policy sequence, then twelve additional no-order turns
PYTHONPATH=. uv run python gamedrive/term_capture.py \
  --country uk --load-name turn0_initial \
  --initial-file parity_cases/dem3saves/turn0_initial.xml \
  --orders-dir parity_cases/dem3saves \
  --capture-prefix autocracy_uk_term_orders_run42 --timeout 180

# Compare a no-order term capture offline
PYTHONPATH=. uv run python gamedrive/capture.py \
  --initial-file parity_cases/dem3saves/turn0_initial.xml --no-orders \
  --native-dir /home/gopostal/.local/share/democracy3/savegames \
  --native-prefix autocracy_uk_term_noorders_run42 --turns 24

# Load only, then edit one live neuron and persist a separate native save
PYTHONPATH=. uv run python gamedrive/inject_drive.py --skip-turn \
  --load-name d3_probe_turn0_copy \
  --edit-node GDP --edit-value 0.123

# Opt-in manager census and refreshed XML (all names/paths must be fresh)
PYTHONPATH=. uv run python gamedrive/inject_drive.py --skip-turn \
  --load-name d3_probe_turn0_copy --manager-audit \
  --manager-save-name d3_probe_managers_run42
```

The wrapper reads and writes the installed user's save directory:
`/home/gopostal/.local/share/democracy3/savegames`. Copy a source capture there
under a new name first; do not use an existing save name. It refuses stale
outputs by default, checks that each native XML has the v1.30.2 sections and a
single final `</xml>`, and returns nonzero on a timeout or incomplete output.
The game process is terminated by gdb, and the source capture is not modified.
The `capture.py` comparison is offline and never launches the game.

`term_capture.py` reads `term_length` from the selected mission. For UK that is
16 turns; its default `--extra-turns 8` therefore produces 24 completed native
saves. Supplying `--orders-dir` replays the available pre-turn order saves and
pads the remaining term with explicit no-order turns. The source name is only
read by the game; every loaded/capture output name is fresh and validated.

`--turn-mode sync` is the reliable default. `--turn-mode async` (or the legacy
`--gameplay-turn`) remains available for experiments, but the GUI launcher is
ptrace-sensitive. `--turn-mode direct` calls the simulation entrypoint without
the gameplay wrapper and is intended only for the single-turn no-order oracle.
Memory edits are intentionally limited to `--skip-turn`.

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
`turn0_orders.xml`; the bounded driver now applies that order save in the live
process rather than treating it as a completed-turn input. Missing order files
are replayed as no-op turns, so the supplied fixtures produce exactly twelve
native output names. `capture.py` aligns each native `<turn>` field with the
simulator snapshot and reports finance, ordinary-node, and policy residuals.

## Current limitations and safety boundaries

The load flow is driven by the game's own asynchronous path
(`SIM_LoadGame::LoadGame` -> `ProcessGameLoad` -> `OpenSavedFile` +
`LoadGameData` + `PreLoadGame`/`PostLoadGame`). Direct calls to
`OpenSavedFile()`/`LoadGameData()` from a breakpoint still hang; the probe uses
the native `LoadGame()` launcher and waits on the completion flag instead.

The game's `SIM_Gameplay::NextTurn()` thread orchestration can stall under
ptrace. The default `--turn-mode sync` replaces that launcher with the same
native `NextTurnThread(void*)` worker and manager order synchronously. The
asynchronous launcher is retained only as an explicit experiment.

The earlier loader “stall” was not repaired by rewriting XML. Static checks
showed that native `SIM_SaveGame::SaveGame` writes a complete file, while
direct `OpenSavedFile`/`LoadGameData` calls bypass the loader thread boundary.
The harness fix is to use the native asynchronous loader in a fresh process,
validate every output boundary, and never reload a probe output in-place. This
also prevents a stale or partially written file from being misdiagnosed as a
successful round trip. A native integration run remains opt-in on this host.

`--manager-audit` invokes `PreCalculateIncome`, `PreJoinParties`,
`CalculateActivists`, and `CalculateVoteRate` after load. The text audit records
party live-list membership/activist fields, poll rate, each voter's party link,
and income-host link count; `--manager-save-name` preserves the refreshed XML.
This is a live-process census, not an assertion that those pointers are
serialized into the simulator state.

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

The injector is pinned to the installed v1.30.2 ELF. Run the static preflight
after any binary update, keep raw/proprietary saves outside this repository,
and use the opt-in smoke test when a disposable installed save is available:

```
AUTOCRACY_RUN_NATIVE_INTEGRATION=1 \
AUTOCRACY_NATIVE_SOURCE_NAME=d3_probe_turn0_copy \
uv run pytest -q -m integration tests/test_gamedrive_integration.py
```
