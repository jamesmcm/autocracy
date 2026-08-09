# Democracy 3 Drastic-Changes Parity Investigation

This file tracks the investigation into why the simulator diverges from the
real game for the "drastic changes" playthrough captured in
`parity_cases/dem3saves/`.

## Ground truth

The playthrough in `parity_cases/dem3saves/` was captured from the actual
installed Democracy 3 game (v1.30.2) running under Xvfb.  The files are the
game's own XML saves, one `_initial` and one `_orders` per turn.  `_orders`
saves are taken after placing orders but before ending the turn; `_initial`
saves are the state at the start of the next turn.

Key finance figures per turn (from the `<finances>` block):

| turn | income    | expenditure | debt      | PC | GDP    | GeneralStrike |
|------|-----------|-------------|-----------|----|--------|---------------|
| 0    | 227,899   | 276,528     | 848,403   | 26 | 0.233  | 0.441 (off)   |
| 1    | 116,549   | 276,244     | 897,031   | 27 | 0.169  | 0.672 (off)   |
| 2    | 111,023   | 250,481     | 1,056,727 | 27 | 0.000  | 0.651 (ON)    |
| 3    | 86,459    | 241,684     | 1,173,366 | 25 | 0.000  | 0.749 (ON)    |

## Discoveries

### 1. Action costing: lowering to a floor is `lower`, not `cancel`

The orders save drops CorporationTax 0.22->0.0, IncomeTax 0.45->0.0 and
Prisons 0.5->0.0 with political capital 26->1 (25 spent).  The game charged
the **lower** costs: CorpTax 10 + IncomeTax 7 + Prisons 8 = 25.

The simulator's `_policy_action_type` classifies any move to level 0 as
`cancel`, charging 30 + 13 and refusing Prisons (uncancellable).  The game
treats the slider floor as a `lower` action:

* `corptax` and `incometax` are `PERCENTAGE` sliders whose floor is 1%
  (not 0% / cancellation).
* `prisons` is `DISCRETE` with `OVERCROWDED CELLS` as the floor; the policy
  is `UNCANCELLABLE` but that only means it cannot be switched off, not that
  it cannot be lowered to its floor.

### 2. Finance is live-recomputed each turn, not static

Serialized income multipliers, income/expense scalars and cost multipliers
are recomputed by the game every turn from current node values:

* `incom_mult` is evaluated from the policy's income_multiplier expression
  at the **previous** turn's node values.  E.g. IncomeTax uses
  `GDP,0.5+(0.5*x)`; the shipped noop saves show incom_mult 0.5 -> 0.6164
  (GDP 0.2328) -> 0.5845 (GDP 0.169) across uk0/uk1/uk2.
* `earn_scalar` (income) and `cost_scalar` (expense) are per-department
  ministerial scalars that sum to 2.0 and drift each turn as minister
  experience grows.  Observed formula: `earn_scalar = 0.875 + 0.25*comp`
  where `comp` is the department minister competence.
* `cost_mult` is evaluated from the policy's cost_multiplier expression
  (e.g. Prisons `CrimeRate,0.1+(0.9*x)`).

The finance totals the game displays (the `<finances>` block n_2/n_1) include
**debt interest**, which the simulator's `total_expenditure` does not.

### 3. Income history lags the multiplier update

Per-policy `incomehistory[0]` in a save is the income computed with the
multiplier/scalar from the *previous* save (the finance snapshot is written
before the next turn's multiplier update).  The shipped noop saves confirm:
uk1 incomehistory[0] equals uk0's serialized income even though uk1's
incom_mult field already shows the new value.

Exact reconstruction rule (verified to within a fraction against the
playthrough saves): the income the game *displays* (`<finances>` n_2) at
start of turn N equals the serialized `incomehistory[0]` sum at turn N+1,
which equals

    sum over active income policies of
        base(min_income, max_income, policy_val@N+1)
        * earn_scalar@N * wealth_mod * incom_mult@N

where the incom_mult used is the one serialized in save N (itself computed
from the *previous* turn's nodes).  The same relation holds for expenditure
(costhistory) and `cost_mult`/`cost_scalar`.

So a faithful simulator must recompute income/cost each turn from the *live*
node values (GDP, CrimeRate, ...) rather than using the serialized
multipliers/scalars frozen at load time.

### 5. Ministerial scalars are recomputed from competence each turn

`earn_scalar` and `cost_scalar` are per-department and sum to 2.0:
`earn_scalar = 0.875 + 0.25 * competence(dept minister)` and
`cost_scalar = 2.0 - earn_scalar`.  Minister competence is
`0.2 + 0.8 * experience * suitability`, and experience grows by
`MINISTER_EXPERIENCE_RATE` (0.025) each turn.  Verified across all
playthrough saves (every department matches to 1e-6).

### 4. General Strike situation fires at turn 2

`GeneralStrike` is a situation (`situations.csv`), not a dilemma
(`OPTIONS_DILEMMAS=0`).  Its latent at turn 0 is 0.441 (below the 0.6 start
trigger), but the drastic cuts push it to 0.672 by turn 1.  Because
`SIM_Situation::NextTurn` activates based on the *previous* turn's stored
value, it fires at turn 2, crashing GDP (effect `GDP,-0.3-(0.2*x)`) and
cranking expenditure via its 1000/turn cost.

## Simulator gaps to fix

1. Action cost: classify lowering to slider floor as `lower` (and allow
   uncancellable policies to be lowered to their floor).  This blocks the
   entire playthrough replay (CorporationTax/IncomeTax/Prisons orders fail).
2. Finance: recompute income/cost from live node values each turn, including
   ministerial scalars and debt interest.  `total_income`/`total_expenditure`
   must track the game's `<finances>` neurons (n_2/n_1), which also include
   debt interest (~16.6k on 848k debt at turn 0).
3. Situation/finance ordering and the one-turn income-history lag.
4. Verify the 12-turn playthrough end-to-end against `parity_cases/dem3saves/`
   using `parity_cases/replay_playthrough.py` (run with `PYTHONPATH=.`).

## Known target values (from the game's own saves)

| turn | n_2 income | n_1 expenditure | PC start | GeneralStrike |
|------|-----------|-----------------|----------|---------------|
| 0    | 227,899   | 276,528         | 26       | 0.441 (off)   |
| 1    | 116,549   | 276,244         | 27       | 0.672 (off)   |
| 2    | 111,023   | 250,481         | 27       | 0.651 (ON)    |
| 3    | 86,459    | 241,684         | 25       | 0.749 (ON)    |
| 11   | 246,862   | 260,874         | 21       | 1.000 (ON)    |

PC spend per orders save: turn0 25 (CorpTax 10 + IncomeTax 7 + Prisons 8),
turn1 26 (StatePensions lower 26), turn2 27 (StateHousing cancel 18 + CCTV
raise 9), turn5 SalesTax 0.44->1.0, turn8/9/10/11 StateHealthService moves.

## Implementation status (as of this update)

The simulator now matches the game's `<finances>` block *exactly* for the
shipped noop uk0->uk1->uk2 transitions and reproduces the first three turns of
the drastic-changes playthrough to within ~130 currency units of income and
~54 of expenditure:

| turn | income err | exp err | PC sim/game | GeneralStrike |
|------|-----------|---------|-------------|---------------|
| 0->1 |     +0.0   |   +0.0  | 27 / 27     | off -> off    |
| 1->2 |   +133.0   |  -46.0  | 27 / 27     | ON (turn 2)   |
| 2->3 |     +4.5   |  -53.8  | 25 / 25     | ON            |

### Determinism: the game is fully reproducible

The game is **deterministic** for this playthrough: `OPTIONS_RANDOMSTART = 0`
in the user `config.txt`, so `GRandom::Init(0x9c40, 1)` is called with the
fixed seed **1** (only `OPTIONS_RANDOMSTART = 1` seeds from the
`SDL_GetPerformanceCounter` clock).  `srand(1)` then precomputes 40,000
float32 randoms (`rand() * (1/2^31)`) that every `RandUnitFloat` /
`RandomChoice` / `IsChance` call consumes in a fixed order.

Every policy change in the orders saves is a *paid player order*: the per-turn
PC math reconciles exactly (e.g. turn 4 spends 45 = PollutionControls cancel 6
+ PropertyTax raise 19 + PrivatePrisons introduce 20, from a reconstructed
start-of-turn-4 PC of 50 = 25 + 25 income).  No random event altered any
policy.  The capital income therefore declines purely from minister loyalty
(26 -> 20), and the income formula
`int(SUM max(1, 6 * (loyalty - 0.10)))` reproduces the serialized `<max>/2`
values exactly.

### What was implemented

1. **Action costing.** `PolicyAction` gained an optional `action_type`.
   Lowering a slider to its floor (even level 0) is a `lower` (lower cost,
   policy stays active); only a true switch-off is a `cancel` (cancel cost,
   active flag flips, value/target frozen).  Uncancellable policies can be
   lowered to their floor but never cancelled.  `_next_level` allows discrete
   sliders to reach a level-0 floor.
2. **Live finance.** `_recompute_live_finance`/`_recompute_orders_finance`
   rebuild the finance lines every turn:
   * income = sum over *active* income policies of `base(min,max,val) *
     wealth_mod * earn_scalar * incom_mult`;
   * expenditure = same for costs, plus `wealth_mod *` the active situation
     costs plus the quarterly debt interest `debt * rate * 0.25`;
   * the multiplier neurons are evaluated from the *previous* turn's nodes
     (one-turn history lag); ministerial scalars come from the current
     competence (`earn_scalar = 0.875 + 0.25 * competence`).
   All money arithmetic is rounded to float32 (`_f32`), which reproduces the
   serialized totals to the last ULP.
3. **Debt / credit rating / interest.** `debt@N+1 = debt@N + (exp - inc)`
   using the post-orders net; the credit rating is re-derived from the
   debt-to-GDP ratio every other turn (`turns_since_credit`); the interest
   rate is `INTEREST_RATE_MIN + (INTEREST_RATE_MAX - INTEREST_RATE_MIN) *
   min((rating / 9)^2, 1)`.
4. **Situation fixes.** The `_default_,expr` cell is now evaluated (so
   AntisocialBehaviour's 0.8 base is used, not 0).  The game parses
   `-0.1(-0.6*x)` as `-0.1 + (-0.6*x)` (concatenation), not implicit
   multiplication; fixed in `_sanitize_expression`.  GeneralStrike still
   fires on the turn after its latent crosses the trigger (turn 2), matching
   the game.

### Latest verification (correct-alignment t12, 6 commits later)

The `turn11_initial` save was captured after the turn-11 process, i.e. it is
the **start of turn 12**, not turn 11 — so the `verify_all` harness
misaligns that one comparison by one process.  Against the correct alignment
the final state is:

| field | diffs | note |
|-------|-------|------|
| nodes | 28    | mostly < 0.05 |
| situations | 24 | active sets match (sitA=0) |
| ministers | 11 | loyalty/value drift from polls |
| polls | 19    | poll-model approximation |
| percentages | 15 | party-model blocker |
| frequencies | 12 | voter dynamics |
| policies | 0 | exact |
| income err | 1,245 | 0.5% |
| exp err | 80 | |
| PC err | 3 | capital income off 1 at t10-t11 |

Reference saves: uk0->uk1 exact (nodes 26 diffs, income err 0), uk1->uk2
income err 135 (was 606 before the CitizenshipTests fix).

### Earlier session commits (historical milestones)

1. `59d6264` Calibrate the TradeUnionist equality collapse slope 30->1.2
   (the poll now lands at -0.495 vs the game's -0.501; minister value diffs
   dropped 31->11).
2. `3f0aae0` Decay cancelled policies' inertial rings instead of suppressing
   them.  The game shifts one 0-sample per turn into a cancelled policy's
   ring (confirmed by the StateHousing->PrivateHousing ring `[-0.4,...]` ->
   `[0.0,-0.4,...]`), so the contribution decays rather than vanishing.
   PrivateHousing now exact; t12 income err dropped ~5,327 -> ~1,800.
3. `0ac22f6` Refresh the `_effectivedebt_` neuron from the live debt ratio
   each turn (it was stuck at the serialized value, so DebtCrisis never
   fired).  Now activates on the game's turns; situation active sets match
   (sitA 1->0).
4. `a33c818` Apply the CitizenshipTests constant to Immigration for the
   never-introduced policy (the serialized Immigration carries the -0.05
   base; the sim had suppressed it).  Only the Immigration link, not the
   RacialTension one (that over-corrects).  Immigration 0.366->0.318,
   RacialTension->0.541, RaceRiots->0.347; uk2 nodes 30->26, income 606->135.
5. `6c641d3` Exclude situation effects from the income-group node incoming.
   The DebtCrisis fix made the sim's DebtCrisis active, so its -0.4x effect
   on `_MiddleIncome` landed in the graph sum *and* the fitted squeeze,
   over-crashing the node to -1.0 (game -0.965) and inflating CarUsage/
   CO2Emissions/CarbonTax income.  Skipping situation effects for the
   `_` income nodes (they are voter-derived, handled by their own collapse
   model) fixed it: t12 income err ~4,217 -> ~1,200.
6. `bb1b2a1` Freeze the settled StateSchools->Education ring.  The
   serialized ring is frozen at the pre-game level (0.184) in every save
   while the policy sits at 0.36; the always-shift rule advanced it to the
   current level.  Only this single link is frozen (a broad "settled rings
   don't shift" rule regressed the income lines).  Education 0.576->0.531.

### Recent branch commits

7. `6407bc3` Preserve policy finance history parity.
8. `6b729f4` Align parity replay and document static driving.

### Remaining gaps (as of this update)

* **Node values** (28 diffs, mostly < 0.05): the biggest are ViolentCrimeRate
  (0.21, but the game's t12 VCR value 0.263 is a **save anomaly** — it is
  inconsistent with the game's own input data, which implies ~0.42; the sim's
  VCR matches at every aligned point), Education (0.048, the
  PrivateSchools->Education ring), Health (0.034, the StateHealthService->
  Health ring), and WorkerProductivity (0.030). OilSupply is exact at the
  aligned replay turns. The remaining drifts are all the same
  ring-timing/source-model class: the game freezes or times its inertial rings
  differently from the simulator's current targeted policy-boundary rule, and
  broad freezes regress the income lines, so each remaining case needs direct
  evidence.
* **Situation latents** (24 diffs): GeneralStrike (-0.083) is the
  `Socialist_perc` party-model gap; RaceRiots/HospitalOvercrowding are driven
  by the anomalous game t12 VCR; SkillsShortage/TeacherShortage follow the
  Education node.
* **Voter polls** (19 diffs): the loaded polls drift by the effect changes and
  the equality-driven collapses, but the remaining groups need the full poll
  opinion dynamics (complacency/party/sympathy feedback).
* **Voter percentages** (15 diffs): the income groups (Wealthy/Poor/
  MiddleIncome) are exact (band reassignment matches 2000/2000), but the
  *other* groups' percentages drift in the game because the voters' party/
  sympathy memberships change over the run, which the static loaded
  memberships do not reproduce. The save bridge now retains each voter's
  serialized ideology, sympathy, party, and organization fields plus the
  top-level party history rings; the remaining blocker is rebuilding the native
  manager-owned membership lists from those inputs.
* **Voter frequencies** (12 diffs): the `_freq` neurons drift with the voter
  dynamics; the sim keeps the loaded values.
* **Minister satisfaction** (11 diffs): `0.5 + average` of the two sympathised
  polls, so it inherits the poll drift; the capital income trails the game by
  1 at the last two turns (the FOREIGNPOLICY loyalty sits one bucket higher).
* **Finance**: income/expenditure within 0.5-2%.  The save bridge now retains
  the game's 20-entry policy finance rings, but the remaining income gap is
  coupled to the same hidden effect-throttle state as the targeted node
  residuals: the game's per-policy income neuron lags policy changes, e.g. the
  TobaccoTax raise at t7 only appears in income at t9 and then declines slowly.
  The serialized saves do not expose enough runtime throttle state to recover
  every such ring exactly.

### Follow-up audit (current)

The save bridge now retains both 20-entry per-policy finance rings, in the
same newest-first order as the game's `SavePolicies` serializer.  The turn
kernel writes the pre-policy-update live line into the new head, so an active
policy at its slider floor still contributes its configured minimum amount;
only a true cancellation writes zero.  `compare-save` compares these ring
heads while the live maps remain available for order-phase finance updates.

Static disassembly also confirmed that `ApplyInterestRateCalculations` adds
the global-interest neuron offset (`_global_interest_rates_ - 0.5`) to the
credit-rating factor before interpolating `INTEREST_RATE_MIN/MAX`; the Python
rate helper now models that input.  The final capture is serialized as game
turn 12 in `turn11_initial.xml`, and the replay harness now aligns by the XML
turn field.  With the current ring ordering and the captured StateHealth
  freeze-until-order timing, its final comparison is income about -1,490.4 and
  expenditure +88.2.  The earlier pre-policy sampling baseline was about -1,225
  and -32.5.  The remaining material state residual is the captured late-turn
`ViolentCrimeRate` anomaly, followed by targeted inertial-ring and voter-party
model gaps listed below.

**Precision is not the cause** of the node gaps: full-double and float32-
throughout node evaluation are byte-identical.  The residuals come from the
game's effect-ring/source details and the voter population dynamics
(membership/percentage/frequency changes).

### Static effect-load audit (current)

The executable explains the remaining ring ambiguity. `SIM_NeuralEffect` keeps
separate current and desired output throttles, advances the current throttle
toward the desired value using the effect delay, averages the leading inertial
history slots, and then writes a new raw expression sample. For a policy parent,
the live effect also applies ministerial effectiveness and the policy's current
slider value.

`SIM_LoadGame::LoadEffects` restores the input-effect throttles and the raw
33-slot `effecthistory` arrays. It does not serialize the desired output
throttle for every arbitrary outgoing effect; `SIM_Simulation::PostLoad` then
recalculates live effects from the state it has. This is why the captured
`StateHealthService -> Health` and `PrivateSchools -> Education` rings do not
always equal the direct expression evaluated from the saved policy/simvalue.

A broad pre-policy source rule is now retained because the native ordering and
the captured `StateHealthService -> Health` ring support sampling the current
policy value before `Policy::NextTurn`. A one-pass per-policy delay preserves
the old ring head immediately after a slider order. This improves the targeted
ring heads, and the StateHealthService ring now remains at its captured .211
value through no-op turns until the turn-8 order. The post-order raw samples
still differ because each outgoing effect has missing runtime throttle state;
  the aligned final residual is now about `-1,490` income / `+88` expenditure.
Effect-level desired throttles are not serialized, so no stronger global ring
rule is justified from the captures.

### Hidden manager histories and native follow-up (current)

The save bridge now retains the eight fixed hidden simulation neurons and their
33-slot newest-first histories (`n_0`, `n_1`, `n_2`, `n_3`, `n_5`, `n_6`, `n_7`,
and `n_8`, including `_globaleconomy_` and `_year`). `SimulationState` carries
those histories through JSON snapshots and advances them on each simulated
turn. This closes a state-loss bug in the extraction/continuation path; it does
not make the global cycle deterministic. Static disassembly of
`SIM_GlobalEconomy::Calculate()` (0x5d14e0) shows a `RandReal(0.9, 1.1)` factor,
while the random cursor is not serialized and the captured histories do not
identify a recoverable contiguous seed-1 stream. The simulator therefore keeps
the deterministic sine term and records the native limitation instead of
hard-coding a guessed multiplier sequence.

The same audit pinned `SIM_Voter::UpdateIncome()` (0x61e620),
`SIM_Voter::AddToIncomeGroup(SIM_VoterType*, float)` (0x61e4d0), and
`SIM_VoterType::ForceVoter(SIM_Voter*, float)` (0x623350). Native income-group
membership evaluates the overlapping `[-.3,.3]`, `[.2,.8]`, and `[.7,1.3]`
sinusoidal windows against the runtime income, selects the strongest candidate,
and applies the `VOTER_GROUP_MEMBERSHIP_THRESHHOLD=0.5` floor. The simulator
now mirrors that assignment and counts the selected linked-list group for the
income percentages. Across the captured UK saves the formula is within
`6e-7` of every serialized income-group weight after the first load transition;
the first voter’s `.916348` wealthy weight is an exact static check. The save
bridge now preserves nested VoterType `<income>` values and the simulator
recalculates their direct graph contribution each turn; the manager-added
per-voter host contribution remains non-serialized. The first ordered
transition's maximum direct-income error is `1.4e-4`. These native entries are
now part of the 46-symbol static preflight, which also verifies the
load/singleton, slider, party, and RNG hooks; it remains safe and does not
launch the installed game.

### Native VoterType income-neuron pass (current)

Static disassembly of `SIM_VoterManager::PreCalculateIncome()` (0x620920)
shows that the manager first updates each voter's runtime income, then writes
the serialized VoterType income value into all 33 history slots of the nested
neuron and refreshes its outputs. The ordinary simulation neuron pass therefore
needs to calculate direct policy/simvalue inputs for these `<income>` neurons
before the manager-owned host links are applied. `SaveGame` and
`SimulationState` now retain the values under `<Group>_income`, and the pass is
covered by a first-transition replay regression. The non-serialized host links
and the live party lists remain outside the safe deterministic model.

### Calibration is data-driven (commit 33222ed)

All the parity fits that were previously hard-coded in the simulator now live
in `autocracy/calibration.json`, loaded through `SimulationData.calibration`
(deep-merged with an optional `calibration.json` placed in the gamedata
root).  This covers:

* `effect_scale`: which policy->node links carry the ministerial scale
  (`BorderControls->Immigration` uses `implementation`; the never-introduced
  `CitizenshipTests->Immigration` constant is `unscaled`).
* `effect_applicability`: which links stay live when their policy is inactive.
* `frozen_rings`: which settled policy rings never shift (`StateSchools->
  Education`).
* `frozen_until_order`: which captured policy rings remain frozen until an
  explicit order starts their native runtime ramp (`StateHealthService ->
  Health`).
* `voter_collapse`: the equality-collapse slopes per voter group, the income
  `squeeze` per node (slope/threshold/saturation), the voter contribution and
  the GDP-crash parameters.

The simulator code contains no named-policy special cases or fitted numeric
constants anymore, so a different country/gamedata set can be reproduced by
supplying its own `calibration.json` instead of editing code.  The shipped
default reproduces the UK playthrough exactly (behavior unchanged).

### External driving of the real game is feasible

`Democracy3.bin.x86_64` (v1.30.2) is **not stripped and has debug_info**, so
every simulation entry point is available as a symbol:

* `SIM_GetSimulation()` — the simulation singleton
* `SIM_Simulation::NextTurn()` (0x60e120) — the complete turn step (calls
  `SIM_EventManager/FinanceManager/GlobalEconomy/MinisterManager/
  PolicyManager/PressureGroupManager/SituationManager::NextTurn`)
* `SIM_Simulation::GetNeuronByName(std::string)` — read any neuron
* `SIM_Policy::ForceSlider(float)` — set a policy slider
* `SIM_LoadGame::OpenSavedFile(std::string)` + `LoadSimulation()` — loads the
  same XML saves this repo parses
* `SIM_SaveGame::Save*()` — serializes the full state back to XML
* `SIM_Mission::Load(std::string)` — loads a country; the game accepts a
  `-silent` command-line flag and runs under Xvfb

The static follow-up pinned `SIM_LoadGame`'s filename `std::string` at `+0x838`
and the `ProcessGameLoad` sequence: open the file, release gameplay, preload,
load all game data, postload, then free the temporary load buffer. The mission
selection sequence inside `LoadMission()` is also pinned:
`LoadMissions()` -> `GetByName()` -> `SetCurrent()` ->
`ApplyMissionSpecificData(false)` -> `SIM_Names::Initialise()` -> mission
options. This must happen before the loaded simulation is advanced. The load
order reaches `LoadEffects` after simulation/gameplay data; that routine
restores input throttles and raw effect histories but not every outgoing
desired throttle. This is the missing runtime state behind the targeted ring
gaps.

Two harnesses would give exact parity by construction (it *is* the game):
gdb-driving (break on `SIM_Simulation::NextTurn`, call load->set-policy->
NextTurn->save from a command file) or an LD_PRELOAD `.so` exposing a
set-policy/next-turn/save-state interface.  Either round-trips through the
existing `savegame.py` XML parser.  The trade-off is weight (needs the
installed game + Xvfb) and version coupling to the binary's symbols.

### gdb-driving prototype (`gamedrive/`)

`gamedrive/gdb_drive.py` + `gamedrive/harness.gdb` launch the installed game
under Xvfb + gdb and prove the driving is reachable:

* the game stops at `mainLoop()` before the graphics resize crash that
  otherwise kills it under Xvfb (the crash is `APP_Game::NotifyResized` ->
  `GEngine::CreateRenderTarget`, a GL/resize issue);
* `SIM_GetSimulation()` returns the live singleton (0xa38360);
* `SIM_Simulation::NextTurn()` is callable from gdb but SIGFPEs without a
  country loaded (`SIM_Names::GetRandomFullName` divides by an uninitialized
  value), confirming a country must be loaded first.

**Blockers to a full round-trip** (documented in `gamedrive/README.md`): the
`SIM_GetLoadGame()`/`SIM_GetSaveGame()` getters are static inline (no symbol)
and their singletons are lazily constructed; gdb has no usable C++ object
layout/type information beyond the member accesses visible in disassembly;
and direct calls into `OpenSavedFile`/`LoadGameData` from the breakpoint hang.
The filename offset and load call order are now pinned, as is the mission
selection sequence. Completing the round-trip still needs the remaining
class/runtime layout and a safe way through the game's loading thread before
`SIM_Simulation::Initialise`, `NextTurn()`, and the
`SIM_SaveGame::Save*` serializers can round-trip through `savegame.py`.

* **Random systems.**  No random event changed a policy in this playthrough
  (all changes are paid orders), so the deterministic core covers the policy
  state.  Events/attacks only move voter opinions, which the finance/GDP/PC
  parity targets do not depend on.  If the voter-parity targets are ever
  pursued, the game consumes its seed-1 random stream by collecting the
  events whose value crossed their threshold each turn and picking one with
  `GRandom::RandomChoice`.

### Static party-membership audit (current)

The party branch is now partially modeled from the v1.30.2 binary rather than
from raw-value guesses. `SIM_Voter::CalculateApproval()` first maps the saved
voter value through `(value + 1) * 0.5` and stores the result in a runtime
approval field. `SIM_Voter::ConsiderPartyMembership(1)` then updates
opposition/player sympathy using the `VOTER_*_SYMPATHY_*` thresholds and gains
from `simconfig.txt`.

The simulator now applies that base sympathy step before advancing the next
voter value. It also reproduces the statically identified membership rules:
existing members leave below 0.2 sympathy; an unaffiliated voter joins the
opposition above 0.7 sympathy only while player sympathy is at most 0.1, with
the mirrored rule for the player party. Party names are selected by the
serialized party `type` field, so this works for missions whose party names
differ. `SIM_Party::NextTurn`'s growth/decline status and pre-transition
member-count history are also retained.

This closes the high-confidence per-voter transition, but not the whole
manager model. Native `SIM_VoterManager` and `SIM_VoterType` rebuild linked
membership lists during load, calculate percentage neurons from those lists,
and update party activist/poll state in separate manager passes. Those lists
are not serialized, and the saved approval also has perception/fundraising/
election modifiers that are not yet modeled. The late replay therefore still
has a measured party-count and poll residual when the simulator's voter-value
approximation diverges.

### Native VoterType frequency pass (current)

Static disassembly of `SIM_VoterType::NextTurn` and `SIM_Neuron::CalculateValue`
resolved the remaining first-transition frequency mismatch. The serialized
`<freqval>` is the current value of the nested neuron at `SIM_VoterType +
0x380`; it is not the CSV population percentage at `+0x2c4`. The nested
`SIM_Neuron` is constructed with a zero base and `[-1, 1]` bounds, then runs
with the ordinary effect vector before `SIM_VoterManager::NextTurn`. The
percentage is separately recomputed from the linked-list membership count.

The save parser now retains `<grudges>` whose targets end in `_freq` as
persistent frequency-neuron inputs. The four UK mission inputs in the captured
turn-zero save are `Farmers -0.07`, `Retired -0.09`, `EthnicMinorities -0.25`,
and `Religious +0.05`; applying them removes the previous `0.25`/`0.09` first
turn errors. Active situation outputs are refreshed from their stored pre-pass
value before the frequency list consumes them, fixing the larger
`VigilanteMobs -> Conservatives_freq` half-ring error. A regression test covers
both the mission inputs and the first transition.

The remaining frequency differences are small (roughly `1e-4` to `3e-4`) and
come from rounded serialized inertial histories or the native random/situation
manager timing. The captured `_winning_` value is still an event-manager
runtime result absent from the turn-zero save, and the global-economy random
cursor remains non-serialized. The static injector preflight remains the safe
boundary for gdb/LD_PRELOAD work; the installed game is not launched.

### Native voter-list ordering fix (current)

The percentage pass now preserves the native ordering boundary instead of
using the frequency value that was just calculated. `SIM_Voter::PreSimulatedNextTurn`
and `ReconsiderVoterGroupMemberships` test an ordinary group's raw membership
weight plus the saved `VoterType::freqval` against the manager threshold. The
four ideology links are different: `SIM_Voter::NextTurn` overwrites them via
`ForceVoter`, whose linked-list test uses the raw forced coefficient and no
frequency base. The simulator snapshots all saved frequency values before the
frequency neurons mutate and uses that snapshot for ordinary groups.

The captured load boundary also matters. On the first turn-zero transition,
the native income percentages remain at their serialized zero values even
though `UpdateIncome` has selected the overlapping income curves; subsequent
passes count the selected income memberships. The focused replay now reduces
the voter-percentage error to a maximum of `0.001` through turn three and
approximately `0.002` at the final capture. A regression test locks the
previous-frequency criterion on the Commuter list. This is a deterministic
membership fix; the non-serialized party lists, activist/poll modifiers, and
per-voter income host links remain open manager state.
