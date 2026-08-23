## Simulation Architecture

This document explains how the Democracy 3 simulator embedded in this repository interprets the original game data and evolves the world state turn by turn. Refer to the Python sources alongside this file for concrete details:

- `autocracy/data_loader.py` – parsing of CSV/INI assets
- `autocracy/simulator.py` – state machine, DAG traversal, and action validation
- `main.py` – CLI façade that exercises the simulator

All paths below are relative to `gamedata/data/`, which mirrors the Democracy 3 installation.

---

### Input Assets

| File/Folder | Purpose | Notes |
|-------------|---------|-------|
| `simulation/simulation.csv` | Core statistics (“nodes”) such as GDP, CrimeRate, CO₂, etc. | Each row defines metadata (`name`, `guiname`, `description`, `min`, `max`) and then a list of `source,equation,inertia` triplets representing inbound influences (edges) for the DAG. |
| `simulation/votertypes.csv` | Voter happiness + membership nodes. | Similar structure: default support (`default`), linked-list membership seed (`percentage`), and `Influences` columns that create edges. Membership nodes (`<Group>_freq`) are synthesized automatically as zero-base native frequency neurons; nested `<income>` neurons are restored from saves and their direct links are evaluated by the simulator. |
| `simulation/policies.csv` | Policy sliders that the player can adjust. | Columns include slider identifier (`slider`), action costs (`introduce`, `cancel`, `raise`, `lower`), department, `mincost`/`maxcost` (and analogous income columns), `cost multiplier` / `incomemultiplier` expressions that now get parsed into live budget modifiers, implementation lag, and `#Effects` columns describing outbound edges toward other nodes. |
| `simulation/sliders.csv` | Slider metadata. | Declares whether a slider is `DISCRETE` (enum-like) or `PERCENTAGE` (continuous). Discrete labels (`NONE`, `LOW`, `MEDIUM`, …) provide normalized action levels, while the executable still stores the target as a float. Percentage sliders use continuous 0 – 1 values. |
| `simulation/situations.csv` | Dynamic modifiers (“situations”). | Each row defines trigger thresholds, optional upkeep cost, a list of input effects (how existing nodes/policies influence the latent value), and output effects (how the active situation feeds back into the DAG). Inputs/outputs can also specify inertia so their impact ramps over multiple turns. |
| `missions/<country>/<country>.txt` | Country-specific configuration. | Sections include `[config]` (currency, demographics), `[options]` (game modifiers), `[stats]` (display text), and `[policies]` (initial slider positions normalized 0 – 1). |
| `missions/<country>/overrides/*.ini` | Graph overrides per country. | Inside `[override]` blocks: remove (`Equation = DELETE`) or inject new edges between hosts and targets, optionally overriding inertia. |
| `data/simconfig.txt` | Global tuning constants. | Political capital accrual, complacency rules, credit rating thresholds, etc. The simulator currently uses the capital-related fields. |

All CSV files are read with `latin-1` encoding to match the original game data. Equations occasionally contain malformed constructs (e.g., `0.0.5` or stray parentheses); the simulator normalizes them before evaluation.

---

### DAG Construction

1. **Nodes**: All entries from `simulation.csv` and `votertypes.csv` become nodes. For voter groups, an extra `<Group>_freq` node represents membership share.
2. **Policies**: Every policy from `policies.csv` is added as a node of kind `policy`. Their slider identifiers are used to look up slider metadata when validating moves.
3. **Edges**: Each parsed effect triple (`source`, `target`, `expression`, optional `inertia`) becomes a directed edge. Policy effects are appended after the base DAG to ensure policy nodes influence the same graph.
4. **Overrides**: Country overrides remove or add edges on top of the shared DAG before a new playthrough begins.

NetworkX represents the DAG in memory, which makes it easy to walk predecessors for any node when applying updates.

---

### Equation Grammar

Effect expressions live in the CSV files and are evaluated safely inside `autocracy/simulator.py`:

* Allowed operations: `+`, `-`, `*`, `/`, `^` (converted to `**`), unary `+/-`, and modulo.
* Allowed helper functions: `min`, `max`, `abs`.
* Variables:
  * `x` – the normalized value of the source node/policy (0 – 1).
  * Any other identifier (e.g., `GDP`, `TobaccoUse`) is looked up dynamically in the current state when present, so cross-node references inside equations continue to work.
* Broken patterns such as `0.0.5` or unmatched parentheses are automatically sanitized before `ast.parse` validates the expression tree. If an expression still fails, the simulator surfaces the exception.

Common behavioural patterns handled correctly:

- **Linear**: `0.05 + (0.12*x)` – straight proportional influence.
- **Polynomial / threshold**: `0.1*(x^6)` – higher-order sensitivity near extremes.
- **Piecewise via `max/min`**: e.g., `max(0, (x - 0.3))`. These are fully supported through the allowed built-ins.
- **Frequency/membership updates**: Equations targeting `<Group>_freq` nodes are treated like any other effect, so they participate in the same DAG pass. The native nested frequency neuron has a zero base and `[-1, 1]` bounds; the CSV percentage is the initial linked-list population share, not that neuron's base. Persistent savegame `CreateGrudge` inputs targeting a frequency are added separately on every pass.

---

### Turn Progression

1. **Initial State** (`get_initial_state`):
   - Node values are seeded from their defaults.
   - Policy sliders are filled with mission-provided levels, clamped to `[0, 1]`.
   - If a matching baseline save exists (`gamedata/saves/<country>0.xml`), its `simvalues`/`policies` override the defaults so the simulator starts from the exact in-game conditions.
   - Situation latent values are evaluated from their inputs to determine which situations start active (respecting their start/stop trigger thresholds).
   - Each serialized inertial link is restored as a raw 33-slot effect ring. On load, the live effect is the average of the leading `inertia` slots, with policy links additionally scaled by ministerial effectiveness.
   - The loader also restores hidden global neurons and their serialized 33-slot histories, nested VoterType `<income>` values, voter histories, serialized party metadata/history rings, full per-voter party/sympathy inputs (`<milit>`, `<invotech>`, `<insocial>`, `<inliberal>`, `<oppsymp>`, `<playsymp>`, `<party>`, and `<orgs>`), policy runtime fields, delayed policy throttles, ministerial effectiveness, situation state, finance snapshots, and the `<inherited>` simvalue block (the values from two turns ago). Policy runtime distinguishes the current policy-neuron value (`<val>`) from the requested slider target (`<targ>`); the current value moves toward the target by a fixed `1 / implementation_time` step each turn. Hidden histories are advanced newest-first in snapshots; the native global-economy random cursor is not serialized, so its per-turn multiplier remains an explicit parity boundary. The simulator also carries calibrated outgoing-ring start state through action and JSON snapshots; the captured StateHealthService ring stays frozen until its policy is explicitly ordered. The data-driven calibration records the native `BorderControls` minister scale and the installed parser's `Unemployment -> Immigration` offset; no hidden final-node correction is applied to normal simulator runs.
   - Political capital is restored from `<politicalcapital><points>`. The
     simulator also retains the baseline active-minister accrual recovered
     from the initial save; for the shipped UK start this is 26 points per
     turn with a 52-point cap. Countries without a baseline save use the
     configured minister-count fallback. Loyalty-aware runs use a zero-floor
     contribution for each active minister; deterministic below-threshold
     removal is opt-in because the native resignation roll is not serialized.

2. **Action Phase** (`apply_actions`):
   - Each `PolicyAction` contains a `policy_name`, a normalized `delta`, and
     an optional `action_type`.
   - Costs depend on what is happening:
     * **Introduce** (inactive → active): use `introduce_cost`.
     * **Cancel** (a true switch-off, active flag flips): use `cancel_cost`.
       The neuron value and slider target are left where they are.
     * **Raise / Lower** (slider move): use `raise_cost` / `lower_cost`.
       Dragging a slider down to its *floor* — even level 0 — is a `lower`,
       the policy stays active.  Uncancellable policies can be lowered to
       their floor but never cancelled.
   - Validation steps:
     - Policy must exist and enough political capital must remain.
     - Slider metadata determines what levels are legal.
       * **Discrete** sliders use the enumerated labels for action suggestions and validation. The executable still stores a normalized target float, so a captured UI drag can land between labels; the simulator accepts such normalized targets for controlled parity experiments.
       * **Percentage / continuous** sliders can take any value within `[0, 1]`.
     - Attempting to move beyond the boundary (no actual change) raises an error.
   - Capital is deducted per accepted action; the finance lines are
     recalculated against the post-orders active policy set (the game
     recomputes them when the player confirms the orders).
   - Native `gamedrive` replays opt into `native_order_runtime`: a complete
     order save is applied as one batch, the visible policy value and input
     throttle jump to the requested target, and the next debt roll consumes
     the previous policy-history level. Existing policies retain their
     serialized finance multipliers; a newly introduced policy seeds a
     midpoint history sample and evaluates its new multiplier from the
     current node snapshot. Direct interactive calls retain the delayed
     current-versus-target behavior by default.

3. **End of Turn** (`process_end_of_turn`):
   - Advance ministers first, then policy runtime: minister experience/effectiveness is updated before policy implementation fractions, while current policy values and their policy-input throttles move toward requested targets by the fixed `1 / implementation_time` step. The action-phase policy map therefore remains the current `<val>` until this phase.
   - Roll finance-manager debt before evaluating the effect vector. The post-roll debt feeds `_effectivedebt_`, credit-rating updates, and situation inputs for the pass.
   - Advance the effect vector using the pre-turn policy values as source snapshots. Direct links use the current source throttle; inertial links shift a raw expression sample into their ring and average the leading window. A policy-owned ring retains its old head for the first process after a target change, then samples the pre-`Policy::NextTurn` value while the policy ramps. Saved rings contain raw samples, not minister-scaled live values.
      The executable writes a fresh ring sample for every applicable inertial link each turn — including a settled policy (a lowered IncomeTax keeps shifting 0-samples, and a raised TobaccoTax keeps shifting −0.8 samples until the leading-window average converges). The simulator mirrors this ordering and the observed one-pass target-change delay where the saved state is sufficient.
      `SIM_LoadGame::LoadEffects` restores raw effect histories and input throttles, but not the desired output throttle for every outgoing effect. The captured StateHealthService -> Health ring is frozen through no-op turns and begins ramping only after its explicit order; that timing is represented by the data-driven `frozen_until_order` calibration. Its post-order numeric samples, and the targeted Education/Health/WorkerProductivity residuals, still depend on outgoing throttle state that cannot be reconstructed from the serialized policy or simvalue alone. OilSupply is exact in the current aligned replay.
   - Two equation-level parity observations are data-driven in `calibration.json`: `BorderControls -> Immigration` carries the active FOREIGNPOLICY minister scale, and the installed parser evaluates `Unemployment -> Immigration` with a −0.06 offset relative to the CSV expression. The first two shipped no-op transitions then match every ordinary node within 0.001.
    - Refresh the manager-owned hidden global neurons (`_global_socialism`, `_global_liberalism`, `_security_`, and `_winning_`) and their outputs before the ordinary neuron list. Active situation outputs are also refreshed from the stored pre-pass situation value before downstream neurons consume them.
    - Walk ordinary simulation nodes in data order. Each node is `default + Σ current incoming effects`, clamped to its declared `[min, max]`; after a node is calculated, its direct outgoing links are recalculated immediately, matching `SIM_Neuron::CalculateValue`.
    - Derive the income-group nodes (`_LowIncome`/`_MiddleIncome`/`_HighIncome`) from the voter population: each of the 2000 loaded voters' value drifts by the change in the policy + economy-node effects on its voter types, and native `SIM_Voter::UpdateIncome` reassigns income groups using the three overlapping sinusoidal windows with the configured 0.5 membership floor. The income node is the graph sum plus a contribution from that income group's voters, and the middle-income node additionally collapses on the middle-class squeeze (Equality below ~0.3). Income-group percentages count the selected native group membership rather than applying a raw `inincome` band cutoff.
    - Evaluate the nested VoterType income neurons from their direct graph inputs, then evaluate the native VoterType frequency neurons, including persistent frequency grudges, before refreshing the voter manager. The manager's later per-voter income host links are not serialized. Ordinary group percentages use the previous saved `<group>_freq` base for the membership test; the four ideology pairs follow the raw `ForceVoter` weights, and the first income pass retains the captured zero-percentage startup state. Active situation outputs targeting voter types are included in the same current-versus-previous effect delta as policy and ordinary-node links. Then advance the evidence-backed base party step from `SIM_Voter::ConsiderPartyMembership`: approval uses `(value + 1) * 0.5`, sympathy uses the shipped `VOTER_*` thresholds/gains, membership changes use the 0.2/0.7 thresholds and cross-party guard, and serialized party member/activist histories are shifted in the native order. Native manager-owned party lists and activist-count/poll modifiers remain a documented parity boundary.
    - Recompute situation latent values from their input links and retain the manager’s start/stop decision for the pass. Situation outputs are gated by the active set and participate in the same effect vector.
    - Add the active-minister political-capital accrual and clamp at the
      corresponding `POLITICAL_CAPITAL_MAX_MULTIPLIER` cap.  When the
      `SimulationConfig.minister_loyalty` flag is enabled the accrual is
      re-derived each turn from the ministers' loyalty (which itself drifts
      through `SIM_Minister::ProcessLoyalty` as ministers gain/lose loyalty
      based on their satisfaction with the enacted policies — a minister's
      satisfaction is `0.5 + average` of the two voter groups they
      sympathise with, and the gain/drop thresholds interpolate by the
      minister's experience); contributions have a zero floor. Otherwise the
      accrual stays at the value loaded from the save. The explicit
      `SimulationConfig.minister_resignations` mode removes below-threshold
      portfolios and applies `MINISTER_RESIGNS_LOYALTY_CHANGE`; it is not part
      of the default deterministic replay because native resignation is
      probabilistic and its RNG cursor is not serialized.
    - Recompute the finance lines from the advanced policy values and the
      advanced ministerial scalars, with the multiplier neurons evaluated at
      the previous turn's nodes and the debt interest charged on the
      freshly-rolled debt. Native order replay uses its delayed
      policy-history sample only for the debt preview; the completed save's
      displayed totals remain live values. A department without an active
      minister uses the data-driven `minister_fallback.competence` (the
      captured UK value is `0.25`, yielding earn/cost scalars
      `0.9375`/`1.0625`).
   - Electoral countdowns follow the headless native turn worker: a positive
     countdown decrements, while a zero countdown starts the next serialized
     interval at `term_length - 1`. The headless worker does not invoke the GUI
     result screen, so it does not change the current term. Call
     `resolve_election(state, data=None)` when the countdown is zero to count
     native-style expected turnout, persist a deterministic rounded result and
     vote totals, increment the term, reset the countdown, and apply the
     native player-win loyalty boost. `forecast_election(state)` exposes the
     unrounded player/opposition/absent expectation used by election oracles;
     `resolve_election_if_ready` is the no-op-safe bridge used by the oracle
     agents at every observed boundary.

In a corrected UK simulator experiment (seed `20260813`, beam 6, horizon 5,
two actions per turn, 16 sampled candidates, 64 legal batches, and a
15-second operational budget per decision), the passive control lost its first
election at turn 16 with 26 player / 1,142 opposition / 832 absent and a 6.05%
poll. The election oracle recovered from a lagged turn-5 dip, crossed to a
positive expected margin at turn 13, and won the first election with 865 player
/ 281 opposition / 854 absent and an 81.85% poll. A full-roster 120-second
search selected the same first batch (`TelecommutingInitiative` plus
`FoodStamps`) and completed two lookahead layers after 589 simulator branches.
The production example uses the same search with a 900-second wall-clock
budget; beam states are retained incrementally so a long search does not grow
memory with every discarded voter population.

4. **Random Systems** (`SimulationConfig`):
   - `process_events`, `process_dilemmas`, `process_attacks` and the
     pressure-group threat loop are implemented in `autocracy/events.py`,
     driven by the shipped `events/*.txt`, `dilemmas/*.txt`,
     `attacks/*.txt` and `pressuregroups.csv` data.
   - Every system defaults to **off**. `SimulationConfig` carries
     `random_events`, `dilemmas`, `pressure_group_events`, `assassinations`
     and a `random_seed`; a config with everything off is a bit-for-bit no-op,
     which is what the deterministic save-parity runs rely on.
   - When enabled, event trigger chances are approximated from each file's
     `_random_` base constant plus its evaluated condition influences, rolls
     use the seeded RNG, and `CreateGrudge(...)` scripts apply one-shot
     opinion shifts to voter values/simvalues while frequency targets are
     retained as persistent nested-neuron inputs. Dilemmas
     fire when their latent influence sum crosses 0.5 and resolve with a
     seeded option choice. Plots and assassinations require the matching
     extremist pressure group's support to reach the file's `MinStrength`.
   - The executable keeps unstored random-system state (event cooldowns,
     group strength), so live-game timing is not reproduced exactly;
     enabled runs are reproducible through the seed.
   - CLI: `uv run main.py simulate --turns 4 --events --dilemmas
     --pressure-groups --assassinations --random-seed 42` (the `agent`
     command accepts the same flags).

### Full-node parity audit

The drastic-change replay compares every ordinary node exposed by the captured
save's `<simvalues>` block, not only headline metrics such as GDP and Health.
All 40 observed nodes are present in the simulator at the available checkpoints
(game turns 1, 2, 3, and 12). At the final checkpoint, 31/40 are within 0.01
of the game and 39/40 are within 0.05. The largest raw difference is the
captured `ViolentCrimeRate` value (0.1976); the game's turn-12 value is
inconsistent with its own inputs and is therefore recorded as a save anomaly.
Excluding that anomaly, the largest ordinary-node residual is Education at
0.0490, followed by WorkerProductivity and Health.

Voter values, voter percentages/frequencies/incomes, hidden global neurons,
and situation latents are serialized in separate manager-owned sections rather
than the ordinary `<simvalues>` block. The long-run audit can feed those
serialized checkpoint values, together with grudges, effect histories,
policy-manager runtime, and finance-manager state, back into the corresponding
turn boundary. This closes the serialized sections exactly without making
native checkpoint data an input to ordinary simulator runs. The remaining
ordinary-node differences are therefore isolated to continuous effect/source
state and a small number of native save anomalies.

The simulator returns a new state object, but its core update is intentionally ordered rather than fully synchronous: direct effects can cascade to later nodes in the same pass, as they do in the game. The 33-pass `PreCalcCoreSimulation` settling routine used by the executable during initialization is distinct from the normal one-pass turn path.

### Native ground-truth bridge

`gamedrive/` now contains a version-pinned, headless gdb/`LD_PRELOAD` probe for
the installed Democracy 3 v1.30.2 binary. It uses the game's asynchronous
`SIM_LoadGame::LoadGame` path, waits for the native loading-complete flag, and
serializes through `SIM_SaveGame`. The reliable default turn mode invokes the
game's `NextTurnThread(void*)` worker synchronously; this preserves the native
manager order while avoiding the GUI thread orchestration that can stall under
ptrace.

Run `make -C gamedrive` and `uv run python gamedrive/preflight.py` before using
the probe. `inject_drive.py --turn-mode sync --orders-dir parity_cases/dem3saves`
translates pre-turn slider/implement/cancel saves, inserts no-op turns for
missing captures, and writes a bounded native capture with fresh names.
`gamedrive/capture.py` replays the same twelve turns through the simulator and
compares the native XML offline. `--skip-turn --edit-node NAME --edit-value
VALUE` performs an explicit, process-local write to the live neuron value slot
and saves a separate edited copy. Always copy source captures to fresh names
and keep raw outputs outside the repository.

For long-run parity, `gamedrive/term_capture.py --country uk` reads the UK's
16-turn mission term and, by default, captures eight additional turns (24
completed saves). It can replay the existing captured orders and pad their tail
with no-order turns. The offline comparator accepts `--no-orders --turns 24`
for a clean baseline sequence.

The direct boundary gives a same-input oracle: loading `turn0_initial.xml`
reproduced the parser's extracted load snapshot exactly. A no-order native
turn matched simulator finance, policies, and effect throttles exactly; across
all observed sections the ordinary-node residual was 14/40 nodes (maximum
0.021644), while voter percentages remained the largest gap (maximum 0.496).
The captured `turn1_initial.xml` includes `turn0_orders.xml`, so it is not a
valid comparison for that no-order native run. The bounded order driver keeps
that distinction explicit by applying the pre-turn save before its native
worker and aligning each output by its serialized `<turn>` field.

### Multi-term native audit

The term-aware driver reads the UK's 16-turn electoral term and generated two
24-turn chains from the unchanged `parity_cases/dem3saves/turn0_initial.xml`:

```text
autocracy_uk_term_noorders_chain2_20260810_step{1..24}_turn1.xml
autocracy_uk_term_orders_chain_20260810_step{1..24}_turn1.xml
```

The first chain is no-order throughout; the second applies the captured policy
sequence through turn 12 and then pads the tail with no-order turns. Every
checkpoint passed native-save validation and serialized turns 1–24. Run the
complete offline audit with:

```bash
PYTHONPATH=. uv run python gamedrive/term_audit.py \
  --initial-file parity_cases/dem3saves/turn0_initial.xml \
  --native-dir /home/gopostal/.local/share/democracy3/savegames \
  --native-prefix autocracy_uk_term_noorders_chain2_20260810 --turns 24

# The captured-order chain uses the same audit with its pre-turn order files.
PYTHONPATH=. uv run python gamedrive/term_audit.py \
  --initial-file parity_cases/dem3saves/turn0_initial.xml \
  --native-dir /home/gopostal/.local/share/democracy3/savegames \
  --native-prefix autocracy_uk_term_orders_chain_20260810 --turns 24 \
  --orders-dir parity_cases/dem3saves
```

The audit compares finance, ordinary and hidden nodes, situations, voter
aggregates and individual values, policy current/target/runtime fields, effect
histories, party metadata, active situations, and serialized minister/election
fields at every turn. Policy targets are exact in both chains. The quiet chain
loses the TAX minister at turn 15; the audit now applies that serialized native
roster and the measured missing-minister fallback, including the remaining
loyalty penalty. The policy chain retains TAX, showing that the native
resignation decision is probabilistic. `--minister-resignations` remains a
legacy deterministic fallback for replays without native roster checkpoints.
Live party lists, activist counts, income-host links, and some poll modifiers
are manager-owned pointers rebuilt in process, so the audit reports them as
runtime boundaries rather than pretending they are serialized simulator inputs.

The same launcher also supports a 128-turn no-order stress chain. The
2026-08-10 run produced 128 validated terminal saves plus 128 intermediate load
saves from `turn0_initial.xml`, with zero policy-target differences. The
expanded audit supplies each checkpoint's serialized minister roster, active
situations, grudges, hidden values/histories, voter/party/poll state, effect
histories, policy-manager runtime, and finance-manager state. Election
countdown/current-term deltas, active roster/situation membership, hidden and
situation fields, serialized voter fields, policy runtime, finance totals/debt,
and effect-history comparisons are now zero at every checkpoint.

The audit now restores each checkpoint's serialized `<simvalues>` after the
model pass and before the next continuation turn. The ordinary checkpoint
comparison is therefore zero at all 128 turns, while the pre-overlay model
snapshots remain visible in `ordinary_max_delta`: maximum 0.1156 at turn 54
(`PrivateHealthcare`), mean per-checkpoint maximum about 0.0940. The persistent
`PrivateHealthcare` value is a native save anomaly because it disagrees with
that save's own serialized effect histories. The simvalue schedule is an
explicit audit-only oracle and is not part of the default simulator.

---

### Actions API

Two functions make the simulator easy to integrate with AI agents:

| Function | Description |
|----------|-------------|
| `list_available_actions(state, data=None)` | Returns `PolicyActionOption` objects describing every feasible single-step policy adjustment, including the action type (`introduce`/`cancel`/`raise`/`lower`), political capital required, and the policy’s implementation delay. Suggestions use a default 5 % normalized step, while the state retains separate current and desired policy values. |
| `apply_actions(state, actions, data=None)` | Applies validated actions, raising informative errors if the move is impossible (unknown policy, insufficient capital, invalid target, or no change at boundary). It changes the desired slider target and spends capital; the current policy value remains unchanged until `process_end_of_turn` advances implementation. |

The CLI mirrors these endpoints:

```bash
uv run main.py describe --verbose
uv run main.py actions --country uk
# or, after pre-adjusting a policy
uv run main.py actions --country uk -p IncomeTax:-0.05
```

### Agent Loop + State Persistence

`autocracy/agent.py` contains `BaseAgent`, which wires together the usual turn loop:

1. Inspect the current state.
2. Call `list_available_actions` to see legal moves.
3. Optionally apply one or more validated actions.
4. Advance the DAG via `process_end_of_turn`.

`PassiveAgent` inherits from the base scaffold and simply never spends political capital; real agents can override `choose_actions` to implement policies or learned behaviour.

`SimulatorOracleAgent` is a best-case beam-search implementation. At each
depth it evaluates a no-op and legal policy-action batches by calling
`apply_actions` followed by the real `process_end_of_turn` transition and
`resolve_election_if_ready` boundary. The winning plan is ranked by a
caller-supplied `objective(state)` (the default combines headline metrics and
poll rate), but only its first-turn batch is returned; the agent replans from
the resulting observed state. `beam_width` controls retained branches,
`search_horizon` controls forecast turns (`None` means the next election),
`candidate_limit=None` enables exhaustive single-option enumeration, and
`max_actions_per_turn`/`batch_candidate_limit` control legal combinations in a
turn. Supplying a `random_seed` samples the bounded policy roster without
replacement at each beam node, giving a reproducible Monte-Carlo/beam hybrid
across long runs; without a seed, bounded selection is deterministic.
`time_budget_seconds` can stop a large search at a wall-clock deadline; the
result reports `timed_out`, `completed_depth`, and `elapsed_seconds` and still
contains a safe first-turn simulator transition.

An election loss is terminal: losing branches are removed from the beam, and
`OracleElectionLoss` is raised if no legal continuation can win the pending
election. The native oracle applies the same rule after parsing each real XML
turn. Because headless GameDrive leaves the result screen unresolved, its
temporary branch save is updated with the new term/countdown and per-voter vote
enums before it is used as a parent for a later native turn.

`ElectionOracleAgent` is the election-tuned simulator oracle. Its defaults
are the documented winning configuration (`PROVEN_ELECTION_SEARCH` in
`autocracy/oracle.py`: beam 6, five-turn lookahead, two policy moves per
turn, 16 sampled candidates, up to 64 legal pairs, 15-second decision
budget), which recovers from a lagged turn-5 dip and wins the first UK
election from turn 0. Override parameters only deliberately — weaker
hand-rolled settings have produced misleading "unwinnable" conclusions in
past experiments.

```python
from autocracy.agent import ElectionOracleAgent

agent = ElectionOracleAgent(
    beam_width=6,
    search_horizon=None,
    candidate_limit=64,
    max_actions_per_turn=2,
    batch_candidate_limit=256,
    time_budget_seconds=900,
)
result = agent.search()
agent.step()
```

`gamedrive.oracle.GameDriveOracleAgent` has the same result/beam/batch
semantics but
uses `gamedrive/inject_drive.py` for every branch. It loads a real save, sends
the native order through the installed Democracy 3 binary, parses the fresh
XML output, and uses that output as the next branch state. This is the
ground-truth version and is substantially slower; its native source must
already be present in the configured save root and the version-pinned probe
must be built. Its default `candidate_limit` is 16; use `None` for exhaustive
native enumeration. Its `random_seed` option has the same reproducible
candidate-sampling behavior as the simulator agent.
`ElectionGameDriveOracleAgent` selects the matching full-term election-margin
objective for native validation.

### Autoregressive time-series forecasting

`autocracy.timeseries` defines a dependency-free contract for testing a
time-series foundation model against the simulator. `StateFeatureEncoder`
fixes the state columns; `AutoregressiveContext` keeps the observed rows and
the action batch responsible for each transition; and
`ForecastModelInput` exposes those rows, action history, pending actions, and
the requested horizon as plain Python data. `TimeSeriesPolicyAgent` scores
candidate actions with a forecaster, commits the selected action through the
real simulator, then appends the new observed state before the next choice.

The included `PersistenceForecaster` and `EmpiricalActionForecaster` are
CPU-safe baselines. `Chronos2SmallForecaster` (in `autocracy.chronos`) is a
working multivariate backend for `autogluon/chronos-2-small`: it forecasts
every player-visible observed feature jointly, supplies the policy sliders as
known-future treatment covariates, and batches all candidate actions into one
pipeline call. It needs the optional `chronos` extra (`uv sync --extra
chronos`). The injected-backend `Chronos2Forecaster` remains available and
does not import torch. Run the current scaffold with:

```bash
uv run main.py timeseries --turns 32 --model empirical --forecast-horizon 8 \
  --trace-out forecast.json

# Foundation-model forecaster with pre-game save histories as covariates
uv run --extra chronos main.py timeseries --turns 16 --model chronos2-small \
  --forecast-horizon 8 --trace-out forecast.json
```

The trace contains each predicted trajectory, selected action, actual next
state, and one-step mean absolute error, so persistence, empirical, and
Chronos-2 runs can be compared over the same observed episode.

See [`TIMESERIES.md`](TIMESERIES.md) for the complete model-input contract,
covariate roles, trace schema, and the closed-loop control comparison against
the simulator oracle (`reports/chronos-2-small-vs-oracle.md`). The same
section documents the single-life active-learning loop
(`autocracy.learning` + `experiments/chronos_learning.py`), in which the
Chronos agent learns a treatment-effect memory purely from its own observed
transitions and wins its first election without any scripted moves or oracle
look-ahead.

Simulation snapshots can be persisted for later comparison with Democracy 3 by using:

| Function | Description |
|----------|-------------|
| `state_to_dict` / `state_from_dict` | Convert a `SimulationState` to/from a serializable dictionary. |
| `save_state(path)` / `load_state(path)` | Write/read JSON snapshots to disk. |

CLI equivalents:

```bash
# Single loop using the passive agent, saving the resulting snapshot
uv run main.py agent --country uk --turns 1 --state-out saved_state.json

# Resume from a previous point and inspect feasible actions
uv run main.py actions --state-in saved_state.json

# Inspect every influence for a node (e.g., Health) while comparing to health.png
uv run main.py node Health --country uk
```

### Budget Tracking

- Each `SimulationState` carries `policy_costs`, `policy_incomes`,
  `total_expenditure`, and `total_income`. It also preserves
  `policy_cost_histories` and `policy_income_histories`, matching the game's
  20-entry newest-first policy finance rings. The live maps are used during
  order/turn calculations; the ring head is the value serialized into a save.
  An active policy at its slider floor still uses its configured minimum
  amount, while a cancelled policy contributes zero.
- Income and expenditure are **live-recomputed every turn** to match the game's `<finances>` block:
  - income = sum over *active* income policies of `base(min,max,val) * wealth_mod * earn_scalar * incom_mult`;
  - expenditure = the analogous cost sum, **plus** `wealth_mod ×` the active
    situation costs, **plus** the quarterly debt interest
    `debt * rate * 0.25`.
- The multiplier neurons are evaluated from the *previous* turn's node values
  (the game's one-turn history lag), and the ministerial scalars come from the
  current competence (`earn_scalar = 0.875 + 0.25 · competence`, with
  `cost_scalar = 2 - earn_scalar`).  All money arithmetic is rounded to
  float32, which reproduces the serialized totals exactly. The interest rate
  also includes the global-interest neuron offset from its 0.5 baseline.
- The state also tracks the national `debt` (rolled forward by the last net),
  the `credit_rating` (recomputed every other turn from the debt-to-GDP
  ratio) and the `interest_rate` derived from it.
- A loaded save uses its serialized cost/income multipliers and ministerial
  scalars to reproduce the observed finance snapshot. Fresh or explicitly
  dynamic calculations can evaluate the CSV modifiers against the current
  context, so health-driven pension costs and situation-driven healthcare
  overruns remain available for experiments.
- `main.py` prints political capital, total income, total expenditure, and the net balance after every metric table, and `describe --verbose` augments the policy table with live cost/income columns for quick audits.

### Savegame Interop

`autocracy/savegame.py` adds helpers to compare against real Democracy 3 save files:

| Function | Description |
|----------|-------------|
| `parse_savegame(path)` | Parses the (slightly malformed) XML save into a `SaveGame` dataclass containing all `simvalues`, current policy values, requested policy targets, live finance lines, 20-entry per-policy finance rings, hidden neurons, voter fields, policy runtime fields, the `<inherited>` simvalue block, and serialized effect rings. |
| `load_state_from_savegame(path, data=None)` | Builds a `SimulationState` seeded from the save (node values, current/desired policy values, runtime fields, and current turn) so the simulator can continue from an in-game snapshot. |
| `compare_state_to_savegame(state, save, tolerance)` | Produces a `StateComparison` listing value/policy discrepancies and missing entries, making it easy to sanity-check the simulator against the real game. |

Example:

```python
from autocracy.savegame import parse_savegame, load_state_from_savegame, compare_state_to_savegame

save = parse_savegame("gamedata/saves/uk0.xml")
state, _ = load_state_from_savegame("gamedata/saves/uk0.xml")
comparison = compare_state_to_savegame(state, save)
assert not comparison.value_diffs  # simulator matches save snapshot
```

---

### Extending the Simulator

1. **Additional Node Types**: Drop new CSV rows into the `simulation/` folder; the loader automatically picks them up and extends the DAG.
2. **New Countries**: Add missions under `gamedata/data/missions/<name>` – as long as the file structure mirrors the originals, `get_initial_state` can load them.
3. **Random Events**: Fill in `process_dilemmas`, `process_attacks`, `process_events` to mutate the `SimulationState` between turns.
4. **Policy Granularity**: Adjust `DEFAULT_PERCENTAGE_STEP` in `autocracy/simulator.py` or enrich `sliders.csv` to provide explicit step counts for continuous sliders if finer control is required.

With these building blocks the repository now serves as a deterministic, data-driven Democracy 3 sandbox suitable for reinforcement-learning experiments or automated balancing.
