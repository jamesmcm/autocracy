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
| `simulation/votertypes.csv` | Voter happiness + membership nodes. | Similar structure: default support (`default`), membership share (`percentage`), and `Influences` columns that create edges. Membership nodes (`<Group>_freq`) are synthesized automatically. |
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
- **Frequency/membership updates**: Equations targeting `<Group>_freq` nodes are treated like any other effect, so they participate in the same DAG pass.

---

### Turn Progression

1. **Initial State** (`get_initial_state`):
   - Node values are seeded from their defaults.
   - Policy sliders are filled with mission-provided levels, clamped to `[0, 1]`.
   - If a matching baseline save exists (`gamedata/saves/<country>0.xml`), its `simvalues`/`policies` override the defaults so the simulator starts from the exact in-game conditions.
   - Situation latent values are evaluated from their inputs to determine which situations start active (respecting their start/stop trigger thresholds).
   - Each serialized inertial link is restored as a raw 33-slot effect ring. On load, the live effect is the average of the leading `inertia` slots, with policy links additionally scaled by ministerial effectiveness.
   - The loader also restores hidden global neurons, voter histories, policy runtime fields, delayed policy throttles, ministerial effectiveness, situation state, finance snapshots, and the `<inherited>` simvalue block (the values from two turns ago). Policy runtime distinguishes the current policy-neuron value (`<val>`) from the requested slider target (`<targ>`); the current value moves toward the target by a fixed `1 / implementation_time` step each turn. No per-node response calibration is applied; parity differences remain observable.
   - Political capital is restored from `<politicalcapital><points>`. The
     simulator also retains the baseline active-minister accrual recovered
     from the initial save; for the shipped UK start this is 26 points per
     turn with a 52-point cap. Countries without a baseline save use the
     configured minister-count fallback.

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

3. **End of Turn** (`process_end_of_turn`):
   - Advance policy runtime first: implementation fractions increase by minister effectiveness divided by implementation time, while current policy values and their policy-input throttles move toward requested targets by the fixed `1 / implementation_time` step. The action-phase policy map therefore remains the current `<val>` until this phase.
   - Advance the effect vector using the pre-turn policy values as source snapshots. Direct links use the current source throttle; inertial links shift a raw expression sample into their ring and average the leading window. Saved rings contain raw samples, not minister-scaled live values.
     The executable only writes a fresh ring sample for simvalue and situation sources (situations while active) every turn; a settled policy's ring keeps its older samples, which is why serialized policy rings still hold values from earlier, lower slider levels. The simulator mirrors that rule: policy rings advance only while the policy is moving toward its target or still rolling out.
   - One parity calibration is applied to `BorderControls -> Immigration`: the shipped save pair implies that link is *not* ministerially scaled (its implied contribution is the raw −0.4 ring value). Every other policy effect on the simvalue nodes does carry the ministerial scale.
   - Walk ordinary simulation nodes in data order. Each node is `default + Σ current incoming effects`, clamped to its declared `[min, max]`; after a node is calculated, its direct outgoing links are recalculated immediately, matching `SIM_Neuron::CalculateValue`.
   - Recompute situation latent values from their input links and retain the manager’s start/stop decision for the pass. Situation outputs are gated by the active set and participate in the same effect vector.
    - Add the active-minister political-capital accrual and clamp at the
      corresponding `POLITICAL_CAPITAL_MAX_MULTIPLIER` cap.  When the
      `SimulationConfig.minister_loyalty` flag is enabled the accrual is
      re-derived each turn from the ministers' loyalty (which itself drifts
      through `SIM_Minister::ProcessLoyalty` as ministers gain/lose loyalty
      based on their satisfaction with the enacted policies); otherwise the
      accrual stays at the value loaded from the save.
    - Recompute the finance lines from the advanced policy values and the
      advanced ministerial scalars, with the multiplier neurons evaluated at
      the previous turn's nodes and the debt interest charged on the
      freshly-rolled debt.

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
     opinion shifts to voter values/frequencies and simvalues. Dilemmas
     fire when their latent influence sum crosses 0.5 and resolve with a
     seeded option choice. Plots and assassinations require the matching
     extremist pressure group's support to reach the file's `MinStrength`.
   - The executable keeps unstored random-system state (event cooldowns,
     group strength), so live-game timing is not reproduced exactly;
     enabled runs are reproducible through the seed.
   - CLI: `uv run main.py simulate --turns 4 --events --dilemmas
     --pressure-groups --assassinations --random-seed 42` (the `agent`
     command accepts the same flags).

The simulator returns a new state object, but its core update is intentionally ordered rather than fully synchronous: direct effects can cascade to later nodes in the same pass, as they do in the game. The 33-pass `PreCalcCoreSimulation` settling routine used by the executable during initialization is distinct from the normal one-pass turn path.

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

Simulation snapshots can be persisted for later comparison with Democracy 3 by using:

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

- Each `SimulationState` carries `policy_costs`, `policy_incomes`, `total_expenditure`, and `total_income`.
- Income and expenditure are **live-recomputed every turn** to match the game's `<finances>` block:
  - income = sum over *active* income policies of `base(min,max,val) * wealth_mod * earn_scalar * incom_mult`;
  - expenditure = the analogous cost sum, **plus** `wealth_mod ×` the active
    situation costs, **plus** the quarterly debt interest
    `debt * rate * 0.25`.
- The multiplier neurons are evaluated from the *previous* turn's node values
  (the game's one-turn history lag), and the ministerial scalars come from the
  current competence (`earn_scalar = 0.875 + 0.25 · competence`, with
  `cost_scalar = 2 - earn_scalar`).  All money arithmetic is rounded to
  float32, which reproduces the serialized totals exactly.
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
| `parse_savegame(path)` | Parses the (slightly malformed) XML save into a `SaveGame` dataclass containing all `simvalues`, current policy values, requested policy targets, finance lines, hidden neurons, voter fields, policy runtime fields, the `<inherited>` simvalue block, and serialized effect rings. |
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
