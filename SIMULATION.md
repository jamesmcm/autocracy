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
| `simulation/sliders.csv` | Slider metadata. | Declares whether a slider is `DISCRETE` (enum-like) or `PERCENTAGE` (continuous). For discrete sliders, the textual options (`NONE`, `LOW`, `MEDIUM`, …) imply evenly spaced normalized levels between 0 and 1. For percentage sliders the numeric bounds (e.g., `0`, `75`) are recorded for reference, but the simulator treats them as continuous 0 – 1 values. |
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
   - Each effect (graph edge or situation link) is given an initial entry in the effect-memory dictionary so inertia has state to blend toward.
   - When `...1.xml` exists as well, the simulator computes per-node calibration factors by comparing the predicted no-op turn against the observed save. Each node’s delta is then scaled by this factor every turn, anchoring passive runs to what the real game produces.
   - Political capital starts at `POLITICAL_CAPITAL_PER_MINISTER * 5`, capped by `POLITICAL_CAPITAL_MAX_MULTIPLIER`.

2. **Action Phase** (`apply_actions`):
   - Each `PolicyAction` contains a `policy_name` and a normalized `delta`.
   - Costs depend on what is happening:
     * **Introduce** (0 → >0): use `introduce_cost`.
     * **Cancel** (>0 → 0): use `cancel_cost`.
     * **Modify** (>0 → >0): use `raise_cost` or `lower_cost` depending on the direction.
   - Validation steps:
     - Policy must exist and enough political capital must remain.
     - Slider metadata determines what levels are legal.
       * **Discrete** sliders use the enumerated labels to derive evenly spaced levels, and the new value must exactly match one of them.
       * **Percentage / continuous** sliders can take any value within `[0, 1]`.
     - Attempting to move beyond the boundary (no actual change) raises an error.
   - Capital is deducted per accepted action; state values are not recalculated yet.

3. **End of Turn** (`process_end_of_turn`):
   - For each node, sum all incoming effects: `current_value + Σ delta`. Every effect keeps its own inertia buffer (`value += (target - value)/inertia`) so delayed influences ramp gradually over multiple turns.
   - Recompute every situation’s latent value, update `active_situations` based on hysteresis (start ≥ `start_trigger`, stop < `stop_trigger`), and inject their outputs into the DAG using the same inertia rule.
   - Clamp to each node’s declared `[min, max]`.
   - Refill political capital by half of `POLITICAL_CAPITAL_PER_MINISTER`, capped again.

4. **Random Systems**:
   - `process_dilemmas`, `process_attacks`, and `process_events` exist as explicit stubs so that future work can plug in the missing systems without changing the core APIs today.

The simulator uses a synchronous update: every node reads from the previous state and writes to a new state object, preventing cascading artefacts within the same turn.

---

### Actions API

Two functions make the simulator easy to integrate with AI agents:

| Function | Description |
|----------|-------------|
| `list_available_actions(state, data=None)` | Returns `PolicyActionOption` objects describing every feasible single-step policy adjustment, including the action type (`introduce`/`cancel`/`raise`/`lower`), political capital required, and the policy’s implementation delay. Slider values are treated as continuous 0‑1 with a default 5 % step, so agents can make incremental adjustments even if the UI labels are discrete. |
| `apply_actions(state, actions, data=None)` | Applies validated actions, raising informative errors if the move is impossible (unknown policy, insufficient capital, invalid discrete level, or no change at boundary). The returned state is ready for `process_end_of_turn`. |

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

- Each `SimulationState` carries `policy_costs`, `policy_incomes`, `total_expenditure`, and `total_income`, all recomputed whenever policies or node values change.
- Policy cost/income multipliers are read from the CSV (`cost multiplier`, `incomemultiplier`) and evaluated every turn so health-driven pension costs or situation-driven healthcare overruns are reflected automatically.
- `main.py` prints political capital, total income, total expenditure, and the net balance after every metric table, and `describe --verbose` augments the policy table with live cost/income columns for quick audits.

### Savegame Interop

`autocracy/savegame.py` adds helpers to compare against real Democracy 3 save files:

| Function | Description |
|----------|-------------|
| `parse_savegame(path)` | Parses the (slightly malformed) XML save into a `SaveGame` dataclass containing mission metadata, all `simvalues`, and policy slider levels. |
| `load_state_from_savegame(path, data=None)` | Builds a `SimulationState` seeded from the save (node values, policy sliders, current turn) so the simulator can continue from an in-game snapshot. |
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
