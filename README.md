## Democracy 3 Simulator (WIP)

This repository contains a lightweight simulator for Democracy 3 that operates directly on the original `data` assets.

Place the Democracy 3 `data` folder inside `gamedata`, so it should include
`./gamedata/data/simconfig.txt`. The game data is encoded with latin-1 rather
than UTF-8.

The aim is to create an environment for online-learning AI agents: can they
learn to govern without being given the underlying DAG and equations?

### Running the CLI

Use [uv](https://github.com/astral-sh/uv):

```bash
# Inspect the initial UK setup
uv run main.py describe --country uk

# Dump the entire DAG state and policy roster
uv run main.py describe --country uk --verbose

# Apply a policy tweak, advance one turn, and persist the resulting state
uv run main.py simulate --country uk --turns 1 -p IncomeTax:-0.05 --state-out snapshot.json

# Resume from a saved snapshot with the passive agent loop
uv run main.py agent --state-in snapshot.json --turns 3

# Load or compare a Democracy 3 save directly
uv run main.py load-save gamedata/saves/uk0.xml
uv run main.py compare-save gamedata/saves/uk0.xml --state-in snapshot.json

# Inspect the causes/effects for a specific node
uv run main.py node Health
```

Useful options include `--metric`/`-m`, repeatable `--policy`/`-p`
(`Name:delta`), `--gamedata`, and `--state-in`/`--state-out`. Metrics include
political capital, total income, total expenditure, and net balance. The
`describe --verbose` command also prints policy cost/income columns.

### Architecture

- `autocracy/data_loader.py` parses the CSV/INI files under `gamedata/data/`.
- `autocracy/simulator.py` builds the DAG, exposes state/turn/action helpers,
  and leaves dilemmas, attacks, and events as explicit stubs.
- `autocracy/agent.py` provides the `BaseAgent`/`PassiveAgent` turn-loop
  scaffold.
- `autocracy/savegame.py` parses Democracy 3 XML saves and compares simulator
  output against real snapshots.
- `main.py` provides the Typer/Rich CLI.
- `SIMULATION.md` documents the data formats, update ordering, persistence,
  and public APIs in more depth.

### Situation and inertia handling

- `situations.csv` is parsed alongside the other simulation assets. Latent
  situation values are evaluated each turn using trigger thresholds and
  per-link inertia.
- Serialized inertial links are restored as raw 33-slot effect rings. The live
  effect is the leading-window average; policy links apply ministerial
  effectiveness to the live value, not to saved raw samples.
- State snapshots include situations, active situations, hidden global
  neurons, voter histories, policy runtime/multiplier fields, delayed policy
  throttles, ministerial effectiveness, political-capital points, policy
  finance history rings, and effect histories.
- When available, `get_initial_state` seeds node and policy values from
  `gamedata/saves/<country>0.xml`, matching the shipped baseline.
- Policy runtime keeps the current neuron value (`<val>`) separate from the
  requested slider target (`<targ>`). The current value moves toward the
  target by a fixed `1 / implementation_time` step per turn.
- Political capital is seeded from the save and accrues using the captured
  active-minister baseline; for the shipped UK start this is 26 points per
  turn with a 52-point cap.

### Parity extraction

The XML saves are the authoritative observation corpus. Extract a reviewable
snapshot with:

```bash
uv run python -m autocracy.parity gamedata/saves/uk0.xml --output /tmp/uk0-parity.json
uv run pytest tests/test_parity.py -q
```

`parity_cases/uk_noop.json` records both deterministic shipped-save
no-op transitions (`uk0.xml` → `uk1.xml` → `uk2.xml`), including continuous
state, finance, situations, hidden values, voters, policy runtime, the
`<inherited>` snapshot, and sample records from the 303-ring effect-memory
corpus. `parity_cases/uk_bus_lanes_live.json` records a controlled Xvfb
capture of a Bus Lanes intervention and its current/target runtime fields.

Both shipped no-op turns now stay inside a mean continuous-state error of
about 0.003, with a single remaining outlier (`Immigration`, ~0.03): the
save pair implies `BorderControls -> Immigration` is applied without the
ministerial scale, and even then a small residual remains because the game's
unstored random-system state cannot be reproduced. The ring-update rule
(simvalue/situation rings advance each turn; policy rings wait at target
changes and then drain with post-update samples) and that calibration are
described in `SIMULATION.md`.

### Stochastic systems

Random events, dilemmas, pressure-group events and extremist
plots/assassinations are data-driven (`gamedata/data/simulation/events`,
`dilemmas`, `attacks`, `pressuregroups.csv`) but **off by default**, so
parity runs are deterministic:

```bash
# Fully deterministic (default)
uv run main.py simulate --turns 4

# Enable systems with a reproducible seed
uv run main.py simulate --turns 4 --events --dilemmas \
    --pressure-groups --assassinations --random-seed 42
```

Programmatic callers pass a `SimulationConfig` to `process_end_of_turn` or
`BaseAgent`; an all-off config is a bit-for-bit no-op. See `SIMULATION.md`
for the semantics.

### Savegame integration

```python
from autocracy.savegame import (
    compare_state_to_savegame,
    load_state_from_savegame,
    parse_savegame,
)

save = parse_savegame("gamedata/saves/uk0.xml")
state, graph = load_state_from_savegame("gamedata/saves/uk0.xml")
comparison = compare_state_to_savegame(state, save)
print("Differences?", comparison.has_differences())
```

Use this to validate the simulator against real Democracy 3 saves or to
bootstrap a simulation from an in-game snapshot.

### Current parity limits

- The one-pass core now agrees closely with the shipped UK no-op transition,
  with targeted residuals in inertial rings and the captured late-turn
  `ViolentCrimeRate` value (which is inconsistent with the game's own inputs).
- Finance is live-recomputed, including debt interest and the global-interest
  neuron; save parsing also preserves each policy's 20-entry cost/income
  history ring. The drastic replay's aligned final residual is about -1,225
  income / -33 expenditure.
- Party/sympathy poll dynamics, percentages, and frequencies remain the main
  manager-owned parity gap; stochastic systems stay opt-in and off by default.

### Next parity work

- Reduce targeted Education/Health/OilSupply/WorkerProductivity ring drift.
- Reconstruct party/sympathy voter dynamics from static binary analysis and
  captured saves.
- Use the binary manager call order and save snapshots to continue the native
  gdb/LD_PRELOAD path; do not launch the installed game on this server.
- Keep raw proprietary saves outside the repository; commit only reviewable
  extraction fixtures and reproducible tests.
