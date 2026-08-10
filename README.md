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
- The general effect-ring replay is necessarily a proxy for arbitrary loaded
  saves: the executable restores raw histories and input throttles, but does
  not serialize every outgoing effect's desired throttle. Targeted
  policy/simvalue ring residuals therefore remain until that runtime state can
  be recovered. The captured StateHealthService ring's no-op freeze is
  reproduced until an explicit order starts that policy's ring.
- State snapshots include situations, active situations, hidden global
  neurons and their 33-slot histories, voter histories, party metadata/history rings, full per-voter
  party/sympathy inputs, policy runtime/multiplier fields, delayed policy
  throttles, ministerial effectiveness, political-capital points, policy
  finance history rings, and effect histories.
- Voter income groups follow the native overlapping sinusoidal curves and
  `VOTER_GROUP_MEMBERSHIP_THRESHHOLD` floor. Nested VoterType `<income>` values
  are loaded from saves and their direct graph links are evaluated each turn;
  the manager's non-serialized per-voter host contribution remains a runtime
  boundary.
- VoterType frequency neurons use the native zero-base `[-1, 1]` pass; CSV
  membership percentages are calculated from the native linked lists. Ordinary
  groups use the previous saved `<group>_freq` value as their membership base,
  while the four `ForceVoter` ideology links use their raw forced weights;
  persistent `CreateGrudge(..., <group>_freq, ...)` inputs are restored from
  saves and included on every pass. The captured turn-zero income percentages
  retain the game's pre-first-pass startup state.
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
- The drastic replay has also been audited against all 40 ordinary serialized
  `<simvalues>` nodes at the available game-turn checkpoints (1, 2, 3, and 12).
  At turn 12, 31/40 nodes are within `0.01` and 39/40 are within `0.05`; the
  largest observed error is `ViolentCrimeRate` (`+0.1976`), followed by
  `Education` (`+0.0490`). The saved turn-12 crime value is inconsistent with
  the game's own inputs, so it is tracked separately from the genuine
  effect-ring drift. Voter values and situation latents are serialized in
  separate manager-owned sections and remain less exact than the ordinary DAG.
- Finance is live-recomputed, including debt interest and the global-interest
  neuron; save parsing also preserves each policy's 20-entry cost/income
  history ring. The drastic replay's aligned final residual is about -1,490
  income / +88 expenditure under the current pre-policy effect sampling.
- The remaining continuous-state residuals are concentrated in outgoing
  effect-ring throttle/load state (including the post-order StateHealth ramp)
  and the non-serialized global-economy random cursor. The current model
  retains the evidence-backed ring freeze, one-pass policy delay, and
  pre-policy source ordering documented in `SIMULATION.md`; the finance
  trade-off is measured in `parity_cases/DRASH_NOTES.md`.
- Base party/sympathy membership transitions now use the binary-confirmed
  approval transform, simconfig thresholds, party-type lookup, and serialized
  member-count history; the serialized activist ring also advances with its
  loaded current head. Native manager-owned party lists, activist-count/poll
  modifiers and dynamic per-voter income-neuron host links remain the main
  parity gap; the
  underlying per-voter fields and VoterType frequency/grudge state are retained
  when loading or snapshotting state.
- Active situation output links targeting voter types now join the same
  current-versus-previous effect delta as policy and ordinary-node links. This
  closes the captured GeneralStrike-to-Conservatives omission; native party
  lists, approval modifiers, and activist/poll manager state still limit later
  membership parity.
  Stochastic systems stay opt-in and off by default.

### Next parity work

- Reduce targeted Education/Health/WorkerProductivity ring drift; OilSupply is
  exact in the current aligned replay.
- Continue simulator-side party/sympathy parity work; the native probe now has
  an opt-in manager census for live party, activist, poll, and income-host
  state.
- Run `gamedrive/preflight.py` to verify the 75 version-specific native symbols
  before using the version-pinned injector.
- Use `gamedrive/inject_drive.py --turn-mode sync` under Xvfb to load a copied
  save, replay pre-turn `_orders` files, and save a bounded native capture.
  Compare it offline with `gamedrive/capture.py`; see
  [`gamedrive/README.md`](gamedrive/README.md) for fresh-output validation,
  manager census, and the memory-editing boundary.
- Keep raw native output saves outside the repository and use fresh save names;
  the installed binary is version-pinned to Democracy 3 v1.30.2.
- Keep raw proprietary saves outside the repository; commit only reviewable
  extraction fixtures and reproducible tests.
