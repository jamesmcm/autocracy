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

# Run the CPU-safe autoregressive action-forecast scaffold and save its trace
uv run main.py timeseries --turns 8 --model empirical --trace-out forecast.json

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
  scaffold and the simulator-backed beam-search oracle.
- `autocracy/timeseries.py` provides the fixed-schema autoregressive context,
  CPU baselines, and optional Chronos2 backend boundary for action forecasts.
- `autocracy/savegame.py` parses Democracy 3 XML saves and compares simulator
  output against real snapshots.
- `gamedrive/oracle.py` provides a native beam-search oracle that evaluates
  branches through the installed Democracy 3 executable.
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
  not serialize every outgoing effect's desired throttle. The long-run audit
  can supply the native checkpoint's serialized effect, hidden, situation,
  policy-manager, voter-manager, and finance-manager state explicitly; that
  bridge is audit-only and does not change ordinary simulator runs. The
  captured StateHealthService ring's no-op freeze is reproduced until an
  explicit order starts that policy's ring.
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
  turn with a 52-point cap. Loyalty-aware runs derive each minister's
  contribution from loyalty with a zero floor; deterministic resignation is
  opt-in because the native resignation roll is not serialized.

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

Both shipped no-op turns now reproduce all 40 ordinary nodes within 0.001
(mean errors about 0.00012 and 0.00001). The native executable scales
`BorderControls -> Immigration` with the active FOREIGNPOLICY minister, while
its installed equation parser gives `Unemployment -> Immigration` a measured
−0.06 offset; both observations are data-driven in `calibration.json`. The
ring-update rule (simvalue/situation rings advance each turn; policy rings
wait at target changes and then drain with post-update samples) is described
in `SIMULATION.md`.

### Long-run native corpus

The `gamedrive` bridge has now generated two fresh 24-turn UK chains from the
unchanged `parity_cases/dem3saves/turn0_initial.xml` source: a no-order chain
and the captured policy sequence followed by a no-order tail. Raw native XML
is kept outside the repository under the installed save root; the reproducible
names are recorded in [`gamedrive/MULTI_TERM_TODO.md`](gamedrive/MULTI_TERM_TODO.md).
Every checkpoint passed native-save validation and serialized turns 1–24.

The same unchanged source also has a 128-turn no-order stress chain recorded
in `gamedrive/MULTI_TERM_TODO.md`; all 128 terminal and intermediate saves
passed validation, with zero policy-target differences. The expanded audit
now feeds each checkpoint's serialized minister roster, active situations,
grudges, hidden values/histories, voter/party/poll state, effect histories,
policy-manager runtime, and finance-manager state back into the replay. With
those explicit checkpoint inputs, situation values, hidden values and
histories, voter fields, policy runtime, finance totals/debt, and effect
histories match exactly at all 128 checkpoints.

The audit now also restores each checkpoint's serialized `<simvalues>` before
continuing. That closes the ordinary checkpoint comparison to zero across all
128 turns while leaving a pre-bridge model residual in the report: maximum
0.1156 (turn 54, `PrivateHealthcare`) and mean per-checkpoint maximum about
0.0940. The persistent `PrivateHealthcare` value is inconsistent with that
native save's own effect histories, so it remains explicitly marked as a save
anomaly rather than folded into the normal equation model. The checkpoint
bridge is audit-only; normal simulator runs remain model-derived.

`gamedrive/term_audit.py` compares the complete offline state at every
checkpoint. Policy targets are exact across both chains. The long run exposed
the native missing-minister finance fallback (competence `0.25`), a zero-floor
political-capital contribution, and an aliased voter snapshot; these are now
covered by the simulator. Native roster checkpoints take precedence over the
non-serialized resignation roll. The simulator also models the headless
election countdown and provides `resolve_election` for explicit vote counting,
result persistence, term advancement, and player-win loyalty effects.
The remaining default-path differences are continuous ordinary-node/effect
state and native save anomalies; the audit-only checkpoint bridge does not
alter ordinary simulator runs.

Captured order saves are replayed as one native order batch. The order-phase
finance preview consumes the previous policy-history sample, while a newly
introduced policy starts from its midpoint history sample. This keeps the
24-turn intervention-chain policy targets exact and reduces its peak finance
residual to about 1,084 income / 2,053 expenditure.

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

### Oracle agents

The simulator oracle evaluates a no-op and legal policy-action batches at every
beam layer by applying the batch and running the actual simulator turn
transition. It executes only the first batch from the winning forecast, then
replans from the observed state on the next turn:

```python
from autocracy.agent import SimulatorOracleAgent

agent = SimulatorOracleAgent(
    beam_width=4,
    search_horizon=2,
    candidate_limit=32,
    objective=lambda state: state.values["GDP"],
)
result = agent.search()
print(result.first_actions, result.score, result.evaluated)
agent.step()
```

`candidate_limit=None` makes the simulator search exhaustive. The default
objective combines GDP, Health, Education, CrimeRate, Unemployment, and poll
rate; pass `objective` to optimize a different score. Set `random_seed` to
sample a fresh reproducible subset of candidates at each beam node instead of
using the deterministic first candidates. `max_actions_per_turn` enables
multi-policy batches, and `batch_candidate_limit=None` enumerates every legal
combination of the selected options.

`time_budget_seconds` is a wall-clock search budget. When it expires, the
agent returns the best branch completed so far (or a safe no-op fallback) and
records `timed_out`, `completed_depth`, and `elapsed_seconds` on the result.

For an election-focused best-case baseline, `ElectionOracleAgent` searches to
the next election by default, uses the expected native-style turnout model,
and scores the expected player-minus-opposition vote margin:

```python
from autocracy.agent import ElectionOracleAgent

election_oracle = ElectionOracleAgent(
    beam_width=6,
    search_horizon=None,       # full remaining term
    candidate_limit=64,
    max_actions_per_turn=2,
    batch_candidate_limit=256,
    time_budget_seconds=900,
)
result = election_oracle.search()
print(result.first_actions, result.score, result.evaluated)
```

The native `CastVote` path samples turnout before choosing a candidate. The
simulator exposes `forecast_election(state)` so the oracle optimizes expected
votes rather than one arbitrary random draw; `resolve_election` rounds that
expectation deterministically for reproducible experiments.

For ground-truth action evaluation, `GameDriveOracleAgent` performs the same
beam search by launching the native GameDrive probe for every branch and
parsing each fresh XML save:

```python
from gamedrive.oracle import GameDriveOracleAgent

native = GameDriveOracleAgent(
    "oracle_source",
    beam_width=2,
    search_horizon=2,
    candidate_limit=16,
    random_seed=20260811,
    objective=lambda save: save.simvalues.get("GDP", 0.0),
)
native.step()  # commits the first native save from the winning path
```

`ElectionGameDriveOracleAgent` provides the matching native election-margin
configuration (`search_horizon=None`, wider beam, two-action batches, and
`score_savegame_election`). It is useful for validating the simulator oracle,
but every native branch launches the real executable and is consequently much
slower.

`oracle_source.xml` must be a copied save in the configured native save root;
build the probe with `make -C gamedrive` first. Native search is intentionally
expensive and keeps only fresh save artifacts belonging to its winning path.
Both oracle paths resolve a pending election at the zero-countdown boundary
and discard branches that lose. If no searched branch survives, they raise
`OracleElectionLoss` instead of continuing a campaign after the game has
ended. The native path persists the resolved term/countdown and voter vote
enums into its temporary winning checkpoint before launching another branch;
the original source save is left untouched.

### Time-series action forecasting

`autocracy.timeseries` supplies the experiment boundary for a foundation model
without adding a torch or CUDA dependency. `AutoregressiveContext` stores a
fixed feature schema, the observed state rows, and the action batch that led
from each row to the next. `TimeSeriesPolicyAgent` evaluates legal actions by
asking an `ActionConditionedForecaster` for a future trajectory, executes the
selected action in the simulator, and appends the real resulting state before
the next decision. This is the intended autoregressive loop:

```python
from autocracy.timeseries import EmpiricalActionForecaster, TimeSeriesPolicyAgent

agent = TimeSeriesPolicyAgent(
    EmpiricalActionForecaster(),
    forecast_horizon=8,
    candidate_limit=32,
)
for _ in range(32):
    agent.step()
agent.save_trace("forecast.json")
```

The empirical and persistence forecasters run on the VPS and provide CPU
baselines. `Chronos2Forecaster.from_callable(...)` accepts a later GPU-backed
Chronos2 predictor using the same `ForecastModelInput`; no model weights are
downloaded by this repository. Each trace records predictions and the actual
next state, including one-step mean absolute error for later model comparison.

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
  history ring. The expanded 128-turn audit restores the serialized
  finance-manager runtime at each checkpoint, closing the long-run finance
  comparison to zero. That restoration is explicit audit input; normal
  simulator runs still calculate finance from their own state.
- The remaining continuous-state model residuals are concentrated in ordinary
  effect-source state (including late `CrimeRate`, `RacialTension`, and
  `PovertyRate` channels) and in native save values that contradict their own
  serialized inputs. The audit uses the serialized simvalue checkpoint bridge
  to prevent those unstored native cursors from cascading into later turns;
  the normal simulator still retains the evidence-backed ring freeze, one-pass
  policy delay, and pre-policy source ordering documented in `SIMULATION.md`.
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
- Use `gamedrive/term_capture.py --country uk --one-process-per-turn` for the
  configured 16-turn UK term plus eight extra turns; its `--orders-dir` mode
  appends a no-order tail to an existing action sequence. Audit the resulting
  chain with `gamedrive/term_audit.py`; use
  `SimulationConfig(minister_resignations=True)` only when the replay should
  model deterministic below-threshold minister removal.
- Keep raw native output saves outside the repository and use fresh save names;
  the installed binary is version-pinned to Democracy 3 v1.30.2.
- Keep raw proprietary saves outside the repository; commit only reviewable
  extraction fixtures and reproducible tests.
