# Autoregressive time-series experiments

This document describes the forecasting experiment boundary in
`autocracy.timeseries` and the Chronos-2 backend in `autocracy.chronos`. The
CPU baselines run anywhere; the foundation-model backend needs the optional
`chronos` extra (`uv sync --extra chronos`) and runs on GPU or CPU.

## What is being forecast?

The simulator has a multivariate state rather than one scalar series. A
`StateFeatureEncoder` turns the selected simulator fields into a stable row of
named floating-point features.

Two schema builders exist:

- `StateFeatureEncoder.from_state(...)` encodes every ordinary value, policy
  slider, finance field, and poll/election field. It exists for parity work
  and full-state diagnostics.
- `StateFeatureEncoder.from_visible_state(...)` encodes exactly what the
  player can see in-game. Values are restricted to ordinary simulation nodes:
  every ``_``-prefixed meta neuron (`_globaleconomy_`, `_year`, `_Terrorism`,
  ...), all `HIDDEN`/`PLACEHOLDER` nodes, and derived runtime mirrors
  (`*_perc`, nested voter incomes) are excluded. The auxiliary finance and
  election columns keep only displayed figures (political capital, income,
  expenditure, debt, interest rate; poll rate; election countdown).
  Foundation-model experiments must use this schema so no hidden or derived
  variable leaks into covariates.

The encoder sorts value and policy names once when it is created. That fixed
column order is stored in the context and must be used by every forecast
backend:

```python
from autocracy import simulator
from autocracy.timeseries import StateFeatureEncoder

state, _ = simulator.get_initial_state("uk")
encoder = StateFeatureEncoder.from_visible_state(state)
```

## Covariate roles

`FeatureRoleSchema.from_encoder(encoder)` classifies each column:

| Role | Columns | Meaning |
| --- | --- | --- |
| Target | every non-policy column, led by `politics/poll_rate` | observed features whose future is unknown and must be predicted |
| Treatment | all `policy/*` columns | player-controlled sliders; known at every future step because the agent chooses them |

The headline target is `politics/poll_rate` (`ELECTORAL_SUPPORT_FEATURE`),
the game's published total voter electoral support. Everything else the
player sees (economy gauges, budget lines, election countdown) is predicted
jointly in multivariate mode while the treatment path conditions the
forecast.

## Pre-game covariate history

A Democracy 3 save stores a newest-first `<history>` ring for every recorded
simvalue. `parse_savegame` exposes these as `save.simvalue_histories`,
`get_initial_state` copies them onto the state
(`state.value_histories`, ending at `state.value_histories_turn`), and
`AutoregressiveContext.from_state(..., include_value_histories=True)` replays
them as context snapshots ending at turn 0. This means a forecasting model
starts the game with roughly 20–34 turns of real covariate history instead of
a single row.

Preprocessing notes:

- Features without rings (policies, finance fields) are held constant over
  the pre-game window; that matches what a fresh game reveals about them.
- Exact zeros inside value rings are serialization artifacts (nodes record
  only once live; some statistics such as GDP resample periodically), so they
  carry the most recent non-zero reading forward — the series the player
  actually sees.
- Seeding only applies while the context still ends at
  `value_histories_turn`; once real turns advance, live observations take
  over.

## Chronos-2-small backend

`autocracy.chronos.Chronos2SmallForecaster` implements the
`ActionConditionedForecaster` protocol with `autogluon/chronos-2-small`
(28M parameters) through the `chronos-forecasting` package:

```python
from autocracy.chronos import Chronos2SmallForecaster
from autocracy.timeseries import TimeSeriesPolicyAgent

agent = TimeSeriesPolicyAgent(
    Chronos2SmallForecaster(),          # device_map defaults to cuda when available
    visible_features_only=True,
    seed_pre_game_history=True,
    forecast_horizon=4,
)
```

Per decision turn it converts all candidate actions into one batched
`predict_df` call: each candidate is an independent item sharing the same
observed history but carrying its own known-future policy frame (the pending
slider move applied once, then held). Observed features are returned as
median forecasts per step, treatments are restored from the deterministic
path, and the agent scores the final row. Candidate batching makes the cost
roughly independent of the candidate count (tens of milliseconds per
candidate on an RTX 4080).

The CLI enables the backend directly:

```bash
uv run --extra chronos main.py timeseries --model chronos2-small \
  --turns 16 --forecast-horizon 4 --trace-out traces/chronos.json
```

## Autoregressive control loop

At every decision turn, `TimeSeriesPolicyAgent` performs the following loop:

```text
observed state[t]
       │
       ├─ encode the complete observed context (+ pre-game history if seeded)
       │
       ├─ for each legal candidate action a[t]:
       │      forecast state[t+1 ... t+h] conditioned on a[t]
       │      score the final forecast row
       │
       ├─ select the highest-scoring candidate
       │
       ├─ apply that action to the real simulator and advance one turn
       │
       └─ append the actual state[t+1] and a[t] to the context
```

Only the selected action is executed. Forecast rows are never fed back as if
they were real game state. Backends may implement `predict_batch(inputs)` to
score every candidate in one call; the agent uses it automatically and falls
back to per-candidate `predict`.

## Warm-up and flip-flop damping

Zero-shot foundation models have no interventional experience in context, so
their first choices can be erratic. `TimeSeriesPolicyAgent` supports two
mitigations:

- `warmup_plan` + `warmup_batch_size`: a scripted sequence of `PolicyAction`s
  executed before model-driven control starts (illegal or unaffordable
  entries are skipped). With `warmup_batch_size=2` the agent executes two
  planned moves per turn, doubling interventional coverage per turn of
  control. Every warm-up transition is appended to the context, so the
  forecaster observes real treatment/response pairs — it learns the impact
  of moves over time instead of acting on priors alone.
  `diverse_warmup_plan(state)` builds such a program deterministically:
  cheapest-first alternating raises and lowers across distinct policies
  within a capital budget share.
- `reverse_window` + `reverse_penalty`: after warm-up, a candidate that
  reverses an action taken within the last `reverse_window` transitions on
  the same policy loses `reverse_penalty` score points, suppressing
  short-horizon flip-flops.
- `max_actions_per_turn` + `batch_candidate_limit`: enumerate multi-action
  candidate batches (pairs by default) under the hard per-turn budget — the
  summed political-capital cost of a batch may never exceed the capital
  available at the start of the turn.
- `max_action_delta`: restrict candidates to slider steps of at most this
  size, so the agent compounds many modest moves instead of a few large ones.
- `debt_growth_penalty`: subtracts a penalty proportional to the predicted
  rise in debt burden (relative debt growth over GDP growth) from the latest
  observed state to the final forecast step, so scoring acts on the
  predicted fiscal path exactly as the oracle evaluates the real one.
- `score_horizon_mean`: rank candidates by the objective averaged over the
  whole predicted path instead of the final step alone.

## Run the CPU baselines

The CLI does not require torch, CUDA, or model weights:

```bash
# Recent action-conditioned delta baseline
uv run main.py timeseries \
  --turns 32 \
  --model empirical \
  --forecast-horizon 8 \
  --candidate-limit 32 \
  --random-seed 20260813 \
  --trace-out traces/empirical.json

# Persistence baseline for comparison
uv run main.py timeseries \
  --turns 32 \
  --model persistence \
  --forecast-horizon 8 \
  --candidate-limit 32 \
  --random-seed 20260813 \
  --trace-out traces/persistence.json
```

The command can also save the final simulator snapshot with `--state-out` and
resume from a snapshot with `--state-in`. Keep traces and snapshots outside
the repository, or add an experiment-specific directory to the ignore list;
they are generated data rather than source fixtures.

`PersistenceForecaster` repeats the latest row. `EmpiricalActionForecaster`
uses recent observed deltas, keyed by policy and action type when examples are
available, and rolls its predicted rows forward for the requested horizon.
Neither is intended to be a strong policy model. They provide quick sanity
checks for the context plumbing, action conditioning, and evaluation scripts.

## Trace format

`agent.save_trace("forecast.json")` writes a JSON document with format marker
`autocracy-timeseries-trace-v1`:

```text
trace
├── format
├── context
│   ├── encoder
│   ├── states[]       # observed feature rows and turns
│   └── actions[]      # actions between consecutive observed states
└── decisions[]
    ├── turn
    ├── actions[]
    ├── forecast       # all requested future rows
    ├── score
    ├── candidate_count
    ├── observed       # actual next simulator row
    └── one_step_mae
```

`one_step_mae` is the raw mean absolute error across the encoded columns for
the first predicted row versus the observed next state. Because GDP, debt,
and normalized gauges have different units, use per-feature normalization or
report separate metric groups for serious comparisons; the trace retains the
raw rows needed to calculate those metrics later.

## Comparing models fairly

For a policy-performance comparison, run each model from the same initial
snapshot with the same simulator configuration, seed, feature schema,
candidate limit, and forecast horizon. Compare at least:

1. election wins and vote margin at every term boundary;
2. poll rate and peak poll rate;
3. debt, budget balance, and interest-rate trajectory;
4. per-feature one-step and multi-step forecast errors; and
5. cumulative policy actions and political-capital use.

The policy trajectories will diverge after the first different action, so
forecast accuracy and policy outcome should be reported separately. For a
strict one-step model comparison, replay a fixed action trace through the
simulator and compare each model's forecast with the same observed next
state. For a policy comparison, let each agent choose actions online and
compare the resulting campaigns from identical starting states.

Election resolution remains the simulator's deterministic expected-turnout
model. A forecast agent should be evaluated over complete terms rather than
only intermediate poll improvements: delayed effects such as income loss,
debt accumulation, and interest costs may not appear within a short horizon.

## Current limitations and next steps

- Forecasts currently cover encoded state features, not a reconstructed full
  `SimulationState`; only real simulator states are used for subsequent turns.
- The baseline agent evaluates no-op and single actions. Multi-action
  conditioning can be added after the single-action protocol is calibrated.
- Chronos-2 conditions on slider levels only; implementation delays are
  learned implicitly from history rather than encoded as separate columns.
- Situation intensities are visible in-game but not yet encoded; adding them
  is the natural next schema extension.
- Forecast quality for periodically resampled statistics depends on the
  placeholder-zero carry-forward described above.

## Closed-loop control comparison

`experiments/control_comparison.py` plays one full UK term (16 turns) with
the no-op, persistence, empirical, chronos-2-small, and simulator-oracle
agents from the same start, then reports election result, margin, mean poll
rate, and the weighted composite for each:

```bash
uv run --extra chronos python experiments/control_comparison.py \
  --turns 16 --horizon 5 --seed 20260813 \
  --out reports/control_comparison_h5.json
```

The oracle is `ElectionOracleAgent` with its documented winning defaults
(`PROVEN_ELECTION_SEARCH`: beam 6, depth-5 lookahead, two policy moves per
turn) — it branches the real simulator and wins from turn zero, so it is an
upper bound rather than a fair player-visible agent. See
`reports/chronos-2-small-vs-oracle.md` for the latest results.

## Single-life active learning (`autocracy.learning`)

Zero-shot candidate rankings from Chronos are nearly flat (the
`chronos-2-small-vs-oracle.md` diagnostic measured a predicted poll spread of
7.5e-5 against true single-move swings of ±6 points), so a purely model-ranked
campaign is noise. `TreatmentEffectMemory` supplies the missing signal using
only what the agent itself observes while playing one continuous life:

* every executed transition appends the observed poll change de-trended
  against the forecaster's **no-op counterfactual** (what the world model
  expected without any action) to a per-action table; repeats converge on
  measured treatment effects;
* an unseen signature inherits shrunk evidence (`family_shrinkage`) from its
  `(policy, direction)` family, so the next step of a proven slider ranks as
  promising instead of unknown;
* a second channel records each action's measured budget-balance effect
  (expenditure-normalised), which `fiscal_prudence_weight` applies while the
  visible budget runs a deficit;
* candidate pools are prioritised by learned effect plus curiosity
  (uniformly random among never-tried moves), concentrating expensive
  forecasts on informative candidates;
* curiosity is scaled by the share of the term still ahead
  (`exploration_countdown=True`), so early turns explore and late turns
  exploit before the election.

Leakage rules the design respects: no policy names or hand-picked moves
anywhere in the strategy path, no scripted warm-ups in the learning
experiment, fresh empty memory at turn 0, and no cross-episode reuse
(replaying the same save and carrying learned effects back would be oracle
look-ahead).

Run it:

```bash
uv run --extra chronos python experiments/chronos_learning.py \
  --runs 20 --terms 3 --out reports/chronos_single_life.json
```

Each run is one life from `uk0.xml`: win the first election, then survive two
re-elections with the same live campaign. At the shipped defaults, 4 of 20
lives win the first election (margins +17 to +86) and every winner then
sweeps both re-elections with growing margins (+592/+677, +758/+938,
+956/+1051, +851/+1192) and final polls of 0.72–0.85 — the online memory
keeps compounding once past the first-election hump. Lives that hit the
mid-term DebtCrisis still lose.

`experiments/diagnose_learning.py` quantifies *why*: probe mode measures
Spearman ρ between predicted and true per-candidate poll deltas (≈0 for both
chronos-2-small and the 120M chronos-2 base — and the base also wins less,
1/20 vs 4/20, when driving the full learning loop), while truth mode shows
that ranking the same candidate pools by their true one-turn effect wins 8/8
lives. The strategy is model-limited, not sampling-limited; any future
forecaster should be admitted by measuring its probe-ρ first. The declared
£ cost of each slider move rides on every option (`financial_delta`) but
using it for deficit scoring or exploration ordering proved
counterproductive — the winning path spends heavily and outruns the debt
spiral rather than avoiding it. With `--terms 5`, every first-election
winner survives all five elections with escalating margins, so survival past
the first boundary is solved; the open gap is reliably winning that first
vote.
