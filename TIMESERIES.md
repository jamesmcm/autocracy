# Autoregressive time-series experiments

This document describes the forecasting experiment boundary in
`autocracy.timeseries`. It is deliberately independent of a particular model
library so the same simulator rollout can be used with the CPU baselines now
and a GPU-backed Chronos2 implementation later.

## What is being forecast?

The simulator has a multivariate state rather than one scalar series. A
`StateFeatureEncoder` turns the selected simulator fields into a stable row of
named floating-point features. By default it includes:

- all ordinary simulator values;
- all policy slider values;
- finance fields such as political capital, income, expenditure, debt, and
  interest rate; and
- poll, election countdown, term, result, and last-winner fields.

The encoder sorts value and policy names once when it is created. That fixed
column order is stored in the context and must be used by every forecast
backend. A smaller schema can be selected for an initial model run:

```python
from autocracy import simulator
from autocracy.timeseries import StateFeatureEncoder

state, _ = simulator.get_initial_state("uk")
encoder = StateFeatureEncoder.from_state(
    state,
    value_names=["GDP", "Health", "Education", "CrimeRate", "Unemployment"],
    policy_names=["IncomeTax", "StateHealthService"],
)
```

## Autoregressive control loop

At every decision turn, `TimeSeriesPolicyAgent` performs the following loop:

```text
observed state[t]
       │
       ├─ encode the complete observed context
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
they were real game state. The next prediction sees the actual simulator
state, which makes the rollout autoregressive while preventing model error
from silently replacing the environment.

The default candidate set is the no-op plus a bounded set of legal single
policy actions. `candidate_limit` includes the no-op. Candidate sampling is
reproducible when `random_seed` is supplied. The current scaffold does not
enumerate multi-policy batches; the existing beam-search oracle remains the
appropriate experiment for that action space.

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

## Model input and backend contract

Every candidate is converted into a `ForecastModelInput` containing:

| Field | Meaning |
| --- | --- |
| `feature_names` | Fixed ordered feature columns. |
| `history` | Observed rows, oldest to newest, as tuples of floats. |
| `turns` | Simulator turn for each observed row. |
| `action_history` | Actions that produced each observed transition. |
| `pending_actions` | Candidate action being evaluated now. An empty tuple is no-op. |
| `horizon` | Number of future rows required. |

A backend implements the small `ActionConditionedForecaster` protocol:

```python
from collections.abc import Mapping, Sequence

from autocracy.timeseries import (
    Chronos2Forecaster,
    ForecastModelInput,
    TimeSeriesPolicyAgent,
)


def gpu_predict(
    model_input: ForecastModelInput,
) -> Sequence[Mapping[str, float]]:
    # Convert model_input.history to the tensor/layout expected by the
    # installed model, condition on model_input.pending_actions, and return
    # one named row for every requested future step.
    raise NotImplementedError


agent = TimeSeriesPolicyAgent(
    Chronos2Forecaster.from_callable(gpu_predict),
    forecast_horizon=8,
)
```

The callable must return at least `horizon` rows. Each row should contain all
`feature_names`; missing fields are currently filled with zero by the adapter,
so a production backend should validate its output before returning it. The
adapter also accepts a pre-built `StateForecast` when the backend wants to
attach additional metadata.

`Chronos2Forecaster` intentionally does not import Chronos2, PyTorch, or CUDA.
This VPS can therefore run the surrounding experiment and collect traces. On
a GPU machine, the future integration only needs to implement the conversion
between `ForecastModelInput` and the installed Chronos2 pipeline. The agent,
feature schema, action records, and trace format do not need to change.

The CLI's `--model chronos2` option currently fails explicitly with a message
directing callers to the Python adapter. This prevents an accidental model
download or an opaque CPU out-of-memory attempt on the VPS.

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
- The context stores action names, types, and deltas. A future model may also
  encode action costs, implementation delay, or candidate metadata as extra
  fixed columns.
- Chronos2 integration is intentionally deferred until a GPU is available.
  The adapter boundary is the place to add batching, scaling, checkpoint
  loading, and device management without coupling those concerns to the
  simulator.
