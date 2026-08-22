# Project plan: an in-context causal world model for few-shot sequential control

## 1. Project objective

Build a pretrained tabular foundation model that can enter a previously unseen dynamic system, observe a short history of states and interventions, infer useful causal dynamics in context, and choose interventions that maximise a long-term objective under costs and hard constraints.

The deployed model should not require:

* access to the underlying causal graph;
* gradient-based training on the new environment;
* environment resets;
* large numbers of exploratory interactions;
* explicit causal-graph reconstruction.

The intended control loop is:

[
\text{observed history}
\rightarrow
\text{intervention-conditioned world model}
\rightarrow
\text{search over future actions}
\rightarrow
\text{execute first action}
\rightarrow
\text{append observation to context}.
]

The primary test environment will be Democracy 3 through the existing `autocracy` simulator. The simulator already parses the original game assets, exposes policy actions and persistent snapshots, models situations and inertia, and can compare simulated states against real save files. Its current known problem is substantial divergence from actual game states and financial calculations, so simulator validation is the first project gate.

---

# 2. Core research hypotheses

The project should test four progressively stronger claims.

## H1: causal prediction

A model pretrained on synthetic temporal SCMs can infer from context:

[
p(S_{t+1}\mid H_t,do(A_t=a))
]

for an unseen dynamic system better than:

* an observational forecaster;
* a generic pretrained tabular regressor;
* conventional regressors fitted only to the short context.

## H2: few-shot adaptation

As interventions accumulate in the context, the model improves its predictions without changing its weights.

For example, observing policy levels 0–3 should allow prior-informed extrapolation to levels 4–6, while subsequent observations at those levels should update the inferred response curve.

## H3: control

Explicit search over the learned causal world model produces better long-term outcomes than:

* random legal actions;
* no-op;
* immediate-reward greedy control;
* search over a non-causal observational model.

## H4: transfer

A model pretrained only on broad synthetic dynamic SCMs transfers to Democracy 3 without training on Democracy 3 trajectories.

This is the main research contribution. Success should not depend on including the Democracy 3 graph or equations in the pretraining distribution.

---

# 3. What to reuse

## 3.1 `autocracy`

The current repository provides a suitable environment interface:

* original Democracy 3 asset parsing;
* graph construction;
* turn processing;
* policy changes;
* state serialization;
* savegame parsing;
* save-to-simulator comparison;
* an agent scaffold;
* situation thresholds and delayed inertia effects.

However, its own current comparison shows large first-turn discrepancies in important variables and especially policy incomes and expenditures. Until those discrepancies are understood, it cannot be treated as a trustworthy oracle for long-horizon evaluation.

## 3.2 CausalTimePrior

CausalTimePrior is a strong base for synthetic pretraining. It already supports:

* time-lagged temporal SCMs;
* observational and interventional trajectory pairs;
* hard, soft and time-varying interventions;
* multiple nonlinear mechanism families;
* configurable graph priors;
* regime changes;
* stability checks and burn-in;
* forward simulation from known SCMs.

Its existing API returns observational and interventional series together with the intervention and underlying SCM. It also contains a proof-of-concept PFN implementation and includes TempoPFN and a Do-PFN prior as submodules.

The generator should be forked and extended rather than rewritten.

## 3.3 TabICLv2

TabICLv2 is the better architectural starting point than original TabICL because it supports both classification and regression, exposes pretraining code, includes a minimal architecture implementation, and supports repeated inference with a fixed context through KV caching.

Its architecture is still fundamentally a tabular supervised learner: it represents rows using column-then-row processing and performs in-context prediction from labelled context rows.

That is enough for the first MVP, but not necessarily for the final model.

---

# 4. Environment interface

All environments—synthetic SCMs and Democracy 3—should implement the same minimal API.

```python
class ControlledEnvironment(Protocol):
    def reset(self, seed: int) -> Observation: ...
    def state(self) -> Observation: ...
    def legal_actions(self) -> list[Action]: ...
    def step(self, action: Action) -> StepResult: ...
    def clone(self) -> ControlledEnvironment: ...  # synthetic/oracle evaluation only
```

A step result should contain:

```python
@dataclass
class StepResult:
    next_state: dict[str, float | int]
    reward_components: dict[str, float]
    action_cost: float
    terminated: bool
    failure_reasons: list[str]
    metadata: dict[str, object]
```

`clone()` is needed for generating counterfactual branches and oracle search during synthetic training and evaluation. The deployed agent must not rely on it.

---

# 5. Define the control problem explicitly

For each environment, define:

* observable state variables (S_t);
* controllable action variables (A_t);
* legal discrete levels for each action;
* which policies may change per turn;
* action-change costs;
* target variables;
* cumulative reward;
* hard constraints;
* terminal failure conditions;
* planning horizon or discounting.

A suitable generic objective is:

[
G_{t:H}
=======

\sum_{k=1}^{H}
\gamma^{k-1}
r(S_{t+k},A_{t+k-1})
+
V_{\mathrm{terminal}}(S_{t+H}).
]

Hard constraints should be represented separately rather than only as large negative rewards:

[
S_{t+k}\in\mathcal S_{\mathrm{safe}},
\qquad
A_{t+k}\in\mathcal A(S_{t+k}),
\qquad
C(A_{t+k})\le B_{t+k}.
]

For Democracy 3, the first objective should be intentionally narrow:

> Maximise one selected target metric over a fixed number of turns while remaining solvent and avoiding electoral or other terminal failure.

Do not begin with an elaborate weighted social-welfare function.

---

# 6. Simulator-repair phase

## MVP 0A: deterministic one-turn parity

### Goal

Given the same initial save and no policy change, reproduce the next real save state closely enough to trust the basic update equations.

### Work

Create a structured parity suite containing:

* initial save;
* action or no-op;
* next real save;
* expected differences for every state variable;
* expected policy cost and income;
* situation activation changes;
* effect-memory and inertia state where inferable.

Test one subsystem at a time:

1. base node initialization;
2. policy slider normalization;
3. policy income and expenditure formulas;
4. direct influences;
5. link inertia;
6. situation trigger state;
7. situation effects;
8. voter/population updates;
9. political capital;
10. random events and dilemmas.

Record the first update stage at which each variable diverges.

### RNG handling

Run two validation modes:

* **deterministic core:** disable or freeze random events;
* **stochastic parity:** reproduce the game RNG where possible, or compare distributions over repeated runs.

Do not explain every discrepancy as RNG. The current financial differences are too large and systematic for that to be the default assumption.

### Expected MVP result

* Most non-random continuous state variables agree within a predefined tolerance after one no-op turn.
* Policy income and spending totals agree closely.
* Remaining mismatches are isolated to named unsupported systems.

### Gate

Do not proceed to long-horizon Democracy evaluation until one-turn parity is credible.

---

## MVP 0B: controlled intervention parity

### Goal

Validate the simulator’s response to known policy changes.

### Work

Capture real-game transitions for a small intervention matrix:

* no-op;
* increase one tax by one level;
* decrease it by one level;
* increase one spending policy;
* introduce or cancel a policy if practical;
* repeat selected actions for several turns to expose inertia.

Prefer a small set of policies with clear direct and delayed effects.

### Expected MVP result

For each tested action:

* direction of effect is correct;
* immediate changes are close;
* delayed response curves are approximately reproduced;
* costs and political-capital effects are correct.

### Deliverable

A machine-readable `parity_cases/` corpus that later becomes a held-out simulator-regression suite.

---

# 7. Synthetic control prior

## MVP 1A: controlled CausalTimePrior

Extend CausalTimePrior with explicit state and action roles.

### Original form

A temporal SCM generates:

[
X_t^{(i)}
=========

f_i(X_t,X_{t-1},\ldots,X_{t-K})+\epsilon_t.
]

### Controlled form

Use:

[
S_{t+1}^{(i)}
=============

f_i(
S_t,\ldots,S_{t-K},
A_t,\ldots,A_{t-K_A},
U_{t+1}
).
]

Actions are supplied externally and must not be generated by the SCM’s ordinary endogenous mechanisms.

### Required generator additions

1. **Discrete ordered actions**

   Each action has perhaps 3–8 levels.

2. **Persistent settings**

   A policy remains at its previous level until explicitly changed.

3. **Change actions**

   Distinguish setting a level from changing by (+1) or (-1).

4. **Action costs**

   Include costs for activation, cancellation and movement.

5. **Delayed implementation**

   Sample fixed delays, gradual phase-in and decay.

6. **Action-response families**

   Sample mixtures of:

   * linear;
   * monotonic nonlinear;
   * saturating;
   * thresholded;
   * piecewise;
   * U-shaped;
   * sign-changing;
   * interaction effects.

7. **Stock variables**

   Include quantities such as debt or capital that accumulate over time.

8. **Hard constraints and absorbing failures**

9. **Reward definitions**

   Randomly designate target variables and objective forms.

10. **Partial observability**

Add later, not in the first generator.

### Expected MVP result

The generator can produce thousands of stable, heterogeneous controlled systems where:

* legal discrete interventions change future states;
* effects can be immediate or lagged;
* action sequences have meaning;
* the true simulator can branch from any observed prefix.

CausalTimePrior already provides the graph, nonlinear mechanisms, intervention machinery and regime-changing foundation needed for this extension.

---

## MVP 1B: held-out prior families

Do not randomly split individual trajectories from the same generator configuration.

Create distinct test families:

* unseen graph sizes;
* denser and sparser graphs;
* longer lags;
* held-out mechanism families;
* unseen action-level counts;
* stronger thresholds;
* different noise distributions;
* different reward definitions;
* regime changes;
* partial observation.

This measures whether the model learned a reusable inference algorithm rather than memorising the exact prior.

---

# 8. Training-data representation

The first training format should use ordinary scalar regression. Avoid a multi-output world-model head initially.

## 8.1 Long-form scalar target representation

Represent each prediction query as:

[
(H_t,;A_t^{\mathrm{candidate}},;j)
\rightarrow
S_{t+1}^{(j)},
]

where (j) identifies the state variable being predicted.

One full next-state prediction therefore requires one query per target variable.

### Context rows

Each observed transition becomes a supervised row:

```text
environment-local variable schema
history features
action at t
target variable id
next target value
```

A conceptual row:

```json
{
  "state_t.GDP": 0.42,
  "state_t.Unemployment": 0.31,
  "state_t.Debt": 0.66,
  "state_t_minus_1.GDP": 0.40,
  "action.IncomeTax": 4,
  "action.StateSchools": 3,
  "changed_action": "IncomeTax",
  "time_since_change.IncomeTax": 0,
  "target_variable": "GDP",
  "y": 0.435
}
```

The query row supplies the same features and candidate action but hides `y`.

## 8.2 Why include a target-variable identifier?

TabICL’s regression head predicts a scalar. Encoding `target_variable` lets one shared model predict every next-state dimension without adding a vector output head.

It also permits:

* varying numbers of state variables between SCMs;
* one unified loss;
* selective prediction of only planner-relevant variables;
* easy batching.

## 8.3 Encode time explicitly

TabICL rows are not intrinsically a temporal sequence. Therefore, the row features must contain the relevant temporal state:

* current values;
* lagged values;
* action history;
* time since action changes;
* absolute or relative timestep;
* active regimes;
* missingness masks.

For the first model, use fixed lag columns:

[
S_t,;S_{t-1},\ldots,S_{t-K},
\quad
A_t,;A_{t-1},\ldots,A_{t-K_A}.
]

This makes the problem ordinary tabular regression.

## 8.4 Intervention semantics

Include explicit metadata:

* feature role: state, action, constraint or target identifier;
* action is observed historical or proposed;
* action level;
* whether the level changed;
* persistence duration.

A proposed intervention must not be indistinguishable from a passively observed feature value.

## 8.5 Recommended stored episode format

Store native episodes, not only flattened rows:

```json
{
  "environment_id": "...",
  "schema": {
    "state_variables": [...],
    "action_variables": [...],
    "action_levels": {...},
    "constraints": [...]
  },
  "objective": {...},
  "steps": [
    {
      "t": 0,
      "state": {...},
      "action": {...},
      "next_state": {...},
      "reward": {...},
      "terminated": false
    }
  ]
}
```

Flatten into TabICL rows dynamically during pretraining. This avoids coupling the generated dataset permanently to one architecture.

---

# 9. Can the ordinary TabICL regression head be used?

## MVP answer: yes

For one-step scalar prediction, the existing regression head is sufficient.

Train the model to estimate:

[
\hat y
======

# \hat S_{t+1}^{(j)}

f_\theta(H_t,A_t,j).
]

TabICLv2 already has a released regression model and pretraining support, although the repository notes that the migrated v2 pretraining scripts have not yet been fully tested end-to-end for checkpoint reproduction.

However, the released generic regression checkpoint should be treated only as:

* an initialization candidate;
* a generic-regression baseline;
* a preprocessing and architecture reference.

It has not been pretrained to understand dynamic interventions.

## What must change even for the MVP

The architecture can stay mostly unchanged, but the **pretraining task and prior must change**:

* context rows must come from the same dynamic SCM;
* action columns must have intervention semantics;
* temporal lags must be encoded;
* query rows must specify candidate actions;
* the target-variable identity must be represented;
* train/test sampling must happen at the SCM level.

## Probable small architecture changes

Add learned embeddings for:

* column role;
* target-variable role;
* lag index;
* observed versus candidate value;
* state versus action.

These should be additive metadata embeddings around the existing feature representation, not a complete redesign.

## Likely later limitation

TabICL’s row-based supervised interface does not naturally preserve the structured relationship:

[
\text{time}\times\text{variable}\times\text{action role}.
]

If the flattened version plateaus, move to a hierarchical architecture:

1. encode variables within each timestep;
2. encode timesteps over history;
3. condition on candidate action;
4. decode queried next-state variables.

Do not make this change before showing that flattened TabICL is inadequate.

---

# 10. Predict deltas, levels, or both?

Train two targets:

[
S_{t+1}^{(j)}
]

and

[
\Delta S_{t+1}^{(j)}
====================

S_{t+1}^{(j)}-S_t^{(j)}.
]

For slowly moving systems, delta prediction gives a stronger learning signal for intervention effects. Absolute-level prediction anchors the rollout and prevents accumulated drift.

A combined loss is appropriate:

[
\mathcal L
==========

\lambda_{\mathrm{level}}
\mathcal L_{\mathrm{level}}
+
\lambda_{\Delta}
\mathcal L_{\Delta}.
]

For the simplest implementation, begin with delta prediction and reconstruct the level. Add an absolute-state auxiliary head if rollout drift is severe.

---

# 11. World-model MVPs

## MVP 2A: one-step synthetic prediction

### Task

Given a context of observed transitions from an unseen synthetic SCM and a candidate action, predict the next state.

### Comparisons

* copy-last-state;
* linear autoregression;
* VAR;
* CatBoost fitted to the context;
* generic TabICLv2 regressor without causal pretraining;
* causal-pretrained TabICL;
* observational-only ablation that is not told which columns are interventions;
* CausalTimePrior’s proof-of-concept PFN where adaptable.

CausalTimePrior itself includes VAR-OLS and PCMCI+ baselines, which can be retained for causal-prediction comparisons.

### Metrics

* normalized RMSE or robust-scaled error;
* delta prediction error;
* error by intervention distance from observed support;
* error by lag horizon;
* error by context size;
* sign accuracy for intervention effects;
* ranking accuracy across candidate actions.

### Expected result

The causal-pretrained model should:

* improve as context interventions accumulate;
* outperform the generic regression checkpoint in low-context settings;
* outperform the observational ablation on counterfactual candidate actions;
* degrade gracefully when extrapolating to unseen action levels.

---

## MVP 2B: extrapolation and prior updating

Construct explicit tests where the context contains only action levels 0–3 and queries levels 4–6.

Evaluate:

* prediction error before observing higher levels;
* uncertainty or ensemble disagreement;
* error after one high-level intervention enters context;
* error after several such interventions.

This directly tests the intended prior-driven extrapolation behaviour.

Even if calibration is not a final product goal, ensemble variation is useful for detecting unsupported planner queries.

---

## MVP 2C: multistep open-loop prediction

The model remains one-step, but is rolled forward recursively.

Evaluate horizons:

[
h\in{1,2,4,8,16}.
]

Compare:

* recursive one-step rollout;
* direct horizon-conditioned prediction;
* one-step training with 2- and 4-step auxiliary losses.

### Expected result

The first version will likely degrade sharply with horizon. The goal is not perfect long-term forecasting; it is enough local accuracy for receding-horizon search to improve actions.

---

# 12. Search and control

## MVP 3A: greedy one-step control

Enumerate every legal next action and select:

[
a_t
===

\arg\max_a
r(\hat S_{t+1}(a),a).
]

### Purpose

This is not expected to solve the game. It validates:

* action enumeration;
* batched candidate prediction;
* reward calculation;
* constraint filtering;
* closed-loop context updates.

### Expected result

It should beat random action selection on objectives dominated by immediate effects, while failing on delayed-benefit decisions.

That failure is useful evidence for multistep planning.

---

## MVP 3B: open-loop action-sequence search

Use beam search over action sequences.

For each candidate sequence:

[
a_{t:t+H-1},
]

recursively predict future states and score:

[
\hat G
======

\sum_{k=1}^{H}
\gamma^{k-1}
r(\hat S_{t+k},a_{t+k-1}).
]

Execute only the first action, observe the real next state and replan.

### Start with

* horizon: 3–5 turns;
* small beam width;
* one policy change per turn;
* deterministic mean predictions;
* no-op always included;
* hard rejection of predicted constraint violations.

Because all sequences at a given depth use the same context structure, candidate predictions should be batched.

### Expected result

Beam search should outperform greedy control on synthetic environments containing delayed effects, assuming rollout error remains manageable.

---

## MVP 3C: uncertainty-aware search

Use multiple model ensemble members or posterior-style prediction variants.

Score sequences using:

[
J(a)
====

## \mathbb E[\hat G(a)]

## \lambda_{\mathrm{unc}}\operatorname{SD}[\hat G(a)]

\lambda_{\mathrm{fail}}\widehat P(\text{failure}\mid a).
]

This prevents the planner from systematically selecting actions that exploit one optimistic extrapolation.

### Expected result

It may slightly reduce average return in easy environments but should reduce catastrophic failures and unsupported extreme actions.

---

# 13. Democracy 3 transfer

## MVP 4A: frozen-model one-step prediction

Do not train on Democracy trajectories.

Supply a short context collected from `autocracy`, then query the model with candidate actions.

Evaluate against simulator transitions:

* next-state error;
* effect-direction accuracy;
* candidate-action ranking;
* improvement as the context grows.

Run two versions:

1. a faithful repaired simulator;
2. deliberately perturbed simulator variants.

Perturbations are important because a model should infer the encountered system rather than rely on one fixed Democracy configuration.

## Expected result

A model pretrained only on broad synthetic SCMs will probably not initially predict every game variable accurately. A credible MVP would be:

* useful action ranking for a selected subset of variables;
* visible adaptation from 1, 2, 4 and 8 observed interventions;
* better low-context performance than regressors fitted from scratch.

---

## MVP 4B: constrained single-target control

Choose one target and a reduced action set, for example:

* 5–10 policies;
* one action change per turn;
* fixed horizon;
* budget constraint;
* no stochastic events initially.

Compare:

* no-op;
* random;
* heuristic;
* greedy learned model;
* beam-search learned model;
* beam search using the true simulator;
* search using a generic non-causal model.

### Core metric

Use normalized regret:

[
\operatorname{Regret}
=====================

\frac{
V_{\mathrm{oracle}}-V_{\mathrm{agent}}
}{
|V_{\mathrm{oracle}}-V_{\mathrm{baseline}}|+\epsilon
}.
]

Also report:

* achieved return;
* failure rate;
* cumulative action cost;
* prediction queries;
* wall-clock decision time;
* performance against context length.

### Expected result

The learned planner should sit between the heuristic baseline and oracle simulator planner. It need not approach oracle performance in the first version.

---

## MVP 4C: full-game constraints

Add:

* more policies;
* multiple objectives;
* elections;
* political capital;
* debt;
* situations;
* stochastic events;
* terminal failure;
* longer horizons.

This is the first point at which “playing Democracy 3” becomes a fair description.

---

# 14. Criteo Uplift benchmark

## Why include it

Criteo is useful as a **static causal-estimation sanity check**, not as evidence that the sequential world model works.

The dataset was created from randomized advertising incrementality tests and contains millions of user rows with features, treatment and visit/conversion outcomes.

A recent UpliftBench comparison evaluates S-, T- and X-learners with LightGBM and an EconML causal forest on the approximately 14-million-row Criteo v2.1 data.

## Proposed task

Convert each Criteo example into:

[
(X,T)
\rightarrow
Y.
]

Use the causal-pretrained model as:

* an S-learner-style outcome model;
* or two treatment-conditioned outcome queries.

Estimate:

[
\hat\tau(x)
===========

## \hat P(Y=1\mid x,do(T=1))

\hat P(Y=1\mid x,do(T=0)).
]

## Metrics

* AUUC;
* Qini;
* uplift in top decile;
* policy value for fixed treatment budgets;
* factual outcome log loss as a secondary metric.

## Baselines

* LightGBM S-learner;
* LightGBM T-learner;
* LightGBM X-learner;
* causal forest;
* generic TabICL;
* causal-pretrained TabICL.

## Expected result

A respectable static uplift result would show that causal pretraining has not damaged ordinary treatment-effect learning.

It is not necessary to beat highly tuned industrial uplift models in the first project. Criteo differs substantially from the dynamic few-shot objective:

* it has a very large randomized dataset;
* one binary treatment;
* one decision;
* no lagged state;
* no interactive context;
* no sequential planning.

Therefore, Criteo should be an auxiliary benchmark after synthetic one-step prediction, not MVP 1.

---

# 15. Dreamer comparison

DreamerV3 learns a latent world model from replayed experience and trains an actor–critic from imagined trajectories. Its world model predicts future latent states and rewards conditioned on actions.

The comparison must be designed around the actual advantage claimed by this project: **pretraining and in-context adaptation under severe interaction limits**.

## Unfair comparison to avoid

Do not compare:

* a heavily pretrained causal PFN;
* against Dreamer initialized randomly;
* after giving both only one transition;

and present that as a general superiority result.

That only shows the value of prior knowledge, which is expected.

## Fair comparison axes

### Axis A: interactions in the new environment

Evaluate after:

* 0 action transitions;
* 1;
* 2;
* 4;
* 8;
* 16;
* 32;
* 64;
* larger online budgets.

### Axis B: adaptation mechanism

Compare:

1. pretrained causal model, context updates only;
2. Dreamer from scratch;
3. Dreamer pretrained across the same synthetic environment distribution, then fine-tuned;
4. contextual/meta world-model baseline where available;
5. causal model plus search;
6. ablation without causal intervention semantics.

### Axis C: total compute

Report separately:

* pretraining compute;
* adaptation gradient steps;
* real environment interactions;
* imagined rollouts;
* inference/search compute.

## Expected result

The causal PFN should have its largest advantage at 0–16 interactions.

Dreamer may become stronger after enough task-specific experience and optimization because it continually adapts its parameters and actor. Dreamer’s published implementation is explicitly designed to train its world model and actor–critic concurrently from replayed experience.

The strongest defensible claim would be:

> Better return and lower failure in the very-low-interaction regime, while remaining competitive as more observations accumulate.

---

# 16. Other relevant comparison families

## One-Shot World Model

One-Shot World Model is directly relevant: it proposes a transformer world model learned through in-context learning from synthetic data. It should be included if its implementation and environment assumptions can be adapted to structured tabular state.

This is likely a more conceptually direct comparison than a causal-graph learner.

## Contextual latent world models and meta-RL

Recent contextual world-model work studies task inference from transition histories in offline meta-RL. This is relevant to the representation and adaptation problem, though not necessarily to explicit SCM priors or zero-gradient deployment.

## In-context RL transformers

Recent theoretical and empirical work shows transformers can implement policy-improvement algorithms in context on distributions of random tabular MDPs. This is relevant later if adding a direct policy head, but not needed for the initial world-model-plus-search design.

## Causal discovery models

Arrow, CDFM and DAG-FM provide useful causal-discovery comparisons if graph recovery becomes an auxiliary evaluation. However, they solve a different problem: explicitly predict a graph from data. Your proposed method deliberately avoids paying that cost unless graph reconstruction improves control.

A good experiment later would compare:

1. discover a temporal graph;
2. fit mechanisms on the discovered graph;
3. plan through the fitted SCM;

against direct intervention-conditioned world modelling.

---

# 17. Evaluation hierarchy

Do not judge the project only by prediction error.

Use four levels.

## Level 1: factual prediction

Can it predict held-out observed next states?

Necessary, but not causal.

## Level 2: intervention prediction

Can it predict outcomes under actions not taken in the factual history?

This is the first real causal test.

## Level 3: action ranking

Does it correctly rank candidate interventions even when absolute predictions are imperfect?

This may be sufficient for useful control.

## Level 4: closed-loop return

Does model-based search produce high cumulative reward with low failure?

This is the final criterion.

A model can have mediocre RMSE but good action ranking. Conversely, a model can predict common no-op transitions accurately while choosing disastrous interventions.

---

# 18. Suggested repository structure

```text
autocracy/
├── autocracy/                 # Democracy simulator
├── parity_cases/              # Real-game validation transitions
├── environments/
│   ├── protocol.py
│   ├── democracy.py
│   └── synthetic.py
├── causal_prior/
│   ├── generator.py           # CausalTimePrior extensions
│   ├── actions.py
│   ├── rewards.py
│   ├── constraints.py
│   └── episode_schema.py
├── data/
│   ├── generate.py
│   ├── flatten_tabicl.py
│   └── validation_splits.py
├── models/
│   ├── tabicl_world_model.py
│   ├── generic_tabicl.py
│   └── baselines.py
├── planning/
│   ├── greedy.py
│   ├── beam.py
│   ├── rollout.py
│   └── objectives.py
├── agents/
│   ├── world_model_agent.py
│   ├── random_agent.py
│   └── heuristic_agent.py
├── benchmarks/
│   ├── synthetic_prediction.py
│   ├── synthetic_control.py
│   ├── democracy_control.py
│   ├── criteo_uplift.py
│   └── dreamer_adapter.py
└── reports/
    └── experiment_cards/
```

The CausalTimePrior fork could initially remain a dependency or submodule, but the controlled-environment extensions should live behind your own stable interface.

---

# 19. Recommended order of work

## Phase 0: trustworthy environment

1. Add detailed update-stage tracing to `autocracy`.
2. Build no-op save parity cases.
3. Fix financial scaling and major node discrepancies.
4. Add controlled intervention parity cases.
5. Freeze a deterministic simulator mode.

**Exit condition:** one-turn simulator transitions are trustworthy.

## Phase 1: synthetic prior and dataset

1. Fork CausalTimePrior.
2. Add persistent discrete action inputs.
3. Add costs, objectives and constraints.
4. Add delayed and piecewise action effects.
5. Define the native episode format.
6. Generate held-out prior families.

**Exit condition:** synthetic environments support branching intervention queries and stable multistep simulation.

## Phase 2: scalar causal regression

1. Build long-form target-variable queries.
2. Run generic TabICLv2 regression without retraining.
3. Retrain or pretrain the same architecture on controlled SCM episodes.
4. Add role, lag and intervention embeddings if needed.
5. Benchmark one-step intervention prediction.

**Exit condition:** causal pretraining improves few-shot interventional prediction over generic and observational baselines.

## Phase 3: model-based control

1. Add greedy candidate enumeration.
2. Add recursive world-model rollout.
3. Add batched beam search.
4. Add hard constraints.
5. Compare against oracle search on synthetic SCMs.
6. Measure regret and planner exploitation.

**Exit condition:** learned-model search beats greedy and random policies on delayed-effect synthetic tasks.

## Phase 4: Democracy transfer

1. Define a reduced action and state subset.
2. Evaluate frozen one-step prediction.
3. Run closed-loop constrained control.
4. Expand action and state spaces.
5. Add elections, events and full constraints.

**Exit condition:** the synthetic-pretrained model produces measurable few-shot control gains without Democracy-specific weight training.

## Phase 5: external benchmarks

1. Criteo Uplift static sanity check.
2. One-Shot World Model comparison.
3. Dreamer interaction-efficiency curves.
4. Optional graph-discovery-plus-planning baseline.
5. Direct policy-head or amortised-planning experiment.

---

# 20. What each milestone should realistically demonstrate

| MVP | Expected demonstration                          | What it does **not** yet prove          |
| --- | ----------------------------------------------- | --------------------------------------- |
| 0A  | Simulator matches a no-op game turn             | Correct action effects                  |
| 0B  | Simulator matches selected interventions        | Long-horizon fidelity                   |
| 1A  | Broad controlled dynamic SCM generation         | Learned causal inference                |
| 2A  | One-step interventional prediction              | Useful control                          |
| 2B  | Prior-driven extrapolation and context updating | Safe extrapolation everywhere           |
| 2C  | Some multistep rollout stability                | Good planning                           |
| 3A  | End-to-end agent plumbing                       | Long-term rationality                   |
| 3B  | Search handles delayed effects synthetically    | Democracy transfer                      |
| 3C  | Lower catastrophic planner exploitation         | Perfect uncertainty                     |
| 4A  | Zero-shot/few-shot Democracy prediction         | Winning the game                        |
| 4B  | Useful reduced-scope Democracy control          | General real-world transfer             |
| 4C  | Full constrained game performance               | General causal intelligence             |
| 5   | Competitive external positioning                | Identical assumptions across benchmarks |

---

# 21. Early architecture decision

The initial model should be:

[
\boxed{
\text{TabICLv2-style regressor}
+
\text{causal temporal pretraining}
+
\text{scalar queried target}
+
\text{external beam search}
}
]

Not:

* a direct action policy;
* a full vector-generating sequence model;
* a graph decoder;
* a Dreamer-style actor–critic;
* a custom temporal transformer from the start.

The ordinary scalar regression head is enough for the first scientific question. The likely necessary modifications are metadata embeddings and the training-data construction, not the output head.

Only redesign the architecture after measuring one of these concrete failures:

* temporal information is lost despite explicit lags;
* scalar target queries are too expensive;
* predictions across state variables are incoherent;
* imagined rollouts become inconsistent;
* variable counts or histories exceed practical context limits;
* context updates cannot be cached efficiently.

---

# 22. Final success criterion

The most important final figure should show average return against the number of real interactions in an unseen environment:

[
\text{return}
\quad\text{versus}\quad
\text{new-environment interactions}.
]

Include:

* random;
* heuristic;
* generic TabICL world model;
* causal-pretrained world model;
* causal-pretrained world model plus search;
* Dreamer from scratch;
* pretrained or meta-trained Dreamer where feasible;
* oracle simulator planner.

The project succeeds if causal pretraining produces a substantial leftward shift:

> The agent reaches useful control performance with dramatically fewer real interactions, especially in the one-shot and few-shot region.

That is more meaningful than merely recovering the hidden DAG or obtaining the best one-step prediction score.
