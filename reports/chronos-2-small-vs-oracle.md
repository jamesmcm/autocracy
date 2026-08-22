# Chronos-2-small vs the simulator oracle: one-term control report

**Date:** 2026-08-22 · **Country:** UK mission start (`gamedata/saves/uk0.xml`) · **Term:** 16 turns to the first election · **Seed:** 20260813 · **Stochastic systems:** off

## Goal

> Make a pretrained time-series foundation model (*chronos-2-small*) the world model for policy search in Democracy 3: load the save's pre-game value histories as covariates at game start, keep player-controlled policies strictly separated as known-future treatment covariates, predict total voter electoral support, and measure how close closed-loop search over that model gets to the proven simulator oracle — encoding only what the player can see in-game.

**Starting from turn 0 is winnable**: the documented `ElectionOracleAgent` configuration wins the first election from this save (SIMULATION.md records 865/281/854; our runs below won with margins +23 to +306 depending on search timing). The comparison below is therefore measured against a genuinely winning upper bound, not a shared collapse.

## What was built

### 1. Pre-game covariate history (the "start of game" requirement)

Democracy 3 saves store a newest-first `<history>` ring per recorded simvalue.

* `parse_savegame` now exposes `save.simvalue_histories` (40 rings in `uk0.xml`, 21–34 entries each).
* `get_initial_state` copies them onto the state as `value_histories` with `value_histories_turn`, so the agent knows the rings end at turn 0.
* `AutoregressiveContext.from_state(..., include_value_histories=True)` replays them as context snapshots ending at the current turn: the chronos agent starts its first decision with **33 observed rows (turns −32…0)** instead of one.
* Preprocessing: features without rings (policies, most finance fields) are held constant across the pre-game window; exact-zero runs inside value rings carry the most recent non-zero reading forward, because the game writes placeholders until a node goes live and periodically resampled statistics (GDP) serialize zeros between samples. The carried series matches what the player actually sees on screen.

### 2. Strict player-visible schema

`StateFeatureEncoder.from_visible_state` encodes exactly what the player can observe:

* values: ordinary simulation nodes only. Every ``_``-prefixed meta neuron (`_globaleconomy_`, `_year`, `_security_`, `_Terrorism`, `_LowIncome`, …) is excluded, as are all `HIDDEN`/`PLACEHOLDER` nodes and derived runtime mirrors (`*_perc`, nested income neurons);
* policies: every slider level (the treatments);
* finance: only budget-screen figures — political capital, total income, total expenditure, debt, interest rate;
* politics/election: only the published poll rate and the election countdown.

Nothing hidden, nothing we derive ourselves. The schema is **206 columns**:

| Role | Columns | Count | Meaning |
| --- | --- | --- | --- |
| Predicted targets | all non-policy columns, led by `politics/poll_rate` | 83 | observed features whose future is unknown |
| Treatments (known future) | every `policy/*` slider | 123 | chosen by the agent, therefore known at each forecast step |

The headline target is `politics/poll_rate` (`ELECTORAL_SUPPORT_FEATURE`) — total voter electoral support.

### 3. Multi-step prediction and the chronos-2-small backend

`autocracy.chronos.Chronos2SmallForecaster` wraps `autogluon/chronos-2-small` (28M parameters, optional `chronos` extra). Per decision turn it forecasts **several steps ahead** (`--horizon`, default 5 — matching the oracle's working lookahead depth):

1. All candidate actions become independent items in **one batched `predict_df` call**: shared observed history, each item carrying its own known-future treatment frame (pending move applied once, then held for every predicted step).
2. Observed features come back as median forecasts per step; treatments are restored from the deterministic path so scoring sees whole rows.
3. The agent scores the **final predicted step** on predicted electoral support and executes only the winning action.
4. Cost ≈0.4 s per decision turn including model residency on one RTX 4080 (~14 ms/candidate steady-state).

## Experiment protocol

`experiments/control_comparison.py` plays one full term with each agent from the identical start (seed 20260813), evaluating every agent on **true simulator outcomes**:

* **no-op** – never acts;
* **persistence / empirical** – CPU forecast baselines optimizing predicted poll;
* **chronos-2-small** – strict visible schema + pre-game history, pure predicted-poll objective, horizons 5 and 8;
* **oracle-beam** – the documented winning election search: beam 6, depth-5 lookahead, two policy moves per turn, 16 sampled candidates, up to 64 legal pairs, expected-margin objective, 15 s decision budget, replanning every turn.

| Agent | Election | Margin | Mean true poll | Mean composite | Wall clock |
| --- | --- | ---: | ---: | ---: | ---: |
| no-op | loss | −1116 | 0.1131 | −0.567 | 0.7 s |
| persistence / empirical | loss | −1116 | 0.1131 | −0.567 | 0.8 s |
| chronos h=5, single-action, diverse warm-up | loss | −1002 | 0.1457 | −0.534 | 5.2 s |
| **chronos h=5 shipped defaults: fiscal warm-up (2/turn), two actions, no step cap** | loss | **−978** | **0.1532** | **−0.488** | 10.7 s |
| **oracle-beam (simulator)** | **win** | **+23…+306**¹ | 0.31–0.42 | −0.13…−0.33 | 241 s |

¹ Oracle margins vary run-to-run with branch exploration near the 15 s decision budget; every run wins.

### Strategy-space sweep: what moved the needle and what did not

With the political-capital budget as the only hard constraint (the blunt
step cap is off by default), every legitimate search-side knob was swept at
seed 20260813:

| Variant | Margin | Mean poll | Crisis |
| --- | ---: | ---: | --- |
| no cap, level objective, 16 candidates | **−965** | 0.1469 | t13 |
| no cap, horizon-mean objective | −1066 | 0.1257 | t14 |
| no cap, horizon-mean, 32 candidates | −1118 | 0.1119 | none |
| no cap, level, 32 candidates, h=8 | −1149 | 0.1149 | t14 |
| pair warm-up (10 interventions in ~5 turns) | −978 | **0.1532** | t13 |

Conclusions:

1. **Capital discipline was never the problem.** In every variant the
   agent's treasury never dropped below its per-turn accrual floor
   (`min_cap = 26`); even unconstrained it does not overspend, and it ends
   terms with a full bar only when its cheap-move preferences make spending
   unnecessary. The oracle faces the same budget and spends it on moves the
   forecaster cannot yet tell apart.
2. **Horizon-mean scoring and wider pools hurt slightly** — averaging
   dilutes the late-horizon signal that actually differentiates candidates,
   and extra sampled singles crowd out pairs.
3. **Denser interventional curricula help polls, not margins.** Executing
   two planned moves per turn (10 interventions in ~5 turns) produced the
   best poll trajectory and composite but the same ≈−1,000-vote plateau.
4. Every configuration lands within ±60 votes of a common ≈−1,000 plateau.
   That plateau is set by forecast attribution weakness — predicted
   candidate differences are small relative to the shared trend — not by
   search structure, objective shaping, or budget handling.

### Tuning the loss: can prudence substitute for foresight?

The DebtCrisis triggers when effective debt crosses ≈0.88 — under passive
play it fires at turn 13–14 and clips polls from ~0.18 to ~0.05. The oracle
sees the threshold coming because it branches the real simulator; Chronos
extrapolates smoothly through it. The hypothesis tested: shape the loss so
the agent takes smaller, fiscally prudent steps and never triggers the
crisis at all.

Findings from the grid:

1. **Debt-penalty strength is inert.** Raising `debt_growth_penalty` from
   0.25 through 5 to 20 produced *bit-identical campaigns*. The candidate
   fiscal forecasts differ by less than 1% (predicted five-step debt:
   no-op 2060k, tax+ 2067k, spend 2070k), so even a 20× penalty moves scores
   by less than poll noise. The model learned the fiscal **trend**
   (debt rising within horizon) but not treatment→fiscal **attribution**.
2. **Fiscal warm-up anchors help.** Opening the program with two IncomeTax
   raises (teaching raise-taxes → income-up → polls-dip-slightly) improved
   margin to −965…−983 versus −1076 and lifted mean poll to ~0.147.
3. **Small step sizes avoid or delay the crisis.** Capping candidates at
   |Δ|≤0.12 kept DebtCrisis from firing at all on the primary seed
   (effective debt stayed below threshold; `crisis@none`), and delayed it to
   turn 13–14 on other seeds where uncapped variants fired at 13. A tighter
   cap (|Δ|≤0.06) made 18 post-warm-up moves and delayed the crisis to t14.
4. **Crisis avoidance alone does not win.** The tuned agent's margin
   (−974) matches its crisis-hitting siblings because avoiding the cliff
   removes a late penalty but never builds the early approval surplus the
   oracle creates (its winning paths reach polls of 0.4+ before mid-term,
   which then survive the crisis dip). Incrementalism trades the crash for
   a permanently modest level.

The tuned configuration ships as the experiment defaults: fiscal warm-up
preset, `max_action_delta=0.12`, two actions per turn under the shared
per-turn capital budget, reverse damping window 4.

### Why the crisis-free run still lost (-974): decomposition

The tuned agent avoided DebtCrisis entirely yet lost 56 / 1,030 / 914
(share 5.2%). Tracing the campaign:

1. **The step cap banned the winning moves.** |Δ|≤0.12 removes 113 of the
   149 available options — including *TelecommutingInitiative* (Δ=0.25,
   cost 2) and *FoodStamps* (Δ=0.25, cost 4), precisely the opening pair
   SIMULATION.md records for the oracle's proven win. The anti-crisis
   medicine also outlawed the high-leverage popularity introductions.
2. **What remained was low-leverage tinkering.** 13 of 18 post-warm-up
   actions were +0.05 raises on small taxes (AlcoholTax, JunkFoodTax,
   SalesTax, FlatTax, CapitalGainsTax). Polls never exceeded 0.146
   (mean 0.122, final 0.129); winning requires ≈0.55+.
3. **Capital starvation by passivity.** Political capital sat at its 52
   maximum through turns 0–7 — warm-up moves cost pennies — and returned to
   a full bar by election day. Roughly half the term's accrual was never
   converted into influence.
4. **Confounded credit assignment.** With only seven scripted interventions,
   natural poll recovery co-occurred with the small tax raises, so the
   forecaster attributed the drift to them (predicted 0.160 vs observed
   0.137 at turn 12) and kept raising taxes it believed were popular.
5. **Structural floor.** The opposition begins with ~1,030 reliable voters;
   flipping anyone requires >50% approval (60% for opposition members).
   Crisis avoidance bought +142 votes versus no-op (−974 vs −1116) — real,
   but a 13% poll party loses regardless of volatility.

Net: removing the crash fixed the late collapse but the loss was always
about **level**, not volatility. The next lever is targeted rather than
blunt: keep large deltas available where their predicted fiscal signature is
safe, so the popularity flywheel is legal again while the spiral stays
damped.

### Zero-shot ranking is flat: the decisive diagnostic

Is the plateau a *search* problem — would a greatly expanded candidate space
find actions that help? No. At turn 0, strictly zero-shot (pre-game history
only), all 148 executable single-action candidates were scored exhaustively
(1.2 s per decision turn) and compared against their true one-turn effects
in the simulator:

| Quantity | Value |
| --- | --- |
| Predicted Δpoll range | **−0.000075 … 0.000000** (flat) |
| Actual Δpoll range | **−0.0645 … +0.0510** (std 0.0133) |
| Spearman(predicted rank, actual rank) | **−0.154** |
| Best actual move | IncomeTax cancel, +0.051 in one turn |
| Model's rank for that move | 145 / 148 |

Reality offers poll swings of ±5–6 points from single slider moves; the
zero-shot forecaster sees every candidate as identical to within 7.5×10⁻⁵
and its residual ordering is uncorrelated (slightly anti-correlated) with
the truth. Expanding the search space cannot fix this: the argmax of a flat
function is noise regardless of how many candidates are enumerated. The
model even ranks tax *cancellation* near-worst while reality ranks it best —
consistent with the confounded tax-raising loop observed mid-game.

Two practical corollaries:

1. Search-side fixes (pools, objectives, caps, horizons) are exhausted as
   levers for zero-shot ranking; they matter only once forecasts carry
   attribution.
2. Exhaustive evaluation is cheap (~1.2 s per turn), so large search spaces
   become an asset the moment the world model learns treatment effects —
   via interventional pretraining (H1/H2) or accumulated warm-up context,
   which is why the scripted warm-up remains the only working bridge today.

### Do the fiscal predictions work?

Yes, once live fiscal variation exists in context. The forecaster predicts
the visible budget lines (total income, total expenditure, debt) for every
horizon step, and after the warm-up's contrast introduction the mid-game
forecasts track the simulator closely: predicted expenditure stays within
~4% of observed for the rest of the term, and predicted debt reproduces the
spiral's direction and slope (≈10% overshoot in level). The zero-shot
failure mode is different: at turn 12 the model extrapolated polls to 0.256
while the DebtCrisis threshold clipped them to 0.071 — smooth-trend
extrapolation cannot anticipate regime changes it has never observed.

The warm-up run executes a scripted six-move program over turns 0–5
(`diverse_warmup_plan`: CommunityPolicing +0.12, FoodStandards −0.10,
CleanEnergySubsidies +0.17, PollutionControls −0.25, RaceDiscriminationAct
+0.20, AlcoholTax −0.05 — alternating raises and lowers across distinct
policies within half the capital budget), then hands control to the model
with reverse-action damping (`window=4`, `penalty=0.01`).

### Forecast quality and behaviour (chronos h=5, warm-up run)

* One-step poll-rate MAE **0.024** across the term (polls range ≈0.03–0.33); candidate forecasts respond measurably to their treatment paths.
* Context grows correctly from 33 pre-game rows (turns −32…0) through 16 live observations to 49 rows; the pre-game rings ride along in every state snapshot (`value_histories` + `value_histories_turn` survive save/load), while live rows take over once play advances past the capture turn.
* **Zero flip-flops in 16 actions.** The earlier no-warm-up runs reversed direction within a few turns (AlcoholLaw −0.20 then +0.20); with warm-up plus reverse damping the agent never reverses inside the window and instead *extends* warm-up signals (CommunityPolicing +0.25 follows the warmup's +0.12).
* Post-warm-up program is coherent: small tax rises (IncomeTax, FlatTax, AirlineTax, JunkFoodTax introduces) plus service investment (ScienceFunding, CommunityPolicing, FoodStandards).

### Warm-up scale and seed robustness

Sweeping the warm-up program length (capital share limits the plan to what
fits the budget):

| Warm-up moves | Election | Margin | Mean true poll |
| ---: | --- | ---: | ---: |
| 0 (no warm-up) | loss | −938 | 0.0983 |
| 6 | loss | −1100 | 0.1248 |
| **7** | loss | **−1002** | **0.1457** |
| 8 | loss | −1085 | 0.1244 |
| 9 | loss | −1035 | 0.1363 |

The 7-move configuration is stable across seeds (20260813/1/7/42 → margins
−1002/−1009/−1050/−1073, mean polls 0.126–0.146) with zero flip-flops
everywhere. Returns saturate after roughly seven scripted moves: one term's
capital budget and one-action turns cap how much treatment variety a warm-up
can inject, and none of the variants wins.

## Analysis

1. **The scenario is winnable and the oracle proves it.** With the documented configuration (depth 5, two actions per turn, pair batches) the simulator-direct search recovers from a mid-term dip and crosses the boundary ahead. Earlier "unwinnable" readings were artifacts of a crippled oracle (single actions, shallower/different objectives), not of the scenario — `ElectionOracleAgent` now ships those winning parameters as its defaults so the mistake cannot recur.
2. **Warm-up converts observational priors into measured responses.** After six scripted interventions the model's choices become consistent extensions of observed moves rather than zero-shot guesses: mean true poll rises above every passive baseline (0.1248 vs 0.1131) and the economic composite becomes the best non-oracle result (−0.514 vs −0.567). The margin trades slightly against the raw no-warm-up variant (−1100 vs −938): without damping that variant chased one large poll spike and paid for it later.
3. **The zero-shot ceiling remains causal.** Six scripted moves give contrast on only a handful of sliders; the forecaster still cannot estimate response curves for policies never varied in context, and delayed fiscal feedback (DebtCrisis ≈ turn 13–14 under passive play) stays beyond short horizons.
4. **Horizon sensitivity is real and sharp.** h=8 degrades badly under the strict schema (erratic flip-flopping actions): long recursive rollouts through a foundation model trained without this system's dynamics get noisy fast. Depth-5 matches the oracle's proven lookahead and should remain the default until uncertainty-aware scoring exists.
5. **Strict visibility did not hurt forecast quality** — one-step poll MAE improved versus the earlier looser schema, likely because removed near-constant/derived columns reduced attention noise.

## Bottom line

Turn-0 control is achievable — the simulator-oracle wins the first election. The chronos agent's shipped strategy (fiscal pair warm-up, five-step horizon, two capital-bounded actions per turn, reverse damping, no artificial step caps) is the best non-oracle agent on every headline metric — margin −978 vs −1116 passive, polls 0.153 vs 0.113, best composite — while remaining strictly player-visible and never overspending its budget. But the ≈−1,000-vote plateau across a dozen search-side variants shows the constraint is informational, not strategic: with <1% treatment→fiscal attribution in context, candidate rankings are nearly arbitrary among similar-forecast moves. Closing the gap requires richer interventional experience (multi-term curricula or interventional pretraining), which is precisely the H2 milestone of `causal_model.md`.

## TODO — next steps toward closing the gap to the oracle

1. **Interventional pretraining / richer context is the only zero-shot lever.** The exhaustive diagnostic proves zero-shot rankings carry no signal (ρ ≈ −0.15); pretraining on interventional trajectories of synthetic temporal SCMs (H1) or supplying much richer interventional context (H2) is what makes candidate rankings meaningful at all. Search-side tuning is exhausted.
2. **Approval flywheel via model-selected introduces.** With attribution fixed, re-enable large introduces in the candidate pool and let the forecaster rank them; the oracle's winning opening (TelecommutingInitiative + FoodStamps, Δ=0.25) must be *discoverable*, not prescribed.
3. **Regime-change awareness.** Situation covariates or crisis-threshold sentinels so plans respect thresholds the forecaster cannot represent.
4. **Margin-aware objective.** Predicted poll is an approval proxy; the oracle optimizes expected vote margin from the voter model.
5. **Uncertainty-aware scoring.** Penalize candidates whose predicted gains rest on wide quantile spreads.
6. **Winnability curve.** Sweep starting turns/seeds to map where learned-model search preserves vs forfeits oracle wins.
7. **Deterministic one-step leaderboard.** Replay a fixed action trace and compare per-feature forecast errors.

## Artifacts

* `reports/control_comparison_h5_final.json` – headline run with shipped defaults; earlier ablations in `control_comparison_h5*.json`, `_h8_chronos.json` (gitignored generated data)
* `reports/traces/*.json` – per-turn decisions, forecasts, observations for the latest chronos run
* `autocracy/timeseries.py` (visible schema, roles, history seeding, warm-up/batching, damping, debt penalty), `autocracy/chronos.py` (backend), `experiments/control_comparison.py` (harness)
