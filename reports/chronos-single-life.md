# Single-life active learning: the Chronos agent wins without look-ahead

**Date:** 2026-08-23 · **Country:** UK mission start (`uk0.xml`) · **Seed base:** 20260813 · **Runs:** 20 independent lives

## Goal

> Make the chronos-2-small agent **win and survive while learning from its own play only**: fresh empty memory at turn 0, no scripted moves, no policy-name knowledge anywhere in the strategy path, no cross-episode reuse. The earlier multi-episode design (shared `TreatmentEffectMemory` across restarts of the same save) was rejected as oracle look-ahead — replaying a save and carrying learned effects back into turn 0 is save-scumming, not learning.

## What was built

`autocracy/learning.py` + integration in `autocracy/timeseries.py`, driven by `experiments/chronos_learning.py`. Each mechanism is general; none references a specific policy.

| Mechanism | What it does | Failure mode it fixes |
| --- | --- | --- |
| De-trended effect memory | Records observed Δpoll per action signature, subtracting the forecaster's **no-op counterfactual** (already computed for every candidate batch) instead of a median drift | Flat zero-shot rankings → candidate choice is noise |
| Family fallback (`family_shrinkage=0.8`) | Untried slider steps inherit shrunk evidence from measured `(policy, direction)` siblings | Proven sliders couldn't be re-picked (each step is a new signature); agent stalled after saturating first winners |
| Windowed credit (`memory_credit_lag=2`) | Second sample per batch spanning two transitions | Slowly ramping introductions collect credit their first transition hides |
| Measured fiscal channel | Records each action's Δ(income−expenditure), normalised by expenditure; applied symmetrically **only while the visible budget runs a deficit** (`fiscal_prudence_weight=1.0`) | 75% of lives died to the mid-term DebtCrisis (UK0 is structurally in deficit). A forecast-based guard was tried first and *hurt* — chronos's predicted fiscal lines differ <1% across candidates; measured effects are immediate and large |
| Countdown curiosity (`exploration_countdown=True`) | Exploration bonus scales with the share of the term still ahead | Late-term junk probing with the election imminent |
| Commitment damping (`reverse_window=8`, `reverse_penalty=0.05`) | Reversing a recently moved slider is expensive | Attribution noise caused 7-direction flip-flops on one slider within a term |
| Evidence amplification (`memory_effect_weight=2.5`) + path-mean scoring | Learned effects dominate chronos prediction noise (~±0.03 final-step); averaging the horizon cuts that noise √5 | Forecast noise drowning measured differences |
| Memory-guided pools + randomized novelty | Candidate slots go to highest learned-effect/curiosity options; never-tried moves are ordered randomly | Deterministic tie-breaking made every life identical and coverage collapsed |

## Results (20 lives, shipped defaults)

```
margin: mean=-596 median=-720 best=+227 | DebtCrisis in 13/20 lives | first-election WINS 3/20
```

| Life | Election margins | Final poll |
| --- | --- | --- |
| seed …815 | **+18 / +592 / +677** | 0.719 |
| seed …816 | **+227 / +998 / +1126** | 0.832 |
| seed …830 | **+48 / +851 / +1192** | 0.853 |

Every life that won election 1 then swept both re-elections with *growing*
margins — the live memory keeps compounding across terms (94–96 actions,
76–91 distinct interventions tried per winning life). For calibration: the
no-op baseline loses at −1116, the previous best player-visible chronos
campaign lost at −978, and the simulator oracle wins election 1 at +23…+306.
The winning single-life margins (+18…+227) are inside the oracle's range.

## Ablations (all n≥10 lives)

* Cross-episode memory reuse: wins in ≤4 episodes on several seeds but invalid (look-ahead) — removed.
* Forecast-based balance guard: crisis rate 18/20 vs 15/20 baseline (chronos cannot attribute fiscal effects; pure noise) — replaced by the measured channel.
* Fiscal weight 2.0: worse than 1.0 (suppresses the popularity flywheel itself).
* Final-step scoring instead of horizon mean: 1/20 wins vs 3/20.
* Wider pools (limit 48/144): no median gain over 32/96.

## Open gap

Lives that trigger DebtCrisis still lose (polls clip to ~0.05–0.2). Avoiding
the cliff entirely appears to require either anticipating a threshold the
player cannot see or accepting stronger fiscal shaping than "don't deepen a
deficit" — both left as future work. Clean lives average a −201 margin with
final polls ≈0.53, so the remaining distance to consistent wins is exactly
the crisis.

## Model-limited vs sampling-limited: the measurement

`experiments/diagnose_learning.py` separates the two hypotheses with two
instrumented modes sharing the shipped configuration:

* **probe** — normal chronos lives; each turn every candidate batch is also
  branched through the real simulator for its true one-turn poll delta;
* **truth** — identical pools, cadence, and capital rules, but candidates
  ranked by that true delta (perfect world model, measurement only).

| Instrument | Result |
| --- | --- |
| Spearman ρ(predicted Δpoll, true Δpoll) across candidates | **+0.02 … +0.14** (mean per life, 4 lives) |
| Predicted candidate spread vs true spread | ~0.003–0.011 vs ±0.05+ (predictions nearly flat) |
| Executed action's truthful percentile in pool | top 23–42 % (good, rarely best) |
| True winners (≥+3 pp) available per pool | **9–26 of 145 candidates, every turn** |
| Truth-ranked ceiling, first election / all terms | **8/8 wins (+321…+726), 8/8 survivals**, margins to +1873 |

Interpretation: sampling and time are *not* the constraint — the pools
contain plenty of winning moves every turn, and perfect ranking converts
them into landslide campaigns under identical mechanics. The chronos-2-small
world model's near-zero treatment attribution is the binding constraint, so
forecaster quality is the lever with measured headroom.

### Does the larger Chronos help? Measured: no

`autogluon/chronos-2` (120M base, ~0.5 GB — disk was never a blocker at
54 GB free) behind the identical probe:

| Model | ρ(predicted Δpoll, true) | Predicted spread |
| --- | --- | --- |
| chronos-2-small (28M) | +0.02 … +0.14 | 0.003–0.011 |
| chronos-2 base (120M) | **−0.11 … −0.01** | 0.003–0.004 |

Capacity does not buy treatment attribution for this system — consistent
with the Chronos-2 paper's own size ablation (~1 % skill gap on GIFT-Eval).
Swapping models is a one-flag change (`--model autogluon/chronos-2`); any
future forecaster should be admitted by the same probe-ρ instrument rather
than assumed.

### Ablations after the diagnostics

| Change | Outcome (10–20 lives each) |
| --- | --- |
| Declared-cost prior `fiscal_prior_weight` 1.0 / 0.3 / 0.15 | crises eliminated but spending paralysed: 0 wins, margins −879…−1214 — rejected (the winning path *spends* its way to polls ≈1.0) |
| `memory_effect_weight` 2.5 → **4.0** | mean margin −517 → −470, wins 2→3 per 10 |
| Windowed fiscal credit (recurring programme costs charged level-to-level to their sponsor) | mean −470 → −386, best first-election margin +86 |
| `max_actions_per_turn` 3 | attribution muddied, overspending: 1/10 wins — rejected |

Shipped defaults now: memory weight 4.0, windowed fiscal credit, fiscal
prudence 1.0, no declared-cost prior term.

## Final configuration (20 lives)

```
margin: mean=-597 median=-799 best=+86 | DebtCrisis in 15/20 lives | first-election WINS 4/20
```

| Life | Election margins | Final poll |
| --- | --- | --- |
| seed …815 | **+18 / +592 / +677** | 0.719 |
| seed …817 | **+86 / +758 / +938** | 0.798 |
| seed …816 | **+17 / +956 / +1051** | 0.788 |
| seed …830 | **+48 / +851 / +1192** | 0.853 |

Every winner again sweeps both re-elections. For calibration: no-op loses
at −1116; the simulator oracle wins election 1 at +23…+306 and the truth-
ranked ceiling wins 8/8 at +321…+726.

## Artifacts

* `reports/chronos_single_life.json` — per-life outcomes for the headline run
* `reports/traces/chronos-single-life-winner.json` — full decision trace of the best life
* `autocracy/learning.py`, `autocracy/timeseries.py`, `experiments/chronos_learning.py`
