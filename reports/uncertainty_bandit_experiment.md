# Chronos-2 uncertainty and UCB-style exploration experiment

Date: 2026-09-12  
Proposal: [`UNCERTAINTY.md`](../UNCERTAINTY.md)  
Implementation: `autocracy/uncertainty.py`, `autocracy/chronos.py`, and `autocracy/timeseries.py`

## Result

For this first fixed configuration, uncertainty plus a Bayesian-bandit-style
exploration bonus did **not** help. It reduced mean poll and win rate relative
to the matched recent ungated Chronos-2 runs in both countries, while leaving
action volume almost unchanged and allowing more DebtCrisis turns in the USA.

The experiment did demonstrate that the uncertainty signal is wired into
selection: the UCB bonus was positive for essentially every selected
acquisition, and the context disagreement became large in some seeds. The
negative result is therefore not evidence that the code path was inactive. It
is evidence that this particular context-sensitivity heuristic, with these
weights, was not a useful exploration policy.

This is UCB-style rather than fully Bayesian. The implementation does not fit
a posterior over action rewards. It uses Chronos-2 quantiles plus disagreement
across context windows as an epistemic proxy, then scores candidates using

```text
score = mean(candidate - noop effect)
        + beta * context_disagreement
        - risk_weight * q05_downside_shortfall
```

That distinction matters when interpreting the result.

## Protocol

- Model: `autogluon/chronos-2`, with its native quantiles retained; q05 was
  used for the downside-risk term.
- Countries: United States and Germany.
- Seeds: `20260813`, `20260814`, `20260815` for each country.
- Horizon: 20 elections × 16 turns = 320 turns per seed.
- Scheduling: all three seeds for a country ran sequentially in one process;
  Germany began only after the USA set was complete. No concurrent campaign
  jobs were launched.
- Agent profile: the normal ungated Chronos profile, with at most two policy
  actions per turn.
- Context ensemble: three chronological suffixes (full history, 75%, and
  50% of the available history), paired candidate-vs-no-op.
- Uncertainty parameters: `beta=0.5`, `members=3`,
  `min_context_fraction=0.5`, `risk_weight=0.25`, `risk_quantile=0.05`,
  `risk_floor=0.5`.
- Full quantile output: enabled.

The complete traces are under
`reports/campaigns/chronos/uncertainty-b05-r025/{usa,germany}/`.

## New runs

“Crisis turns” counts turns whose saved state contains `DebtCrisis`; “crisis
terms” counts electoral terms in which that crisis flag was present at the
term summary.

| country | seed | wins | mean poll | crisis terms | crisis turns | actions | runtime |
|---|---:|---:|---:|---:|---:|---:|---:|
| USA | 20260813 | 16/20 | 0.6003 | 20 | 304 | 622 | 95.9 min |
| USA | 20260814 | 19/20 | 0.6181 | 18 | 286 | 631 | 91.6 min |
| USA | 20260815 | 17/20 | 0.6598 | 16 | 247 | 632 | 95.6 min |
| Germany | 20260813 | 13/20 | 0.5400 | 18 | 283 | 629 | 95.0 min |
| Germany | 20260814 | 16/20 | 0.6371 | 11 | 167 | 620 | 95.1 min |
| Germany | 20260815 | 12/20 | 0.6124 | 10 | 150 | 608 | 96.1 min |

Aggregate results:

| country | wins | mean poll | crisis terms | crisis turns/life | actions/life | runtime/life |
|---|---:|---:|---:|---:|---:|---:|
| USA | 52/60 (86.7%) | 0.6260 | 54/60 | 279.0 | 628.3 | 94.3 min |
| Germany | 41/60 (68.3%) | 0.5965 | 39/60 | 200.0 | 619.0 | 95.4 min |

## Comparison with recent runs

The fairest baseline is the three matched seeds (`20260813`–`20260815`),
since the new experiment has three seeds. The four-seed baseline and no-op
figures are included as the recent headline references from
[`cross_country_campaigns.md`](cross_country_campaigns.md); the conservative
gate figures are from [`noop_gate_cross_country.md`](noop_gate_cross_country.md).

| country / profile | wins | mean poll | crisis turns/life | actions/life |
|---|---:|---:|---:|---:|
| USA — uncertainty/UCB, 3 seeds | 52/60 | 0.6260 | 279.0 | 628.3 |
| USA — ungated Chronos, matched 3 seeds | 57/60 | 0.7287 | 188.3 | 633.3 |
| USA — ungated Chronos, recent 4 seeds | 77/80 | 0.7482 | — | 634* |
| USA — conservative gate, 3 seeds | 60/60 | 0.6795 | 0 | 8 |
| USA — no-op, recent 4 seeds | 0/80 | 0.4294 | 0 | 0 |
| Germany — uncertainty/UCB, 3 seeds | 41/60 | 0.5965 | 200.0 | 619.0 |
| Germany — ungated Chronos, matched 3 seeds | 52/60 | 0.6682 | 183.3 | 628.7 |
| Germany — ungated Chronos, recent 4 seeds | 70/80 | 0.6802 | — | 627* |
| Germany — conservative gate, 3 seeds | 0/60 | 0.4572 | 0 | 8 |
| Germany — no-op, recent 4 seeds | 0/80 | 0.4320 | 0 | 0 |

`*` The four-seed action figures are rounded headline values from the prior
report; matched three-seed action counts were recalculated from the saved
traces above. The prior report’s crisis headline is term-level rather than
turn-level, so the table uses the directly recomputed matched crisis-turn
figures where available.

The uncertainty runs were about 2.5× slower than matched ungated Chronos runs:
94.3 versus 37.9 minutes per USA life, and 95.4 versus 37.7 minutes per
Germany life. That cost is expected from the additional context-window
forecasts and full quantile outputs, but the measured outcome did not justify
it in this configuration.

## What the diagnostics say

The selection diagnostics were aggregated over the 317 acquisition decisions
recorded per life (the initial warm-up turns do not have a selected
acquisition).

| country | mean selected epistemic σ | mean all-candidate σ | mean UCB bonus | mean q05 risk penalty | positive-bonus selections |
|---|---:|---:|---:|---:|---:|
| USA | 0.04315 | 0.10881 | 0.02158 | 0.000156 | 100% |
| Germany | 0.01630 | 0.02659 | 0.00815 | 0.000203 | 99.9% |

The selected bonus was therefore active, but it did not produce a meaningful
exploration/exploitation improvement:

1. **Action volume did not change much.** The uncertainty policy chose 628.3
   actions/life in the USA and 619.0 in Germany, almost the same as the
   matched ungated baseline (633.3 and 628.7). This was not a reduction in
   churn or a conservative policy.
2. **The downside term was weak on the score scale.** The average q05 penalty
   was around 0.0002, versus UCB bonuses of 0.008–0.022. The risk term was
   present, but it was unlikely to control rankings at `risk_weight=0.25`.
3. **Context disagreement was not reliably useful epistemic information.**
   The USA third seed had mean selected σ=0.1088 and mean candidate σ=0.3012,
   much larger than the first two USA seeds, without a corresponding outcome
   improvement. That looks more like sensitivity to shortened history and
   non-stationarity than calibrated uncertainty about action value.
4. **The approach did not fix the fiscal failure mode.** Compared with the
   matched baseline, crisis turns increased from 188.3 to 279.0 per USA life
   and from 183.3 to 200.0 per Germany life. The extra uncertainty machinery
   did not prevent the policy agent from spending into a debt spiral.

## Interpretation

The answer to “does uncertainty plus Bayesian-bandit exploration versus
exploitation help?” is **not in this first experiment**. In the USA, the
uncertainty/UCB profile lost 8.3 percentage points of mean poll relative to the
matched baseline and 5 wins over 60 terms. In Germany it lost 7.2 points of
mean poll and 11 wins. The confidence intervals are not estimated from only
three seeds, so these are directional findings rather than a definitive
benchmark conclusion.

The result is consistent with the earlier conservative-gate experiment. The
gate reduced action volume to roughly eight warm-up actions and won every USA
term but none in Germany; uncertainty/UCB preserved the ungated agent’s high
action volume without recovering its baseline performance. The missing piece
is not simply “act more” or “act less.” It is a calibrated estimate of the
causal value and downside of a candidate policy relative to doing nothing,
including fiscal state and delayed effects.

## Recommended next experiment

The next test should be a small, cheaper sweep before another full six-hour
country pair:

- `beta ∈ {0, 0.1, 0.5, 1.0}` and `risk_weight ∈ {0, 0.25, 1.0}` on fewer
  elections or a short horizon;
- an explicit fiscal/debt penalty or action-cost term in the acquisition
  score, so UCB cannot dominate the ranking when uncertainty is caused by
  history truncation;
- a proper Bayesian linear or generalized-linear residual head over
  candidate-minus-no-op effects, with posterior covariance and Thompson/UCB
  selection, instead of treating context-window disagreement as a posterior;
- calibration checks: whether larger predicted σ actually predicts larger
  out-of-sample candidate-vs-no-op error;
- only then, a larger seed count on the most promising setting.

## Reproduction

```bash
uv run --extra chronos python experiments/campaign_trace.py \
  --mode chronos --model autogluon/chronos-2 --country usa \
  --seeds 3 --seed-base 20260813 --elections 20 \
  --uncertainty-beta 0.5 --uncertainty-members 3 \
  --risk-weight 0.25 --risk-quantile 0.05 \
  --label uncertainty-b05-r025

uv run --extra chronos python experiments/campaign_trace.py \
  --mode chronos --model autogluon/chronos-2 --country germany \
  --seeds 3 --seed-base 20260813 --elections 20 \
  --uncertainty-beta 0.5 --uncertainty-members 3 \
  --risk-weight 0.25 --risk-quantile 0.05 \
  --label uncertainty-b05-r025
```
