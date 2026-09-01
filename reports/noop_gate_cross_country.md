# Experiment 1 follow-up: conservative (no-op-gate) agent across all countries — stopped early

Follow-up to `reports/noop_gate_experiment.md`: run the same conservative
profile (δ=0.02, λ=2.0, 8-transition chill, Chronos-2 q20/q80 bands) on all
six countries with 4 seeds, 20 electoral terms each. **The experiment was
stopped after the completed seeds showed the profile fails outside France.**
Traces for all completed lives remain under
`reports/campaigns/chronos/conservative/<country>/<seed>/`; aborted partial
traces exist for the still-running seed directories and are excluded (no
`summary.json`).

## Completed data

| country | gate seeds | baseline seeds | noop seeds |
|---|---|---|---|
| uk | 3 (20260813-15) | 10 | 4 (20260813-16) |
| australia | 2 | 4 | 4 |
| canada | 2 | 4 | 4 |
| france | 3 (13-15) | 4 | 1 |
| germany | 3 | 4 | 4 |
| usa | 3 | 4 | 4 |

## Headline: gate vs noop vs ungated baseline (mean over completed seeds)

| country | gate wins | noop wins | base wins | gate poll | noop poll | base poll | gate actions/life | base actions/life | gate margin | noop margin | base margin |
|---|---|---|---|---|---|---|---|---|---|---|---|
| uk | **0/60** | 0/80 | **106/200 (53%)** | 0.089 | 0.072 | **0.500** | 7 | 622 | -1379 | -1466 | +29 |
| australia | **0/40** | 0/80 | **50/80 (63%)** | 0.111 | 0.022 | **0.552** | 8 | 759 | -1544 | -1909 | +252 |
| canada | **0/40** | 0/80 | **51/80 (64%)** | 0.112 | 0.038 | **0.522** | 10 | 783 | -1360 | -1722 | +103 |
| france | **50/50** | 20/20 | 51/80 (64%) | **0.857** | 0.845 | 0.593 | 5 | 712 | **+1314** | +1249 | +324 |
| germany | 0/60 | 0/80 | **70/80 (88%)** | 0.457 | 0.432 | **0.680** | 8 | 627 | -120 | -191 | **+632** |
| usa | **60/60** | 0/80 | 77/80 (96%) | 0.679 | 0.429 | **0.748** | 8 | 634 | +531 | -209 | **+898** |

Per-seed detail (completed lives):

```
uk         gate: 20260813 0/20 poll=0.089  20260814 0/20 poll=0.089  20260815 0/20 poll=0.089
australia  gate: 20260813 0/20 poll=0.128  20260814 0/20 poll=0.094
canada     gate: 20260813 0/20 poll=0.104  20260814 0/20 poll=0.120
france     gate: 20260813 10/10 poll=0.857 20260814 20/20 poll=0.858 20260815 20/20 poll=0.857
germany    gate: 20260813 0/20 poll=0.457  20260814 0/20 poll=0.464  20260815 0/20 poll=0.452
usa        gate: 20260813 20/20 poll=0.668 20260814 20/20 poll=0.684 20260815 20/20 poll=0.686
```

## Interpretation

1. **The gate does not transfer.** France and the USA are the only wins
   (France 50/50, USA 60/60) — and both were already winnable or nearly
   winnable by doing nothing much smarter (France 20/20 no-op; USA no-op
   0/20 but a *fiscal* problem, see below). Every other country: 0 wins,
   and the gate profile is barely distinguishable from no-op
   (uk poll 0.089 vs 0.072; australia 0.111 vs 0.022; canada 0.112 vs 0.038;
   germany 0.457 vs 0.432) while the ungated agent wins 53-88% there.

2. **8-10 actions then silence — even when the country is dying.** Every
   gate life spends its whole first actions in turns 0-2 (always the same
   safe warm-up-adjacent moves: CommunityPolicing, CleanEnergySubsidies,
   RaceDiscriminationAct raises; PollutionControls/FoodStandards lowers),
   then never acts again. In the UK the mean poll is 0.089 with a
   -45M/turn balance and 307/320 crisis turns: the agent sits on its hands
   through the entire death spiral. The gate converts intervention bias
   into **action paralysis** wherever the world model's flat candidate
   spread (±7.5e-5 poll) cannot produce evidence that clears δ=0.02.

3. **Why France/USA "worked"**: the gate is indistinguishable from a very
   good no-op there. France is the documented no-op-wins country; the USA
   no-op has a positive balance (+522k/turn) and simply drifts down slowly,
   so a handful of early small raises keep the poll above the line all 20
   terms. Neither case demonstrates evidence-gated *control*; both are
   consistent with "the starting position saved it".

4. **The gate margin is mis-calibrated relative to the world model's
   signal, not the game's.** The measurement the gate consumes (Chronos-2
   candidate spread) is ~1e-5, but real action effects are ~1e-2
   (memory_effect_weight × measured effects). δ=0.02 was chosen from
   France's scale and happens to sit *above* what flat forecasts can
   justify anywhere. With λ=2.0 the uncertainty term only adds margin.
   The profile therefore degenerates to "act only during warm-up, then
   never" in every country whose early-term poll trajectory isn't
   self-sustaining.

5. **GPU economics**: 6 concurrent campaigns made each ~6x slower; a
   4-seed × 6-country sweep at ~40-90 min/life solo was never going to
   finish in a night, and stopping at 2-3 seeds/country was sufficient:
   every non-France country is 0/N with near-identical per-seed curves, so
   more seeds would not change the conclusion.

## Conclusion and what this buys us

- The status-quo prior alone is **not** a general fix: it wins exactly where
  no-op was already strong, and it loses everything the ungated agent won.
  The honest summary: *the gate removes intervention bias by removing
  intervention; it does not add the ability to detect when action is
  genuinely required.*
- NEXT_STEPS §1's framing was correct that the agent over-acts, but the
  fix cannot be a static threshold on a forecaster whose candidate spread
  is flat. The missing ingredient is a *credible evidence channel* — which
  is what experiment 2's learning lines (previous-life context, crisis
  retrieval) and the horizon ablation (§8) target. A promising concrete
  direction from this run: gate on the **treatment-memory LCB**
  (measured, per-action) instead of the Chronos spread, with δ scaled to
  memory σ, and keep exploration_countdown so early-term learning can
  still gate in actions later.
- Data preserved: conservative traces (completed lives) alongside the
  baseline/noop references for the same seeds; comparison tool:
  `experiments/compare_conservative.py`.

## Reproducing

```bash
# aggregate what exists
uv run python experiments/compare_conservative.py --elections 20

# single completed life
uv run --extra chronos python experiments/campaign_trace.py \
    --mode chronos --country usa --conservative --seeds 1 --elections 20
```
