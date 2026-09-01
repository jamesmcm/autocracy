# Experiment 1: no-op-aware action selection (status-quo gate), France, 10 terms

> **Cross-country follow-up (stopped early):** the same profile does *not*
> generalise — 0 wins in uk/australia/canada/germany where the ungated agent
> wins 53-88%. See `reports/noop_gate_cross_country.md`. The France result
> below stands but demonstrates "the gate is as good as a good no-op", not
> evidence-gated control.

Hypothesis (NEXT_STEPS §1): the agent's primary failure is **sequential
over-intervention under forecast uncertainty**, not missing world knowledge.
If an evidence-gated, conservative Chronos beats the no-op baseline
consistently on France, the intervention bias is confirmed as the dominant
failure mode.

## Method

`TimeSeriesPolicyAgent` gained a status-quo gate: a candidate batch is taken
only when its **evidence** — the forecast objective (horizon-mean poll) plus
the measured treatment-memory effect, **excluding exploration bonuses** —
beats the no-op candidate's evidence by

```
margin = delta + lambda * (band_forecast + band_noop + memory_effect_weight * sigma_memory)
```

where `band_*` are Chronos-2 q20/q80 half-widths of the poll forecast
(`Chronos2SmallForecaster(with_bands=True)`) and `sigma_memory` is the
recency-weighted std-dev of the memory samples for the batch
(`TreatmentEffectMemory.effect_uncertainty`). This is the uncertainty-aware
form LCB(a) > UCB(noop) + delta. A post-intervention chill
(`reversal_cooldown=8` transitions) doubles the margin for any policy touched
recently. When every action fails the margin, the agent holds the status quo
that turn and `ForecastDecision.evidence_margin` records what was cleared.

Gate settings: `delta=0.02`, `lambda=2.0`, `reversal_cooldown=8`,
warm-up preserved (8 scripted moves for interventional context), everything
else identical to the baseline chronos preset. Run:
`uv run --extra chronos python experiments/campaign_trace.py --mode chronos
--country france --conservative --seeds 1 --elections 10`
(seed 20260813, 20-turn terms, full state saved every turn; replay verified).

## Results, France, 10 electoral terms (200 turns)

| agent | wins | mean poll | final poll | actions | DebtCrisis turns | avg balance/turn |
|---|---|---|---|---|---|---|
| **chronos + no-op gate** | **10/10** | **0.857** | 0.851 | **5** | **0** | **+110k** |
| no-op | 10/10 | 0.844 | 0.836 | 0 | 0 | +100k |
| chronos baseline (ungated) | 7/10 | 0.788 | 0.557 | 387 | 53 | -145k |

Margins per term (player - opposition votes):

```
conservative: 1264 1316 1342 1320 1318 1328 1308 1320 1318 1304
noop:         1185 1246 1280 1244 1248 1270 1235 1259 1257 1236
baseline:     wins terms 1-7, then -27 ... deep losses as DebtCrisis locks in
```

Term-mean polls:

```
conservative: 0.828 0.865 0.863 0.858 0.864 0.857 0.857 0.861 0.856 0.860
noop:         0.815 0.851 0.848 0.842 0.852 0.843 0.845 0.850 0.842 0.849
baseline:     0.831 0.879 0.890 0.866 0.891 0.895 0.866 0.607 0.533 0.624
```

Macro state at turn 200 (normalised sim values):

| metric | conservative | no-op | baseline |
|---|---|---|---|
| GDP | 0.247 | 0.224 | 0.038 |
| Unemployment | 0.460 | 0.479 | 0.506 |
| CrimeRate | 0.122 | 0.154 | 0.255 |

## What the gate did

- 5 total actions in 200 turns (all in the first 3 turns, right after warm-up:
  CommunityPolicing +0.25 introduce, PollutionControls -0.25,
  RaceDiscriminationAct +0.20, ArtsSubsidies -0.16, CleanEnergySubsidies
  +0.25 introduce), then silence: the measured effects of those moves never
  cleared delta again, so the agent held the status quo.
- The five moves were *improvements* over pure no-op: GDP +10% relative,
  unemployment and crime lower, every election margin above the no-op's
  (mean 1316 vs 1246), mean poll +0.013 above no-op.
- Zero DebtCrisis turns; the treasury ran a persistent surplus.
- Replay verification passed: the stored full states + actions reproduce the
  campaign exactly (200/200 turns).

## Answer to the experiment's question

**Yes — with an evidence gate, conservative Chronos now out-performs no-op
consistently on France (10/10 wins, higher margins and polls every term, no
crisis), where the ungated agent was destroying a winning position.** The
comparison also confirms the diagnosis: same forecaster, same memory, same
warm-up — the only change is requiring actions to prove themselves, and the
~78x action reduction (5 vs 387) plus 0 crisis turns converted a 64%-win
churning agent into a 100%-win holder that *beats* doing nothing.

The residual risk is now visible for experiment 2: the agent acted only in
the first 3 turns and never again. In a country where the starting position
forces action (UK: 0/20 no-op wins), a pure status-quo prior will need the
memory-guided exploration channel to justify *necessary* interventions — the
gate as-is may under-act there. That is exactly the France-vs-UK contrast
NEXT_STEPS predicts, and the next experiment (horizon ablation, then
previous-life context) should be run on UK as the action-forcing case.

## Reproducing

```bash
uv run --extra chronos python experiments/campaign_trace.py \
    --mode chronos --country france --conservative --seeds 1 --elections 10
uv run --extra chronos python experiments/campaign_trace.py \
    --mode noop --country france --seeds 1 --elections 10 --seed-base 20260813
uv run --extra chronos python experiments/campaign_trace.py \
    --mode chronos --country france --seeds 1 --elections 10  # ungated baseline
uv run pytest tests/test_gate.py tests/test_chronos.py tests/test_timeseries.py
```

Data: `reports/campaigns/chronos/conservative/france/20260813/`
(`turns.jsonl.gz` full states + actions, `summary.json`), next to the
existing `noop/france` and ungated `chronos/france` traces for the same seed.
