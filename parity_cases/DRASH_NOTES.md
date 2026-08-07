# Democracy 3 Drastic-Changes Parity Investigation

This file tracks the investigation into why the simulator diverges from the
real game for the "drastic changes" playthrough captured in
`parity_cases/dem3saves/`.

## Ground truth

The playthrough in `parity_cases/dem3saves/` was captured from the actual
installed Democracy 3 game (v1.30.2) running under Xvfb.  The files are the
game's own XML saves, one `_initial` and one `_orders` per turn.  `_orders`
saves are taken after placing orders but before ending the turn; `_initial`
saves are the state at the start of the next turn.

Key finance figures per turn (from the `<finances>` block):

| turn | income    | expenditure | debt      | PC | GDP    | GeneralStrike |
|------|-----------|-------------|-----------|----|--------|---------------|
| 0    | 227,899   | 276,528     | 848,403   | 26 | 0.233  | 0.441 (off)   |
| 1    | 116,549   | 276,244     | 897,031   | 27 | 0.169  | 0.672 (off)   |
| 2    | 111,023   | 250,481     | 1,056,727 | 27 | 0.000  | 0.651 (ON)    |
| 3    | 86,459    | 241,684     | 1,173,366 | 25 | 0.000  | 0.749 (ON)    |

## Discoveries

### 1. Action costing: lowering to a floor is `lower`, not `cancel`

The orders save drops CorporationTax 0.22->0.0, IncomeTax 0.45->0.0 and
Prisons 0.5->0.0 with political capital 26->1 (25 spent).  The game charged
the **lower** costs: CorpTax 10 + IncomeTax 7 + Prisons 8 = 25.

The simulator's `_policy_action_type` classifies any move to level 0 as
`cancel`, charging 30 + 13 and refusing Prisons (uncancellable).  The game
treats the slider floor as a `lower` action:

* `corptax` and `incometax` are `PERCENTAGE` sliders whose floor is 1%
  (not 0% / cancellation).
* `prisons` is `DISCRETE` with `OVERCROWDED CELLS` as the floor; the policy
  is `UNCANCELLABLE` but that only means it cannot be switched off, not that
  it cannot be lowered to its floor.

### 2. Finance is live-recomputed each turn, not static

Serialized income multipliers, income/expense scalars and cost multipliers
are recomputed by the game every turn from current node values:

* `incom_mult` is evaluated from the policy's income_multiplier expression
  at the **previous** turn's node values.  E.g. IncomeTax uses
  `GDP,0.5+(0.5*x)`; the shipped noop saves show incom_mult 0.5 -> 0.6164
  (GDP 0.2328) -> 0.5845 (GDP 0.169) across uk0/uk1/uk2.
* `earn_scalar` (income) and `cost_scalar` (expense) are per-department
  ministerial scalars that sum to 2.0 and drift each turn as minister
  experience grows.  Observed formula: `earn_scalar = 0.875 + 0.25*comp`
  where `comp` is the department minister competence.
* `cost_mult` is evaluated from the policy's cost_multiplier expression
  (e.g. Prisons `CrimeRate,0.1+(0.9*x)`).

The finance totals the game displays (the `<finances>` block n_2/n_1) include
**debt interest**, which the simulator's `total_expenditure` does not.

### 3. Income history lags the multiplier update

Per-policy `incomehistory[0]` in a save is the income computed with the
multiplier/scalar from the *previous* save (the finance snapshot is written
before the next turn's multiplier update).  The shipped noop saves confirm:
uk1 incomehistory[0] equals uk0's serialized income even though uk1's
incom_mult field already shows the new value.

### 4. General Strike situation fires at turn 2

`GeneralStrike` is a situation (`situations.csv`), not a dilemma
(`OPTIONS_DILEMMAS=0`).  Its latent at turn 0 is 0.441 (below the 0.6 start
trigger), but the drastic cuts push it to 0.672 by turn 1.  Because
`SIM_Situation::NextTurn` activates based on the *previous* turn's stored
value, it fires at turn 2, crashing GDP (effect `GDP,-0.3-(0.2*x)`) and
cranking expenditure via its 1000/turn cost.

## Simulator gaps to fix

1. Action cost: classify lowering to slider floor as `lower` (and allow
   uncancellable policies to be lowered to their floor).
2. Finance: recompute income/cost from live node values each turn, including
   ministerial scalars and debt interest.
3. Situation/finance ordering and the one-turn income-history lag.
4. Verify the 12-turn playthrough end-to-end against `parity_cases/dem3saves/`.
