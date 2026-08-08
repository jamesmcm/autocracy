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

Exact reconstruction rule (verified to within a fraction against the
playthrough saves): the income the game *displays* (`<finances>` n_2) at
start of turn N equals the serialized `incomehistory[0]` sum at turn N+1,
which equals

    sum over active income policies of
        base(min_income, max_income, policy_val@N+1)
        * earn_scalar@N * wealth_mod * incom_mult@N

where the incom_mult used is the one serialized in save N (itself computed
from the *previous* turn's nodes).  The same relation holds for expenditure
(costhistory) and `cost_mult`/`cost_scalar`.

So a faithful simulator must recompute income/cost each turn from the *live*
node values (GDP, CrimeRate, ...) rather than using the serialized
multipliers/scalars frozen at load time.

### 5. Ministerial scalars are recomputed from competence each turn

`earn_scalar` and `cost_scalar` are per-department and sum to 2.0:
`earn_scalar = 0.875 + 0.25 * competence(dept minister)` and
`cost_scalar = 2.0 - earn_scalar`.  Minister competence is
`0.2 + 0.8 * experience * suitability`, and experience grows by
`MINISTER_EXPERIENCE_RATE` (0.025) each turn.  Verified across all
playthrough saves (every department matches to 1e-6).

### 4. General Strike situation fires at turn 2

`GeneralStrike` is a situation (`situations.csv`), not a dilemma
(`OPTIONS_DILEMMAS=0`).  Its latent at turn 0 is 0.441 (below the 0.6 start
trigger), but the drastic cuts push it to 0.672 by turn 1.  Because
`SIM_Situation::NextTurn` activates based on the *previous* turn's stored
value, it fires at turn 2, crashing GDP (effect `GDP,-0.3-(0.2*x)`) and
cranking expenditure via its 1000/turn cost.

## Simulator gaps to fix

1. Action cost: classify lowering to slider floor as `lower` (and allow
   uncancellable policies to be lowered to their floor).  This blocks the
   entire playthrough replay (CorporationTax/IncomeTax/Prisons orders fail).
2. Finance: recompute income/cost from live node values each turn, including
   ministerial scalars and debt interest.  `total_income`/`total_expenditure`
   must track the game's `<finances>` neurons (n_2/n_1), which also include
   debt interest (~16.6k on 848k debt at turn 0).
3. Situation/finance ordering and the one-turn income-history lag.
4. Verify the 12-turn playthrough end-to-end against `parity_cases/dem3saves/`
   using `parity_cases/replay_playthrough.py` (run with `PYTHONPATH=.`).

## Known target values (from the game's own saves)

| turn | n_2 income | n_1 expenditure | PC start | GeneralStrike |
|------|-----------|-----------------|----------|---------------|
| 0    | 227,899   | 276,528         | 26       | 0.441 (off)   |
| 1    | 116,549   | 276,244         | 27       | 0.672 (off)   |
| 2    | 111,023   | 250,481         | 27       | 0.651 (ON)    |
| 3    | 86,459    | 241,684         | 25       | 0.749 (ON)    |
| 11   | 246,862   | 260,874         | 21       | 1.000 (ON)    |

PC spend per orders save: turn0 25 (CorpTax 10 + IncomeTax 7 + Prisons 8),
turn1 26 (StatePensions lower 26), turn2 27 (StateHousing cancel 18 + CCTV
raise 9), turn5 SalesTax 0.44->1.0, turn8/9/10/11 StateHealthService moves.

## Implementation status (as of this update)

The simulator now matches the game's `<finances>` block *exactly* for the
shipped noop uk0->uk1->uk2 transitions and reproduces the first three turns of
the drastic-changes playthrough to within ~130 currency units of income and
~54 of expenditure:

| turn | income err | exp err | PC sim/game | GeneralStrike |
|------|-----------|---------|-------------|---------------|
| 0->1 |     +0.0   |   +0.0  | 27 / 27     | off -> off    |
| 1->2 |   +133.0   |  -46.0  | 27 / 27     | ON (turn 2)   |
| 2->3 |     +4.5   |  -53.8  | 25 / 25     | ON            |

### Determinism: the game is fully reproducible

The game is **deterministic** for this playthrough: `OPTIONS_RANDOMSTART = 0`
in the user `config.txt`, so `GRandom::Init(0x9c40, 1)` is called with the
fixed seed **1** (only `OPTIONS_RANDOMSTART = 1` seeds from the
`SDL_GetPerformanceCounter` clock).  `srand(1)` then precomputes 40,000
float32 randoms (`rand() * (1/2^31)`) that every `RandUnitFloat` /
`RandomChoice` / `IsChance` call consumes in a fixed order.

Every policy change in the orders saves is a *paid player order*: the per-turn
PC math reconciles exactly (e.g. turn 4 spends 45 = PollutionControls cancel 6
+ PropertyTax raise 19 + PrivatePrisons introduce 20, from a reconstructed
start-of-turn-4 PC of 50 = 25 + 25 income).  No random event altered any
policy.  The capital income therefore declines purely from minister loyalty
(26 -> 20), and the income formula
`int(SUM max(1, 6 * (loyalty - 0.10)))` reproduces the serialized `<max>/2`
values exactly.

### What was implemented

1. **Action costing.** `PolicyAction` gained an optional `action_type`.
   Lowering a slider to its floor (even level 0) is a `lower` (lower cost,
   policy stays active); only a true switch-off is a `cancel` (cancel cost,
   active flag flips, value/target frozen).  Uncancellable policies can be
   lowered to their floor but never cancelled.  `_next_level` allows discrete
   sliders to reach a level-0 floor.
2. **Live finance.** `_recompute_live_finance`/`_recompute_orders_finance`
   rebuild the finance lines every turn:
   * income = sum over *active* income policies of `base(min,max,val) *
     wealth_mod * earn_scalar * incom_mult`;
   * expenditure = same for costs, plus `wealth_mod *` the active situation
     costs plus the quarterly debt interest `debt * rate * 0.25`;
   * the multiplier neurons are evaluated from the *previous* turn's nodes
     (one-turn history lag); ministerial scalars come from the current
     competence (`earn_scalar = 0.875 + 0.25 * competence`).
   All money arithmetic is rounded to float32 (`_f32`), which reproduces the
   serialized totals to the last ULP.
3. **Debt / credit rating / interest.** `debt@N+1 = debt@N + (exp - inc)`
   using the post-orders net; the credit rating is re-derived from the
   debt-to-GDP ratio every other turn (`turns_since_credit`); the interest
   rate is `INTEREST_RATE_MIN + (INTEREST_RATE_MAX - INTEREST_RATE_MIN) *
   min((rating / 9)^2, 1)`.
4. **Situation fixes.** The `_default_,expr` cell is now evaluated (so
   AntisocialBehaviour's 0.8 base is used, not 0).  The game parses
   `-0.1(-0.6*x)` as `-0.1 + (-0.6*x)` (concatenation), not implicit
   multiplication; fixed in `_sanitize_expression`.  GeneralStrike still
   fires on the turn after its latent crosses the trigger (turn 2), matching
   the game.

### Known remaining gaps

A full game-state audit (nodes, situations, ministers, voters, policies,
finance) across every saved game shows the policy levels, active flags,
political capital and capital income all match, and the remaining gaps are
concentrated in the voter population and the node values that feed off it:

* **Node values** (26-30 diffs, mostly < 0.03): Immigration (sim 0.413 vs
  game 0.385), Wages, Health, PrivateHealthcare, Unemployment,
  WorkerProductivity, Education, etc.  The Immigration gap comes from the
  never-introduced policies' constant terms (CitizenshipTests' -0.05 base on
  Immigration is partially applied in the game) interacting with the node
  values that feed it; adding the full constants over-corrects.  This drift
  feeds the ~+5k (2%) final-turn income error via the CO2Emissions/CarUsage
  multipliers and the ~-1.3k turn-8 error.
* **Situation latents** (25-31 diffs): the situation values inherit the node
  drift, though the active sets match at every captured turn.
* **Voter polls** (18-19 diffs): the loaded polls drift by the effect changes
  and the equality-driven collapses, but the remaining groups (Conservatives
  -0.45 vs -0.82, MiddleIncome +0.03 vs -0.01) need the full poll opinion
  dynamics (complacency/party/sympathy feedback) that the approximation
  cannot reproduce.
* **Voter percentages** (13-15 diffs): the income groups (Wealthy/Poor/
  MiddleIncome) are now exact — each voter is placed in their income band
  (inincome < 0.25 / 0.25-0.75 / > 0.75) with the band membership set to
  `sin((income - boundary)/0.6·π)` (income = 1.2·inincome - 0.1), matching
  the game's serialized percentages (0.252/0.252/0.496) and the voters'
  reassigned income-group memberships 2000/2000.  The *other* groups'
  percentages still drift in the game because the individual voters'
  memberships change over the run (the party/sympathy membership changes),
  which the static loaded memberships do not reproduce.
* **Voter frequencies** (7-12 diffs): the `_freq` neurons drift with the
  voter dynamics; the sim keeps the loaded values.
* **Minister satisfaction values** (5-10 diffs): these are `0.5 + average` of
  the two sympathised voter polls, so they inherit the poll drift; the
  per-turn capital income still matches every turn because the loyalty
  regions are robust to the small poll differences.
* **Finance**: income/expenditure within ~2%, debt within ~0.03%, and the
  income/cost multipliers are re-derived from the (slightly drifted) nodes.

**Precision is not the cause** of the node gaps: testing the node computation
in full double precision and in float32-rounding throughout changes nothing.
The residuals come from the game's effect-value details and the voter
population dynamics (membership/percentage/frequency changes), which are a
large subsystem beyond the finance/loyalty parity targets.

* **Random systems.**  No random event changed a policy in this playthrough
  (all changes are paid orders), so the deterministic core covers the policy
  state.  Events/attacks only move voter opinions, which the finance/GDP/PC
  parity targets do not depend on.  If the voter-parity targets are ever
  pursued, the game consumes its seed-1 random stream by collecting the
  events whose value crossed their threshold each turn and picking one with
  `GRandom::RandomChoice`.
  (e.g. Immigration is 0.413 vs the game's 0.385 from the combination of
  the never-introduced CitizenshipTests base term and the node values that
  feed it); the reference saves still reproduce uk1 exactly and uk2 within
  0.3%.
