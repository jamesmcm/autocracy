# Multi-term native capture and parity TODO

The UK mission uses a 16-turn electoral term. The first long-run target is 24
turns: one complete term plus eight additional quarterly turns. Raw native XML
stays outside the repository under the installed save root; only summaries,
fixtures, and reproducible tooling belong here.

- [x] Add a term-aware launcher that derives mission term length and pads a
      captured order sequence with explicit no-order turns.
- [x] Extend the offline comparator to replay a no-order-only capture for an
      explicit number of turns.
- [x] Generate and validate a 24-turn UK no-order capture from `turn0_initial`.
- [x] Generate and validate a 24-turn UK capture using the existing 12-turn
      policy sequence followed by a no-order tail.
- [x] Compare every captured turn across finance, ordinary nodes, situations,
      policies, voters, parties, polls, and hidden histories.
- [x] Identify the first sustained divergence and its native manager/order
      cause instead of tuning to a single end-of-term checkpoint.
- [x] Improve the simulator from the long-run evidence and add regression
      tests for each durable fix.
- [x] Record capture names, commands, parity summaries, and remaining limits in
      `gamedrive/README.md`, `README.md`, and `SIMULATION.md`.
- [x] Commit and push the capture tooling, parity fixes, and documentation in
      reviewable milestones.

## Completed capture record

Both chains were generated on 2026-08-10 from the unchanged
`parity_cases/dem3saves/turn0_initial.xml` source. The raw XML remains outside
the repository under the installed save root.

```text
autocracy_uk_term_noorders_chain2_20260810_step{1..24}_turn1.xml
autocracy_uk_term_orders_chain_20260810_step{1..24}_turn1.xml
```

The first chain contains 24 explicit no-order turns. The second applies the
captured policy sequence through turn 12, then continues with no orders to
turn 24. Every file passed `validate_native_save`, and the serialized turns
are monotonic from 1 through 24.

`gamedrive/term_audit.py` compares every checkpoint offline. It found exact
policy targets across both chains. The durable simulator fixes were the
native no-minister finance fallback (competence `0.25`, producing earn/cost
scalars `0.9375`/`1.0625`) and a zero-floor political-capital contribution.
An explicit `SimulationConfig(minister_resignations=True)` mode models the
quiet chain's turn-15 TAX vacancy; native resignation is probabilistic, so the
flag remains opt-in for other replays.

Remaining differences are concentrated in manager-owned voter/party lists,
effect-ring state not serialized by the game, the global-economy random cursor,
and the order-chain expenditure residual after policy changes.

Acceptance requires complete native XML with monotonic serialized turns 1–24,
an unchanged source save, an offline simulator comparison for every checkpoint,
and focused tests that pass without the installed game.

## Follow-up order-runtime parity record

- [x] Generate a fresh 24-turn policy-intervention chain containing slider
      changes, a cancellation, and new policy introductions.
- [x] Replay captured orders as one native order batch rather than recalculating
      the debt preview after each individual action.
- [x] Model the native previous-policy-history finance sample and midpoint
      seed for newly introduced policies.
- [x] Add regression coverage for native batch previews and introductions.

The fresh intervention chain is stored outside the repository as
`autocracy_uk_policy_rollout_fresh_20260810_step{1..24}_turn1.xml`. The current
offline audit keeps policy targets exact and reduces the order-chain finance
residual to at most about 1,084 income and 2,053 expenditure across the 24
turns. Remaining continuous-state gaps are concentrated in native
effect-ring/load state and manager-owned voter and party data.

## 128-turn no-order stress record

- [x] Generate a native 128-turn UK no-order chain from the unchanged
      `turn0_initial.xml` source.
- [x] Validate all terminal and intermediate native saves and confirm
      monotonic serialized turns.
- [x] Audit every checkpoint with the offline simulator using the documented
      deterministic minister-resignation mode.
- [x] Record the long-horizon residual envelope as the next parity target.

The raw chain is stored outside the repository as
`autocracy_uk_128turn_noorders_20260810_step{1..128}_turn1.xml`. All 128
terminal saves and 128 intermediate load saves passed native validation, with
zero policy-target differences. Over the 128 checkpoints, the audit measured
maximum absolute residuals of about 1,737 income, 299,074 expenditure, 0.600
ordinary-node value, 0.603 situation value, and 0.891 voter value before the
closure work below. That original envelope is retained here as the baseline;
the post-fix residuals are recorded separately.

## 128-turn residual and election closure record

- [x] Parse and serialize election countdown/current-term fields, poll state,
      and poll histories through savegame/state JSON round trips.
- [x] Mirror the headless native countdown and add explicit election-result
      handling for party/sympathy votes, term advancement, vote totals, and
      player-win minister loyalty.
- [x] Use serialized native minister rosters and active-situation schedules as
      explicit audit inputs, keeping the default simulator deterministic.
- [x] Restore serialized voter/party/poll checkpoints and effect-history rings
      during the long-horizon audit; fix prior-turn voter snapshot aliasing.
- [x] Add focused regression coverage for election transitions, roster
      removal, idless effect-history matching, and immutable prior snapshots.
- [x] Re-run the complete 128-turn audit and document the residuals that remain
      in the finance, income-group, situation, and hidden-state model.
- [x] Commit and push the closure batch with synchronized README, simulator,
      and gamedrive documentation.

The post-fix 128-turn audit has zero election countdown/current-term deltas,
zero active-roster and active-situation membership differences, exact
serialized voter fields, and zero effect-history delta. The remaining maximum
absolute residuals are 1,722.2 income, 15,469.3 expenditure, 0.5883 ordinary
node value (`_MiddleIncome`), 0.5471 situation value, 0.0600 hidden-node value,
and 30.75 hidden-history value. At turn 128 the signed finance residual is
about +1,543.2 income / −15,469.3 expenditure; these are model residuals, not
missing serialized manager state.
