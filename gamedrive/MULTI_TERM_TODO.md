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
