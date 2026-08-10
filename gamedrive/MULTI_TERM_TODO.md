# Multi-term native capture and parity TODO

The UK mission uses a 16-turn electoral term. The first long-run target is 24
turns: one complete term plus eight additional quarterly turns. Raw native XML
stays outside the repository under the installed save root; only summaries,
fixtures, and reproducible tooling belong here.

- [x] Add a term-aware launcher that derives mission term length and pads a
      captured order sequence with explicit no-order turns.
- [x] Extend the offline comparator to replay a no-order-only capture for an
      explicit number of turns.
- [ ] Generate and validate a 24-turn UK no-order capture from `turn0_initial`.
- [ ] Generate and validate a 24-turn UK capture using the existing 12-turn
      policy sequence followed by a no-order tail.
- [ ] Compare every captured turn across finance, ordinary nodes, situations,
      policies, voters, parties, polls, and hidden histories.
- [ ] Identify the first sustained divergence and its native manager/order
      cause instead of tuning to a single end-of-term checkpoint.
- [ ] Improve the simulator from the long-run evidence and add regression
      tests for each durable fix.
- [ ] Record capture names, commands, parity summaries, and remaining limits in
      `gamedrive/README.md`, `README.md`, and `SIMULATION.md`.
- [ ] Commit and push the capture tooling, parity fixes, and documentation in
      reviewable milestones.

Acceptance requires complete native XML with monotonic serialized turns 1–24,
an unchanged source save, an offline simulator comparison for every checkpoint,
and focused tests that pass without the installed game.
