# Native game-driving TODO

- [x] Verify the Democracy 3 v1.30.2 ELF entrypoints and singleton addresses
      with static preflight.
- [x] Load a copied save through the game's asynchronous load path and save
      the native post-load state under a fresh name.
- [x] Run the native `NextTurnThread(void*)` manager sequence synchronously
      from a bounded headless gdb session.
- [x] Add an opt-in, process-local neuron read/write probe and verify a
      persisted GDP edit without changing the source capture.
- [x] Compare the direct native no-order turn against the simulator across
      ordinary nodes, situations, voters, policies, and finance fields.
- [x] Drive native slider/order entrypoints so a captured `_orders` save can be
      replayed without treating it as a completed-turn input.
- [x] Explain and fix the loader stall when reloading probe-produced or edited
      native output XML.
- [x] Resolve or replace the ptrace-sensitive asynchronous
      `SIM_Gameplay::NextTurn()` launcher path.
- [x] Recover manager-owned party membership, activist/poll, and per-voter
      income host links from the live process.
- [x] Build a bounded multi-turn native capture and compare it with the
      simulator's full 12-turn action replay.
- [x] Add a lightweight opt-in integration smoke test that skips cleanly when
      the installed game binary is unavailable.

The completed harness is deliberately split into safe, reproducible pieces:
`order_plan.py` translates pre-turn `_orders` saves, `native_probe.cpp` applies
the native policy methods and bounded `NextTurnThread` loop, and `capture.py`
performs the offline 12-turn comparison. Fresh-output and XML-boundary checks
in `inject_drive.py` prevent stale or truncated files from being mistaken for
successful reloads. Manager refresh/census is opt-in because it mutates live
manager lists before the audit save. The native integration test is also
opt-in; normal CI skips it when the installed game and display tooling are not
available.
