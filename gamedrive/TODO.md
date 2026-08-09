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
- [ ] Drive native slider/order entrypoints so a captured `_orders` save can be
      replayed without treating it as a completed-turn input.
- [ ] Explain and fix the loader stall when reloading probe-produced or edited
      native output XML.
- [ ] Resolve or replace the ptrace-sensitive asynchronous
      `SIM_Gameplay::NextTurn()` launcher path.
- [ ] Recover manager-owned party membership, activist/poll, and per-voter
      income host links from the live process.
- [ ] Build a bounded multi-turn native capture and compare it with the
      simulator's full 12-turn action replay.
- [ ] Add a lightweight opt-in integration smoke test that skips cleanly when
      the installed game binary is unavailable.
