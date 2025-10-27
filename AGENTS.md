# Repository Guidelines

## Project Structure & Module Organization
The core package lives under `autocracy/`, covering the simulator, agents, data loaders, and savegame bridge; import paths should stay rooted there. The Typer CLI entrypoint (`main.py`) sits at the repo root, while reusable assets are under `gamedata/data` with reference saves in `gamedata/saves`. Tests live in `tests/`, and both `README.md` and `SIMULATION.md` must be kept in sync whenever you change public behavior.

## Build, Test, and Development Commands
- `uv run main.py describe --country uk` – quick validation that the DAG loads and prints metrics.
- `uv run main.py simulate --turns 2 -p IncomeTax:-0.05` – exercise the update loop with sample policy nudges.
- `uv run pytest` – run the suite; append `-k simulator` or `-vv` when chasing a regression.

## Coding Style & Naming Conventions
Target Python 3.12 with full type annotations, mirroring the existing `from __future__ import annotations` imports. Use 4-space indents, snake_case for functions, and short imperative Typer command names; keep data classes and helpers small and composable. Prefer `Path` objects for filesystem access, guard side effects behind functions, and add concise comments only when behavior is non-obvious.

## Testing Guidelines
Pytest drives coverage, so drop new files beside `tests/test_simulator.py` using the `test_<feature>.py` convention. Build fixtures with `simulator.get_initial_state()` or `tmp_path`, assert using `pytest.approx` where floats are involved, and wrap exceptions with `pytest.raises` for clarity. When modifying policy math or serialized state, add regression cases that fail before your change and pass after.

## Commit & Pull Request Guidelines
The branch currently lacks Git history, so establish good habits now: one logical change per commit, imperative subjects under 72 chars, and detailed bodies when reasoning is non-obvious. PRs should link to any discussions, describe user-visible impact, and include the exact `uv run pytest` (or CLI) output you used for validation. Add screenshots or short tables when tweaking GDP/policy metrics so reviewers can compare before/after quickly.

## Security & Configuration Tips
Treat `gamedata` as read-only reference assets; point experiments at an alternate directory via `--gamedata` or `BaseAgent(..., gamedata_root=...)` instead of editing in place. Avoid committing transient JSON snapshots (`tmp_state*.json`, `ukout*.json`) or proprietary saves, and scrub any sensitive annotations from shareable logs.
