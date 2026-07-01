# AGENTS.md

## Project Context

- This repository is a Python 3.10 project for multi-agent reinforcement learning experiments on microgrid and related environments.
- Main source code lives under `src/`.
- Environment and algorithm configuration lives under `src/config/`.
- Generated experiment outputs live under `results/` and log files such as `run_*.log`; treat these as run artifacts unless the user explicitly asks to inspect or modify them.

## Development Rules

- Follow the global Codex rules from `C:\Users\xyx\.codex\AGENTS.md`.
- Keep changes surgical and tied to the user's request.
- Do not refactor training, environment, learner, or controller code unless it is required for the task.
- Preserve existing naming and style, including current module names such as `EA_*`, `facmac_*`, and graph/microgrid terminology.
- Do not silently hide runtime errors. Let failures surface unless handling an external boundary such as user input, files, or subprocesses.

## Python And Dependencies

- Use Python 3.10, matching `.python-version`.
- Use `uv` for Python environment and dependency work.
- Run Python commands through `uv run` when practical.
- Do not introduce `pip`, `conda`, `poetry`, or other Python package managers unless the user explicitly approves.
- If dependencies change, update the project dependency metadata and lockfile consistently.

## Common Commands

```powershell
uv sync
uv run python simple_test.py
uv run python test_microgrid.py
uv run python src/main.py --config=facmac_smac --env-config=microgrid with batch_size_run=1 use_cuda=False t_max=5000
```

## Validation

- For environment or reward changes, run a small deterministic smoke test when practical.
- For training-loop changes, prefer a short `t_max` run with `use_cuda=False` unless GPU behavior is explicitly in scope.
- For config changes, verify the target YAML can be loaded by the existing entry point.
- Report any verification skipped because it would require long training, unavailable GPU, StarCraft II assets, or external services.

## Repository Hygiene

- Do not modify generated outputs in `results/`, TensorBoard logs, model checkpoints, or large run logs unless the user asks.
- Do not delete or overwrite user experiment artifacts.
- Avoid committing local virtual environments, caches, generated models, or log files.
- Before committing, check `git status` and confirm the intended author name and email with the user.
