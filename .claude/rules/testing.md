# Testing

- `tasks.py` is the only test interface. Three levels, each including the one before: `test`, `test:ci`, `test:ci:e2e`. Pull requests run `test:ci` on Linux; `test:ci:e2e` runs nightly on macOS and Windows (`.github/workflows/nightly.yml`).
- Agents and local runs use `test` only: `.venv/bin/python tasks.py test` for everything, or `.venv/bin/python tasks.py test tests.test_x` for named modules. It runs ruff and `tests/test_*.py`. (On Windows, `.venv/Scripts/python.exe`.)
- `test:ci` adds compileall; `test:ci:e2e` adds the real training run (`tests.e2e_train`). Run them only to debug a CI failure.
- Do not call `python -m unittest` or `ruff` directly when `tasks.py test` does the job.
- Tests never open a window: `test_*.py` stays window-free. Put page logic in plain functions and test those.
