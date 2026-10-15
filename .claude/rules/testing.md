# Testing

- `tasks.py` is the only test interface. Three levels, each including the one before: `test`, `test:ci`, `test:ci:e2e`.
- Agents and local runs use `test` only: `.venv/bin/python tasks.py test` for everything, or `.venv/bin/python tasks.py test tests.test_x` for named modules. It runs ruff and the window-free `tests/test_*.py`. (On Windows, `.venv/Scripts/python.exe`.)
- `test:ci` adds compileall and the window tests (`tests/gui_*.py`); `test:ci:e2e` adds the real training run (`tests.e2e_train`). They open real Tk and Panda3D windows, so they are CI-only and refuse to run unless `CI` is set. Never run them, never set `CI` or `GROUNDWORK_GUI_TESTS=1` to get around that, and never open an app window to check a page: a page change is covered locally by its `test_ia_*.py` logic tests and in CI by its `gui_ia_*.py` tests.
- Do not call `python -m unittest` or `ruff` directly when `tasks.py test` does the job.
- New test that opens a window: `gui_*.py`, with `from tests import windows` as its first import. `test_*.py` stays window-free. Prefer putting page logic in plain functions and testing those in `test_*.py`.
