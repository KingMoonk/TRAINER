# Agent rules

- Never run `tests/gui_*.py`, and never open an app window locally (no `app.py`, `studio_ia.shell`, Panda3D viewer, `tk.Tk()`), not even to check.
- Locally run only `unittest` on `tests/test_*.py` modules and `ruff check`; while iterating, run just the affected tests (see `.claude/rules/affected-tests.md`).
- Window tests are CI-only: each `gui_*.py` starts with `from tests import windows`, which skips the module unless `CI` or `GROUNDWORK_GUI_TESTS=1` is set. Do not set either.
- A new test that opens a window goes in `gui_*.py` with that import as its first import; `test_*.py` must stay window-free.
