# Agent rules

- Test with `.venv/bin/python tasks.py test` (all window-free tests and ruff) or `.venv/bin/python tasks.py test tests.test_x tests.test_y` (just those modules). That is the only test command you run. Details: `.claude/rules/testing.md`.
- Only run `tasks.py test:ci` or `test:ci:e2e`, `tests/gui_*.py`, or `tests.e2e_train` if trying to debug CI issues. 
- A new test that opens a window goes in `gui_*.py` with that import as its first import; `test_*.py` must stay window-free.
- While iterating, run just the affected tests (see `.claude/rules/affected-tests.md`).
