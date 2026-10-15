# Affected tests only

- While iterating on a change, run only the tests the change affects, not the whole suite. CI runs everything (`tasks.py test:ci:e2e`: compileall, ruff, unit, `gui_*.py` and e2e) on every push and pull request.
- Affected means: the `tests/test_*.py` modules that import a changed module, directly or through the module you edited, plus the test files you changed. Find them with `git grep -l -E "import <module>|from <module>" tests/test_*.py`.
- Run them by name: `.venv/bin/python tasks.py test tests.test_queue tests.test_rules` (on Windows, `.venv/Scripts/python.exe`). It also runs ruff.
- Never add `gui_*.py` modules to the list (see CLAUDE.md and `.claude/rules/testing.md`); a change to a page is covered by its `test_ia_*.py` logic tests locally and its `gui_ia_*.py` tests in CI.
- Run the full `.venv/bin/python tasks.py test` only when the user asks, or when a change touches something most modules import (`studio_core.py`, `studio_ia/services.py`, `studio_ia/copy.py`).
- There are no git hooks: nothing runs on commit or push, so CI is the gate.
