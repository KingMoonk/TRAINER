# Affected tests only

- While iterating on a change, run only the tests the change affects, not the whole suite. CI runs everything (unit, `gui_*.py`, e2e and ruff) on every push and pull request.
- Affected means: the `tests/test_*.py` modules that import a changed module, directly or through the module you edited, plus the test files you changed. Find them with `git grep -l -E "import <module>|from <module>" tests/test_*.py`.
- Run them by name: `.venv/bin/python -m unittest tests.test_queue tests.test_rules` (on Windows, `.venv/Scripts/python.exe`).
- Never add `gui_*.py` modules to the list (see CLAUDE.md); a change to a page is covered by its `test_ia_*.py` logic tests locally and its `gui_ia_*.py` tests in CI.
- Run `.venv/bin/python -m ruff check <changed files>` on the files you touched.
- Run the full `unittest discover -s tests` only when the user asks, or when a change touches something most modules import (`studio_core.py`, `studio_ia/services.py`, `studio_ia/copy.py`).
- There are no git hooks: nothing runs on commit or push, so CI is the gate.
