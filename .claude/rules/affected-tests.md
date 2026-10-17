# Affected tests only

- While iterating on a change, run only the tests the change affects, not the whole suite. CI runs everything on every push and pull request (`tasks.py test:ci`: ruff, unit and compileall); the real training run (`test:ci:e2e`) runs nightly.
- Affected means: the `tests/test_*.py` modules that import a changed module, directly or through the module you edited, plus the test files you changed. Find them with `git grep -l -E "import <module>|from <module>" tests/test_*.py`.
- Run them by name: `.venv/bin/python tasks.py test tests.test_queue tests.test_rules` (on Windows, `.venv/Scripts/python.exe`). It also runs ruff.
- Run the full `.venv/bin/python tasks.py test` only when the user asks, or when a change touches something most modules import (`studio_core.py`, `studio_ia/services.py`, `studio_ia/copy.py`).
- Web changes: run the Vitest files beside what you changed, e.g. `pnpm test src/app/queue` in `web/`, then the whole `pnpm test` once before the PR, with `pnpm lint` and `pnpm typecheck`. Playwright (`pnpm test:e2e`) is CI-only. A changed API response model also needs `python tasks.py web:fixtures` (see `testing.md`).
- There are no git hooks: nothing runs on commit or push, so CI is the gate.
