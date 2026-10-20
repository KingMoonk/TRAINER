# Affected tests only

- Run only the tests the change affects, never the whole suite, while iterating and before a PR alike. CI runs everything on every push and pull request (`tasks.py test:ci`: ruff, unit and compileall); the real training run (`test:ci:e2e`) runs nightly.
- Affected means: the test files you changed, plus the tests that exercise the code you changed (the `tests/test_*.py` modules or Vitest files that call the changed function, route or page). Find them with `git grep -l <name> tests/test_*.py` or `git grep -l <name> -- 'web/src/**/*.test.ts*'`. A module that only imports a changed file, without using what changed, is not affected.
- Run them by name: `make test T="tests.test_queue tests.test_rules"`. It also runs ruff.
- Never run the full `make test`, the whole `pnpm test` or `make web-check` unless the user asks. A change to a widely imported file (`studio_core.py`, `studio_ia/services.py`, `studio_ia/copy.py`) is no exception: run the tests of what changed and let CI run the rest.
- Web changes: run the Vitest files for what you changed, e.g. `pnpm test src/app/queue` in `web/`, plus `pnpm lint` and `pnpm typecheck`. Playwright (`pnpm test:e2e`) is CI-only. A changed API response model also needs `make web-fixtures` and the contract tests (`pnpm test src/contract`), see `testing.md`.
- If CI fails, run the failing tests locally by name, not the suite.
- There are no git hooks: nothing runs on commit or push, so CI is the gate.
