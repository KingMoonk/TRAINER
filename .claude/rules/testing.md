# Testing

- `tasks.py` is the only test interface. Three levels, each including the one before: `test`, `test:ci`, `test:ci:e2e`. Pull requests run `test:ci` on Linux; `test:ci:e2e` runs nightly on macOS and Windows (`.github/workflows/nightly.yml`).
- Agents and local runs use `test` only, through the Makefile: `make test T="tests.test_x"` for the affected modules (`affected-tests.md`); bare `make test` runs everything, only when the user asks. It runs ruff and `tests/test_*.py`. (The Makefile picks `.venv/Scripts/python.exe` on Windows.)
- `test:ci` adds compileall; `test:ci:e2e` adds the real training run (`tests.e2e_train`). Run them only to debug a CI failure.
- Do not call `python -m unittest` or `ruff` directly when `make test` does the job.
- Tests never open a window: `test_*.py` stays window-free. Put page logic in plain functions and test those.

## Web tiers

The web app (`web/`) is tested in tiers; put a test in the lowest tier that can show the behaviour.

1. Unit (Vitest): pure logic. Every `web/src/app/*/logic.ts` and `web/src/lib/*` module with logic has a `*.test.ts` beside it.
2. Component (Vitest, jsdom, Testing Library): a page against a mocked API. Use `renderPage`, `mockApi` and `emit` from `web/src/test/`; the existing `page.test.tsx` files are the model. jsdom has no layout, so overflow and viewport checks stay in Playwright.
3. Contract (Vitest): `web/src/contract/` replays API responses recorded over `FakeServices` through the client and the real pages. The recorder is `tests/web_fixtures.py`; `tests/test_web_fixtures.py` (part of `make test`) fails while `web/src/contract/fixtures/*.json` is stale. After changing a response model, run `make web-fixtures`, then `pnpm test src/contract` in `web/`, and commit the fixtures.
4. e2e (Playwright, Chromium, every PR): the few flows that need a real browser (drag and keyboard reorder, streaming, phone-width sweep, the `live` specs against `studio_api --fake`). Do not add a page-level assertion here that a component test can make.
5. Nightly: the same e2e flows in WebKit (`web-webkit` in `nightly.yml`), and the macOS and Windows runs.

Locally run the affected Vitest files (`pnpm test <path>`). Never run Playwright locally; CI runs `pnpm test:e2e`. Test files must not contain palette classes, named colours or hex/rgb literals (`pnpm lint` runs the theme guard over all of `web/src`).
