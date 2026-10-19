# Groundwork Studio web UI

The browser front end that replaces the Tk app (docs/rewrite.md). Next.js app router, TypeScript, Tailwind, shadcn/ui,
TanStack Query. Static export: `pnpm build` writes `out/`, which the API server serves.

```bash
pnpm install
pnpm dev            # http://localhost:3000
pnpm lint && pnpm typecheck && pnpm build
pnpm test           # Vitest: unit, component (jsdom) and contract tests; no browser, no build needed
pnpm exec playwright install chromium   # once, CI only
pnpm test:e2e       # Playwright e2e flows against out/, so build first; no API needed. CI runs it; do not run it from an agent
```

`NEXT_PUBLIC_API_URL` points `src/lib/api.ts` at the API server; leave it unset when the API serves `out/` itself.

## Theming

Every colour is a token in `src/app/globals.css` (the token list and how to add a theme are at its top); components use
semantic utilities (`bg-card`, `text-warning`, `bg-surface-overlay`) and never a palette class or a colour literal.
`pnpm lint` runs `scripts/check-theme.mjs`, which fails on either one outside that file. Light / dark / system and the
accent are chosen under Appearance on the Settings page (the gear in the sidebar, `/settings`).

## Running with the API

On Windows, after `python tasks.py web:build`, run `start_web.bat` from the repository root and open
`http://127.0.0.1:8787/chat/`. The launcher uses the main checkout's Python environment and workspace when started from a
Git worktree. Keep its console open while using the app; startup errors remain visible there.

To choose the real workspace explicitly from another checkout:

```bash
python -m studio_api --workspace /path/to/TRAINER --port 8787
```

Use one running app per workspace. `--workspace` cannot be combined with `--fake`.

```bash
# repo root: the API over empty in-memory data (drop --fake for a real workspace)
.venv/bin/python -m studio_api --fake --port 8787
# web/: the dev server, calling it (the API allows http://localhost:* origins while it is bound to loopback or --fake)
NEXT_PUBLIC_API_URL=http://127.0.0.1:8787 pnpm dev
```

The status strip says **Connected** once `GET /events` is open. `docs/node.md` covers running the API as a service.

## The API client

`src/lib/api.ts` has `api`, an [openapi-fetch](https://openapi-ts.dev/openapi-fetch/) client typed from the server's OpenAPI
schema, and `unwrap()` to turn a call into its data or an `ApiError`:

```ts
const job = await unwrap(api.GET("/jobs/{job_id}", { params: { path: { job_id } } }));
```

A route or field the server does not have fails `pnpm typecheck`. `fetchJson` and `subscribe()` (the `/events` stream) stay for
anything outside the schema.

The types are generated, in two committed files:

| File | Made by | From |
| --- | --- | --- |
| `openapi.json` | `python -m studio_api --dump-openapi web/openapi.json` | the routes of `create_app(FakeServices())` |
| `src/lib/api-schema.ts` | `pnpm gen:api` ([openapi-typescript](https://openapi-ts.dev/)) | `openapi.json` |

**Any PR that adds, removes or changes a route or a response model regenerates both files** and commits them:

```bash
.venv/bin/python -m studio_api --dump-openapi web/openapi.json && (cd web && pnpm gen:api)
```

Two checks keep them honest. `tests/test_api_schema.py` (part of `python tasks.py test`) fails when `openapi.json` differs from
the app's schema and prints the command above. The CI `web` job runs `pnpm gen:api` and `git diff --exit-code` on the generated
TypeScript, so a hand-edited `api-schema.ts` fails there. Do not edit either file by hand.

## Tests

Vitest (`src/**/*.test.ts(x)`) holds the unit, component and contract tiers; Playwright (`tests/`) holds the e2e flows that need a real browser. See `.claude/rules/testing.md` for which tier a test belongs in and for re-recording the contract fixtures (`python tasks.py web:fixtures`).

## Tests against the live API

`tests/live.spec.ts` runs against `python -m studio_api --fake` serving `out/` (the `live` Playwright project), and checks that
the status strip goes from Connecting to Connected. Any `tests/live.<page>.spec.ts` runs in the same project, against a
`--fake` API that runs a queued job when `POST /queue/start` is called: `step` rises on `train` events, then a version is added. Uploaded files are parsed into memory; no training weights or workspace records are written. Live specs share one worker, and the server uses `--fake-step-seconds 0.25` so progress can be observed. `live.phase2.spec.ts` imports a text file, selects it for pretraining, starts the queue and checks the trained version, including after a refresh. It needs the Python requirements, so it is skipped unless `API_LIVE=1`:

```bash
API_LIVE=1 API_PYTHON=$PWD/../.venv/bin/python pnpm test:e2e   # API_PYTHON defaults to `python`
```
