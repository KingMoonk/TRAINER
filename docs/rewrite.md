# Rewrite: web UI, API server, compute cluster

Branch `wedd-compute-cluster` turns the desktop app into two things that talk over HTTP:

- **API** — the Python app minus Tk. FastAPI in front of `studio_ia.services.WorkspaceServices`, run as a long-lived
  service on each machine that has compute. Jobs still run in a worker process (`studio_worker.py`, #84).
- **Web** — a Next.js app (shadcn/ui, Tailwind, dark mode by default) that any browser on the tailnet opens. It holds no
  state the server needs; two browsers can be open against one server.

Later, servers find each other over Tailscale and hand whole jobs to each other (a *cluster*), and versions are published
to a shared bucket so a run on one machine can continue on another.

`master` is the baseline. The branch is a long-lived integration branch: each phase lands as one or more PRs into it, it merges
`master` in regularly (merge, never rebase), and it merges back to `master` when the web UI reaches parity (end of Phase 5).
The Tk app keeps working on the branch until Phase 5 removes it, so nothing on the branch is ever "half an app".

## Goals and non-goals

Goals: a UI that cannot freeze (it is another process, usually another machine); no flicker (React reconciles keyed trees
in place); one URL for the whole app; the same server code on macOS/Apple Silicon and Windows/NVIDIA; pooled compute
between a handful of laptops over WAN.

Non-goals: per-step gradient sync across machines (DDP over WAN is slower than the fastest single box; one run spans
machines with DiLoCo instead, see `docs/distributed.md`); a public
deployment (the tailnet is the perimeter until Phase 7); keeping the desktop packaging (`packaging/`, frozen builds) alive.

## Repository layout (monorepo)

```
/                       Python API and training code, as today (studio_*.py, studio_ia/, tests/, tasks.py)
studio_api/             Phase 1: FastAPI app, routes, models, SSE, node registry
web/                    Phase 0: the Next.js app (pnpm, TypeScript, Tailwind, shadcn/ui, TanStack Query, Playwright)
docs/                   this plan and the operator docs (tailnet setup, running a node)
```

The Python code stays at the root until the Tk app is gone: moving fifty `studio_*.py` modules into a package while two
UIs still import them is churn for no gain. Phase 5 may move them under `server/` once there is one UI.

Conventions: Conventional Commits with scopes `api`, `web`, `cluster` (plus the existing ones). `tasks.py` stays the one
entry point and gains `web:lint`, `web:test`, `web:build`; CI gets an ubuntu `web` job (lint, typecheck, build, Playwright
against `FakeServices`) alongside the macOS/Windows `smoke` job. The macOS/Windows job stays Python-only.

## Phases

Each phase lists what it delivers and what "done" means. Boxes are ticked on the branch as PRs land.

### Phase 0 — scaffold (#91)

- [x] Branch `wedd-compute-cluster` from `master`; this document.
- [x] `web/`: Next.js (app router), TypeScript strict, Tailwind, shadcn/ui with a dark theme as the default, TanStack Query,
      an `api` client module (fetch + SSE). It is hand-written until Phase 1 generates it from the API's OpenAPI schema.
- [x] `tasks.py web:*`, CI `web` job, pnpm lockfile committed. README section: how to run the web UI locally (the API
      joins it in Phase 1).

Done: `python tasks.py web:build` passes in CI; `pnpm dev` shows an empty shell with nav and dark theme.

### Phase 1 — API server (#93)

- [ ] `studio_api/`: FastAPI + uvicorn. One router per `Services` scope: `versions`, `queue`, `storage`, `passive`, `data`,
      `pretraining`, `post_training`, `chat`, `inspect`, `benchmark`, `viewer`, `node`. One route per public `Services`
      method; Pydantic models derived from the dataclasses (`Version`, `Job`, `QueueState`, `Plan`, `Gate`).
- [x] Events: `GET /events` is an SSE stream of every `Services.emit(kind, payload)`; broadcast to every subscriber.
      `Services` gets a thread-safe subscriber path that does not go through Tk (`bind_ui`/`pump` stay for the Tk app until
      Phase 5).
- [ ] Long reads become jobs: `lr_range_test`, `pretrain_preview`, `inspect_*`, `run_benchmark`, `compare_runs`, scoring.
      `POST` returns a job id; progress and result arrive on `/events`; `GET /jobs/{id}` for late joiners. The #82 snapshot
      store, if it lands on `master`, is the read model these `GET`s serve. `ready()`/`_Slow`/`_Cold` are not used by the API.
- [x] Files cross the wire, not paths: `import_data`, `import_source`, `import_pretrained` accept uploads (and a server-side
      folder picker for corpora already on the box); `export_version` is a download.
- [x] Per-client state leaves the server: `stage_draft`/`draft_spec` (the Builder's unsaved design) lives in the browser.
- [x] `node` router: capabilities (`studio_backend.gpu_status()`: device kind, memory), busy/free, version, hostname.
- [x] Serve `web/out` (static export) from the API so the app is one URL; `--bind` defaults to `127.0.0.1` (open only on
      loopback or with `--fake`; `docs/node.md` says how to bind the tailnet address), and `--fake` serves `FakeServices` for
      frontend work without a workspace.
- [x] Auth: trust `Tailscale-User-Login` from `tailscale serve`; refuse requests without it unless `--bind 127.0.0.1`.
- [x] Tests: `tests/test_api_*.py` with FastAPI's `TestClient` over `FakeServices`; one test per router that every route
      answers and every emitted event reaches `/events`.
- [x] `web/src/lib/api.ts` is typed from the API's OpenAPI schema (`pnpm gen:api`; the CI `web` job checks it is current),
      replacing the hand-written client from #91.
- [x] `docs/node.md`: run the API as a service (launchd on macOS, a scheduled task on Windows), tailnet setup, ACL.

Done: the Tk app still passes CI; `python -m studio_api --fake` serves a schema with every `Services` method; the
generated TypeScript client compiles against it.

### Phase 2 — web shell and the always-visible pages (#94)

- [ ] Layout: sidebar nav mirroring today's routes, status strip (current version, queue, resources) driven by SSE,
      loading skeletons instead of spinners, toasts for `problem_report`.
- [ ] Queue (jobs, plan, hold/move/remove, pause/resume, checkpoint now) and Storage (clean, free space).
- [ ] Your data and Pretraining (the two pages #87 patched for flicker): collections, import/download, previews, word lists,
      LR range test as a job.
- [ ] Copy: `studio_ia/copy.py` strings move to `web/` page by page as each page is built; the Python copy of a string is
      deleted when its last Tk user goes (Phase 5).
- [ ] Playwright smoke per page against `--fake`.

Done: a user can import data, queue a pretraining run, watch it progress and see the version appear, all from the browser.

Verified by `web/tests/live.phase2.spec.ts` against the real `--fake` API and SSE stream in CI
(PR #220, web run 38853281308). This covers the done-line workflow; the remaining page-level checklists above stay separate.

### Phase 3 — the rest of the pages (#95)

- [ ] Builder (largest page, 2k lines of Tk): tower, start-from, controls, costs/expectations, tray states (edit, saved,
      queued, training), charts, look-inside links, and the #83/#85 learning loop (guess-before-saving, one change per
      version, lab notebook, explain panel, parameter bar, experiments). Draft design is browser state.
- [ ] Post-training (LoRA options from #79), Chat (streamed tokens over SSE; KV cache from #86), Training run, Loss, Trace,
      Value map, Head lab / Head zoom, Compare, Benchmark, Quick test, Passive, Make data.
- [ ] Charts: one chart component (recharts or visx) for loss/BPC/compare; no page draws its own.

Done: every route in the Tk shell has a web page with the same capabilities; the route check test lists both.

### Phase 4 — 3D viewer in the browser (#96)

- [ ] `viewer` router returns scene geometry and stats from `studio_scene.SceneModel` as JSON (plus a binary endpoint
      for large tensors if JSON is too slow).
- [ ] `web/` Inside 3D page in react-three-fiber: modes, head stages, tile kinds, play/step, probe step.
- [ ] Delete `studio_brain3d.py`, `studio_viewer_ui.py`, `studio_vizbridge.py`'s process plumbing, the `panda3d` dependency
      and the headless-display step in CI.

Done: the viewer works in Safari and Chrome against a live run; CI no longer needs Mesa.

### Phase 5 — remove Tk, merge to master (#97)

- [ ] Delete `studio_ia/shell.py`, `widgets.py`, `theme.py`, `pages/`, `studio_widgets.py`, `studio_ux.py`,
      `studio_ui_guard.py`'s Tk parts, `tests/gui_*.py`, `packaging/`, `start_studio.bat`, the `--run-module` trick in
      `app.py`. `app.py` starts the API and opens the browser.
- [ ] `Services` loses `bind_ui`/`pump`/`ready`; `copy.py` is gone; `tasks.py test:ci` drops the window level.
- [ ] Optional: move the Python package under `server/`.
- [ ] Merge the branch to `master`; `master` CI runs the Python matrix plus the `web` job.

Done: no `tkinter` import anywhere; CI is faster than today.

### Phase 6 — cluster (#98)

- [ ] Node registry: every API is a coordinator and a worker; nodes are listed by hand (tailnet names) in the workspace
      config, and each polls the others' `/node` for capabilities and load. A node alone is a cluster of one.
- [ ] Jobs carry requirements (device preference, minimum memory, dtype); placement picks the best free node, or the one
      the user pinned. Defaults: pretraining and RLHF prefer CUDA; eval, scoring, benchmarks, data generation, LoRA and
      passive training take any free node; memory-bound inspect jobs prefer the node with the most memory.
- [ ] Transient nodes: a job whose node disappears is marked interrupted at its last checkpoint and resumes when the node
      returns or on another node (`checkpoint_now`/`resume` already exist; the optimizer restarts on a different node).
- [ ] Version store interface: local folder (always) plus an S3-compatible bucket (Cloudflare R2 or Backblaze B2 free
      tier). "Publish" uploads bf16 weights only (no optimizer state); other nodes list it as "available, not local" and
      pull on demand or when a job needs it. LAN peers copy directly.
- [ ] Checkpoints are `.npz` NumPy archives (`studio_model.load_checkpoint`), so they already load on any device; a version
      records the device that trained and scored it, and a resume on the other device kind is a test.
- [ ] Distributed pretraining: one run on several nodes with DiLoCo (local AdamW steps, an outer Nesterov step on the
      averaged pseudo-gradients every H steps), elastic membership and round checkpoints (`docs/distributed.md`, #132, #133).
- [ ] The web UI shows the merged queue and merged versions across nodes, with the node on each row.
- [ ] `docs/cluster.md`: the tailnet, inviting a second user, ACL, bucket setup.

Done: a pretraining job queued from a Mac runs on the Windows/NVIDIA node, its version is published, and a benchmark of it
runs on a Mac.

### Phase 7 — hosting and auth, later and optional (#99)

- [ ] Web on Vercel with real sign-in (Auth.js or Clerk); the API stays on the tailnet behind `tailscale serve` (or Funnel
      with a token check), CORS for the Vercel origin.
- [ ] Until then the API serves the static export and the tailnet is the perimeter.

## What carries over from current work

- #84 worker process: becomes the remote worker contract in Phase 6 unchanged.
- #82 snapshot store and command channel: the API's read model (Phase 1). The Tk render scheduler and the keyed-lists
  helper (#87) are Tk-only and go in Phase 5.
- #83 Builder learning loop (#85 and the later items): the features are rebuilt in Phase 3; the logic functions
  (`expectations`, `design_expectation`, `sizes`, `change_chips`, `attribution_inputs`) are plain Python and stay.
- #86 KV cache: used by streamed chat in Phase 3.
- Every `tests/test_*.py` logic test stays; only `gui_*.py` goes (replaced by Playwright in Phase 2 onwards).

## Risks

- The Builder is a third of the UI code; Phase 3 is the long pole. Build it last in Phase 3 and keep the Tk Builder
  usable until then.
- Starlink upload (~10–20 Mbps, CGNAT): checkpoint transfer is minutes, so the bucket is the exchange point and nothing
  in the UI waits on a transfer. Tailscale may relay through DERP; `tailscale ping` shows which.
- Two device kinds: every kernel path (`torch.compile`, fused attention) needs a plain fallback on both; the macOS/Windows
  CI matrix is the guard.
- Long-lived branch drift: merge `master` into the branch at least weekly; the Tk app on the branch must keep passing CI
  until Phase 5.
