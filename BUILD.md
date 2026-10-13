# BUILD — Groundwork Studio IA (final design)

Source of truth for the new app: `design/ia/*.dc.html` (16 boards, read the visible copy and the
`renderVals()` logic), `design/ia/SPECS.md` (BUILD SPEC notes, DESIGN RULES, DATA CONTRACT) and
`design/ia/boards.txt` (boards stripped to copy + logic, easiest to read).

Decisions that override anything older in the repo or the canvas notes:

- **Engine: PyTorch** (`torch`), CPU / CUDA / Apple MPS. The canvas note "No PyTorch" is superseded by
  the owner's instruction. Checkpoints stay `.npz` (numpy arrays) so every consumer keeps working.
- **New UI lives in `studio_ia/`** (package). Old pages (`studio_gui.py`, `studio_flow.py`, `studio_network.py`,
  `studio_align.py`, `studio_curriculum_page.py`, `studio_passive_page.py`) stay untouched by builders and are
  retired at cut-over. `app.py` switches to `studio_ia.shell.main` once the cut-over gate passes.
- **Pages never import engine modules directly.** They call `studio_ia.services.Services` (one façade).
  Tests use `studio_ia.services.FixtureServices` over the fixture workspace (scope S10).
- **Design rules** (`design/ia/SPECS.md`, "DESIGN RULES · three depths") apply to every page: one question,
  one picture, one primary action; plain words with technical names as small grey glosses; every number is
  exact or `[N]`; one unit (bits per character; nats only on the Loss page); k of n, never a bare percentage
  on small samples; a guess before every run; seeds reported as a spread; every automatic switch gated on a
  measured check; state always visible (not saved amber · saved/needs training coral · training violet · live blue).
- **No mock states in the build.** The boards' "MOCK" / "example shape" switches are review aids; a page
  shows real values or `[N]` / an empty state.

## Theme tokens (`studio_ia/theme.py`)

bg `#0e1014` · panel `#15181e` · panel2 `#1b1f27` · border `#2a2f3a` · border2 `#20242c` · text `#e6e8ec` ·
text2 `#c3c8d1` · muted `#8f97a6` · dim `#5b6271` · violet `#9f8cff` (selected / running) · amber `#e5b567`
(not saved) · coral `#ef8a7a` (needs training / fail) · blue `#6cb4ee` (live / pass) · pink `#e58fb0` (attention,
guess chips) · sage `#9fd36b` (word pieces) · amber-bg `#211c12` · coral-bg `#26171a` · blue-bg `#121a22` ·
violet-bg `#1a1730`. Fonts: system UI 15px body, 32px h1, 22px h2, 12px kickers with 0.08em tracking,
monospace for numbers, Georgia for model output.

## Shared services façade (`studio_ia/services.py`)

```
class Services:                       # every method is synchronous; long work runs through the queue
    workspace: studio_core.Workspace
    # versions (scope S3)
    versions() -> list[Version]       # number, name, change, parent, spec, status, best_checkpoint, scorecard, verdict
    current_version() -> Version | None
    best_version() -> Version | None  # lowest held-back bits/char among versions with a checkpoint
    save_version(spec, change_text) -> Version
    # queue (scope S2)
    queue() -> QueueState             # now (job, step, total, phase, lr, eta_s, paused), up_next [jobs], plan
    enqueue(job) / move_up(id) / remove(id) / pause() / resume() / checkpoint_now()
    set_plan(plan)
    # data (scope S5)
    data(kind) -> DataSummary         # kind in ("text", "conversations", "problems")
    # evaluation (scope S4)
    references(version) -> {uniform_bpc, unigram_bpc, best_so_far_bpc, noise_band}
    run_curve(run_dir) -> Curve       # per-step, smoothed, held-back dots, lr lane, schedule phases
    # inspect (scope S7)  thin wrappers over studio_inspect, numpy out
    # chat
    chat(version, messages, on_text, stop) -> str
    subscribe(callback)               # UI thread callback for queue / run events
```

Scopes extend this file; add methods, never rename existing ones.

## Scopes, owners and gates

Every scope: minimal code, no new dependencies, `ruff check .` clean, its tests green, do not commit, do not
edit files outside the scope's list. Report what was built, what was skipped and the verbatim test tail.

### S1 · UI foundation (sonnet) — `studio_ia/__init__.py, theme.py, widgets.py, shell.py, services.py, pages/__init__.py, pages/base.py`, `tests/gui_ia_shell.py`
Shell: 220px sidebar (4 areas: Network designer · Pretraining · Post-training · Chat, each with its gloss line),
machine strip at the bottom of the sidebar (what is training, progress bar, time left + queued count, which
version Chat uses, link to Queue), main area with page router `shell.show("builder")`. Widgets: Card, Kicker,
Chip(state), Segmented, Toggle, GuessChips(labels) -> value, KofN, StateTray, HowItWorks(collapsible),
BpcPlot (per-step faint, smoothed bold, held-back dots, reference lines with an axis break, shaded unrun
remainder, LR lane with warm-up/steady/cool-down bands), HeatGrid, Progress. Page base class with
`refresh()` and `on_event(kind, payload)`. Routes for all 16 boards registered with placeholder pages that
render the board's h1 + lead line.
Gate: `gui_ia_shell.py` builds the window with `FixtureServices` (stub it with an in-memory fake until S10 lands),
visits all 16 routes, asserts no Tk error and that the machine strip text updates from a fake queue event.

### S2 · Queue and plan (sonnet) — `studio_queue.py`, `studio_jobs.py`, `studio_ia/pages/queue.py`, `tests/test_queue.py`
One runner thread, one job at a time, persisted `data/queue.json` (atomic). Job = {id, kind, label, note,
params, status, eta_s, created_at}. Kinds: pretrain (catch_up | from_scratch), quicktest, compare,
post_round (1..4), restore_answers, checkpoint_now (cool-down branch of the running job, main run continues).
Runner API: pause() (resumes exactly: uses train()'s pause event), resume(), stop_after_cooldown(), reorder, remove.
Plan: hours (any | nights | set range), memory (half | most | all of physical RAM, enforced as a batch-size cap
passed to jobs), idle_only, pause_on_battery, keep_awake, chat_follows_best; reuse platform helpers from
`studio_passive_watch` (battery, keep-awake, idle) by import, never duplicate them. Automatic job: after a
pretrain job whose data included user text, enqueue restore_answers (post_round 3, teacher = previous best).
Handlers in `studio_jobs.py` take (job, services, emit, stop, pause); the pretrain handler calls
`studio_model.train`; others may be stubs that emit "not built yet" until their scope lands.
Gate: tests with fake handlers: sequential execution, persistence across runner restart, pause/resume,
move_up/remove, plan hour window blocks start, idle/battery pause via patched helpers, checkpoint_now does not
stop the main job. Queue page renders Now / Up next / When it may train from the state.

### S3 · Versions and warm start (sonnet) — `studio_versions.py`, `studio_warmstart.py`, `tests/test_versions.py`, `tests/test_warmstart.py`
Versions: `data/versions.json`; each version = {number, change (plain words from a spec diff), parent, spec,
created_at, runs[], best_checkpoint, status (draft|saved|queued|training|trained), scorecard, verdict}.
Spec diff in plain words ("more blocks (4 → 6)", "look-back heads 4 → 8", "wide thinking room",
"small word list"). Warm start on numpy param dicts: `warm_start(old_params, old_spec, new_spec) ->
(params, report)`; add block = new block's residual-out projections (attention proj / MLP f2 / swiglu down /
ssm out) zero so the function is preserved; widen MLP = copy old rows/cols, zero the new output rows (preserved);
re-split heads at fixed width = reuse qkv/proj as-is (not preserved, say so); remove parts = drop; tokenizer
change = fresh ("needs training from scratch"). Report lists copied/fresh/zeroed and whether the result is
function-preserving.
Gate: for add-block and widen-MLP, logits of the warm-started model equal the old model's on random ids
(atol 1e-5) using `studio_compile.CompiledModel` + `load_weights`; version numbering and diff text tests.

### S4 · Evaluation and references (sonnet) — `studio_eval.py`, `tests/test_eval.py`
`heldout_split(docs, fraction)`: by document, exact duplicates removed first. `bits_per_char(checkpoint, docs)`
= total nats over total characters ÷ ln 2 (characters the word list cannot encode scored at the uniform rate and
counted). `uniform_bpc(tokenizer, docs)`, `unigram_bpc(tokenizer, docs)` (exact from token counts),
`noise_band(checkpoint, docs)` = spread of per-document bpc estimates (bootstrap over documents, report lo/hi).
`run_status(history)` -> learning | flat | memorising per the Training board rule. `smooth(curve)`.
`training_saved(curve_new, curve_old)` -> {percent, comparable: bool, reason} with the same-length-same-schedule
rule. `sanity_checks(run_dir)` -> the four quick-test checks. Model calls go through `studio_model.LocalModel`
/ `ScratchTransformer.forward` + `backend.to_cpu` only.
Gate: unit tests on synthetic token streams for the formulas; one tiny real run for bpc/noise band.

### S5 · Data kinds and rules (sonnet) — `studio_rules.py`, `studio_data_kinds.py`, `studio_ia/pages/your_data.py`, `studio_ia/pages/make_data.py`, `tests/test_rules.py`
Rules: copy, reverse, sort, count, small_addition, max_length, ends_cleanly; `check(rule, prompt, answer) -> bool`
and `generate(rule, rng, n)`; problems stored as examples {prompt, rule, params}. Data kinds: classify workspace
datasets into text / conversations / problems (text rows → text; messages → conversations; rows with a rule →
problems); conversations never enter pretraining. Near-duplicate detection (MinHash or shingle Jaccard ≥ 0.8)
reported as a count. Pages per the YourData and MakeData boards (three tabs; generate problems; drafts routed
to review; local-model rewrite and prose stay "for review only").
Gate: rule tests (positive/negative per rule), classification tests, pages render on FixtureServices.

### S6 · Compare runs and Quick test (sonnet, wave 2) — `studio_experiments.py` (extend), `studio_ia/pages/compare.py`, `studio_ia/pages/quicktest.py`
Compare: ExperimentSpec gains seeds, lr_retune (3-point sweep per variant first), hold-equal modes (existing
`match()`), paired data order, same schedule scaled to length; results = final bpc mean + spread per variant,
pieces/s, peak memory; guess chip before queueing; runs are queued as one `compare` job. Quick test: two short
runs (current design vs draft), same tiny dataset and order, same wall-clock, optional ¼ width with μP; the
four sanity checks; verdict sentence repeats the guess; weights discarded.
Gate: tests on the matcher and the spread/verdict logic with synthetic curves; pages render.

### S7 · Look-inside pages (sonnet, wave 2) — `studio_ia/pages/trace.py, head_zoom.py, value_map.py, loss.py, head_lab.py`, `studio_inspect.py` additions
Inspect additions (coordinate: the engine owner adds hook `attn.logits` = post-scale, post-bias, pre-mask):
`trace(checkpoint, text, pos)` -> per stage ‖x‖, ‖Δx‖/‖x‖, cosine, logit-lens top; `prepared_text(tokenizer)` =
random pieces repeated twice; `head_tests(pattern, ids)` = previous-word share, first-word share ("parks on the
first word"), induction score on the repeated half; `rank_heads(checkpoint, docs)` = mean-ablate each head alone
on a fixed held-out batch, return error rise; mean ablation must be causal (mean over positions ≤ t or over a
reference batch, say which). Inside a head derives steps 2–5 client-side from raw scores and asserts the derived
softmax matches `attn.pattern` to 1e-4.
Gate: tests on a fixture checkpoint; derived-softmax assertion; pages render.

### S8 · Post-training rounds and gate (sonnet, wave 2, after the engine port) — `studio_rl.py`, `studio_gate.py`, `studio_ia/pages/post_training.py`, `studio_ia/pages/chat.py`
R1 SFT = `studio_model.train` with objective "assistant" on conversations. R2 = group sampling (4–8) via
`LocalModel.generate`, rule reward (+ optional gated rubric), advantage = score − group mean (÷ std optional),
policy-gradient on sequence log-probs from `CompiledModel.sequence_logps(ids, targets, tape=True)`; all-pass /
all-fail groups skipped. R3 = on-policy distillation: student samples, teacher per-token log-probs (teacher =
previous best checkpoint, or the same model with the correction in context), per-token reverse KL, heat map data.
R4 = existing `studio_rlhf.dpo_train` on human pairs only. Gate: `studio_gate.passes(before, after)` = held-back
bpc did not rise AND checked-problem pass rate did not fall; judge k of n shown, never gates. Chat page per board:
versions shelf, scorecard, compare two versions, verdict sentence only for same-length-same-schedule runs,
"Teach a correction" → round 3 item, picking a reply → round 4 pair.
Gate: R2/R3 loss finite-difference checks on a tiny model; gate tests; pages render.

### S9 · Copy module (haiku) — `studio_ia/copy.py`, `tests/test_ia_copy.py`
Every visible string from the 16 boards, grouped by page, as Python constants (plain words, glosses, kickers,
how-it-works paragraphs, verdict sentences, state tray texts). Keep `[N]`, `[k] of [n]` placeholders verbatim.
Gate: test that each page's h1 and lead line are present and that no string contains a bare percentage
template for small samples.

### S10 · Fixture workspace (haiku) — `tests/fixtures/ia_workspace.py`, `tests/test_ia_fixture.py`
`build(directory) -> Workspace` with: 2 versions (version 1 = default spec trained ~2 s on the committed sample,
version 2 = version 1 with 2 more blocks, warm-started if `studio_warmstart` exists else trained 1 s), 20
conversations, 10 checkable problems, 5 held-back documents; writes versions.json and an empty queue.json in the
formats S2/S3 define (coordinate by reading their modules; if absent, write the schema from this file).
Gate: builds in < 20 s on CPU, re-entrant, used by every `gui_ia_*` test.

### S11 · 3D page (later) — hosts the existing Panda3D viewer with the three modes; not in wave 1 or 2.

### Cut-over gate
All `gui_ia_*` tests and `tests/test_*.py` green; `app.py` starts the new shell; the old pages are deleted
in a final clean-up scope.
