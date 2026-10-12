# Network designer rebuild — implementation plan

Prepared from issue #9 and its corrections, draft PR #10 at `b444fe3`, the local Groundwork Studio IA HTML export, and the current repository.

Sources: https://github.com/KingMoonk/TRAINER/issues/9 · https://github.com/KingMoonk/TRAINER/pull/10

## Starting state

- The working tree is clean on `master`. The engine draft is on local and remote `network-designer`; inspect or implement against that branch, not the old engine on master.
- The draft supplies the registry, slots, ModelSpec, validator, estimator, compiler, 18 modules, frozen legacy engine and parity tests. It does not supply module contract tests or the inspection/scene services.
- The committed parity test already contains the learnable counting sequence proposed by the issue correction. Run it before changing it; the reported random-input failure is stale relative to the commit.
- No draft tests were executed during this planning pass. Importability and handoff statements are not evidence that the engine is correct.
- The HTML export contains 19 artboards, including all designer views and the full module-system technical spec. Its nested bundles were decoded for review. `canvas.json` BUILD SPEC notes are not present in the exported material reviewed here; recover them before final visual acceptance. Values and narratives in mockups are illustrative, not runtime data.

## Constraints and boundaries

Keep the native Tkinter app, handwritten NumPy/CuPy forward and backward passes, and existing dependencies. Preserve the public config API and existing saved data. One module per file, discovered through registry metadata. UI forms, compatibility messages, estimates and view availability must derive from that metadata.

The compiler owns the residual block template and tapes. Modules own local weights, gradients, caches and hooks. Engine services never import Tkinter or Panda3D. Training and inspection run off the Tk main thread; events reach the UI through its queue. Keep Panda3D in its existing separate process: each process owns its own UI/event loop, with communication through `studio_vizbridge.py`.

Treat these as separate concepts: architecture training objective (`ce`, `ce_ls`, `ce_z`) and the existing dataset target mask (`assistant` versus all tokens). Preserve both in run configuration and resume checks.

## Milestones and gates

### 0. Prove the engine replacement

Work on `network-designer` without merging the draft. Run `.venv/bin/python -m unittest tests.test_parity`, then `.venv/bin/python -m unittest discover -s tests`, and `.venv/bin/ruff check .`. Record skips and distinguish unavailable GPU/display tests from passes.

Investigate failures in the compiler or adapters before adding modules. Cover default and hybrid blocks, exact initial parameter arrays and RNG draw order, 100-step loss parity below 1e-6, dropout, multiple outstanding tapes, shared embedding/head gradient accumulation and telemetry. Add old-checkpoint inference/resume coverage against fixtures produced by the frozen engine, including optimizer and RNG restoration. Verify RLHF, passive training and curriculum callers because they also depend on `studio_model.py`.

Exit: parity, full suite and lint pass; checkpoint compatibility is demonstrated. GPU parity is executed where CUDA is available or explicitly outstanding.

### 1. Make module correctness executable

Add `tests/test_module_contract.py`, parametrized by registry entries with slot-specific fixtures. First check fp64 central finite differences for activation gradients and every trainable parameter, relative error below 1e-6 with a defined near-zero handling rule. Then check shapes including T=1, causality, deterministic seeds, exact parameter estimates, gradient reach, config validation/JSON round trips and declared hook counts/shapes. Check gradients accumulate rather than overwrite.

Use separate contracts for tokenizer, objective, optimizer, schedule, initialization and precision; these are not all differentiable layers. Objectives require loss-gradient checks; optimizers require hand-computed update/state references. Exempt constant/absent branches from non-zero gradient requirements. Route MoE fixtures through every expert and avoid router discontinuities during finite differences. Conditional GPU checks compare outputs and gradients within 1e-5.

Audit estimator conventions: total versus active parameters, tied weights counted once, skipped branches omit their norms, initialization and dtype, FLOP approximation and cache applicability. Gate the issue's exact counts at d=256, L=4, H=4, V=50,257, T=256 using explicit `vocab=50257`; real tokenizer vocabulary includes seven specials.

Exit: all existing modules satisfy the contract; baseline is 16,090,880 parameters, with documented compute/cache units and assumptions.

### 2. Expand the registry in dependency order

1. Sinusoidal/RoPE/ALiBi positions, RMSNorm, factorized embedding, untied head, SwiGLU, GQA (including MQA configuration).
2. Linear attention, causal convolution, GRU, LSTM, selective SSM, then MoE.
3. Label-smoothed/z-loss objectives, SGD/Lion/Muon, constant/WSD schedules, muP initialization.
4. GPT-2 and custom BPE tokenizers; GPT-2 vocabulary downloads on demand into `data/`, with explicit offline/error behavior and no added dependency.

Each module ships only after its contract passes. Define algorithm/configuration details and numerical references before implementation, especially linear attention normalization, SSM recurrence, MoE routing/auxiliary losses, Muon matrix eligibility, muP scaling and custom BPE training/persistence. Hook emission must return patched values and preserve backend/dtype. Include glyph metadata and unique mixer/channel parameter names.

Verify all issue count deltas and per-block mixer counts as the relevant modules land. Reject incompatible position/mixer and head/embedding combinations with actionable reasons rather than silently substituting modules.

### 3. Persist and propagate ModelSpec

Update `studio_archdb.py`, presets in `studio_arch.py`, and curriculum engine/page adapters. Introduce an idempotent, transactional migration of legacy architecture JSON while preserving revision relationships, weight references, hashes and backup/restore behavior. Test on copied temporary databases; do not migrate the user's live database during development.

Write each run's canonical `spec.json` alongside existing configuration and checkpoints. Persist module configs, tokenizer assets, seeds, optimizer state and schema versions. Exercise save/load, export/import, presets, hybrid overrides, weight transplant, resume, curriculum stages and RLHF. Resume must compare all behavior-relevant spec/training settings, with clear errors for unsupported state.

### 4. Build inspection and bounded summaries

Implement `studio_inspect.py` and `studio_summary.py` independently of UI: checkpoint forward hooks, selected prediction backward, logit lens, head statistics, zero/mean ablation, detectors, binned attention and downsampled run paths. Use immutable probe inputs; inspection must not change training RNG, gradients, caches or weights. Define result shapes, units, normalization axes and selection coordinates before building views.

The engine's attention layout is query-by-key; the artboards show keys as rows and queries as columns. Transpose only in presentation, consistently across 2D/3D. Validate causal masking, probability sums, ablation losses, tied/untied logit lens and actual configured objective gradients. Label detector findings as measured heuristics, without asserting mockup explanations as facts.

Model-level summaries return [H,K,K] and stream norms. Do not claim memory bounded in T merely by binning an already dense [T,T] hook: add a chunked summary/probe path if necessary, and verify against dense results at small T. Bound Window/Head to 32 tokens, Weights to 16×16, and Run paths to 2,000 points. Keep the dense trainer's practical context limits visible.

### 5. Experiments and replay data

Add `studio_experiments.py`: persistent ExperimentSpec, common dataset/split/tokenizer handling, seed pairing, worker execution, cancellation, queue events and aggregate results. Match parameters/compute through explicit dimension adjustments within ±2%, reporting every adjustment and impossible matches. Token and wall-clock budgets control runtime stopping; wall-clock fairness requires measured timings rather than a static estimate. Store actual achieved budgets, throughput and available peak-memory measurements.

Implement `studio_snapshot.py` and integrate with training at configurable intervals: fixed probe hooks, gradient norms, actual update ratios, evaluation loss, detector markers and versioned numeric `.npz` files under `runs/<id>/viz/`. Distinguish unavailable evaluation loss from training loss. Replaying snapshots must require no training. Exact one-step mode must use one real optimizer step and recorded forward/backward values, with a defined pause/step state machine; interpolation is only animation between recorded snapshots.

Exit: deterministic CPU experiments/replay, budget matching tests, cancellation/error delivery and exact one-step comparisons pass. Seed spread is displayed descriptively; “within noise” is not presented as a statistical significance test.

### 6. Build the desktop views

Extend `studio_widgets.py` with schema-generated forms and validated edits. Convert `studio_network.py` into the host for Design, Sandbox, Variants, Experiments, Trace a token, Inside a head, Value map, Loss, Head lab, 3D and retained Versions. Use one `studio_net_*.py` file per view and a shared canonical spec/selection owned by the host.

Start with Design/Sandbox/Variants and `studio_circuit.py`'s vertical residual stream. Debounce validation/estimates at 150ms. Next connect Experiments and deep views to tested services. Disable unsupported views with reasons tied to the selected block's hooks; model-wide availability alone is insufficient for hybrids. Carry checkpoint/layer/token/head selection through navigation. Render computed values, honest empty/loading/error states and actual formulas for the chosen module/objective.

Exit: Tk smoke tests cover edits, undo, incompatible combinations, hybrid selection, view navigation, saved architectures and responsive training; inspect against artboards and recovered BUILD SPEC notes.

### 7. Scene model and 3D renderer

Define `studio_scene.py` before renderer changes: renderer-independent geometry/data for Run, Model, Window, Head and Weights, each supporting Build/Train/Ride a token where meaningful. Preserve selection across zoom thresholds, cap geometry, support slot defaults for missing glyphs, and rebuild only the affected gate on a module swap. Build mode changes the same canonical spec through bridge messages, with validation on the host.

Rewrite `studio_brain3d.py` to render this scene through the existing process bridge. Preserve ordered control messages and coalesce replaceable snapshots without unbounded queue growth. Add replay/scrubbing, numbers, reduced motion, billboard labels, keyboard controls and equivalent 2D interactions, including weight inspection/editing and recomputed forward results.

Exit: deterministic scene/replay tests; one gate per block; bounded lanes/beams independent of full context; exact recorded slow-motion values; gate-local updates; process disconnect/reconnect behavior. Measure the 60fps target at 32 lanes × 8 blocks on available integrated graphics rather than claiming it from unit tests.

### 8. Release readiness

Run the full suite and lint after integration, a fresh-session design/spec review per milestone, and live CPU smoke flow: build → train → pause/step → inspect → ablate → compare seeds → replay → save/reopen/resume. Run GPU checks on suitable hardware. Update README and PyInstaller hidden imports for registry discovery, then smoke-test packaged builds. Keep PR #10 draft until its gates and any agreed follow-up scope are complete; do not merge or publish during preparation.

## First implementation increment

Start with milestones 0–1 only: establish the PR branch baseline, resolve demonstrated regressions, prove legacy checkpoint resume, and land the registry-driven contract harness plus default estimator gate. This provides a reviewable engine foundation before the module catalogue and UI expand.

## Decisions to retain or resolve

Retain the issue's small byte-tokenizer defaults, separate GPT-2-style preset, seven special tokens, Versions tab, 0=auto size fields, norm placement in config and legacy parameter prefixes. Report these design deviations in the implementation PR.

Resolve from design notes or explicit technical decisions: precise new-module algorithms, custom BPE semantics, snapshot schema/storage limits, wall-clock matching policy, detector definitions, and the bounded attention-summary path. Recover the original canvas notes for visual acceptance; the provided HTML is sufficient to begin engine correctness work.
