# Build specs and design rules (from the canvas notes)

## spec3D

BUILD SPEC · Look inside · 3D — panda3d, studio_brain3d.py (confirm)
Three zoom levels (Window · Head · Weights) and three modes (Build · Train · Ride a position).
• Window: x = position lane, depth = layers, gate per block, beams to the left only (the causal mask), towers for MLPs.
• Train leads with two facts: one forward pass predicts every position at once and the mask keeps each from seeing its answer; one backward wave starts from all positions’ errors together.
• SSM gates: no beams; a state runs across the lanes left to right.
• Weights level teaches that single weights rarely mean anything: the action scales a whole head’s output, not one weight.
• Build stages edits in the same versioned spec as the designer. Values are placeholders; the mask shape is exact.

## specBuilder

BUILD SPEC · 1 Network designer — one interactive builder; Look-inside views, Compare runs and Quick test link from its footer
Plain words on screen; technical names only as small grey glosses. The picture is the editor: block tower with a stepper, look-back heads as dots on every block, thinking room as a bar, the prompt cut into pieces at the bottom.
States, always visible in the tray under the picture:
• Not saved (amber): edits exist only on this page. Discard, Quick test first, or Save as version N+1.
• Saved (coral): “needs catch-up training” by default. Warm start: added blocks start with zeroed output projections and widened MLPs with zeroed new rows, so the new version starts exactly at the old one’s error (function-preserving). Head-count change at fixed width reuses the weights but is not function-preserving: error jumps, then recovers. Removing parts drops weights: error rises, then recovers. Only a word-list change forces “needs training from scratch”. “Start from scratch instead” stays available.
• Queued / training: shown in the tray and the machine strip. Chat keeps the old version until the new one beats it on held-back text (outside the noise band), or the user picks it.
Chart: bits per character, with three reference lines: uniform guess, counting word frequencies (unigram entropy), best so far. The new version’s dashed line continues from the old one’s end point.
Costs: relative bars from exact formulas: size P, time per step C = 6·(L·(4d²+2d·ff) + V·d) + 6·L·T·d, memory while training = 16 bytes/param (fp32 weights, gradients, Adam m and v) plus activations (labelled). “Last time this changed, error moved by [N]” comes from the version history, never a guess.
Palette: neutral greys, violet = selected/running, amber = not saved, coral = needs training, blue = live, pink = attention, sage = word pieces.

## specChat

BUILD SPEC · 4 Chat — use any trained version; compare two
Header: which version and checkpoint answers, and whether Chat is following the best checkpoint (default) or one the user picked. Switching is live.
Shelf rows: version, change in plain words, verdict chip (better / within the spread / worse / catching up). A version still training answers from its best checkpoint so far. Scorecard one level down: held-back error (bits/char), answers prompts (yes / not yet: has passed Post-training round 1), checked problems passed (k of n), judge preferred it (k of n, context only), size and speed.
Compare: same message to both; picking one saves a rated pair for round 4. Verdict sentence: “To reach the same error, X needs N% of the training Y needed”, only when both runs used the same length and schedule; otherwise “no verdict yet” with the reason. Chip “better/worse” only when the saving holds across the size ladder and exceeds the spread across seeds.
Teach a correction: saved for round 3 (teacher sees the correction, student doesn’t); a whole new answer goes to round 1. Mock chips are example verdicts; measured values are [N].

## specContract

DATA CONTRACT · what the engine must expose for the Look-inside views
Hook names below are declared by modules; views request them by name through inspect.forward(ckpt, text, hooks), in-process. Attention hooks exist only when the block uses an attention mixer.
All numbers on these artboards are illustrative placeholders; the real ones come from a trained checkpoint through these calls.
• tokenize(text) → ids, token strings (shown with their leading spaces).
• forward(ids, hooks) returning per layer ℓ: resid_pre/mid/post [T, d]; attn pattern [H, T, T] (post-softmax); attn scores [H, T, T] (pre-mask, pre-scale); q, k, v [H, T, d_head]; head outputs before W_O [H, T, d_head]; final logits [T, V].
• logit_lens(resid) = unembed(ln_f(resid)) → top-k per layer.
• ablate(layer, head, mode = mean | zero) → loss on a fixed held-out batch. Mean = the head's mean output over the batch.
• backward(ids, target_pos) → grad norm per parameter group.
• stats over held-out batch, per head: mean entropy (nats), mean attention distance (tokens).
Batch: [N] sequences fixed per checkpoint so numbers are comparable. Cache per (checkpoint, prompt).
Checkpoints: every view takes a checkpoint id, so a scrubber can be added later (watch induction heads form).

## specExperiments

BUILD SPEC · Compare runs
ExperimentSpec = base ModelSpec + variants (JSON Merge Patches) + hold-equal mode + seeds + LR retune flag + eval set.
• Hold equal (all approximate, the column says by how much): parameters (match re-solves width to ~±2%), compute, text seen, wall-clock.
• Seeds vary initial weights only; the data order is identical across all runs, so differences are paired. Same schedule for all, scaled to length.
• Retune learning rate per variant (default on): a short 3-point sweep per variant first; when off, each row says “learning rate not retuned”.
• Result: final held-back bits/char with the spread across seeds; a difference inside the spread is grey. Prediction chip before queueing. Button: “Queue 12 runs (+ 9 short sweeps)”.
• Each run also reports pieces/s and peak memory. Any row reopens in the builder.

## specHeadLab

BUILD SPEC · Look inside · Head lab — an experiment bench, not a table
Surface: the version’s heads as tiles (blocks × heads). Tiles and the large view run on prepared text: a random sequence repeated twice, so previous-word, first-word and copying patterns separate cleanly. One primary action: Rank heads by importance (each head mean-ablated alone; tiles show the error rise).
Selected head: prediction chip first (“how much will the error rise: a lot / a little / not at all”), then Switch it off, Test its attention pattern. Tests: previous-word share; “parks on the first word” labelled as often meaning the head is idle (attention sink); induction score on the repeated half.
What changed: same prompt with all heads vs the switched-off set, held-back error before → after (bits/char), and the guess. Wording rule: never a head’s “job”; patterns say where it looks, ablation says how much the model relies on it, neither says why; heads can cover for each other.

## specHeadZoom

BUILD SPEC · Look inside · Inside a head
Same orientation as the reference: rows = keys k_i, columns = queries q_j. Softmax runs DOWN each column; mask is i > j (lower-left). Use ≤ 12 tokens (truncate from the left, keep the query).
Five steps, one button per step: 1 raw q·k, 2 ÷√d_head, 3 mask (−∞), 4 softmax, 5 weighted sum (dim all columns but the query's). Colors: amber positive, blue negative, black masked, white = weight.
Clicking a cell shows its value at every stage. The mock computes steps 2–5 from step 1, so the numbers agree; keep that: fetch raw scores and q/k, derive the rest client-side, and assert the derived softmax matches the model's pattern to 1e-4 (catches hook bugs).
Step 5 panel lists the top-5 weighted value contributors and links to Value map.

## specLoss

BUILD SPEC · Look inside · Loss
The one place that shows nats. One prediction position: top-5 + “all others”, p, one-hot target, ∂loss/∂logit = p − y. Step 0 is exact: p ≈ 1/V, loss ≈ ln V. Conversion card: bits per character = nats per piece ÷ (characters per piece × ln 2); over a text, total nats ÷ (total characters × ln 2). Characters the word list can’t encode are scored at the uniform rate and counted.

## specPost

BUILD SPEC · 3 Post-training — four rounds, fixed order, one card, gate and visual each
Stage line: four clickable rounds with status. Expectation chip always shown: at this size the gain from each round is small; the shape of the pipeline is what you are learning.
R1 Practice answers (SFT). A few thousand conversations, corrections, practice problems with answers; loss on answer tokens only; no reference copy. Verdict: “answers now look like answers”.
R2 Practice with checks (RL with rule-checked rewards, GRPO-style). Per prompt 4–8 samples (mock shows 6); a rule scores each (copy, reverse, sort, count, small addition, length limit, ends cleanly). Advantage = score − group mean (optionally ÷ group std). Visual: the group as a column with scores, the mean line, an up/down push per answer. All-pass or all-fail groups grey out: nothing to learn. Optional teacher rubric adds a smaller second reward, gated: failing a rule gives the minimum. Verdict: “answers that passed the check became more likely than ones that failed”.
R3 Learn from a teacher (on-policy distillation). Student samples; teacher log-probs per token; per-token reverse KL. Visual: heat map, “how surprised the teacher was by this word”. Default teacher: previous best version of this model (same word list). Chat’s “Teach a correction” lands here: teacher = same model with the correction in its context, student = without. The local Ollama model can rewrite and judge only. Verdict: “its answers moved toward the teacher’s at [k] of [n] words”.
R4 Your votes (DPO). Human-rated pairs only, optional, last. Frozen-copy diagram, leash = β; copy says the gap widens, never “more likely” alone. Verdict: “widened the gap on [k] of [n] held-back pairs”.
One gate for every switch: held-back bits per character did not rise AND checked-problem pass rate did not fall. Judge win rate is shown as k of n and never gates alone. Failed gate: checkpoint kept for comparison, Chat unchanged. The MOCK switch is for review only.

## specPretraining

BUILD SPEC · 2 Pretraining — Data → Batches → Train for the current design version
Data: plain text only (library sets, downloads, your files, generated prose as drafts). Conversations go to Post-training round 1, problems with a checkable answer to round 2. “Enough text?”: tokens per parameter for the planned run, shown against the ~20 tokens/parameter rule of thumb, labelled as a rule of thumb.
Batches: tokenize with the version’s word list, end-of-text between documents, rows of context length, rows per step. Held-back text: split by document, exact duplicates removed first, never trained on.
Train: “Continue from version N” is the default (warm start, see Network designer). “Start from scratch” only when asked, or forced by a word-list change. Learning rate: auto (μP transfer from the last tuned version); schedule: warm-up, steady, cool-down (WSD). Button: Add to the queue.

## specPrinciples

DESIGN RULES · three depths (apply to every page)
1 Surface: one question, one picture, one primary action, plain words. At most ~3 controls per card and one number per card. Every number is exact or shown as [N]; illustrative shapes carry a visible tag. The state is always visible: not saved (amber) · saved, needs training (coral) · training (violet) · live (blue).
2 One click in: the 'why and how' attached to the thing itself — explanation, exact definitions in small grey text, the next layer of options as named choices. This is where learning lives. Name what is inside ('How this is measured', 'Who votes'), never a bare 'Advanced'.
3 Engineer depth: Compare runs, Look-inside views, per-part module choices, the build specs — every knob, raw values, hooks. Reached by name from depth 2.
Every control says what it affects: live now, needs saving, or needs a training run.
Technical terms appear only as small grey glosses next to the plain word (look-back heads · attention).
Sophistication goes under the hood: the engine computes exact costs, efficiency gain, ladders and noise checks; the surface shows one sentence and one verdict.
Science rules: one unit, bits per character (nats only on the Loss view). Every automatic switch is gated on a measured check. Small samples show k of n, never a bare percentage. A guess before every run. Seeds are reported as a spread.

## specSandbox

BUILD SPEC · 1 Network designer · Quick test — a throwaway micro run of a design idea
Compares the builder’s staged change against the current version. Both runs: same tiny dataset in the same data order, same wall-clock budget (fair by time). Options: tiny dataset, time each, size (full or ¼ width; learning rate carried with μP; labelled a hint).
Prediction chip before running (lower / same / higher); the result sentence repeats the guess. Plot with uniform-guess and counting-word-frequencies lines; guard sentence: a design that never gets below counting word frequencies hasn’t learned to use context.
Sanity checks: starts near the uniform guess; numbers finite; memorizes one batch; error falls. Actions: Save idea as next version, Run a real comparison, Discard. Weights always discarded. The MOCK switch is for review only.

## specTechSpec

FOR THE AGENTS
The Engineering spec board was removed; the BUILD SPEC notes next to each board and the data contract below are the requirements for the NumPy engine and the Tkinter pages.
Build order:
0. Read studio_model.py, studio_network.py, studio_arch.py, train.py and studio_core.py; confirm file roles before writing code.
1. Slots / registry / spec / estimate / validate + module contract tests (gradient check first).
2. Move today’s GPT into modules with no behaviour change: same seed, same data → same loss curve to 1e-6 over 100 steps. Default spec gives 16,090,880 params.
3. Job queue + machine strip (one runner, resumable, checkpoints with cool-down branches). Warm start transforms (add block, widen MLP, re-split heads).
4. Builder page; Quick test on the same engine.
5. Hooks + inspect.*; Look-inside views.
6. Compare runs + matcher + LR sweep.
7. Post-training rounds 1–4 and the shared gate.
Stack: Python + NumPy (optional GPU backend via xp), Tkinter. No PyTorch.

## specTrace

BUILD SPEC · Look inside · Trace a position
One row per stage: ‖x‖, relative change ‖Δx‖/‖x‖, cosine between the stream before and after the stage, logit-lens top guess. Size alone grows with depth and says little; the relative change and the cosine say how much a stage rewrote. With rotating positions the embedding row is the token vector only. Early-layer guesses flagged as approximate. All values from inspect.forward + logit_lens.

## specTraining

BUILD SPEC · Pretraining run — readable at a glance
Header: version and what changed; the warm start sentence. Progress: step k of n, pass number, schedule phase, learning rate auto, time left. Buttons: Pause; Stop after a short cool-down.
Plot, bits per character only: per-step (faint), smoothed (bold), held-back dots, uniform-guess and counting-word-frequencies lines above an axis break, the previous version’s best with its noise band (spread of the held-back estimate). The run starts at that line, may bump while the learning rate warms up again, and the unrun remainder is shaded. Under the plot: the learning-rate lane with warm-up / steady / cool-down bands; note that error usually drops most in the cool-down.
Status, computed: Learning / Flat / Memorising (held-back rising while training falls; likelier after a few passes over the same text; fix = more text or stop at best). Switch box: Chat moves only when held-back error is below the noise band at a checkpoint.
“What it writes now”: greedy, same prompt every N steps; step 0 is identical to the previous version.

## specValueMap

BUILD SPEC · Look inside · Value map
W_V for one head (d_model → d_head) and its row block of W_O (d_head → d_model) as two funnels around a d_head-wide bottleneck. Teaching claim: each head writes into a subspace of at most d_head dimensions, a set of directions, not a block of coordinates. Heads switch (1/2/4/8) re-splits d_model; V+O params stay 2·d². Later: real singular values of W_V^h·W_O^h.

## specQueue

BUILD SPEC · 0 This computer — one machine, one queue, visible on every page
Every page’s sidebar ends with the machine strip: what is training now, a progress bar, time left and how many jobs wait, which checkpoint Chat uses (and that it is the best so far), and a link to this page. Every Train / Run / Queue button in the app appends a job here; nothing starts a second run in parallel.
• Now: the running job, step k of n, schedule phase, learning rate (auto). Pause (resumes exactly) and Make a checkpoint now (branches a short cool-down on a copy; the main run continues).
• Up next: ordered list, Move up / Remove. Jobs: catch-up or from-scratch pretraining, Quick test pairs, Compare-runs batches, Post-training rounds.
• Automatic job: after a night of training on new personal text, the queue appends “Restore how it answers” = Post-training round 3 with last night’s best version as teacher. Copy: new text can erode how it answers; a short lesson from the previous version restores it without losing the new text.
• Plan: hours (any / nights / set), memory it may use (half / most / all), only when idle, pause on battery, keep awake, Chat follows the best checkpoint (default on).
• Answering in Chat never stops training; it shares the machine, so training slows while it answers.
