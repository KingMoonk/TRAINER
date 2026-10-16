"""Post-training rounds 2 and 3, on-policy and with torch gradients.

Round 2, practice with checks (GRPO-style): per prompt the model writes a group of answers, a rule scores each,
advantage = score - group mean (optionally / group std), loss = -mean over answers of advantage * (mean token log-prob).
A group whose answers all score the same has advantage 0 everywhere and teaches nothing; the round counts those
("groups that taught something"), and a quick pass@k check before it skips the round when the model solves nothing.
Round 3, learn from a teacher (on-policy distillation): the student writes an answer, a frozen teacher scores each
of its tokens, per-token reverse KL = log p_student - log p_teacher on the student's own samples. The teacher is either
another trained version that shares the student's word list and context, or the same model shown a hint
(context distillation, which only works if the model can follow a hint).

Both are on-policy (fresh samples every step, one gradient step per batch), so no ratio or clipping is needed.
ponytail: a sampled answer is decoded to text and re-encoded for scoring, so the scored tokens can differ from the
sampled ones on odd spellings; an EOS is always scored. Keep token ids from generate() if that ever matters.
"""
from __future__ import annotations

import math
import random
import threading
from pathlib import Path

import numpy as np
import torch

import studio_rules
from studio_backend import empty_cache
from studio_core import atomic_json, now
from studio_model import AdamW, LocalModel, ScratchTransformer, Tokenizer, batch_arrays, load_checkpoint, train
from studio_rlhf import _encode_pair, _save, token_logps

RUBRIC_WEIGHT = .3   # the teacher's rubric adds at most this much; a failed rule is always the minimum, 0
MEASURE_PROBLEMS = 200  # held-back problems answered per before/after measurement
PROBE_PROBLEMS = 8      # problems answered before round 2 to see whether it can learn anything
PROBE_ANSWERS = 4       # answers sampled per probe problem (the k of pass@k)


def _messages(item):
    """A prompt as chat messages: a string, {"prompt"}, {"messages"} or {"user"}."""
    if isinstance(item, str):
        return [{"role": "user", "content": item}]
    if item.get("messages"):
        return list(item["messages"])
    return [{"role": "user", "content": item.get("prompt") or item.get("user", "")}]


def _prompt_text(messages):
    return next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")


def _setup(checkpoint, device, data=None):
    local = LocalModel(checkpoint, device=device, data=data)
    return local, AdamW(local.model, 0.)


def _begin(output_dir, local, checkpoint, stage):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    cfg = dict(local.config, dropout=0)
    return directory, cfg, {"stage": stage, "base_checkpoint": str(checkpoint), "data_hash": ""}


def _finish(directory, local, optimizer, cfg, step, base_step, best, history, extra, record, stop):
    _save(directory / "last.npz", local.model, optimizer, cfg, local.tokenizer, base_step + step, best, history, extra)
    if not (directory / "best.npz").exists():
        _save(directory / "best.npz", local.model, optimizer, cfg, local.tokenizer, base_step + step, best, history, extra)
    record.update(status="stopped" if stop.is_set() else "completed", step=step, finishedAt=now(), checkpoint=str(directory / "last.npz"),
                  directory=str(directory))
    atomic_json(directory / "run.json", record)
    return record


# --------------------------------------------------------------------------- round 2

def advantages(rewards, divide_by_std=True):
    """reward - group mean (divided by the group std when that is > 0). All-equal groups give all zeros."""
    r = np.asarray(rewards, dtype=np.float64)
    a = r - r.mean()
    std = r.std()
    return a / std if divide_by_std and std > 1e-9 else a


def has_spread(rewards):
    """True when a group holds both a better and a worse answer. All-equal groups (all pass or all fail) give no gradient."""
    return max(rewards) - min(rewards) > 1e-9


def group_spread(groups):
    """(taught, total): how many of the reward groups had any spread, and how many there were."""
    groups = list(groups)
    return sum(1 for g in groups if has_spread(g)), len(groups)


def length_normalised(logp, valid, weights):
    """-mean over answers of weight * (that answer's mean token log-prob).

    Each answer's summed log-prob is divided by its own token count (the GRPO form, not the constant divisor of Dr. GRPO),
    so a long answer does not outweigh a short one just by having more tokens: equal per-token log-probs give the same loss
    at any length. logp is [rows, tokens], zero where valid is False."""
    counts = valid.sum(dim=1).clamp(min=1).to(logp.dtype)
    return -(weights * logp.sum(dim=1) / counts).sum() / len(weights)


def reinforce_loss(policy, rows, adv, backward=False):
    """-mean(advantage * mean token log-prob) over encoded (x, y) rows (see length_normalised); accumulates gradients when backward."""
    x, y = batch_arrays(rows)
    logp, valid, cache = token_logps(policy, x, y, tape=backward)
    weights = torch.as_tensor(np.asarray(adv), dtype=logp.dtype, device=logp.device)
    loss = length_normalised(logp, valid, weights)
    if backward:
        policy.backward_loss(loss, cache)
    return float(loss.detach())


def pass_at_k(checkpoint, problems, count=PROBE_PROBLEMS, k=PROBE_ANSWERS, temperature=.8, seed=7):
    """Sample k answers for `count` checkable problems: {"problems", "k", "solved" (problems with any passing answer),
    "passed" (passing answers), "answers"}. With solved == 0 every group in round 2 would be all-fail and teach nothing."""
    problems = [p for p in problems if p.get("rule") in studio_rules.CHECKABLE]
    problems = random.Random(seed).sample(problems, min(count, len(problems)))
    local = LocalModel(checkpoint, device="cpu")
    local.rng = np.random.default_rng(seed)
    solved = passed = 0
    for problem in problems:
        messages = _messages(problem)
        prompt = _prompt_text(messages)
        budget = studio_rules.max_tokens(problem["rule"])
        ok = sum(bool(studio_rules.check(problem["rule"], prompt, answer, problem.get("params")))
                 for answer in local.generate_group(messages, k, max_tokens=budget, temperature=temperature))
        passed += ok
        solved += ok > 0
    return {"problems": len(problems), "k": k, "solved": solved, "passed": passed, "answers": len(problems) * k}


def practice_with_checks(checkpoint, problems, output_dir, group=6, steps=20, learning_rate=1e-4, prompts_per_step=4,
                         divide_by_std=True, rubric=None, max_tokens=None, temperature=.8, seed=7, device="auto",
                         emit=None, stop=None, pause=None):
    """Round 2. problems: [{"prompt", "rule", ...}]. rubric: optional callable (prompt, answer) -> 0..1, a smaller
    second reward that only counts for answers that pass the rule. Saves last.npz and best.npz (best mean reward)."""
    emit, stop, pause = emit or (lambda event: None), stop or threading.Event(), pause or threading.Event()
    problems = [p for p in problems if p.get("rule") in studio_rules.CHECKABLE]
    if not problems:
        raise ValueError("Round 2 needs problems with a rule the app can check.")
    local, optimizer = _setup(checkpoint, device)
    local.rng = np.random.default_rng(seed)
    policy, tokenizer = local.model, local.tokenizer
    directory, cfg, extra = _begin(output_dir, local, checkpoint, "rl_checks")
    base_step, rng = load_checkpoint(checkpoint)["step"], random.Random(seed)
    record = {"name": "practice with checks", "createdAt": now(), "status": "training", "stage": "rl_checks", "steps": steps,
              "group": group, "base_checkpoint": str(checkpoint)}
    atomic_json(directory / "run.json", record)
    history, best, means = [], -math.inf, []
    groups_total = taught_total = 0
    emit({"type": "info", "text": f"round 2 · {len(problems)} problems · groups of {group}"})
    step = 0
    while step < steps and not stop.is_set():
        while pause.is_set() and not stop.is_set():
            threading.Event().wait(.05)
        rows, adv_all, shown, step_rewards = [], [], [], []
        for problem in rng.sample(problems, min(prompts_per_step, len(problems))):
            messages = _messages(problem)
            prompt = _prompt_text(messages)
            budget = max_tokens or studio_rules.max_tokens(problem["rule"])  # sized by the rule unless the caller fixes it
            answers = local.generate_group(messages, group, max_tokens=budget, temperature=temperature)   # one prompt prefill for the group
            passed = [studio_rules.check(problem["rule"], prompt, a, problem.get("params")) for a in answers]
            rewards = [float(ok) + (RUBRIC_WEIGHT * float(rubric(prompt, a)) if rubric and ok else 0.) for a, ok in zip(answers, passed)]
            flat = not has_spread(rewards)   # all pass or all fail: nothing to learn
            step_rewards.append(rewards)
            adv = np.zeros(group) if flat else advantages(rewards, divide_by_std)
            if not flat:
                rows += [_encode_pair(tokenizer, messages, a, cfg["context"]) for a in answers]
                adv_all += list(adv)
            shown.append({"prompt": prompt, "rule": problem["rule"], "mean": float(np.mean(rewards)), "skipped": flat,
                          "answers": [{"text": a, "pass": ok, "score": r, "push": float(d)} for a, ok, r, d in zip(answers, passed, rewards, adv)]})
            means.append(float(np.mean(rewards)))
        loss = None
        if rows:
            policy.zero_grad()
            loss = reinforce_loss(policy, rows, adv_all, backward=True)
            rate = learning_rate * min(1., (step + 1) / max(1, min(10, steps // 10)))
            optimizer.update(policy, rate)
        step += 1
        mean = float(np.mean([g["mean"] for g in shown]))
        taught, groups = group_spread(step_rewards)
        taught_total, groups_total = taught_total + taught, groups_total + groups
        row = {"step": step, "loss": loss, "mean_reward": mean, "groups": groups, "taught": taught,
               "groups_total": groups_total, "taught_total": taught_total}
        history.append(row)
        for visual in shown:
            emit({"type": "group", **visual})
        emit({"type": "metric", **row, "total": steps})
        if mean > best:
            best = mean
            _save(directory / "best.npz", policy, optimizer, cfg, tokenizer, base_step + step, -best, history, extra)
    half = max(1, len(history) // 2)
    record.update(groups=groups_total, taught=taught_total, first_mean=float(np.mean([h["mean_reward"] for h in history[:half]])) if history else None,
                  last_mean=float(np.mean([h["mean_reward"] for h in history[half:] or history])) if history else None)
    return _finish(directory, local, optimizer, cfg, step, base_step, -best if math.isfinite(best) else None, history, extra, record, stop)


# --------------------------------------------------------------------------- round 3

class Teacher:
    """A frozen model that shares the student's word list. With `correction`, the same weights see it as a system
    message in front of the conversation (the student does not).

    It holds the student's device only while score() runs, unless `resident` is set (learn_from_teacher sets it when the job's
    memory share has room for both). Released: on a CUDA GPU its weights wait in host memory between scores; on an Apple GPU
    (unified memory, so a host copy would free nothing) they are dropped and read from the checkpoint again; on the CPU the
    model just stays."""

    def __init__(self, checkpoint, correction=None, device="auto"):
        data = load_checkpoint(checkpoint, weights_only=True)
        self.tokenizer = Tokenizer.restore(data["tokenizer"])
        self.config = dict(data["config"], device=device, dropout=0)
        self.context = data["config"]["context"]
        self.correction = correction
        self.path, self.weights, self.model = str(checkpoint), data["model"], None
        self.resident = False

    def build(self):
        """The model on its device: the one already built, or a new one from the weights held (or read from the checkpoint again)."""
        if self.model is None:
            if self.weights is None:
                self.weights = load_checkpoint(self.path, weights_only=True)["model"]
            self.model = ScratchTransformer(self.config, self.tokenizer.vocab_size)
            self.model.load_weights(self.weights)
        return self.model

    def release(self):
        """Off the device (a model on the CPU just stays)."""
        device = self.model.backend.device if self.model is not None else "cpu"
        if device != "cpu":
            self.model = None
            self.weights = self.weights if device == "cuda" else None
            self.resident = False
            empty_cache(device)

    def score(self, rows):
        """One 1-D CPU tensor per encoded row: the teacher's log-probs of that row's answer tokens."""
        model = self.build()
        with torch.no_grad():
            logps = [v.cpu() for v in _answer_logps(model, rows)]
        if not self.resident:
            self.release()
        return logps

    def messages(self, messages, correction=None):
        """The conversation as the teacher sees it: `correction` (or the teacher's own) goes first as a system message."""
        correction = correction or self.correction
        return [{"role": "system", "content": correction}, *messages] if correction else list(messages)


def check_teacher(student_tokenizer, student_context, teacher):
    """Raise a plain ValueError unless the teacher reads the same word list and the same reading window as the student:
    its per-token log-probs are only comparable to the student's token for token on the same ids and positions."""
    if teacher.tokenizer.serialize() != student_tokenizer.serialize():
        raise ValueError("This teacher uses a different word list from the model it teaches, so its scores would not line up. "
                         "Pick a version trained with the same word list.")
    if teacher.context != student_context:
        raise ValueError(f"This teacher reads {teacher.context} tokens at a time and the model reads {student_context}. "
                         "Pick a version with the same reading window.")


def privileged(checkpoint, correction, device="auto"):
    """Teacher = the same model with the correction placed in a system message in front."""
    return Teacher(checkpoint, correction, device)


def _answer_logps(model, rows):
    """List of 1-D tensors, one per row: log-probs of that row's answer tokens (autograd when the model has a tape)."""
    x, y = batch_arrays(rows)
    logp, valid, _ = token_logps(model, x, y)
    return [logp[i][valid[i]] for i in range(len(rows))]


def reverse_kl(student, teacher_logps, student_rows, backward=False):
    """Mean per-token reverse KL, sum_t (log p_student - log p_teacher), over the student's sampled tokens.

    teacher_logps: Teacher.score of the same answers, already computed, so the teacher is off the device by now.
    The gradient goes through the student only: loss = mean over tokens of stopgrad(d_t) * log p_student(t), the
    on-policy estimator whose expectation is the gradient of the KL (teacher log-probs are constants).
    Returns (mean_kl, per_row [(student_logps, teacher_logps) as numpy])."""
    xs, ys = batch_arrays(student_rows)
    s_logp, s_valid, cache = token_logps(student, xs, ys, tape=backward)
    s_rows = [s_logp[i][s_valid[i]] for i in range(len(student_rows))]
    d = [s.detach() - t.to(s.device, s.dtype) for s, t in zip(s_rows, teacher_logps)]
    total = max(1, sum(len(v) for v in d))
    if backward:
        loss = sum((dv * s).sum() for dv, s in zip(d, s_rows)) / total
        student.backward_loss(loss, cache)
    mean = float(sum(float(v.sum()) for v in d) / total)
    return mean, [(s.detach().cpu().numpy(), t.cpu().numpy()) for s, t in zip(s_rows, teacher_logps)]


def _decide_residency(teacher, student, tokenizer, batch, share, emit):
    """Once per run, before the first step: keep a teacher on a GPU in memory between steps when the student's training step plus
    the teacher's weights fit `share` of the device's memory, else let it leave after each score (a rebuild every step, which
    costs more than the scoring itself on an Apple GPU). A teacher on the CPU stays anyway, so there is nothing to decide."""
    import studio_memory
    model = teacher.build()   # the first score would build it too; its weights are what is measured
    device = model.backend.device
    if device == "cpu":
        return
    fixed, per_sample = studio_memory.training_bytes(student.config, tokenizer.vocab_size)
    same_memory = studio_memory.pool(student.model.backend.device) == studio_memory.pool(device)
    step, weights = (fixed + per_sample * batch) if same_memory else 0, studio_memory.weight_bytes(model)
    room = studio_memory.budget(device, share)
    teacher.resident = step + weights <= room
    sizes = f"a training step of {_mb(step)} plus the teacher's weights of {_mb(weights)}, against the {share:.0%} ({_mb(room)}) of memory the plan allows this job"
    emit({"type": "info", "text": f"Keeping the teacher in memory between steps: {sizes}." if teacher.resident else
          f"Not keeping the teacher in memory between steps, which makes each step slower: {sizes}."})


def _mb(nbytes):
    return f"{nbytes / 2 ** 20:,.0f} MB"


def learn_from_teacher(checkpoint, prompts, teacher, output_dir, steps=20, learning_rate=1e-4, prompts_per_step=4,
                       max_tokens=None, temperature=.8, probe=8, seed=7, device="auto", emit=None, stop=None, pause=None, memory_share=None):
    """Round 3. teacher: a checkpoint path (any trained version with the same word list and context; check_teacher
    raises otherwise) or privileged(checkpoint, correction), the same model shown a hint.

    Before training it samples `probe` answers and keeps them; the heat map shows the teacher's surprise at each word,
    and the verdict counts the words where the student's log-prob moved closer to the teacher's after the round
    (|student - teacher| shrank), measured teacher-forced on those same answers.

    memory_share: the fraction of the device's memory the plan lets this job use (the queue passes it to every job). Given, a
    teacher on a GPU stays in memory between steps when it fits next to the student's training step, and the choice is emitted
    as an info event. None (a direct call, no plan to measure against) keeps the teacher leaving the device after each score."""
    emit, stop, pause = emit or (lambda event: None), stop or threading.Event(), pause or threading.Event()
    prompts = [(_messages(p), p.get("correction") if isinstance(p, dict) else None) for p in prompts]  # (messages, its own correction)
    if not prompts:
        raise ValueError("Round 3 needs prompts to practise on.")
    if not isinstance(teacher, Teacher):
        teacher = Teacher(teacher, device=device)
    data = load_checkpoint(checkpoint, weights_only=True)
    if teacher.path == str(checkpoint) and teacher.weights is not None:   # the student with a correction in front: one copy of the weights
        teacher.weights = data["model"]
    local, optimizer = _setup(checkpoint, device, data)
    del data
    local.rng = np.random.default_rng(seed)
    student, tokenizer = local.model, local.tokenizer
    check_teacher(tokenizer, local.config["context"], teacher)
    directory, cfg, extra = _begin(output_dir, local, checkpoint, "teacher")
    base_step, rng = local.step, random.Random(seed)
    if memory_share is not None:
        _decide_residency(teacher, local, tokenizer, min(prompts_per_step, len(prompts)), memory_share, emit)

    def encode(item, answer):
        messages, correction = item
        return _encode_pair(tokenizer, messages, answer, cfg["context"]), _encode_pair(tokenizer, teacher.messages(messages, correction), answer, cfg["context"])

    def probe_gaps():
        with torch.no_grad():
            return reverse_kl(student, fixed_logps, [r[0] for r in fixed])[1]

    fixed = [encode(m, local.generate(m[0], max_tokens=max_tokens or studio_rules.max_tokens(), temperature=temperature)) for m in prompts[:probe]]
    fixed_logps = teacher.score([r[1] for r in fixed])   # the same answers are measured before and after, so the teacher scores them once
    before = probe_gaps()
    words = [tokenizer.decode([t]) for t in fixed[0][0][1] if t != -100]
    emit({"type": "heat", "prompt": _prompt_text(prompts[0][0]), "words": words, "surprise": [float(-t) for t in before[0][1]]})
    record = {"name": "learn from a teacher", "createdAt": now(), "status": "training", "stage": "teacher", "steps": steps,
              "base_checkpoint": str(checkpoint), "privileged": bool(teacher.correction)}
    atomic_json(directory / "run.json", record)
    history, best = [], math.inf
    step = 0
    while step < steps and not stop.is_set():
        while pause.is_set() and not stop.is_set():
            threading.Event().wait(.05)
        pairs = [encode(m, local.generate(m[0], max_tokens=max_tokens or studio_rules.max_tokens(), temperature=temperature))
                 for m in rng.sample(prompts, min(prompts_per_step, len(prompts)))]
        teacher_logps = teacher.score([p[1] for p in pairs])   # the teacher's phase: it is off the device again before the student's update
        student.zero_grad()
        kl, _ = reverse_kl(student, teacher_logps, [p[0] for p in pairs], backward=True)
        optimizer.update(student, learning_rate * min(1., (step + 1) / max(1, min(10, steps // 10))))
        step += 1
        row = {"step": step, "loss": kl, "reverse_kl": kl}
        history.append(row)
        emit({"type": "metric", **row, "total": steps})
        if kl < best:
            best = kl
            _save(directory / "best.npz", student, optimizer, cfg, tokenizer, base_step + step, best, history, extra)
    if teacher.resident:
        teacher.release()
    after = probe_gaps()
    gaps = [(abs(float(a) - float(b)), abs(float(c) - float(d))) for (sa, ta), (sb, tb) in zip(before, after)
            for a, b, c, d in zip(sa, ta, sb, tb)]
    record.update(moved=sum(1 for g0, g1 in gaps if g1 < g0 - 1e-9), words=len(gaps),
                  kl_before=float(np.mean([float(np.sum(s - t)) / max(1, len(s)) for s, t in before])),
                  kl_after=float(np.mean([float(np.sum(s - t)) / max(1, len(s)) for s, t in after])))
    emit({"type": "moved", "k": record["moved"], "n": record["words"]})
    return _finish(directory, local, optimizer, cfg, step, base_step, best if math.isfinite(best) else None, history, extra, record, stop)


# --------------------------------------------------------------------------- queue handlers (post_round, restore_answers)

VERDICTS = {1: "Answers now look like answers.", 2: "Answers that passed the check became more likely than ones that failed."}


def _rows(job, context, key, kind):
    return job.params[key] if key in job.params else context.post_rows(kind)


def _doc(row):
    return row if isinstance(row, str) else row.get("text") or f"{row.get('user', '')}\n{row.get('assistant', '')}"


def measure(checkpoint, docs, problems):
    """{bpc, pass_k, pass_n}: bits per character on held-back text; greedy answers checked by each problem's rule."""
    import studio_eval
    local = LocalModel(checkpoint, device="cpu")
    if len(problems) > MEASURE_PROBLEMS:  # imported sets hold back thousands; a fixed sample keeps before/after comparable
        problems = random.Random(0).sample(list(problems), MEASURE_PROBLEMS)
    k = sum(studio_rules.check(q["rule"], q["prompt"], local.generate([{"role": "user", "content": q["prompt"]}], max_tokens=studio_rules.max_tokens(q["rule"]), temperature=0),
                               q.get("params"))
            for q in problems)
    return {"bpc": studio_eval.bits_per_char(checkpoint, [_doc(d) for d in docs]) if docs else None, "pass_k": k, "pass_n": len(problems)}


def skipped_round(probe, checkpoint):
    """The gate result for a round 2 that did not run because the model solved none of the probe problems."""
    detail = f"{probe['passed']} of {probe['answers']} sampled answers passed, on {probe['problems']} problems"
    return {"round": 2, "ok": False, "skipped": True, "verdict": "Not run: this model solves none of these problems yet, so no group would have a better and a worse answer and the round would teach nothing.",
            "checks": [{"name": "The model solves some problems before the round", "detail": detail, "ok": False}],
            "before": None, "after": None, "band": None, "checkpoint": str(checkpoint), "groups": 0, "taught": 0, "probe": probe}


def resolve(number, job, services):
    """App side, before the round's worker starts: the checkpoint it starts from (the current version's) and, for round 3, the
    teacher (the best version's). Both read the version records, which live in the app."""
    p = job.params
    if not p.get("checkpoint"):
        current = services.current_version()
        p["checkpoint"] = str(current.best_checkpoint) if getattr(current, "best_checkpoint", None) else None
        if not p["checkpoint"]:
            from studio_versions import NoCheckpoint
            raise NoCheckpoint(getattr(current, "number", None))
    if number == 3 and p.get("teacher_choice") != "correction" and not p.get("correction") and not p.get("teacher"):
        teacher = getattr(services.best_version(), "best_checkpoint", None)
        p["teacher"] = str(teacher) if teacher else None


def run_round(number, job, context, emit, stop, pause):
    """Run round 1-4 from job.params (resolve() has filled in the checkpoint and teacher), measure before/after, apply the gate
    and emit {"type": "gate", ...}. The result is left in job.params["result"]; the app records it when the job ends."""
    import studio_gate
    import studio_rlhf
    p = job.params
    checkpoint = p.get("checkpoint")
    if not checkpoint:
        from studio_versions import NoCheckpoint
        raise NoCheckpoint(p.get("version"))
    out = Path(p.get("output_dir") or Path(context.run_dir) / f"queue_{job.id}")
    job.params["output_dir"] = str(out)
    if number == 2:   # a model that solves nothing writes all-fail groups, which give no gradient: say so instead of running
        problems = _rows(job, context, "problems", "problems")
        probe = pass_at_k(checkpoint, problems, seed=p.get("seed", 7))
        if probe["problems"] and not probe["solved"]:
            result = skipped_round(probe, checkpoint)
            emit({"type": "gate", **result})
            job.params["result"] = result
            return {"round": number, "ok": False, "checkpoint": checkpoint}
    docs, held = _rows(job, context, "heldout_docs", "heldout_docs"), _rows(job, context, "heldout_problems", "heldout_problems")
    before = measure(checkpoint, docs, held)
    import studio_eval
    band = studio_eval.noise_band(checkpoint, [_doc(d) for d in docs]) if docs else None
    kw = {k: p[k] for k in ("steps", "learning_rate") if p.get(k) is not None}
    verdict = VERDICTS.get(number, "")
    if number == 1:   # the version's own settings carry on, LoRA included: a LoRA version stays LoRA unless the job says {"config": {"lora": None}}
        cfg = {**load_checkpoint(checkpoint)["config"], "objective": "assistant", **kw, **p.get("config", {})}
        record = train(cfg, _rows(job, context, "conversations", "conversations"), out, emit=emit, stop=stop, pause=pause, init_from=checkpoint)
    elif number == 2:
        record = practice_with_checks(checkpoint, problems, out, group=p.get("group", 6), emit=emit, stop=stop, pause=pause, **kw)
        if (record.get("last_mean") or 0) < (record.get("first_mean") or 0):
            verdict = "Answers that passed the check did not become more likely than ones that failed."
        if record.get("groups") and not record.get("taught"):
            verdict = "No group had both a better and a worse answer, so there was nothing to learn from."
    elif number == 3:
        if p.get("teacher_choice") == "correction":   # the same model with each saved correction in front of it
            teacher = Teacher(checkpoint)
            prompts = _rows(job, context, "corrections", "corrections")
        else:
            teacher = privileged(checkpoint, p["correction"]) if p.get("correction") else p.get("teacher")
            prompts = p["prompts"] if "prompts" in p else studio_rlhf.prompts_from(_rows(job, context, "conversations", "conversations"))
        record = learn_from_teacher(checkpoint, prompts, teacher, out, emit=emit, stop=stop, pause=pause, memory_share=p.get("memory_share"), **kw)
        verdict = f"Its answers moved toward the teacher's at {record['moved']} of {record['words']} words."
    elif number == 4:
        pairs = [r for r in _rows(job, context, "pairs", "pairs") if r.get("source") == "human"]  # the judge model's picks never count
        record = studio_rlhf.dpo_train(checkpoint, pairs, out, {"validation": .2, **({"beta": p["beta"]} if "beta" in p else {}), **kw},
                                       emit=emit, stop=stop, pause=pause)
        n, final = record.get("validation_pairs", 0), record.get("final_validation")
        verdict = f"Widened the gap on {round(final['reward_accuracy'] * n)} of {n} held-back pairs." if n and final else "Too few pairs to hold any back."
    else:
        raise ValueError(f"Unknown round: {number}")
    candidate = record.get("checkpoint") or str(out / "last.npz")
    after = measure(candidate, docs, held)
    ok, checks = studio_gate.passes(before, after, band)
    result = {"round": number, "ok": ok, "checks": checks, "verdict": verdict, "before": before, "after": after, "band": band, "checkpoint": candidate}
    if number == 2:   # groups that taught something: the share of reward groups with any spread
        result.update(groups=record.get("groups"), taught=record.get("taught"), probe=probe)
    emit({"type": "gate", **result})
    job.params["result"] = result   # the app keeps it and moves Chat only when ok (WorkspaceServices.record_round)
    return {"round": number, "ok": ok, "checkpoint": candidate}
