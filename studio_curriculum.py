"""Sequential ("curriculum") training: one fixed byte-level model learns datasets stage by stage.

Stages run in the order language -> instructions -> code -> reasoning -> conversation -> knowledge.
Every stage shares one tokenizer (byte, no unknown tokens), one architecture and one context size, and
starts from the previous stage's best weights. After each stage the checkpoint is scored on held-out
examples of that stage AND every earlier stage, so forgetting is measured, not guessed.
The plan state lives in <state_dir>/<plan_id>.json and is rewritten after every stage, so a stopped or
crashed plan continues with resume_plan().
"""
from __future__ import annotations

import copy
import inspect
import json
import math
import random
import threading
from pathlib import Path

from studio_catalog import CATALOG
from studio_core import atomic_json, messages_of, now, uid
from studio_model import train, validate_config
from studio_planning import build_training_set
from studio_spec import is_model_spec

STAGE_ORDER = ("language", "instructions", "code", "reasoning", "conversation", "knowledge")
STAGE_TITLES = {"language": "Plain language", "instructions": "Short instructions", "code": "Code",
                "reasoning": "Reasoning and math", "conversation": "Conversations", "knowledge": "Knowledge and reading"}
# Used when a registered dataset has no catalog stage field.
FALLBACK_STAGE = {"dolly15k": "instructions", "alpaca_cleaned": "instructions", "no_robots": "instructions",
                  "wizard143k": "instructions", "codealpaca20k": "code", "mbpp": "code", "gsm8k": "reasoning",
                  "metamathqa": "reasoning", "commonsenseqa": "reasoning", "oasst1": "conversation",
                  "squad2": "knowledge", "arc_easy": "knowledge", "arc_challenge": "knowledge", "sciq": "knowledge",
                  "tinystories": "language", "tinystories_valid": "language", "wikitext2": "language"}
LONG_SOURCES = {"squad2": 1024}  # catalog id -> smallest context that holds its long passages
STAGE_DEFAULTS = {  # base steps, learning rate, record cap, training objective
    "language": (600, 2e-3, 20000, "all"), "instructions": (600, 1e-3, 20000, "assistant"),
    "code": (400, 8e-4, 10000, "assistant"), "reasoning": (400, 8e-4, 10000, "assistant"),
    "conversation": (400, 6e-4, 10000, "assistant"), "knowledge": (300, 5e-4, 5000, "assistant")}
DEFAULT_CONTEXT = 256
DEFAULT_REPLAY = 0.15
DONE_STATES = {"completed", "skipped"}
HELD_OUT_SPLITS = {"validation", "test"}  # never auto-assigned to a training stage


# --------------------------------------------------------------------------- planning

def catalog_id_of(dataset):
    return dataset.get("catalogId") or (dataset.get("id", "").removeprefix("catalog_") if str(dataset.get("id", "")).startswith("catalog_") else "")


def dataset_stage(dataset):
    """Stage of a registered dataset: catalog field, else known catalog ids, else name hints, else instructions."""
    cid = catalog_id_of(dataset)
    entry = next((e for e in CATALOG if e["id"] == cid), None)
    if entry and entry.get("stage") in STAGE_ORDER:
        return entry["stage"]
    if cid in FALLBACK_STAGE:
        return FALLBACK_STAGE[cid]
    name = str(dataset.get("name", "")).lower()
    for words, stage in ((("code", "python", "program"), "code"), (("math", "reason"), "reasoning"),
                         (("conversation", "chat", "dialog"), "conversation"), (("story", "stories", "wiki", "prose"), "language")):
        if any(word in name for word in words):
            return stage
    return "instructions"


def dataset_split(dataset):
    entry = next((e for e in CATALOG if e["id"] == catalog_id_of(dataset)), None)
    return str(dataset.get("split") or (entry or {}).get("split") or "")


def dataset_records(workspace, dataset):
    count = dataset.get("recordCount")
    if isinstance(count, int) and count >= 0:
        return count
    return int(workspace.count(dataset["id"]))  # SELECT COUNT for disk datasets; no audit DataFrame


def default_plan(workspace, base_config=None, context=None, replay=0.0, dataset_ids=None, name="Train on everything"):
    """An ordered plan with one stage per dataset group. Long-passage sources are skipped below their context size."""
    base_config = dict(base_config or {})
    context = int(context or base_config.get("context") or DEFAULT_CONTEXT)
    batch = max(1, int(base_config.get("batch_size", 4)))
    groups, warnings = {stage: [] for stage in STAGE_ORDER}, []
    for dataset in workspace.state["datasets"]:
        if dataset_ids is not None and dataset["id"] not in dataset_ids:
            continue
        label = dataset.get("name", dataset["id"])
        if dataset_split(dataset) in HELD_OUT_SPLITS:
            warnings.append(f"Skipped {label}: it is a {dataset_split(dataset)} split, kept out of training.")
            continue
        try:
            records = dataset_records(workspace, dataset)
            stage = dataset_stage(dataset)
        except Exception as exc:
            warnings.append(f"Skipped {label}: could not read it ({exc}).")
            continue
        if records <= 0:
            continue
        cid = catalog_id_of(dataset)
        if cid in LONG_SOURCES and context < LONG_SOURCES[cid]:
            warnings.append(f"Skipped {dataset.get('name', cid)}: its passages need a context of at least {LONG_SOURCES[cid]}, "
                            f"but this plan uses {context}. Raise the context to include it.")
            continue
        groups[stage].append((dataset, records))
    stages = []
    for stage in STAGE_ORDER:
        members = groups[stage]
        if not members:
            continue
        total = sum(records for _, records in members)
        base_steps, rate, cap, objective = STAGE_DEFAULTS[stage]
        steps = max(10, min(base_steps, math.ceil(total * 3 / batch)))
        stages.append({"name": STAGE_TITLES[stage], "kind": stage, "dataset_ids": [d["id"] for d, _ in members],
                       "steps": steps, "max_records": int(max(1, min(total, cap, max(200, steps * batch * 2)))),
                       "learning_rate": rate, "context": context, "objective": objective,
                       "eval_records": 100, "status": "pending"})
    return {"name": name, "context": context, "tokenizer": "byte", "replay": float(replay), "stages": stages, "warnings": warnings}


def validate_plan(workspace, plan, base_config=None):
    """Raise ValueError for plans that cannot run; return a list of warnings for ones that can."""
    stages = plan.get("stages") or []
    if not stages:
        raise ValueError("The plan has no stages. Build a default plan or add a stage first.")
    context = int(plan.get("context") or DEFAULT_CONTEXT)
    known = {d["id"]: d for d in workspace.state["datasets"]}
    warnings = []
    for index, stage in enumerate(stages, 1):
        if int(stage.get("context", context)) != context:
            raise ValueError(f"Stage {index} uses context {stage.get('context')} but the plan uses {context}. The context "
                             "cannot change between stages: the model's position embeddings have a fixed size.")
        ids = stage.get("dataset_ids") or []
        if not ids:
            raise ValueError(f"Stage {index} ({stage.get('name', '')}) has no datasets.")
        missing = [key for key in ids if key not in known]
        if missing:
            raise ValueError(f"Stage {index} uses a dataset that no longer exists: {missing[0]}")
        for key in ids:
            need = LONG_SOURCES.get(catalog_id_of(known[key]))
            if need and context < need:
                warnings.append(f"Stage {index}: {known[key].get('name', key)} has long passages (needs context >= {need}); "
                                f"at {context} they are cut into windows and compute is partly wasted.")
        if int(stage.get("steps", 0)) < 1 or int(stage.get("max_records", 0)) < 1:
            raise ValueError(f"Stage {index} needs at least 1 step and 1 record.")
        validate_config(_stage_config({"base_config": base_config or {}, "context": context}, stage, index - 1))
    if not 0 <= float(plan.get("replay", 0)) <= 0.5:
        raise ValueError("Replay must be between 0 and 0.5 (a fraction of the stage's records).")
    return warnings


def _stage_config(state, stage, index):
    cfg = copy.deepcopy(dict(state.get("base_config") or {}))
    context = int(state["context"])
    spec = cfg.get("architecture")
    if isinstance(spec, str):
        spec = json.loads(spec) if spec.strip() not in ("", "None", "null") else None
    if isinstance(spec, dict):
        spec = copy.deepcopy(spec)
        if is_model_spec(spec):
            spec["tokenizer"] = {"id": "byte"}
            spec["dims"]["n_ctx"] = context
        else:
            if not isinstance(spec.get("embedding"), dict):
                raise ValueError("The Blueprint architecture has no embedding section; open it in the Blueprint and save it again.")
            spec["embedding"]["tokenizer"] = "byte"
            spec["embedding"]["context"] = context
    cfg["architecture"] = spec
    steps = int(stage["steps"])
    cfg.update(tokenizer="byte", context=context, steps=steps, learning_rate=float(stage["learning_rate"]),
               objective=stage.get("objective", "assistant"), name=f"{state.get('name', 'Curriculum')} · stage {index + 1} {stage.get('name', '')}".strip())
    cfg["warmup"] = min(int(cfg.get("warmup", 20)), steps // 10)
    cfg["eval_every"] = max(1, min(25, steps // 4))
    return cfg


# --------------------------------------------------------------------------- state files

def state_directory(workspace, state_dir=None):
    return Path(state_dir) if state_dir else Path(workspace.root) / "data" / "curriculum"


def _state_path(directory, plan_id):
    return Path(directory) / f"{plan_id}.json"


def load_plan(workspace, plan_id, state_dir=None):
    path = _state_path(state_directory(workspace, state_dir), plan_id)
    if not path.is_file():
        raise ValueError(f"Unknown curriculum plan: {plan_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def list_plans(workspace, state_dir=None):
    rows = []
    for path in state_directory(workspace, state_dir).glob("*.json"):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return sorted(rows, key=lambda row: row.get("updated", ""), reverse=True)


def latest_plan(workspace, state_dir=None, resumable=True):
    for row in list_plans(workspace, state_dir):
        if not resumable or row.get("status") in {"stopped", "failed", "running", "pending"}:
            return row
    return None


# --------------------------------------------------------------------------- execution

def _fresh_dir(root, base):
    path = Path(root) / base
    attempt = 1
    while path.exists() and any(path.iterdir()):
        attempt += 1
        path = Path(root) / f"{base}_r{attempt}"
    return path


def _build(workspace, ids, count, seed, emit, stop, balance="tokens"):
    kwargs = dict(max_records=int(count), seed=int(seed), stop=stop, emit=emit)
    try:
        if balance and "balance" in inspect.signature(build_training_set).parameters:
            kwargs["balance"] = balance
    except (TypeError, ValueError):
        pass
    return build_training_set(workspace, list(ids), **kwargs)["examples"]


def _write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""), encoding="utf-8")


def _read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return []
    rows = []
    try:
        raw = path.read_bytes()
    except OSError:
        return []
    for line_bytes in raw.split(b'\n'):
        try:
            line = line_bytes.decode(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except (json.JSONDecodeError, ValueError):
            continue
    return rows


def _checkpoint(run_dir):
    for name in ("best.npz", "last.npz"):
        if (Path(run_dir) / name).is_file():
            return str(Path(run_dir) / name)
    return None


def _prompt_key(row):
    """Normalised prompt text, so the same question in two datasets (GSM8K inside MetaMathQA) shares one key."""
    text = row.get("text") or next((m.get("content", "") for m in messages_of(row) if m.get("role") == "user"), "")
    return " ".join(str(text).casefold().split())


def _held_out(workspace, state, eval_dir, seed, emit, stop):
    """Sample each stage's held-out file once, before any training; return the plan-wide prompt-key registry.

    Every stage's training rows are filtered against this registry, so no held-out prompt is trained on
    in any stage, even when stages share a dataset or one dataset contains another's questions.
    """
    stages = state["stages"]
    paths = [eval_dir / f"stage{j + 1}.jsonl" for j in range(len(stages))]
    registry = {_prompt_key(row) for path in paths for row in _read_jsonl(path)}
    missing = [j for j, stage in enumerate(stages) if not paths[j].is_file() and stage["status"] not in DONE_STATES]
    if missing:  # rows an earlier attempt already trained on cannot be held out any more
        trained = {_prompt_key(row) for stage in stages for run_dir in stage.get("run_dirs") or []
                   for row in _read_jsonl(Path(run_dir) / "dataset.jsonl")}
    for j in missing:
        want = int(stages[j].get("eval_records", 100))
        rows = []
        for row in _build(workspace, stages[j]["dataset_ids"], max(1, want * 3), seed + 7919 + j, emit, stop):
            key = _prompt_key(row)
            if len(rows) < want and key and key not in registry and key not in trained:
                rows.append(row)
                registry.add(key)
        _write_jsonl(paths[j], rows)
        if not rows:
            text = f"Stage {j + 1} ({stages[j].get('name', '')}) has no held-out rows; forgetting is not measured for it."
            state.setdefault("warnings", []).append(text)
            emit({"type": "info", "text": "Warning: " + text})
    registry.discard("")
    return registry


def _summary(state):
    return copy.deepcopy({k: v for k, v in state.items() if k not in {"base_config"}})


def _forgetting(state):
    rows = []
    table = {row["after"]: row["evals"] for row in state.get("eval_table", [])}
    stages = state["stages"]
    for j, stage in enumerate(stages):
        learned = (table.get(j, {}).get(str(j)) or {}).get("loss")
        latest_after = max((k for k in table if k >= j and (table[k].get(str(j)) or {}).get("loss") is not None), default=None)
        if learned is None or latest_after is None:
            continue
        now_loss = table[latest_after][str(j)]["loss"]
        rows.append({"stage": j, "name": stage.get("name", ""), "learned_loss": learned, "latest_loss": now_loss,
                     "delta": now_loss - learned, "after_stage": latest_after})
    return rows


def _save(state, emit, event_stage=None):
    state["updated"] = now()
    state["forgetting"] = _forgetting(state)
    atomic_json(_state_path(state["state_dir"], state["id"]), state)
    emit({"type": "curriculum", "plan_id": state["id"], "stage": event_stage, "name": (state["stages"][event_stage]["name"] if event_stage is not None else ""),
          "status": (state["stages"][event_stage]["status"] if event_stage is not None else state["status"]),
          "plan_status": state["status"], "eval_table": copy.deepcopy(state.get("eval_table", [])),
          "forgetting": copy.deepcopy(state["forgetting"]), "state": _summary(state)})


def run_plan(workspace, plan, base_config=None, emit=None, stop=None, pause=None, plan_id=None, init_from=None,
             run_root=None, state_dir=None, register_arch_id=None):
    """Run every stage in order and return the final plan state (also saved as <state_dir>/<plan_id>.json)."""
    emit = emit or (lambda event: None)
    base_config = dict(base_config or {})
    if str(base_config.get("tokenizer", "byte")) != "byte":
        emit({"type": "info", "text": "Curriculum training uses the byte tokenizer in every stage (no unknown characters across datasets)."})
    warnings = validate_plan(workspace, plan, base_config)
    for text in list(plan.get("warnings", [])) + warnings:
        emit({"type": "info", "text": "Warning: " + text})
    plan_id = plan_id or uid("curr")
    directory = state_directory(workspace, state_dir)
    state = {"id": plan_id, "name": plan.get("name", "Curriculum"), "created": now(), "updated": now(), "status": "pending",
             "context": int(plan.get("context") or DEFAULT_CONTEXT), "tokenizer": "byte", "replay": float(plan.get("replay", 0)),
             "base_config": base_config, "init_from": str(init_from) if init_from else None,
             "run_root": str(Path(run_root) if run_root else workspace.run_dir), "state_dir": str(directory),
             "register_arch_id": register_arch_id, "warnings": list(plan.get("warnings", [])) + warnings,
             "stages": [], "eval_table": [], "forgetting": []}
    for stage in plan["stages"]:
        row = copy.deepcopy(stage)
        row.update(status="pending", context=state["context"], run_dir=None, run_dirs=[], checkpoint=None, loss=None, val_loss=None,
                   best_loss=None, steps_done=0, seconds=None, records_used=None, replay_records=0, error=None)
        row.setdefault("eval_records", 100)
        state["stages"].append(row)
    _save(state, emit)
    return _execute(workspace, state, emit, stop, pause)


def resume_plan(workspace, plan_id, emit=None, stop=None, pause=None, state_dir=None):
    """Continue a saved plan from its first incomplete stage (a stopped stage resumes from its last.npz)."""
    state = load_plan(workspace, plan_id, state_dir)
    if state.get("status") == "completed":
        raise ValueError("This plan already finished.")
    for stage in state["stages"]:
        if stage["status"] in {"failed", "running"}:
            stage["status"] = "stopped" if stage.get("run_dir") else "pending"
    return _execute(workspace, state, emit or (lambda event: None), stop, pause)


def _execute(workspace, state, emit, stop, pause):
    stop = stop if stop is not None else threading.Event()
    pause = pause if pause is not None else threading.Event()
    stages, device = state["stages"], (state.get("base_config") or {}).get("device", "auto")
    eval_dir = Path(state["state_dir"]) / f"{state['id']}_eval"
    seed = int((state.get("base_config") or {}).get("seed", 42))
    state["status"] = "running"
    _save(state, emit)
    try:
        registry = _held_out(workspace, state, eval_dir, seed, emit, stop)
    except InterruptedError:
        state["status"] = "stopped"
        _save(state, emit)
        return state
    except Exception:
        state["status"] = "failed"
        _save(state, emit)
        raise

    def unseen(rows):
        return [row for row in rows if _prompt_key(row) not in registry]

    from studio_rlhf import evaluate_checkpoint
    for index, stage in enumerate(stages):
        if stage["status"] in DONE_STATES:
            continue
        if stop.is_set():
            break
        label = f"Stage {index + 1}/{len(stages)} · {stage['name']}"
        try:
            if stage["status"] != "evaluating":
                stage.update(status="running", error=None)
                _save(state, emit, index)
                emit({"type": "info", "text": f"{label}: preparing {stage['max_records']:,} records."})
                resume_from, examples = None, None
                # Newest last.npz over every attempt: a second stop before its first save must not lose the first.
                saved = [Path(d) / "last.npz" for d in stage.get("run_dirs") or [] if (Path(d) / "last.npz").is_file()]
                if saved:
                    newest = max(saved, key=lambda path: path.stat().st_mtime_ns)
                    resume_from = str(newest)
                    # Not filtered: the checkpoint already trained on this exact snapshot, and train() checks its hash.
                    examples = _read_jsonl(newest.parent / "dataset.jsonl") or None
                    emit({"type": "info", "text": f"{label}: resuming from {newest.parent.name}."})
                if examples is None:
                    total = sum(int(workspace.count(key)) for key in stage["dataset_ids"])
                    cap = int(stage["max_records"])
                    # Held-out rows are filtered out, so training is capped at total - held-out, never the whole stage.
                    request = max(1, min(total, cap + int(stage.get("eval_records", 100))))
                    examples = unseen(_build(workspace, stage["dataset_ids"], request, seed + index, emit, stop))[:cap]
                    replay = []
                    if state["replay"] > 0 and index:
                        replay = unseen(_replay_rows(stages[:index], int(state["replay"] * len(examples)), seed + index))
                        examples = examples + replay
                    stage["replay_records"] = len(replay)
                stage["records_used"] = len(examples)
                previous = next((stages[j]["checkpoint"] for j in range(index - 1, -1, -1) if stages[j].get("checkpoint")), None) or state.get("init_from")
                cfg = _stage_config(state, stage, index)
                run_dir = _fresh_dir(state["run_root"], f"{state['id']}_stage{index + 1}")
                stage["run_dir"] = str(run_dir)
                stage["run_dirs"].append(str(run_dir))
                _save(state, emit, index)
                record = train(cfg, examples, run_dir, emit, stop, pause, resume=resume_from,
                               init_from=None if resume_from else previous)
                stage.update(steps_done=record.get("step", 0), loss=record.get("loss"), val_loss=record.get("val_loss"),
                             best_loss=record.get("best_loss"), seconds=(stage.get("seconds") or 0) + (record.get("duration_seconds") or 0),
                             checkpoint=_checkpoint(run_dir))
                if record.get("status") == "stopped":
                    stage["status"] = "stopped"
                    state["status"] = "stopped"
                    _save(state, emit, index)
                    return state
                stage["status"] = "evaluating"
                _save(state, emit, index)
            # held-out scores of this stage and every earlier one (forgetting check)
            evals = {}
            if not stage.get("checkpoint"):
                emit({"type": "info", "text": f"{label}: no checkpoint saved; could not evaluate."})
            else:
                for j in range(index + 1):
                    held_out = _read_jsonl(eval_dir / f"stage{j + 1}.jsonl")
                    if not held_out:
                        continue
                    try:
                        result = evaluate_checkpoint(stage["checkpoint"], held_out, device=device, limit=int(stages[j].get("eval_records", 100)), stop=stop)
                        evals[str(j)] = {k: result[k] for k in ("loss", "perplexity", "token_accuracy", "tokens", "unknown_tokens")}
                    except (ValueError, OSError) as exc:
                        emit({"type": "info", "text": f"{label}: could not score stage {j + 1} data ({exc})."})
            if stop.is_set():
                state["status"] = "stopped"
                _save(state, emit, index)
                return state
            state["eval_table"] = [r for r in state["eval_table"] if r["after"] != index] + [{"after": index, "evals": evals}]
            stage["status"] = "completed"
            _register(workspace, state, stage, index, emit)
            _save(state, emit, index)
            own = evals.get(str(index), {}).get("loss")
            emit({"type": "info", "text": f"{label}: done · held-out loss {own:.3f}" if own is not None else f"{label}: done."})
        except InterruptedError:
            stage["status"] = "stopped" if stage.get("run_dir") else "pending"
            state["status"] = "stopped"
            _save(state, emit, index)
            return state
        except Exception as exc:
            stage.update(status="failed", error=str(exc))
            state["status"] = "failed"
            _save(state, emit, index)
            raise
    state["status"] = "completed" if all(s["status"] in DONE_STATES for s in stages) else "stopped"
    _save(state, emit)
    return state


def _replay_rows(earlier, count, seed):
    """Random rows the earlier stages actually trained on (their run snapshots), spread evenly across stages."""
    pools = [_read_jsonl(Path(s["run_dir"]) / "dataset.jsonl") for s in earlier if s.get("run_dir")]
    pools = [p for p in pools if p]
    if not pools or count <= 0:
        return []
    rng, rows = random.Random(seed), []
    share = max(1, count // len(pools))
    for pool in pools:
        rows.extend(rng.sample(pool, min(share, len(pool))))
    return rows


def _register(workspace, state, stage, index, emit):
    arch_id = state.get("register_arch_id")
    if not arch_id or not stage.get("checkpoint"):
        return
    try:
        from studio_archdb import ArchitectureStore
        store = ArchitectureStore(workspace.root)
        previous = [s.get("weights_version") for s in state["stages"][:index] if s.get("weights_version")]
        stage["weights_version"] = store.add_weights(arch_id, stage["checkpoint"], step=stage.get("steps_done"),
                                                     loss=stage.get("best_loss"), source="curriculum", run_dir=stage["run_dir"],
                                                     note=f"{state['name']} stage {index + 1}: {stage['name']}", parent=previous[-1] if previous else None)
        emit({"type": "info", "text": f"Registered stage {index + 1} weights as v{stage['weights_version']} of the Network architecture."})
    except Exception as exc:
        emit({"type": "info", "text": f"Could not register stage {index + 1} weights: {exc}"})
