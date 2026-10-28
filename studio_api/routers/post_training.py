"""Post-training: what each round has done, the choices a round offers, queueing a round, and starting from a pretrained model."""

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.exceptions import HTTPException
from pydantic import BaseModel, ConfigDict, Field

import studio_registry as registry
import studio_rl
import studio_rlhf
from studio_queue import Job
from studio_api.app import get_services
from studio_api.models_work import JobOut, VersionOut, start_job, version_out


T = {
    "h1": "Post-training",
    "lead": "Four rounds, always in this order. Same design throughout; only the weights shift.",
    "starts_from": "STARTS FROM",
    "reading": "Reading your data…",
    "version_best": "Version 4 · best checkpoint from Pretraining",
    "at_this_size": "At this size the gain from each round is small. The shape of the pipeline is what you are learning.",
    "click_round": "Click a round to see it.",
    "round_1": "ROUND 1 · PRACTICE ANSWERS",
    "r1_method": "supervised fine-tuning",
    "r1_lead": "Teaches the shape of an answer, not the skill. Pretraining taught it to continue text; here it studies prompts paired with good answers.",
    "round_2": "ROUND 2 · PRACTICE WITH CHECKS",
    "r2_method": "reinforcement learning with rule-checked rewards",
    "r2_lead": "For each prompt it writes 6 answers. A rule scores each one. Answers above the group's average are made more likely; answers below it, less likely. A group where every answer scores the same teaches nothing.",
    "round_3": "ROUND 3 · LEARN FROM A TEACHER",
    "r3_method": "distillation from a teacher, or from the same model shown a hint",
    "r3_lead": "It writes its own answer and a teacher scores every word. The teacher is another trained version, or this same model shown a hint, so it learns to answer as if it had the hint. A hint only helps if the model can use one, and tiny models often can't.",
    "move_front": "Move to the front of the queue",
    "round_4": "ROUND 4 · YOUR VOTES",
    "r4_method": "direct preference optimisation (DPO)",
    "r4_lead": "Two answers to one prompt; you pick the better one. Training widens the gap between your pick and the other.",
    "add_votes": "Add votes round to the queue",
    "rate_pairs": "Rate pairs in Chat",
    "add_conversations": "Add conversations in Your data",
    "make_problems": "Make problems a rule can check",
    "prompt_scored": "PROMPT · NOT SCORED",
    "answer_learns": "ANSWER · IT LEARNS TO WRITE THIS",
    "conversations_count": "[N]",
    "conversations_label": "conversations (Dolly, Alpaca, yours)",
    "corrections_count": "[N]",
    "corrections_label": "corrections you wrote",
    "problems_count": "[N]",
    "problems_label": "practice problems with answers",
    "lora_label": "WHAT IT CHANGES",
"lora_off": "Every weight",
"lora_on": "LoRA adapters",
"lora_rank_label": "ADAPTER SIZE (RANK)",
"lora_desc_off": "Ordinary training: every weight in the model can change.",
"lora_desc_on": "The model's own weights stay frozen. Only small add-on layers learn, so the round is lighter and the model drifts less from what it already knows. A bigger rank can learn more and costs more memory.",
"lora_no_checkpoint": "LoRA needs a trained version to start from.",
"lora_mixed": "LoRA needs float32. This version trains in mixed precision.",
"few_thousand": "A few thousand conversations teach the chat format; more keep improving the answers. It trains about two passes over all your conversations, up to 2,000 steps, so a large mix takes a while. Only the answer part is scored. No frozen copy and no leash: this round is ordinary training on good examples.",
    "prompt_rule": "PROMPT · RULE: {rule}",
    "answer_output": "its answer",
    "rule_label": "rule",
    "score_label": "score",
    "group_avg": "vs group average (0.5)",
    "copy_this": "Copy this: hello",
    "passed_count": "6 of 6 passed",
    "nothing_learn": "nothing to learn here: every answer equals the average",
    "rules_check": "Rules it can check: copy, reverse, sort, count, small addition, stay under a length, end cleanly, and the answer key an imported set came with (a final number or an option letter). A teacher's rubric can add a smaller second score, but an answer that fails a rule gets the minimum whatever the rubric says.",
    "prompt_label": "PROMPT",
    "answer_shaded": "ITS ANSWER · SHADED BY HOW SURPRISED THE TEACHER WAS BY EACH WORD",
    "not_surprised": "not surprised",
    "very_surprised": "very surprised · pushed toward what the teacher would write",
    "illustrative": "illustrative words",
    "teacher_label": "Teacher",
    "teacher_note": "The teacher must share this model's word list and reading window; the app checks and says so if it doesn't. Your local model (Ollama) can't teach here. It keeps two jobs: rewriting prompts and judging.",
    "previous_best": "Another trained version",
    "previous_desc": "{version}. Same word list and reading window as this model, and the earlier version with the lowest answer loss among those whose Round 1 passed.",
    "no_default_teacher": "No earlier version has a passed Round 1 yet, so no teacher is preselected: pick one yourself, or run Round 1 on an earlier version first.",
    "no_previous": "No other trained version shares this model's word list yet: train another first, or teach a correction in Chat.",
    "this_model_correction": "This model, shown your correction as a hint",
    "correction_desc": "From Chat's \"Teach a correction\": the same model is shown your correction as a hint and learns to answer as if it had it. This only works if the model can use a hint.",
    "spread_some": "{taught} of {groups} {groups_word} had both a better and a worse answer; only those teach the model.",
    "spread_none": "None of the {groups} {groups_word} had both a better and a worse answer, so nothing was learned.",
    "which_version": "Which version teaches",
    "spread_hint": "Only a group with both a better and a worse answer teaches the model. How many did is shown here as it trains.",
    "for_each_pair": "For each pair you rated, the learning copy widens the gap between your pick and the other, compared with the frozen starting copy. Both can become less likely; the gap is what it is trained on. Bar heights are illustrative.",
    "how_far": "How far it may move",
    "leash_note": "Only pairs a person rated count here; the judge model's picks never do. Optional, and always last.",
    "close": "Close",
    "close_note": "Small, safe changes. Needs more pairs to show an effect.",
    "medium": "Medium",
    "medium_note": "A balance. A good default.",
    "far": "Far",
    "far_note": "Bigger changes from fewer pairs, with more risk of forgetting.",
    "verdict_title": "VERDICT",
    "chat_moves": "Chat moves to the new checkpoint only if both hold:",
    "held_back_check": "Held-back text: error did not rise",
    "held_back_measured": "measured when the round ends",
    "held_back_text_error": "{error} bits per character, was {was_error}",
    "held_back_text_rose": "{error} bits per character, was {was_error}: rose",
    "problems_check": "Checked problems: pass rate did not fall",
    "problems_measured": "measured when the round ends",
    "problems_text": "[k] of [n] held-back problems, was [k] of [n]",
    "overlap_left_out": "{left_out} of {checked} held-back {what} also appear in training and were left out.",
    "overlap_what_problems": "problems",
    "overlap_what_text": "documents",
    "judge_note": "shown for context; never decides alone",
    "result_title": "MOCK",
    "flip_label": "Show a check fails",
    "flip_show_pass": "Show: both checks pass",
    "answers_now": "Answers now look like answers.",
    "passed_became": "Answers that passed the check became more likely than ones that failed.",
    "moved_toward_what": "how many words of its answers moved toward the teacher's",
    "widened_gap_what": "on how many held-back pairs the gap between your pick and the other widened",
    "will_count": "Will count {what}.",
    "so_far_count": "So far: counting {what} (on training prompts; held-back check at the end).",
    "judge_none": "Judge: not run yet",
    "so_far": "So far: {verdict_start}",
    "will_read": "Will read: \"{verdict_full}\"",
    "result_kept": "Kept for comparison, not live. Chat stays on the previous checkpoint.",
    "result_live": "Live: Chat uses {checkpoint} from the next message.",
    "result_passed": "Passed and went live; round 2 started from this checkpoint.",
    "chat_keeps": "Chat keeps {checkpoint} until this round passes both checks.",
    "until_then": "Until then, Chat keeps its current checkpoint.",
}


router = APIRouter(prefix="/post-training", tags=["post-training"])

LEASH = (("Close", .5, T["close_note"]), ("Medium", .1, T["medium_note"]), ("Far", .02, T["far_note"]))  # label, beta, note
LORA_RANKS = (4, 8, 16, 32)  # adapter sizes offered; studio_lora's own default (8) is the middle one
CHUNK = 1 << 20


class RoundState(BaseModel):
    status: str = "idle"  # done | running | queued | idle
    note: str = ""


class OverlapOut(BaseModel):
    model_config = ConfigDict(extra="allow")
    checked: int = 0
    left_out: int = 0
    note: str | None = None


class PostTrainingOut(BaseModel):
    """Services.post_training: missing keys are fine, a client reads them as unknown."""
    rounds: dict[int, RoundState] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)
    gates: dict[int, dict[str, Any]] = Field(default_factory=dict)
    overlap: dict[str, OverlapOut] = Field(default_factory=dict)
    example: Any = None


class PostTrainingJobOut(JobOut):
    result: PostTrainingOut | None = None


class ImportModelOut(BaseModel):
    id: str
    name: str
    size: int


@router.get("/import-models", response_model=list[ImportModelOut])
def import_models():
    """Models this server can import; size is the expected download in bytes."""
    from studio_import import MODELS
    return [ImportModelOut(id=key, name=model["title"], size=model["size"]) for key, model in MODELS.items()]


class LeashOut(BaseModel):
    name: str
    beta: float
    note: str


class TeacherOut(BaseModel):
    number: int
    name: str
    chat_passed: bool | None = None  # whether its Round 1 passed the gate; None = Round 1 never ran on it
    chat_loss: float | None = None  # its measured answer loss on held-back conversations after Round 1; None = not measured


class OptionsOut(BaseModel):
    leash: list[LeashOut]
    lora_ranks: list[int]
    lora_blocker: str | None  # why LoRA cannot be chosen for round 1 from this version, in plain words
    lora_default: bool  # whether the version already trains adapters
    teachers: list[TeacherOut]  # trained versions that could teach this one in round 3
    default_teacher: int | None


class RoundRequest(BaseModel):
    round: int = Field(ge=1, le=4)
    version: int | None = None  # None = the current version
    lora: bool | None = None  # round 1: adapters (True) or every weight (False); None = what the version already does
    lora_rank: int = 8
    teacher: Literal["version", "correction"] = "version"  # round 3: another trained version, or this model shown a correction
    teacher_version: int | None = None  # None = the previous best
    leash: Literal["Close", "Medium", "Far"] = "Medium"  # round 4
    method: Literal["practice", "rejection"] = "practice"  # round 2: practise with rewards, or train once on the answers that passed the check and stopped
    steps: int | None = Field(default=None, ge=1, le=5000)  # round 2: None = the round's default (studio_rl.ROUND2_DEFAULTS)
    prompts_per_step: int | None = Field(default=None, ge=1, le=64)  # round 2: prompts trained on each step
    group: int | None = Field(default=None, ge=2, le=32)  # round 2: answers sampled per prompt
    temperature: float | None = Field(default=None, gt=0, le=2)  # round 2: sampling temperature
    max_tokens: int | None = Field(default=None, ge=1, le=2048)  # round 2: answer length cap; None = sized by the rule
    kl: float | None = Field(default=None, ge=0, le=1)  # round 2: pull back to the start weights; None = studio_rl.KL_DEFAULT (0 for LoRA), 0 = off


class RoundQueued(BaseModel):
    id: str
    label: str
    round: int
    params: dict[str, Any]


class StageRequest(BaseModel):
    stage: Literal["midtrain"]
    version: int | None = None
    steps: int | None = Field(None, ge=1, le=100_000)


class StageQueued(BaseModel):
    id: str
    label: str
    stage: Literal["midtrain"]


class AdoptRequest(BaseModel):
    version: int | None = None  # None = the current version


class AdoptedOut(BaseModel):
    version: int
    round: int
    checkpoint: str  # the checkpoint that is now the version's best


class RowsRequest(BaseModel):
    kind: Literal["conversations", "problems", "pairs", "corrections", "heldout_docs", "heldout_conversations", "heldout_problems"]


class StatusRequest(BaseModel):
    version: int | None = None


def pick_version(services, number):
    """The version asked for, the current one when `number` is None; 404 for a number that is not a version."""
    if number is None:
        return services.current_version()
    found = next((v for v in services.versions() if v.number == number), None)
    if found is None:
        raise HTTPException(404, f"There is no version {number}.")
    return found


def is_mixed_precision(spec):
    """True when the design's precision module runs matrix multiplies in 16 bits (bf16, fp16); LoRA refuses these."""
    entry = ((spec or {}).get("train") or {}).get("precision")
    entry = {"id": entry} if isinstance(entry, str) else entry
    if not isinstance(entry, dict) or not entry.get("id"):
        return False
    try:
        return registry.get("precision", entry["id"]).compute is not None
    except ValueError:
        return False


def lora_blocker(version):
    """Why LoRA cannot be chosen for a round that starts from `version` (plain words), or None; mirrors what studio_model.train
    refuses: LoRA without a checkpoint to start from, and LoRA under mixed precision."""
    if not (version and version.best_checkpoint):
        return T["lora_no_checkpoint"]
    return T["lora_mixed"] if is_mixed_precision(version.spec) else None


def lora_default(version):
    return bool((((version.spec if version else None) or {}).get("train") or {}).get("run", {}).get("lora"))


def _tokenizer_hash(tokenizer):
    """The same serialization studio_rl.check_teacher compares (Tokenizer.serialize), hashed."""
    return hashlib.sha256(json.dumps(tokenizer, sort_keys=True, default=str).encode()).hexdigest()


def _checkpoint_tokenizer_hash(version, root):
    """Hash of the tokenizer stored in the version's checkpoint (only its metadata is read), or None when it cannot be read."""
    try:
        path = Path(version.best_checkpoint)
        path = path if path.is_absolute() or root is None else Path(root) / path
        with np.load(path, allow_pickle=False) as archive:
            return _tokenizer_hash(json.loads(str(archive["metadata"].item()))["tokenizer"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def teacher_options(versions, version, root=None):
    """Trained versions that could teach `version`: any other one with a checkpoint, the same word list and the same reading window.
    The word lists are compared as the job will compare them (the tokenizers stored in the checkpoints, hashed), so a teacher
    listed here does not fail in studio_rl.check_teacher after the job starts; a checkpoint that cannot be read falls back to the
    spec's tokenizer entry. `root` is the workspace folder that relative checkpoint paths sit in."""
    def spec_shape(v):
        spec = v.spec or {}
        return _tokenizer_hash(spec.get("tokenizer")), (spec.get("dims") or {}).get("n_ctx")
    if not version:
        return []
    mine = _checkpoint_tokenizer_hash(version, root) if version.best_checkpoint else None
    out = []
    for v in versions:
        if v.number == version.number or not v.best_checkpoint:
            continue
        theirs = _checkpoint_tokenizer_hash(v, root) if mine else None
        words_match = theirs == mine if mine and theirs else spec_shape(v)[0] == spec_shape(version)[0]
        if words_match and spec_shape(v)[1] == spec_shape(version)[1]:
            out.append(v)
    return out


def chat_result(gates, number):
    """(passed, answer loss) of version `number`'s recorded Round 1 (gates = services.round_gates()); (None, None) when it never ran,
    and a loss of None when the round did not measure one."""
    record = ((gates or {}).get(number) or {}).get(1)
    if not record:
        return None, None
    loss = (record.get("after") or {}).get("conv_loss")
    return bool(record.get("ok")), (float(loss) if isinstance(loss, (int, float)) and loss == loss else None)


def default_teacher(options, version, gates=None):
    """The earlier option (before `version`) whose Round 1 passed its gate with the lowest measured answer loss; None when no
    earlier option passed, so the page preselects nothing rather than an unchecked version."""
    ranked = []
    for v in options:
        passed, loss = chat_result(gates, v.number)
        if version and v.number < version.number and passed and loss is not None:
            ranked.append((loss, -v.number, v))
    return min(ranked, key=lambda r: r[:2])[2] if ranked else None


def round_blocker(round_number, counts, teacher_is_version, has_teacher, checkpoint):
    """Why a round cannot be queued yet (plain words), or None. counts = post_training()["counts"]."""
    need = {1: ("conversations", 1, "Add conversations in Your data first."),
            2: ("problems", 1, "Make problems a rule can check first."),
            4: ("pairs", studio_rlhf.MIN_PAIRS_TO_TRAIN, f"Rate at least {studio_rlhf.MIN_PAIRS_TO_TRAIN} pairs in Chat first: fewer are memorised, not learned.")}
    if not checkpoint:
        return "Train this version in Pretraining first."
    if round_number == 3:
        if teacher_is_version:
            return None if has_teacher else "There is no other trained version to learn from yet."
        return None if counts.get("corrections") else "Teach a correction in Chat first."
    key, least, why = need[round_number]
    return None if (counts.get(key) or 0) >= least else why


@router.post("/status", response_model=PostTrainingJobOut, status_code=202)
def read_status(request: Request, body: StatusRequest):
    """What each round has done for a version, and what is stored for the rounds to use. A job: it counts the stored data.
    The result is a PostTrainingOut."""
    services = request.app.state.services
    version = pick_version(services, body.version)
    return start_job(request, "post_training.status", lambda stop: services.post_training(version), PostTrainingOut)


@router.post("/rows", response_model=JobOut, status_code=202)
def read_rows(request: Request, body: RowsRequest):
    """The rows behind a round (conversations, problems, rated pairs, corrections) or the held-back ones. A job: the held-back
    text is parsed once and its overlap with training is checked."""
    services = request.app.state.services
    return start_job(request, f"post_training.rows.{body.kind}", lambda stop: services.post_rows(body.kind))


@router.get("/options", response_model=OptionsOut)
def options(version: int | None = None, services=Depends(get_services)):
    """The choices a round offers for a version: LoRA for round 1, the teachers for round 3, the leash for round 4."""
    chosen = pick_version(services, version)
    teachers = teacher_options(services.versions(), chosen, getattr(services, "root", None))
    gates = services.round_gates()
    teacher = default_teacher(teachers, chosen, gates)
    return OptionsOut(leash=[LeashOut(name=name, beta=beta, note=note) for name, beta, note in LEASH], lora_ranks=list(LORA_RANKS),
                      lora_blocker=lora_blocker(chosen), lora_default=lora_default(chosen),
                      teachers=[TeacherOut(number=v.number, name=v.name, **dict(zip(("chat_passed", "chat_loss"), chat_result(gates, v.number)))) for v in teachers],
                      default_teacher=teacher.number if teacher else None)


@router.post("/rounds", response_model=RoundQueued)
def queue_round(body: RoundRequest, services=Depends(get_services)):
    """Add a post-training round to the queue, with the choices the page offers. Refused with the page's own sentence when the
    round has nothing to learn from yet."""
    version = pick_version(services, body.version)
    if version is None:
        raise ValueError("There is no version yet. Build one first.")
    counts = (services.post_training(version) or {}).get("counts", {})
    teachers = teacher_options(services.versions(), version, getattr(services, "root", None))
    why = round_blocker(body.round, counts, body.teacher == "version", bool(teachers), bool(version.best_checkpoint))
    if why:
        raise ValueError(why)
    settings = {}
    if body.round == 1:
        if body.lora_rank not in LORA_RANKS:
            raise ValueError(f"LoRA rank is one of {', '.join(map(str, LORA_RANKS))}.")
        wanted = lora_default(version) if body.lora is None else body.lora
        blocked = lora_blocker(version)
        if wanted and blocked and body.lora:
            raise ValueError(blocked)
        if not blocked:
            settings["lora"] = body.lora_rank if wanted else None
    if body.round == 3:
        teacher = None
        if body.teacher == "version":
            teacher = next((v for v in teachers if v.number == body.teacher_version), None) or default_teacher(teachers, version, services.round_gates())
            if body.teacher_version is not None and (teacher is None or teacher.number != body.teacher_version):
                raise ValueError(f"Version {body.teacher_version} cannot teach version {version.number}: "
                                 "it needs a checkpoint, the same word list and the same reading window.")
            if teacher is None:
                raise ValueError(T["no_default_teacher"])
        settings["teacher"] = teacher.best_checkpoint if teacher else None
        settings["teacher_choice"] = "previous best" if body.teacher == "version" else "correction"
    if body.round == 4:
        settings["beta"] = next(beta for label, beta, _note in LEASH if label == body.leash)
    if body.round == 2:
        settings.update(method=body.method, steps=body.steps, prompts_per_step=body.prompts_per_step, group=body.group, temperature=body.temperature,
                        max_tokens=body.max_tokens, kl=body.kl)
    job = studio_rl.round_job(body.round, version, settings)   # the Auto Run builds its rounds the same way
    params = job.params
    services.enqueue(job)
    if body.round == 3:  # a round 3 goes to the front of the queue, as the page does
        for _ in range(len(services.queue().up_next)):
            services.move_up(job.id)
    return RoundQueued(id=job.id, label=job.label, round=body.round, params=params)


@router.post("/stages", response_model=StageQueued, status_code=202)
def queue_stage(body: StageRequest, services=Depends(get_services)):
    """Queue mid-training from a trained version's best weights."""
    version = pick_version(services, body.version)
    if version is None or not version.best_checkpoint:
        raise HTTPException(409, "Choose a trained version with a checkpoint first.")
    label = f"Version {version.number} · {'mid-training'}"
    params = {"version": version.number}
    if body.steps is not None:
        params["config"] = {"steps": body.steps}
    job = Job(body.stage, label, f"{body.stage} stage", params=params)
    services.enqueue(job)
    return StageQueued(id=job.id, label=label, stage=body.stage)


@router.post("/rounds/{round}/adopt", response_model=AdoptedOut)
def adopt_round(round: int, body: AdoptRequest, services=Depends(get_services)):
    """Make the checkpoint a round kept (its gate failed, so it is not live) the version's best checkpoint: Chat and later rounds
    use it from now on. Refused with a sentence when the round passed, has not run, or kept no checkpoint file."""
    if not 1 <= round <= 4:
        raise HTTPException(404, f"There is no round {round}.")
    return AdoptedOut(**services.adopt_round(pick_version(services, body.version).number, round))


class LoopConfigIn(BaseModel):
    """The loop's settings (studio_loop.DEFAULTS); a field left out keeps the default. The first four are the stop rules."""
    max_rounds: int | None = Field(None, ge=1, le=100000)
    patience: int | None = Field(None, ge=1, le=1000)
    max_hours: float | None = Field(None, gt=0)
    audit_flags: int | None = Field(None, ge=1, le=100)
    prompts_per_round: int | None = Field(None, ge=2, le=5000)
    candidates: int | None = Field(None, ge=2, le=8)
    dpo_steps: int | None = Field(None, ge=1, le=5000)
    loss_type: Literal["dpo", "ipo"] | None = None
    beta: float | None = Field(None, ge=.001, le=10)
    max_length_ratio: float | None = Field(None, ge=1)


class LoopStart(BaseModel):
    version: int | None = None  # None = the current version
    resume: str | None = None  # the id of a loop to carry on from its last kept round (its settings count as this many MORE rounds)
    judge_model: str | None = None  # the Ollama model that grades; None = the workspace's teacher, or none (checked problems only)
    config: LoopConfigIn = LoopConfigIn()


class LoopQueued(BaseModel):
    job: str
    loop_id: str
    label: str


class LoopCheckOut(BaseModel):
    name: str
    detail: str = ""
    ok: bool | None = None  # None: could not be measured, never counts as a failure


class LoopRoundOut(BaseModel):
    """One finished round of a loop: the journal line, with all the evidence."""
    model_config = ConfigDict(extra="allow")
    round: int
    status: str  # kept | rolled_back | too_few_pairs
    reason: str = ""
    started: str = ""
    finished: str = ""
    seconds: float = 0.
    node: str = ""  # the machine that collected the round's pairs; "" = this one
    pairs: dict[str, Any] = Field(default_factory=dict)  # total, verified, comparison, rewrite, skipped {reason: n}, errors
    checks: list[LoopCheckOut] = Field(default_factory=list)
    before: dict[str, Any] | None = None  # the held-back evidence of the version the round started from
    after: dict[str, Any] | None = None
    judge: dict[str, Any] | None = None
    length_ratio: float | None = None
    audit: dict[str, Any] | None = None
    audit_flag: bool = False
    dpo: dict[str, Any] | None = None
    checkpoint: str | None = None


class LoopOut(BaseModel):
    """studio_loop.summary: one loop, on disk and live."""
    model_config = ConfigDict(extra="allow")
    id: str
    version: int | None = None
    status: str = "running"  # running | waiting_judge | stopped | finished | failed
    stop_reason: str = ""
    round: int = 1  # the round it is on, or will do next
    kept: int = 0
    failed_streak: int = 0
    audit_flagged: int = 0
    elapsed_s: float = 0.
    created: str | None = None
    updated: str | None = None
    phase: str = ""
    note: str = ""
    config: dict[str, Any] = Field(default_factory=dict)
    anchor: str = ""
    current: str = ""
    anchor_eval: dict[str, Any] | None = None
    current_eval: dict[str, Any] | None = None
    rounds: list[LoopRoundOut] = Field(default_factory=list)
    active: bool = False  # a job for it is queued or running now


class JudgesOut(BaseModel):
    url: str
    models: list[str]
    selected: str  # the workspace's teacher model
    error: str | None = None  # why Ollama could not be read, in words


def loop_jobs(services):
    """The queued or running rlhf_loop jobs."""
    state = services.queue()
    return [j for j in [state.job, *state.up_next] if j is not None and j.kind == "rlhf_loop"]


def loop_blocker(counts, has_judge, resume):
    """Why a new loop cannot start yet (plain words), or None. A loop that carries on already has its data."""
    if resume or has_judge or (counts.get("problems") or 0) >= 1:
        return None
    return "Make problems a rule can check in Your data, or choose an Ollama judge: the loop needs something to learn from."


@router.get("/loops", response_model=list[LoopOut])
def loops(services=Depends(get_services)):
    """Every automated loop, newest first, with each round's evidence."""
    active = {j.params.get("loop_id") for j in loop_jobs(services)}
    return [LoopOut(**{**loop, "active": loop.get("id") in active}) for loop in services.post_loops()]


@router.post("/loops", response_model=LoopQueued)
def start_loop(body: LoopStart, services=Depends(get_services)):
    """Queue the automated loop for a version, or carry on an earlier one from its last kept round. The loop is a long queue job:
    stop it from here and Start again to carry on; it also carries on by itself after an app restart."""
    version = pick_version(services, body.version)
    if version is None:
        raise ValueError("There is no version yet. Build one first.")
    counts = (services.post_training(version) or {}).get("counts", {})
    import studio_rlhf
    root = getattr(services, "root", None)
    has_judge = bool(body.judge_model or (root and studio_rlhf.load_settings(root).get("teacher_model")))
    why = loop_blocker(counts, has_judge, body.resume)
    if why:
        raise ValueError(why)
    config = {k: v for k, v in body.config.model_dump().items() if v is not None}
    return LoopQueued(**services.start_loop(version.number, config, body.resume, body.judge_model))


@router.post("/loops/{loop_id}/stop")
def stop_loop(loop_id: str, services=Depends(get_services)):
    """Stop a loop that is queued or running. It keeps its folder and its last kept round, and Start carries on from there."""
    jobs = [j for j in loop_jobs(services) if j.params.get("loop_id") == loop_id]
    if not jobs:
        raise ValueError("This loop is not queued or running.")
    for job in jobs:
        services.remove(job.id)
    return {"stopped": True}


@router.post("/loops/{loop_id}/adopt", response_model=JobOut, status_code=202)
def adopt_loop(request: Request, loop_id: str):
    """Make a loop's best the version's checkpoint if it holds up on ALL the held-back data. A job: it measures both models. The
    result is the gate result (ok, checks, verdict)."""
    services = request.app.state.services
    return start_job(request, "post_training.adopt_loop", lambda stop: services.adopt_loop(loop_id))


@router.get("/judges", response_model=JudgesOut)
def judges(services=Depends(get_services)):
    """The Ollama models that could grade a loop, read from the server the workspace's settings name."""
    import studio_rlhf
    root = getattr(services, "root", None)
    settings = studio_rlhf.load_settings(root) if root else dict(studio_rlhf.SETTINGS_DEFAULTS)
    try:
        models = studio_rlhf.OllamaTeacher(settings["url"], timeout=3, retries=0).models()
        error = None
    except Exception as exc:  # Ollama not running is an ordinary state, said in words
        models, error = [], str(exc)
    return JudgesOut(url=settings["url"], models=models, selected=settings.get("teacher_model", ""), error=error)


def keep_weights(services, name, upload):
    """Copy an uploaded model file into the workspace's pretrained folder, so the import job finds it instead of downloading it.
    The file must be the published one: its size and sha256 are checked as it is copied, and it is kept only when both match."""
    import studio_import
    root = getattr(services, "root", None)
    if root is None:
        raise ValueError("This server has no workspace to keep the file in.")
    info = studio_import.MODELS[name]
    dest = Path(root) / "data" / "pretrained" / name / "model.safetensors"
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    digest, size = hashlib.sha256(), 0
    try:
        with part.open("wb") as out:
            while chunk := upload.file.read(CHUNK):
                size += len(chunk)
                if size > info["size"]:
                    break
                digest.update(chunk)
                out.write(chunk)
        if size != info["size"] or digest.hexdigest() != studio_import.GPT2_SHA256:
            raise ValueError(f"That file is not the published {info['title']} weights. Check you chose "
                             f"{info['source']}'s model.safetensors, or leave the file out and the server downloads it.")
        part.replace(dest)
    finally:
        part.unlink(missing_ok=True)


@router.post("/imports", response_model=VersionOut)
def import_pretrained(name: str = Form("gpt2"), weights: UploadFile | None = File(None), lora: bool = Form(False),
                      lora_rank: int = Form(8), services=Depends(get_services)):
    """Start from a pretrained open model: saves its design as a new version and queues the conversion; the version trains
    (gets a checkpoint) when the job ends. `weights` is the model's published model.safetensors, for a server that cannot
    download it. `lora` fine-tunes small adapters instead of every weight."""
    import studio_import
    if name not in studio_import.MODELS:
        raise ValueError(f"Unknown pretrained model {name!r}.")
    if lora_rank not in LORA_RANKS:
        raise ValueError(f"LoRA rank is one of {', '.join(map(str, LORA_RANKS))}.")
    if weights is not None and weights.filename:
        keep_weights(services, name, weights)
    return version_out(services.import_pretrained(name, lora={"rank": lora_rank} if lora else None))
