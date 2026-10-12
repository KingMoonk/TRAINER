"""Teacher-LLM synthesis: a local Ollama model writes, mutates, answers or rephrases training data.

Modes (all return exactly what studio_synthesis.generate_batch returns and pass the same QualityGate):

  topic       diverse instruction -> answer pairs on topics/a goal, rotating task types and difficulty
  evolve      Evol-Instruct: mutate prompts from the working dataset, then have the teacher answer them
  answer      distillation: the teacher answers prompts taken from the dataset (or pasted by the user)
  paraphrase  reword prompts, copy the original answers byte-for-byte (safe for computed data)
  dialogue    2-4 exchange conversations on topics

The student is a tiny character-level transformer, so answers are capped (short/medium/long)
and can be forced to plain text. Computed/verified answers are never rewritten: `answer`
skips them, `paraphrase` copies them, and self-grading is not applied to them.

Determinism: topic/task/difficulty/operation choices come from a seeded RNG; the teacher's
own wording is only as deterministic as the sampling temperature allows (Ollama exposes no seed here).
"""
from __future__ import annotations

import json
import math
import random
import re
import threading
import time
import uuid
from collections import Counter, deque

from studio_core import messages_of, normalize
from studio_rlhf import OllamaTeacher
from studio_synthesis import ANSWER_LENGTHS, QualityGate, TEACHER_MODES, _source_id, validate_config

GENERATOR = "teacher-synthesis-v1"
TEMPLATE_IDS = {"topic": "topic-v1", "evolve": "evolve-v1", "answer": "answer-v1",
                "paraphrase": "paraphrase-v1", "dialogue": "dialogue-v1", "grade": "grade-v1"}
RETRIES = 2                 # extra attempts per call after the first one
RETRY_DELAY = .4            # seconds, multiplied by the attempt number
MAX_CONSECUTIVE_FAILURES = 4
PROMPT_CAPS = {"short": 300, "medium": 500, "long": 900}
DEFAULT_TEMPERATURE = {"topic": .8, "evolve": .7, "answer": .3, "paraphrase": .6, "dialogue": .8}

TASK_TYPES = {
    "explain": "explain a concept or fact clearly",
    "how-to": "give practical how-to instructions",
    "compare": "compare two things and say how they differ",
    "classify": "classify an item or sentence into a category and name it",
    "extract": "extract specific information from a short text that is included in the prompt",
    "rewrite": "rewrite or transform a short text that is included in the prompt (simplify, correct, change tone)",
    "brainstorm": "brainstorm a few ideas, names or examples",
    "short QA": "answer a short factual question",
    "step-by-step math": "solve a small math problem, showing the steps in plain sentences, ending with 'The answer is ...'",
}
DIFFICULTIES = {"easy": "easy, for a beginner", "medium": "medium difficulty", "hard": "hard, needing careful thought"}
EVOLVE_OPS = {
    "add_constraints": "Add one concrete extra constraint or requirement (for example a format, a limit or a condition).",
    "deepen": "Make the question deeper: ask for the reason, cause or consequence behind it.",
    "concretize": "Replace general wording with specific concrete details, numbers or named examples.",
    "reasoning": "Rewrite it so that answering needs several explicit reasoning steps.",
    "breadth": "Write a brand-new prompt of the same kind, but about a different, rarer topic in the same field.",
}
PARAPHRASE_STYLES = ["a polite, formal request", "a casual, conversational message", "a short, direct instruction",
                     "a detailed request that spells out what is wanted", "a question-style request"]

_ASCII = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-",
                        "…": "...", " ": " ", "→": "->"})
_REFUSAL = re.compile(r"^\W*(i'?m sorry|i am sorry|sorry,|i apologi[sz]e|as an ai\b|i cannot\b|i can'?t (help|assist|provide)|i'?m unable|i am unable)|as an ai language model", re.I)
_LABEL = re.compile(r"^\W*(answer|response|reply|assistant)\s*:\s*", re.I)
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


# --------------------------------------------------------------------------- text helpers
def plain_text(text):
    """Strip markdown and decoration so a character-level student sees only simple text."""
    text = str(text).translate(_ASCII)
    text = re.sub(r"```[A-Za-z0-9_+-]*\n?", "", text).replace("`", "")
    text = re.sub(r"(?<!\w)(\*\*|__)(?=\S)(.+?)(?<=\S)\1(?!\w)", r"\2", text, flags=re.S)
    text = re.sub(r"(?m)^[ \t]{0,3}#{1,6}[ \t]*", "", text)
    text = re.sub(r"(?m)^[ \t]*[-*•][ \t]+", "", text)
    text = re.sub("[\U00010000-\U0010ffff☀-➿]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def fit_to_cap(answer, cap):
    """Return (answer, trimmed) or (None, False) when it cannot be shortened to `cap` at a sentence end."""
    if len(answer) <= cap:
        return answer, False
    ends = [m.end() for m in re.finditer(r"[.!?](?=\s|$)", answer[:cap + 1]) if m.end() <= cap]
    if ends and ends[-1] >= cap * .4:
        return answer[:ends[-1]].strip(), True
    return None, False


def _records(reply, key):
    """Dict records from a JSON reply shaped as {key:[...]}, a bare list, or one record."""
    if isinstance(reply, list):
        items = reply
    elif isinstance(reply, dict):
        items = reply.get(key)
        if not isinstance(items, list):
            items = next((v for v in reply.values() if isinstance(v, list)), None)
        if items is None and any(k in reply for k in ("prompt", "instruction", "question", "user")):
            items = [reply]
    else:
        return []
    return [item for item in items or [] if isinstance(item, dict)]


def _pick(item, names):
    for name in names:
        value = item.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = str(value)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


PROMPT_KEYS = ("prompt", "instruction", "question", "user", "input", "task")
ANSWER_KEYS = ("answer", "response", "output", "assistant", "completion", "solution")


def _numbers(text):
    return Counter(_NUMBER.findall(text))


def _transcript(context):
    turns = [m for m in context if m["role"] != "system"]
    if len(turns) == 1:
        return turns[0]["content"]
    return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in turns)


# --------------------------------------------------------------------------- teacher calls
class _Calls:
    """Teacher access with retries, stop handling and error accounting."""
    def __init__(self, teacher, stop, stats, temperature):
        self.teacher, self.stop, self.stats, self.temperature = teacher, stop, stats, temperature
        self.consecutive = self.successes = 0
        self.last_error, self.connection_error = "", False

    def ask(self, system, user, json_mode=True, temperature=None):
        """Return the reply, or None when the call failed after retries (counted) or was stopped."""
        temperature = self.temperature if temperature is None else temperature
        error = None
        for attempt in range(RETRIES + 1):
            if self.stop.is_set():
                return None
            try:
                self.stats["teacher_calls"] += 1
                reply = self.teacher.chat(system, user, json_mode=json_mode, temperature=temperature)
                if json_mode and not isinstance(reply, (dict, list)):
                    raise ValueError("The teacher did not return JSON.")
                if not json_mode:
                    reply = reply if isinstance(reply, str) else str(reply)
                self.consecutive, self.successes = 0, self.successes + 1
                return reply
            except Exception as exc:     # network, HTTP, malformed JSON: all handled the same way
                error = exc
                if attempt < RETRIES:
                    self.stats["teacher_retries"] += 1
                    if self.stop.wait(RETRY_DELAY * (attempt + 1)):
                        return None
        self.stats["teacher_errors"] += 1
        self.consecutive += 1
        self.last_error = str(error)[:240]
        self.connection_error = "Cannot reach Ollama" in self.last_error or "not found" in self.last_error.lower()
        return None


# --------------------------------------------------------------------------- sources
def _sources(seeds, prompts):
    """Normalise dataset rows (and pasted prompts) into prompt/answer sources."""
    result = []
    for index, row in enumerate(seeds or []):
        if not isinstance(row, dict):
            continue
        clean = normalize(row)
        if clean and not clean.get("text"):
            messages = [dict(m) for m in clean["messages"]]
            row_id, quality, verified = row.get("id") or _source_id(row, index), row.get("quality", "Review"), bool(row.get("computedVerified"))
        else:
            prompt = next((str(row[k]).strip() for k in ("user", "prompt", "question", "instruction") if isinstance(row.get(k), str) and row[k].strip()), "")
            if not prompt or row.get("text"):
                continue
            messages, row_id, quality, verified = [{"role": "user", "content": prompt}], row.get("id") or _source_id(row, index), "Review", False
        last_user = max((i for i, m in enumerate(messages) if m["role"] == "user"), default=None)
        if last_user is None:
            continue
        answer = next((m["content"] for m in messages[last_user + 1:] if m["role"] == "assistant"), "")
        result.append({"id": str(row_id), "context": messages[:last_user + 1], "prompt": messages[last_user]["content"],
                       "answer": answer, "messages": messages, "quality": quality, "verified": verified,
                       "intent": str(row.get("intent", "general")),
                       "single": sum(m["role"] == "user" for m in messages) == 1})
    for index, prompt in enumerate(prompts or []):
        result.append({"id": f"prompt:{index}", "context": [{"role": "user", "content": prompt}], "prompt": prompt,
                       "answer": "", "messages": [{"role": "user", "content": prompt}], "quality": "Review",
                       "verified": False, "intent": "general", "single": True})
    return result


# --------------------------------------------------------------------------- the run
class _Run:
    def __init__(self, cfg, seeds, existing, stop, emit, teacher):
        self.cfg, self.mode, self.stop, self.emit = cfg, cfg["mode"], stop, emit
        self.rng = random.Random(cfg["seed"])
        self.count = cfg["count"]
        self.cap = ANSWER_LENGTHS[cfg["answer_length"]]
        self.prompt_cap = PROMPT_CAPS[cfg["answer_length"]]
        self.stats = Counter()
        temperature = cfg.get("teacher_temperature", DEFAULT_TEMPERATURE[self.mode])
        self.calls = _Calls(teacher, stop, self.stats, temperature)
        quality = dict(cfg["quality"])
        turns = cfg["max_turns"] if self.mode == "dialogue" else 1
        quality["max_answer_chars"] = min(quality["max_answer_chars"], self.cap * turns)
        self.gate = QualityGate(quality, existing, stop)
        self.batch_id = f"syn_{uuid.uuid4().hex[:12]}"
        self.rows, self.rejected = [], []
        self.reasons, self.coverage, self.difficulty_coverage = Counter(), Counter(), Counter()
        self.attempted = 0
        self.rounds = 0
        self.exhausted = False
        self.recent = deque(maxlen=12)
        self.cursor = 0
        topics = [t for t in cfg["topics"].splitlines() if t.strip()]
        if self.mode in {"topic", "dialogue"}:
            if not topics:
                raise ValueError("Describe what the teacher should write about (a goal or a few topics).")
            self.rng.shuffle(topics)
        self.topics = topics
        self.task_order = list(TASK_TYPES)
        self.rng.shuffle(self.task_order)
        sources = _sources(seeds, cfg["prompts"])
        if self.mode == "evolve":
            self.pool = [s for s in sources if s["single"] and s["prompt"].strip()]
        elif self.mode == "paraphrase":
            self.pool = [s for s in sources if s["single"] and s["answer"].strip()]
        elif self.mode == "answer":
            self.stats["skipped_computed"] = sum(1 for s in sources if s["verified"])
            keep = {"all": lambda s: True, "review": lambda s: s["quality"] != "Good" or not s["answer"],
                    "missing": lambda s: not s["answer"].strip()}[cfg["answer_filter"]]
            self.pool = [s for s in sources if not s["verified"] and keep(s)]
        else:
            self.pool = []
        if self.mode in {"evolve", "paraphrase", "answer"}:
            if not self.pool:
                raise ValueError({"evolve": "Choose a working dataset with examples (or paste prompts) to evolve.",
                                  "paraphrase": "Choose a working dataset with prompt/answer pairs to rephrase.",
                                  "answer": "No prompts match: choose a working dataset, paste prompts, or relax the filter."}[self.mode])
            self.rng.shuffle(self.pool)
        for source in self.rng.sample(sources, min(8, len(sources))) if sources else []:
            self.recent.append(source["prompt"][:80])

    # ---- prompts to the teacher
    def length_rule(self):
        return f"Every answer is at most {self.cap} characters. Be direct; no preamble."

    def style_rule(self):
        return ("Plain text only: no markdown, no asterisks, no backticks, no bullet symbols, no headings, no emoji."
                if self.cfg["plain_text"] else "Keep formatting minimal.")

    def clean(self, text):
        text = _LABEL.sub("", str(text).strip())
        return plain_text(text) if self.cfg["plain_text"] else text.strip()

    def take(self, n, cycle):
        """Next `n` pool sources. Cycling reshuffles after each pass; otherwise the pool is consumed once."""
        picked = []
        for _ in range(n):
            if self.cursor >= len(self.pool):
                if not cycle:
                    break
                self.rng.shuffle(self.pool)
                self.cursor = 0
            picked.append(self.pool[self.cursor])
            self.cursor += 1
        return picked

    def audience(self):
        return self.cfg["audience"] or "general learners"

    # ---- one round per mode: returns a list of candidates (dicts) or [] on failure
    def round_topic(self, n):
        topic = self.topics[self.rounds % len(self.topics)]
        task = self.task_order[self.rounds % len(self.task_order)]
        difficulty = self.rng.choices(["easy", "medium", "hard"], weights=[2, 3, 2])[0]
        avoid = "\n".join(f"- {p}" for p in self.recent) or "- (nothing yet)"
        system = ("You write training data for a very small language model that learns from scratch. "
                  'Reply with JSON only, shaped {"items":[{"prompt":"...","answer":"..."}]}. '
                  "Each prompt is a self-contained request a user could type. Each answer is correct, direct and complete. "
                  f"{self.length_rule()} {self.style_rule()} No placeholders, no mention of being an AI, no refusals. "
                  "Prompts must differ from each other in wording and subject.")
        user = (f"Write {n} different examples.\nSubject area: {topic}\nAudience: {self.audience()}\n"
                f"Task type: {TASK_TYPES[task]} (adapt it to the subject if it does not fit)\nDifficulty: {DIFFICULTIES[difficulty]}\n"
                f"Prompts must be under {self.prompt_cap} characters.\nAlready used - do not repeat or rephrase these:\n{avoid}")
        reply = self.calls.ask(system, user)
        out = []
        for item in _records(reply, "items") if reply is not None else []:
            prompt, answer = _pick(item, PROMPT_KEYS), _pick(item, ANSWER_KEYS)
            out.append({"family": task, "difficulty": difficulty, "prompt": prompt, "answer": answer, "intent": task,
                        "derivation": {"topic": topic, "task_type": task}})
        return out

    def answer_prompt(self, context):
        system = ("You write the ideal reply for a training example of a very small language model. "
                  f"{self.length_rule()} {self.style_rule()} Reply with the answer text only: no label, no quotes, no refusal, no mention of being an AI.")
        reply = self.calls.ask(system, _transcript(context), json_mode=False)
        return None if reply is None else self.clean(reply)

    def round_evolve(self, n):
        picked = self.take(n, cycle=True)
        ops = [self.rng.choice(list(EVOLVE_OPS)) for _ in picked]
        system = ("You rewrite prompts to make better training data. "
                  'Reply with JSON only, shaped {"items":[{"id":0,"prompt":"..."}]} with one item per input, same ids. '
                  f"Each rewritten prompt is self-contained, answerable in text, under {self.prompt_cap} characters, and is NOT answered here.")
        listing = "\n".join(f"id {i}: operation = {EVOLVE_OPS[op]}\nprompt: {s['prompt']}" for i, (s, op) in enumerate(zip(picked, ops)))
        reply = self.calls.ask(system, "Rewrite each prompt using its operation.\n\n" + listing)
        out = []
        if reply is None:
            return out
        evolved = {}
        for position, item in enumerate(_records(reply, "items")):
            try:
                index = int(item.get("id", position))
            except (TypeError, ValueError):
                index = position
            evolved.setdefault(index, _pick(item, PROMPT_KEYS))
        for index, (source, op) in enumerate(zip(picked, ops)):
            if self.stop.is_set():
                break
            prompt = self.clean(evolved.get(index, ""))
            if not prompt:
                out.append({"family": op, "prompt": "", "answer": ""})
                continue
            if prompt.casefold() == source["prompt"].strip().casefold():
                out.append({"family": op, "prompt": prompt, "answer": "x", "unchanged": True})
                continue
            answer = self.answer_prompt([{"role": "user", "content": prompt}])
            if answer is None:
                continue
            out.append({"family": op, "prompt": prompt, "answer": answer, "intent": source["intent"], "source_ids": [source["id"]],
                        "derivation": {"operation": op, "seed_prompt": source["prompt"][:300]}, "system": self._system_of(source)})
        return out

    def round_answer(self, n):
        out = []
        for source in self.take(n, cycle=False):
            if self.stop.is_set():
                break
            answer = self.answer_prompt(source["context"])
            if answer is None:
                continue
            context = [m for m in source["context"]]
            out.append({"family": "answer", "messages": context + [{"role": "assistant", "content": answer}], "prompt": source["prompt"],
                        "answer": answer, "intent": source["intent"], "source_ids": [source["id"]],
                        "derivation": {"method": "distillation", "had_answer": bool(source["answer"].strip()),
                                       "source_quality": source["quality"]}})
        if not out and self.cursor >= len(self.pool):
            self.exhausted = True
        return out

    def round_paraphrase(self, n):
        picked = self.take(n, cycle=True)
        style = self.rng.choice(PARAPHRASE_STYLES)
        system = ("You rephrase requests without changing their meaning. "
                  'Reply with JSON only, shaped {"items":[{"id":0,"prompt":"..."}]} with one item per input, same ids. '
                  "Keep every number, name, quoted text, code and data block exactly as written. Do not answer the requests.")
        listing = "\n".join(f"id {i}: {s['prompt']}" for i, s in enumerate(picked))
        reply = self.calls.ask(system, f"Rephrase each request as {style}.\n\n" + listing)
        out = []
        if reply is None:
            return out
        words = {}
        for position, item in enumerate(_records(reply, "items")):
            try:
                index = int(item.get("id", position))
            except (TypeError, ValueError):
                index = position
            words.setdefault(index, _pick(item, PROMPT_KEYS))
        for index, source in enumerate(picked):
            prompt = self.clean(words.get(index, ""))
            messages = [dict(m) for m in source["messages"]]
            for m in messages:
                if m["role"] == "user":
                    m["content"] = prompt
            reason = None
            if not prompt:
                reason = "malformed_item"
            elif prompt.casefold() == source["prompt"].strip().casefold():
                reason = "unchanged"
            elif _numbers(prompt) != _numbers(source["prompt"]) or any(q not in prompt for q in re.findall(r'"[^"\n]{1,200}"|`[^`\n]{1,200}`', source["prompt"])):
                reason = "meaning_changed"
            elif not .35 <= len(prompt) / max(1, len(source["prompt"])) <= 3.:
                reason = "meaning_changed"
            out.append({"family": "paraphrase", "messages": messages, "prompt": prompt, "answer": source["answer"], "reason": reason,
                        "intent": source["intent"], "source_ids": [source["id"]], "verified": source["verified"], "keep_answer": True,
                        "derivation": {"method": "paraphrase", "style": style, "original_prompt": source["prompt"][:400]}})
        return out

    def round_dialogue(self, n):
        topic = self.topics[self.rounds % len(self.topics)]
        turns = self.rng.randint(self.cfg["min_turns"], self.cfg["max_turns"])
        conversations = min(3, max(1, math.ceil(n / 2)))
        avoid = "\n".join(f"- {p}" for p in self.recent) or "- (nothing yet)"
        system = ("You write short conversations as training data for a very small language model. "
                  'Reply with JSON only, shaped {"conversations":[{"turns":[{"user":"...","assistant":"..."}]}]}. '
                  f"Each assistant message is at most {self.cap} characters, correct and direct. {self.style_rule()} "
                  "The user asks natural follow-up questions that depend on earlier turns. No refusals, no mention of being an AI.")
        user = (f"Write {conversations} different conversations, each with exactly {turns} user/assistant exchanges.\n"
                f"Subject area: {topic}\nAudience: {self.audience()}\nAlready used - do not repeat:\n{avoid}")
        reply = self.calls.ask(system, user)
        out = []
        for item in _records(reply, "conversations") if reply is not None else []:
            exchanges = item.get("turns", item.get("messages", item.get("exchanges")))
            pairs = []
            if isinstance(exchanges, list):
                if all(isinstance(t, dict) and ("user" in t or "assistant" in t) for t in exchanges):
                    pairs = [(_pick(t, ("user", "prompt", "question")), _pick(t, ANSWER_KEYS)) for t in exchanges]
                else:   # role/content list
                    flat = [(str(t.get("role", "")).lower(), str(t.get("content", "")).strip()) for t in exchanges if isinstance(t, dict)]
                    pairs = [(a[1], b[1]) for a, b in zip(flat[0::2], flat[1::2]) if a[0] == "user" and b[0] == "assistant"]
            out.append({"family": "dialogue", "pairs": pairs, "intent": "dialogue", "derivation": {"topic": topic, "turns_requested": turns}})
        return out

    @staticmethod
    def _system_of(source):
        return next((m["content"] for m in source["messages"] if m["role"] == "system"), "")

    # ---- shaping candidates into rows
    def shape(self, cand):
        """Return (messages, reason): clean, validate and cap one candidate."""
        if cand.get("reason"):
            return None, cand["reason"]
        if cand.get("unchanged"):
            return None, "unchanged"
        if self.mode == "dialogue":
            pairs = [(self.clean(u), self.clean(a)) for u, a in cand["pairs"]]
            if any(not u or not a for u, a in pairs) or len(pairs) < self.cfg["min_turns"]:
                return None, "bad_shape" if pairs else "malformed_item"
            pairs = pairs[:self.cfg["max_turns"]]
            messages = []
            for user, answer in pairs:
                if len(user) > self.prompt_cap:
                    return None, "prompt_too_long"
                if _REFUSAL.search(answer):
                    return None, "refusal"
                answer, trimmed = fit_to_cap(answer, self.cap)
                if answer is None:
                    return None, "answer_too_long"
                self.stats["trimmed"] += trimmed
                cand["trimmed"] = cand.get("trimmed", False) or trimmed
                messages += [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]
            return messages, None
        if cand.get("keep_answer"):                 # paraphrase: the answer is copied untouched, never trimmed or cleaned
            return cand["messages"], None
        prompt = cand["prompt"]
        if "messages" in cand:                      # answer mode: the context is already cleaned
            answer = cand["answer"]
            prefix = cand["messages"][:-1]
        else:
            prompt, answer = self.clean(cand.get("prompt", "")), self.clean(cand.get("answer", ""))
            prefix = [{"role": "user", "content": prompt}]
            if cand.get("system"):
                prefix.insert(0, {"role": "system", "content": cand["system"]})
        if not prompt.strip() or not answer.strip():
            return None, "malformed_item"
        if len(prompt) > self.prompt_cap:
            return None, "prompt_too_long"
        if _REFUSAL.search(answer):
            return None, "refusal"
        if answer.strip().casefold() == prompt.strip().casefold():
            return None, "echo"
        answer, trimmed = fit_to_cap(answer, self.cap)
        if answer is None:
            return None, "answer_too_long"
        self.stats["trimmed"] += trimmed
        cand["trimmed"] = trimmed
        return prefix + [{"role": "assistant", "content": answer}], None

    def reject(self, reason, cand, row=None):
        self.reasons[reason] += 1
        if len(self.rejected) < 100:
            shown = row or {}
            self.rejected.append({"reason": reason, "family": cand.get("family", self.mode),
                                  "user": (shown.get("user") or cand.get("prompt") or "")[:1000],
                                  "assistant": (shown.get("assistant") or cand.get("answer") or "")[:1000]})

    def prepare(self, cand):
        messages, reason = self.shape(cand)
        if reason:
            self.reject(reason, cand)
            return None
        system = self.cfg.get("system_prompt")
        if system and messages[0]["role"] != "system":
            messages = [{"role": "system", "content": system}] + messages
        row = normalize({"messages": messages, "intent": cand.get("intent", self.mode)}, source="synthetic")
        reason = self.gate.inspect(row, check_conflicts=False if self.mode == "answer" else None)
        if reason:
            self.reject(reason, cand, row)
            return None
        return row

    # ---- optional self-grading
    def grade(self, pending):
        """pending: [(cand,row)]. Returns the surviving [(cand,row)] after the teacher scores each example 1-10."""
        gradable = [(c, r) for c, r in pending if not c.get("verified")]
        scores = {}
        system = ("You are a strict reviewer of training examples for a small model. Score each example from 1 to 10 for "
                  "correctness, clarity, and whether the answer actually answers the prompt (10 = perfect, 1 = wrong or useless). "
                  'Reply with JSON only, shaped {"grades":[{"id":0,"score":7,"reason":"short reason"}]} with one grade per example id.')
        for _ in range(2):      # a second pass re-asks only for examples the teacher skipped
            todo = [(c, r) for c, r in gradable if id(r) not in scores]
            for start in range(0, len(todo), 5):
                if self.stop.is_set():
                    break
                chunk = todo[start:start + 5]
                listing = "\n\n".join(f"id {i}\nPROMPT: {r['user'][:600]}\nANSWER: {r['assistant'][:800]}" for i, (_, r) in enumerate(chunk))
                reply = self.calls.ask(system, listing, temperature=0.)
                if reply is None:
                    self.stats["grade_errors"] += 1
                    continue
                for position, item in enumerate(_records(reply, "grades")):
                    try:
                        index = int(item.get("id", position))
                        score = max(1., min(10., float(item.get("score"))))
                    except (TypeError, ValueError):
                        continue
                    if 0 <= index < len(chunk):
                        scores[id(chunk[index][1])] = (score, str(item.get("reason", ""))[:200])
        survivors = []
        for cand, row in pending:
            graded = scores.get(id(row))
            if graded is not None:
                cand["score"], cand["rationale"] = graded
                self.stats["graded"] += 1
                self.stats["score_total"] += graded[0]
                if graded[0] < self.cfg["grade_threshold"]:
                    self.reject("low_score", cand, row)
                    continue
            survivors.append((cand, row))
        return survivors

    # ---- accepting
    def accept(self, cand, row):
        row["quality"] = "Review"
        verified = bool(cand.get("verified"))
        row["computedVerified"] = verified
        if self.mode == "paraphrase":
            warnings = ["Prompt reworded by the teacher (numbers and quoted text checked); the answer is copied unchanged."]
            if not verified:
                warnings.append("Target correctness requires human review.")
        else:
            warnings = [] if verified else ["Written by a teacher model; correctness requires human review."]
        if cand.get("trimmed"):
            warnings.append("Answer was shortened to fit the length limit.")
        report = {"status": "needs_review", "automated_checks": "passed", "computed_target": verified, "warnings": warnings,
                  "teacher_model": self.cfg["teacher_model"]}
        if "score" in cand:
            report.update(teacher_score=cand["score"], teacher_rationale=cand.get("rationale", ""))
        elif self.cfg["self_grade"] and not verified:
            report["warnings"].append("Self-grade unavailable for this example.")
        row["qualityReport"] = report
        derivation = dict(cand.get("derivation", {}))
        if cand.get("trimmed"):
            derivation["trimmed"] = True
        row["provenance"] = {"generator": GENERATOR, "recipe": self.mode, "mode": self.mode, "seed": self.cfg["seed"],
                             "batch_id": self.batch_id, "attempt": self.attempted, "difficulty": cand.get("difficulty", ""),
                             "teacher_model": self.cfg["teacher_model"], "template_id": TEMPLATE_IDS[self.mode],
                             "answer_length": self.cfg["answer_length"], "plain_text": self.cfg["plain_text"],
                             "source_ids": cand.get("source_ids", []), "derivation": derivation, "synthetic": True}
        self.gate.remember(row)
        self.rows.append(row)
        self.coverage[cand.get("family", self.mode)] += 1
        if cand.get("difficulty"):
            self.difficulty_coverage[cand["difficulty"]] += 1
        self.recent.append(next((m["content"] for m in row["messages"] if m["role"] == "user"), "")[:80])

    def progress(self, text):
        self.emit({"type": "progress", "completed": len(self.rows), "total": self.count, "attempted": self.attempted, "text": text})

    def run(self):
        started = time.monotonic()
        batch = self.cfg["batch_size"]
        round_limit = math.ceil(self.count / batch) * self.cfg["attempt_multiplier"] + 2
        candidate_limit = self.count * self.cfg["attempt_multiplier"]
        produce = getattr(self, "round_" + self.mode)
        abort = ""
        while len(self.rows) < self.count and not self.stop.is_set() and self.rounds < round_limit and self.attempted < candidate_limit and not self.exhausted:
            n = min(batch, max(self.count - len(self.rows), 3))
            self.progress(f"{len(self.rows):,}/{self.count:,} accepted · asking {self.cfg['teacher_model'] or 'teacher'} ({self.mode})…")
            failures_before = self.stats["teacher_errors"]
            cands = produce(n)
            self.rounds += 1
            if self.stats["teacher_errors"] > failures_before and not cands:
                if self.calls.successes == 0 and (self.calls.connection_error or self.calls.consecutive >= MAX_CONSECUTIVE_FAILURES):
                    raise ValueError(f"The teacher could not be used: {self.calls.last_error}")
                if self.calls.consecutive >= MAX_CONSECUTIVE_FAILURES:
                    abort = f"Stopped after {self.calls.consecutive} failed teacher requests in a row ({self.calls.last_error[:120]})."
                    break
            pending = []
            for cand in cands:
                if len(self.rows) + len(pending) >= self.count:
                    break
                self.attempted += 1
                row = self.prepare(cand)
                if row is not None:
                    pending.append((cand, row))
            if self.cfg["self_grade"] and pending:
                pending = self.grade(pending)
            for cand, row in pending:
                if len(self.rows) >= self.count:
                    break
                reason = self.gate.inspect(row, check_conflicts=False if self.mode == "answer" else None)   # catches repeats inside one round
                if reason:
                    self.reject(reason, cand, row)
                else:
                    self.accept(cand, row)
            if not cands and not self.stats["teacher_errors"] and not self.stop.is_set():
                self.stats["empty_rounds"] += 1
        graded = self.stats["graded"]
        stats = {"requested": self.count, "attempted": self.attempted, "accepted": len(self.rows),
                 "duplicates": self.reasons["duplicate"] + self.reasons["near_duplicate"], "conflicts": self.reasons["prompt_conflict"],
                 "rejected": sum(self.reasons.values()), "reasons": dict(self.reasons),
                 "elapsed": round(time.monotonic() - started, 3), "coverage": dict(self.coverage),
                 "difficulty_coverage": dict(self.difficulty_coverage), "budgets": {self.mode: self.count},
                 "attempts_by_family": {self.mode: self.attempted}, "cancelled": self.stop.is_set(),
                 "shortfall": self.count - len(self.rows), "exhausted_families": [self.mode] if self.exhausted else [],
                 "near_duplicate_scope": "Approximate: recent 10,000 records; up to 256 candidates per record.",
                 "mode": self.mode, "teacher_model": self.cfg["teacher_model"], "rounds": self.rounds,
                 "teacher_calls": self.stats["teacher_calls"], "teacher_errors": self.stats["teacher_errors"],
                 "teacher_retries": self.stats["teacher_retries"], "malformed": self.reasons["malformed_item"],
                 "graded": graded, "mean_score": round(self.stats["score_total"] / graded, 2) if graded else None,
                 "grade_errors": self.stats["grade_errors"], "trimmed": self.stats["trimmed"],
                 "skipped_computed": self.stats["skipped_computed"], "abort_reason": abort}
        self.progress(("Cancelled; partial batch preserved. " if self.stop.is_set() else "Batch ready for review. ") +
                      f"{len(self.rows):,} accepted; {stats['rejected']:,} filtered.")
        return {"rows": self.rows, "rejected": self.rejected, "stats": stats, "config": self.cfg, "batch_id": self.batch_id}


def generate_teacher_batch(config, seeds=None, existing=None, stop=None, emit=None, teacher=None):
    """Generate a review batch with a teacher LLM. Same return shape as studio_synthesis.generate_batch."""
    cfg = validate_config(config)
    if cfg["mode"] not in TEACHER_MODES:
        raise ValueError("Not a teacher mode.")
    stop = stop if stop is not None else threading.Event()
    emit = emit or (lambda event: None)
    if teacher is None:
        if not cfg["teacher_model"]:
            raise ValueError("Choose a teacher model (use Refresh models to list the ones Ollama has).")
        teacher = OllamaTeacher(cfg["teacher_url"], cfg["teacher_model"])
    emit({"type": "progress", "completed": 0, "total": cfg["count"], "attempted": 0, "text": "Indexing existing examples for duplicate checks…"})
    return _Run(cfg, seeds, existing, stop, emit, teacher).run()


def list_teacher_models(url, timeout=6):
    """Model names served by Ollama at `url` (raises RuntimeError with a plain message if unreachable)."""
    teacher = OllamaTeacher(url, "")
    try:
        teacher.timeout = timeout
    except AttributeError:
        pass
    return teacher.models()
