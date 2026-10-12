"""Reproducible synthetic curricula with computed targets and inspectable provenance.

Procedural targets are calculated, source tasks copy supplied evidence, model
drafts use a user-selected local checkpoint, and the optional teacher modes
(see studio_teacher_synth) ask a *local* Ollama model to write examples.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from datetime import date, timedelta
import hashlib
import json
import math
import random
import re
import string
import threading
import time
import uuid

from studio_core import messages_of, normalize, now


FAMILIES = ["arithmetic", "word_problems", "logic", "list_ops", "string_ops",
            "json_extract", "table_qa", "code_trace", "calendar"]
# Source remixes reuse answers that already exist in the sampled dataset.
REMIX_FAMILIES = ["final_answer", "clean_solution", "json_answer", "multi_turn", "context_first"]
RECIPE_PRESETS = [
    {"id": "balanced", "name": "Balanced reasoning curriculum", "description": "Nine computed task families, progressing from easy to hard.",
     "config": {"mode": "mixed", "difficulty": "curriculum", "families": FAMILIES, "count": 2000}},
    {"id": "numbers", "name": "Math and word problems", "description": "Exact arithmetic, multi-step stories, tables and dates.",
     "config": {"mode": "reasoning", "families": ["arithmetic", "word_problems", "table_qa", "calendar"], "count": 1000}},
    {"id": "logic", "name": "Logic and code traces", "description": "Truth tables, sorting, transformations and executable-style traces.",
     "config": {"mode": "reasoning", "families": ["logic", "list_ops", "string_ops", "code_trace"], "count": 1000}},
    {"id": "structured", "name": "Strict JSON extraction", "description": "Nested records with machine-checked JSON answers.",
     "config": {"mode": "structured", "families": ["json_extract"], "quality": {"require_json": True}, "count": 1000}},
    {"id": "remix", "name": "Dataset remix (real sources)", "description": "Final-answer, clean-solution, JSON-format, multi-turn and context-first variants of downloaded records. Targets are reused from the source.",
     "config": {"mode": "remix", "families": REMIX_FAMILIES, "count": 2000}},
    {"id": "grounded", "name": "Grounded source tasks", "description": "Extract exact passages and answers from your selected dataset.",
     "config": {"mode": "grounded", "count": 500}},
    {"id": "facts", "name": "Correlated fact templates", "description": "Each row keeps related facts together; review your supplied facts.",
     "config": {"mode": "templates", "prompt": "What is the capital of {country}?", "response": "{capital}",
                "fact_rows": [{"country": "France", "capital": "Paris"}, {"country": "Japan", "capital": "Tokyo"}], "count": 100}},
    {"id": "variants", "name": "Seed conversation variants", "description": "Conservative request wrappers preserve original answers and conversation turns.",
     "config": {"mode": "seeds", "count": 500}},
    {"id": "teach_topic", "name": "Teacher model: write examples about a topic", "description": "A local Ollama model writes short instruction/answer pairs on your topics, then quality gates filter them.",
     "config": {"mode": "topic", "count": 200, "answer_length": "short", "plain_text": True, "topics": "basic arithmetic word problems"}},
    {"id": "teach_dialogue", "name": "Teacher model: short conversations", "description": "A local Ollama model writes 2-4 turn conversations on your topics.",
     "config": {"mode": "dialogue", "count": 100, "answer_length": "short", "plain_text": True, "topics": "everyday questions"}},
    {"id": "grow_evolve", "name": "Grow dataset: evolve harder prompts", "description": "Mutates prompts from your working dataset (constraints, depth, detail, reasoning) and has the teacher answer them.",
     "config": {"mode": "evolve", "count": 200, "answer_length": "short", "plain_text": True}},
    {"id": "grow_paraphrase", "name": "Grow dataset: rephrase prompts, keep answers", "description": "Safe for computed data: prompts are reworded, answers are copied exactly.",
     "config": {"mode": "paraphrase", "count": 200}},
    {"id": "grow_answer", "name": "Grow dataset: teacher answers my prompts", "description": "Distillation: the teacher answers prompts from your dataset (all, only unreviewed, or only unanswered).",
     "config": {"mode": "answer", "count": 200, "answer_length": "short", "plain_text": True, "answer_filter": "review"}},
    {"id": "local_model", "name": "Your model as a draft teacher", "description": "Sample your own checkpoint with quality gates; all answers need review.",
     "config": {"mode": "model", "count": 100, "temperature": .8, "top_k": 40, "max_tokens": 180}},
]

# Teacher-LLM modes are implemented in studio_teacher_synth (imported lazily).
TEACHER_MODES = ("topic", "evolve", "answer", "paraphrase", "dialogue")
ANSWER_LENGTHS = {"short": 220, "medium": 600, "long": 1500}   # character caps for the tiny student
DEFAULT_TEACHER_URL = "http://localhost:11434"

DEFAULT_QUALITY = {"min_answer_chars": 1, "max_answer_chars": 20000,
                   "max_repeat_fraction": .85, "require_json": False,
                   "near_duplicate_threshold": 0., "check_conflicts": True}


def _integer(value, name, minimum, maximum):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer.")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be an integer.") from None
    if str(value).strip() != str(result) and not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return result


def _number(value, name, minimum, maximum):
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number.") from None
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return value


def _flag(value, name):
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false.")
    return value


def _lines(value, name, limit=5000):
    if value is None:
        return []
    if isinstance(value, str):
        value = value.splitlines()
    if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
        raise ValueError(f"{name} must be text, one item per line.")
    result = [x.strip() for x in value if x.strip()]
    if len(result) > limit:
        raise ValueError(f"{name} may list at most {limit:,} items.")
    return result


def _validate_teacher(cfg):
    """Validate the teacher-model options in place (only used for TEACHER_MODES)."""
    url = cfg.get("teacher_url") or DEFAULT_TEACHER_URL
    if not isinstance(url, str) or not re.match(r"^https?://\S+$", url.strip()):
        raise ValueError("The teacher URL must look like http://localhost:11434.")
    cfg["teacher_url"] = url.strip().rstrip("/")
    model = cfg.get("teacher_model", "")
    if not isinstance(model, str):
        raise ValueError("The teacher model must be a model name.")
    cfg["teacher_model"] = model.strip()
    cfg["topics"] = "\n".join(_lines(cfg.get("topics", ""), "Topics", 200))
    audience = cfg.get("audience", "")
    if not isinstance(audience, str) or len(audience) > 300:
        raise ValueError("The audience must be a short description (at most 300 characters).")
    cfg["audience"] = audience.strip()
    cfg["answer_length"] = cfg.get("answer_length", "short")
    if cfg["answer_length"] not in ANSWER_LENGTHS:
        raise ValueError("Answer length must be short, medium or long.")
    cfg["plain_text"] = _flag(cfg.get("plain_text", True), "plain_text")
    cfg["self_grade"] = _flag(cfg.get("self_grade", False), "self_grade")
    cfg["grade_threshold"] = _number(cfg.get("grade_threshold", 6), "grade_threshold", 1, 10)
    cfg["batch_size"] = _integer(cfg.get("batch_size", 5), "batch_size", 1, 10)
    cfg["answer_filter"] = cfg.get("answer_filter", "all")
    if cfg["answer_filter"] not in {"all", "review", "missing"}:
        raise ValueError("Answer filter must be all, review or missing.")
    cfg["prompts"] = _lines(cfg.get("prompts"), "Prompts")
    cfg["min_turns"] = _integer(cfg.get("min_turns", 2), "min_turns", 1, 6)
    cfg["max_turns"] = _integer(cfg.get("max_turns", 4), "max_turns", cfg["min_turns"], 6)
    if cfg.get("teacher_temperature") is not None:
        cfg["teacher_temperature"] = _number(cfg["teacher_temperature"], "teacher_temperature", 0., 2.)
    else:
        cfg.pop("teacher_temperature", None)


def validate_config(config):
    """Return a JSON-serializable recipe, rejecting ambiguous or unsafe budgets."""
    if not isinstance(config, dict):
        raise ValueError("A recipe must be a JSON object.")
    cfg = dict(config)
    cfg["mode"] = cfg.get("mode", "mixed")
    if cfg["mode"] not in {"mixed", "reasoning", "arithmetic", "structured", "templates", "grounded", "seeds", "model", "remix", *TEACHER_MODES}:
        raise ValueError("Unknown synthesis mode.")
    cfg["count"] = _integer(cfg.get("count", 500), "count", 1, 100000)
    cfg["seed"] = _integer(cfg.get("seed", 42), "seed", 0, 2**63 - 1)
    cfg["attempt_multiplier"] = _integer(cfg.get("attempt_multiplier", 6), "attempt_multiplier", 1, 30)
    cfg["difficulty"] = cfg.get("difficulty", "curriculum")
    if cfg["difficulty"] not in {"easy", "medium", "hard", "curriculum"}:
        raise ValueError("Difficulty must be easy, medium, hard, or curriculum.")
    defaults = ["json_extract"] if cfg["mode"] == "structured" else REMIX_FAMILIES if cfg["mode"] == "remix" else [cfg["mode"]] if cfg["mode"] in TEACHER_MODES else FAMILIES
    cfg["families"] = cfg.get("families", defaults)
    allowed = REMIX_FAMILIES if cfg["mode"] == "remix" else FAMILIES + ["grounded", "seeds", "templates", *TEACHER_MODES]
    if (not isinstance(cfg["families"], list) or not cfg["families"] or
            any(name not in allowed for name in cfg["families"])):
        raise ValueError("Choose at least one valid task family.")
    cfg["families"] = list(dict.fromkeys(cfg["families"]))
    cfg["weights"] = cfg.get("weights", {})
    if not isinstance(cfg["weights"], dict):
        raise ValueError("Task weights must be an object of positive numbers.")
    cfg["weights"] = dict(cfg["weights"])
    for name, weight in cfg["weights"].items():
        if name not in FAMILIES + REMIX_FAMILIES + ["grounded", "seeds", "templates", *TEACHER_MODES]:
            raise ValueError(f"Unknown weighted task family: {name}.")
        cfg["weights"][name] = _number(weight, f"weight for {name}", .001, 1000000)
    quality = cfg.get("quality", {})
    if not isinstance(quality, dict):
        raise ValueError("Quality settings must be an object.")
    quality = {**DEFAULT_QUALITY, **quality}
    quality["min_answer_chars"] = _integer(quality["min_answer_chars"], "Minimum answer characters", 1, 1000000)
    quality["max_answer_chars"] = _integer(quality["max_answer_chars"], "Maximum answer characters", quality["min_answer_chars"], 1000000)
    quality["max_repeat_fraction"] = _number(quality["max_repeat_fraction"], "Maximum repeat fraction", .05, 1.)
    quality["near_duplicate_threshold"] = _number(quality["near_duplicate_threshold"], "Near-duplicate threshold", 0., 1.)
    for flag in ("check_conflicts", "require_json"):
        if not isinstance(quality[flag], bool):
            raise ValueError(f"{flag} must be true or false.")
    cfg["quality"] = quality
    cfg["temperature"] = _number(cfg.get("temperature", .8), "temperature", 0., 3.)
    cfg["top_k"] = _integer(cfg.get("top_k", 40), "top_k", 0, 100000)
    cfg["max_tokens"] = _integer(cfg.get("max_tokens", 180), "max_tokens", 1, 4096)
    cfg["device"] = cfg.get("device", "auto")
    if cfg["device"] not in {"auto", "cpu", "cuda"}:
        raise ValueError("Device must be auto, cpu, or cuda.")
    system_prompt = cfg.get("system_prompt")
    if system_prompt is not None:
        if not isinstance(system_prompt, str) or len(system_prompt) > 2000:
            raise ValueError("The system prompt must be text of at most 2,000 characters.")
        if system_prompt.strip():
            cfg["system_prompt"] = system_prompt.strip()
        else:
            cfg.pop("system_prompt")
    if cfg["mode"] in TEACHER_MODES:
        _validate_teacher(cfg)
    # Reject non-JSON objects before creating an expensive model or starting work.
    return json.loads(json.dumps(cfg, ensure_ascii=False, allow_nan=False))


def _compact(text):
    return " ".join(str(text).casefold().split())


def _content_keys(row):
    turns = messages_of(row)
    if row.get("text"):
        content = [("text", _compact(row["text"]))]
        prompt, answer = content, ""
    else:
        content = [(m["role"], _compact(m["content"])) for m in turns]
        complete = bool(content and content[-1][0] == "assistant")
        prompt, answer = content[:-1] if complete else content, content[-1][1] if complete else ""
    encode = lambda value: hashlib.sha256(json.dumps(value, ensure_ascii=False).encode()).digest()
    return encode(content), encode(prompt), answer


def _shingles(text):
    words = re.findall(r"\w+|[^\w\s]", _compact(text))[:2048]
    # Numbers and punctuation are deliberately retained: 2+3 and 2+4 differ.
    width = 3 if len(words) >= 3 else 1
    return {hashlib.blake2b("\x00".join(words[i:i + width]).encode(), digest_size=8).digest()
            for i in range(max(0, len(words) - width + 1))}


class QualityGate:
    """Exact/conflict checks plus optional bounded, approximate near-duplicate search."""
    def __init__(self, quality=None, existing=None, stop=None):
        self.cfg = {**DEFAULT_QUALITY, **(quality or {})}
        self.seen, self.answers = set(), {}
        self.signatures, self.postings, self.order = {}, defaultdict(set), deque()
        self.serial = 0
        for row in existing or []:
            if stop is not None and stop.is_set():
                break
            self.remember(row)

    def remember(self, row):
        key, prompt, answer = _content_keys(row)
        self.seen.add(key)
        if answer:
            # A sentinel records existing inconsistencies instead of ignoring them.
            previous = self.answers.get(prompt, answer)
            self.answers[prompt] = answer if previous == answer else None
        if self.cfg["near_duplicate_threshold"] <= 0:
            return
        shingles = _shingles("\n".join(m["content"] for m in messages_of(row)))
        serial = self.serial
        self.serial += 1
        self.signatures[serial] = shingles
        selected = sorted(shingles)[:12]
        self.order.append((serial, selected))
        for signature in selected:
            self.postings[signature].add(serial)
        # Bound both memory and candidate comparisons. Exact checks remain global.
        while len(self.order) > 10000:
            old, selected = self.order.popleft()
            self.signatures.pop(old, None)
            for signature in selected:
                self.postings[signature].discard(old)
                if not self.postings[signature]:
                    del self.postings[signature]

    def inspect(self, row, check_conflicts=None):
        """Return the first reason to reject `row`, or None. `check_conflicts` overrides the gate setting."""
        if not row or not row.get("assistant", "").strip():
            return "empty_answer"
        if not row.get("messages") or row["messages"][-1]["role"] != "assistant":
            return "incomplete_conversation"
        answer = row["assistant"]
        if len(answer) < self.cfg["min_answer_chars"]:
            return "answer_too_short"
        if len(answer) > self.cfg["max_answer_chars"]:
            return "answer_too_long"
        if self.cfg["require_json"]:
            try:
                json.loads(answer, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
            except (TypeError, ValueError):
                return "invalid_json"
        words = re.findall(r"\w+", answer.casefold())
        if len(words) >= 12 and max(Counter(words).values()) / len(words) > self.cfg["max_repeat_fraction"]:
            return "repetitive_answer"
        key, prompt, target = _content_keys(row)
        if key in self.seen:
            return "duplicate"
        if (self.cfg["check_conflicts"] if check_conflicts is None else check_conflicts) and prompt in self.answers and self.answers[prompt] != target:
            return "prompt_conflict"
        threshold = self.cfg["near_duplicate_threshold"]
        if threshold > 0:
            shingles = _shingles("\n".join(m["content"] for m in messages_of(row)))
            candidates = set()
            for signature in sorted(shingles)[:12]:
                candidates.update(sorted(self.postings.get(signature, ()), reverse=True)[:64])
                if len(candidates) >= 256:
                    break
            for serial in sorted(candidates, reverse=True)[:256]:
                other = self.signatures[serial]
                union = len(shingles | other)
                if union and len(shingles & other) / union >= threshold:
                    return "near_duplicate"
        return None


def _example(user, answer, family, verified=True, **extra):
    return {"user": user, "assistant": str(answer), "intent": family,
            "computedVerified": verified, **extra}


def _procedural(family, rng, difficulty):
    level = {"easy": 1, "medium": 2, "hard": 3}[difficulty]
    limit = (20, 200, 5000)[level - 1]
    a, b = rng.randint(0, limit), rng.randint(1, limit)
    if family == "arithmetic":
        op = rng.choice(["+", "-", "*", "//", "%"] if level > 1 else ["+", "-", "*"])
        answer = {"+": lambda: a + b, "-": lambda: a - b, "*": lambda: a * b,
                  "//": lambda: a // b, "%": lambda: a % b}[op]()
        semantics = " // means integer floor division; % means remainder." if op in {"//", "%"} else ""
        return _example(f"Calculate {a} {op} {b}.{semantics} Return only the integer.", answer, family,
                        derivation={"operands": [a, b], "operation": op, "result": answer})
    if family == "word_problems":
        name = rng.choice(["Ari", "Mina", "Dev", "Sam", "Noor", "Leo"])
        item = rng.choice(["notebooks", "apples", "tokens", "stickers", "marbles"])
        if level == 1:
            text, answer = f"{name} has {a} {item} and receives {b} more. How many {item} does {name} now have?", a + b
        elif level == 2:
            removed = rng.randint(0, a * b)
            text, answer = f"{name} fills {a} boxes with {b} {item} each, then gives away {removed} {item}. How many remain?", a * b - removed
        else:
            boxes, per_box, removed, people = rng.randint(2, 30), rng.randint(2, 40), rng.randint(0, 20), rng.randint(2, 9)
            total = boxes * per_box + removed
            text = f"{name} has {boxes} boxes of {per_box} {item} and {removed} loose {item}. They share these equally among {people} people, keeping any remainder. How many {item} does each person get?"
            answer = total // people
        return _example(text + " Return only the integer.", answer, family)
    if family == "logic":
        values = {name: bool(rng.getrandbits(1)) for name in "ABCDE"[:level + 2]}
        x, y = rng.sample(list(values), 2)
        z = rng.choice(list(values))
        op = rng.choice(["AND", "OR", "XOR"])
        answer = values[x] and values[y] if op == "AND" else values[x] or values[y] if op == "OR" else values[x] != values[y]
        expression = f"({x} {op} {y})"
        if level > 1:
            expression, answer = f"NOT {expression}", not answer
        if level > 2:
            expression, answer = f"({expression}) AND {z}", answer and values[z]
        table = ", ".join(f"{k}={'true' if v else 'false'}" for k, v in values.items())
        return _example(f"Given {table}, evaluate {expression}. XOR is true when exactly one input is true. Answer true or false.", str(answer).lower(), family)
    if family == "list_ops":
        values = [rng.randint(-limit, limit) for _ in range(level * 3 + 2)]
        op = rng.choice(["sort ascending", "sort descending", "reverse", "keep even numbers", "remove duplicates, preserving first occurrence"])
        if op == "sort ascending": answer = sorted(values)
        elif op == "sort descending": answer = sorted(values, reverse=True)
        elif op == "reverse": answer = list(reversed(values))
        elif op == "keep even numbers": answer = [x for x in values if x % 2 == 0]
        else: answer = list(dict.fromkeys(values))
        return _example(f"For {json.dumps(values)}, {op}. Return only a JSON array.", json.dumps(answer), family)
    if family == "string_ops":
        alphabet = string.ascii_letters + string.digits
        word = "".join(rng.choice(alphabet) for _ in range(level * 4 + 3))
        op = rng.choice(["reverse the characters", "convert to uppercase", "convert to lowercase", "sort the characters by Unicode code point"])
        answer = word[::-1] if op == "reverse the characters" else word.upper() if op == "convert to uppercase" else word.lower() if op == "convert to lowercase" else "".join(sorted(word))
        return _example(f"Input string: {json.dumps(word)}. Please {op}. Return only the resulting string, without quotation marks.", answer, family)
    if family == "json_extract":
        record = {"id": f"item-{rng.randrange(10000000):07d}", "name": rng.choice(["amber", "cedar", "coral", "jade", "silver"]), "quantity": a, "active": bool(rng.getrandbits(1))}
        if level >= 2:
            record["location"] = {"shelf": rng.randint(1, 90), "zone": rng.choice(list("ABCDE"))}
        if level >= 3:
            record["tags"] = rng.sample(["new", "small", "fragile", "indoor", "stocked"], 3)
        paths = list(record)
        if level >= 2: paths += ["location.shelf", "location.zone"]
        field = rng.choice(paths)
        answer = record
        for part in field.split("."):
            answer = answer[part]
        return _example(f"Record: {json.dumps(record, ensure_ascii=False)}\nExtract the value at {field}. Return only valid JSON, including quotation marks around string values.", json.dumps(answer, ensure_ascii=False), family,
                        derivation={"record": record, "field": field, "result": answer})
    if family == "table_qa":
        records = [{"item": f"item_{i + 1}", "price": rng.randint(1, limit), "units": rng.randint(1, 15)} for i in range(level + 2)]
        operation = rng.choice(["total", "maximum", "lookup"])
        if operation == "total":
            question, answer = "What is the total cost of all units (sum of price multiplied by units)? Return only the integer.", sum(r["price"] * r["units"] for r in records)
        elif operation == "maximum":
            best = max(records, key=lambda r: r["price"])
            question, answer = "Which item has the highest unit price? Break ties by earliest row. Return only its item name.", best["item"]
        else:
            selected = rng.choice(records)
            question, answer = f"How many units of {selected['item']} are listed? Return only the integer.", selected["units"]
        return _example("Use only this table:\n" + json.dumps(records) + "\n" + question, answer, family)
    if family == "code_trace":
        start, add, multiplier, loops = rng.randint(-20, 20), rng.randint(1, 8), rng.randint(1, 4), level + rng.randint(1, 4)
        value = start
        for _ in range(loops): value = value * multiplier + add
        code = f"x = {start}\nfor i in range({loops}):\n    x = x * {multiplier} + {add}\nprint(x)"
        return _example("Trace this Python code without executing external commands. What integer is printed? Return only that integer.\n\n" + code, value, family,
                        derivation={"start": start, "add": add, "multiplier": multiplier, "loops": loops, "result": value})
    if family == "calendar":
        start = date(2000, 1, 1) + timedelta(days=rng.randint(0, 18000))
        offset = rng.randint(-365 * level, 365 * level)
        answer = start + timedelta(days=offset)
        return _example(f"Starting from {start.isoformat()}, add {offset} calendar days. Return only the date in YYYY-MM-DD format.", answer.isoformat(), family)
    raise ValueError(f"Unknown computed task: {family}")


def _template_candidates(cfg, rng, limit):
    prompts = cfg.get("prompt_variants") or [cfg.get("prompt", "")]
    answers = cfg.get("response_variants") or [cfg.get("response", "")]
    if not isinstance(prompts, list) or not isinstance(answers, list) or any(not isinstance(x, str) or not x.strip() for x in prompts + answers):
        raise ValueError("Templates need a prompt and answer (or lists of non-empty variants).")
    fields = set()
    for template in prompts + answers:
        for _, field, spec, conversion in string.Formatter().parse(template):
            if field is not None:
                if not re.fullmatch(r"[A-Za-z_]\w*", field) or spec or conversion:
                    raise ValueError("Template placeholders must be plain names such as {country}.")
                fields.add(field)
    facts = cfg.get("fact_rows")
    variables = cfg.get("variables") or {}
    if facts is not None:
        if not isinstance(facts, list) or not facts or any(not isinstance(row, dict) or not fields <= row.keys() for row in facts):
            raise ValueError("Every fact row must contain all template placeholders.")
        size = len(facts)
    else:
        if not isinstance(variables, dict) or any(not isinstance(v, list) or not v for v in variables.values()) or not fields <= variables.keys():
            raise ValueError("Every template variable needs a non-empty list, or supply correlated fact_rows.")
        size = math.prod(len(v) for v in variables.values())
    variants = len(prompts) * len(answers)
    total = size * variants
    # range is limited to Py_ssize_t; a large Cartesian product is sampled by
    # randint with a bounded seen set, never materialized in memory.
    sample_count = min(total, limit)
    if total <= 2**63 - 1:
        indices = rng.sample(range(total), sample_count)
    else:
        chosen = set()
        while len(chosen) < sample_count:
            chosen.add(rng.randrange(total))
        indices = sorted(chosen)
        rng.shuffle(indices)
    for index in indices:
        row_index, variant_index = divmod(index, variants)
        prompt_index, answer_index = divmod(variant_index, len(answers))
        if facts is not None:
            values = facts[row_index]
        else:
            values, remainder = {}, row_index
            for name, choices in reversed(list(variables.items())):
                remainder, choice = divmod(remainder, len(choices))
                values[name] = choices[choice]
        yield _example(prompts[prompt_index].format_map(values), answers[answer_index].format_map(values), "templates", False,
                       source_ids=["fact:" + hashlib.sha256(json.dumps(values, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]] if facts is not None else [],
                       derivation={"fields": values, "correlated_facts": facts is not None})


def _source_id(row, index):
    return str(row.get("id") or row.get("sourceId") or f"source:{index}")


def _grounded_candidate(seeds, rng, difficulty):
    index = rng.randrange(len(seeds))
    row = seeds[index]
    source_ids = [_source_id(row, index)]
    context = row.get("text")
    if not context:
        # The source answer is evidence, not an independently verified fact.
        turns = messages_of(row)
        answers = [m["content"] for m in turns if m["role"] == "assistant"]
        if not answers:
            raise ValueError("Grounded sources need text or an assistant answer.")
        selected = rng.randrange(len(answers))
        context = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in turns)
        answer = answers[selected]
        prompt = f"Reference conversation (copying task):\n<reference>\n{context}\n</reference>\nCopy assistant answer number {selected + 1} exactly. Include the answer text only."
        return _example(prompt, answer, "grounded", True, source_ids=source_ids,
                        derivation={"method": "exact_answer_copy", "assistant_index": selected, "source_evidence": answer})
    context = str(context)
    # Bound context while retaining exact source bytes in the selected excerpt.
    start = rng.randrange(max(1, len(context) - 12000 + 1)) if len(context) > 12000 else 0
    context = context[start:start + 12000]
    pieces = [(m.start(), m.end()) for m in re.finditer(r"[^\n.!?]+(?:[.!?]+|(?=\n)|$)", context) if m.group().strip()]
    if not pieces:
        pieces = [(0, len(context))]
    chosen = rng.randrange(len(pieces))
    begin, end = pieces[chosen]
    answer = context[begin:end].strip()
    # Character spans avoid ambiguous sentence segmentation in the instruction.
    begin += len(context[begin:end]) - len(context[begin:end].lstrip())
    end = begin + len(answer)
    prompt = (f"Source passage:\n<source>\n{context}\n</source>\n"
              f"Copy the source passage characters from index {begin} (inclusive) to {end} (exclusive), counting from zero. Return only the copied text.")
    return _example(prompt, answer, "grounded", True, source_ids=source_ids,
                    derivation={"method": "exact_span", "source_offset": start, "start": begin, "end": end, "source_evidence": answer})


def _seed_candidates(seeds, rng, limit):
    wrappers = ["{q}", "Please respond to the following request:\n{q}", "Here is my request:\n{q}",
                "I need help with this request:\n{q}", "Please address the following:\n{q}",
                "My question is:\n{q}", "Can you respond to this?\n{q}", "Request:\n{q}"]
    sources = [(i, row) for i, row in enumerate(seeds) if not row.get("text") and messages_of(row)[-1]["role"] == "assistant"]
    for index in rng.sample(range(len(sources) * len(wrappers)), min(limit, len(sources) * len(wrappers))):
        source_index, wrapper = divmod(index, len(wrappers))
        original_index, original = sources[source_index]
        turns = [dict(m) for m in messages_of(original)]
        user = next((m for m in turns if m["role"] == "user"), None)
        if user is None:
            continue
        user["content"] = wrappers[wrapper].format(q=user["content"])
        yield {"messages": turns, "intent": "seeds", "computedVerified": False,
               "source_ids": [_source_id(original, original_index)], "derivation": {"method": "request_wrapper", "wrapper": wrapper}}


_FINAL = re.compile(r"\n?####\s*(.+?)\s*$", re.S)
_CALC = re.compile(r"<<[^<>]*>>")
_CONTEXT = re.compile(r"^(?P<question>.+?)\n\nContext:\n(?P<context>.+)$", re.S)


def _single_turn(row):
    turns = messages_of(row)
    if row.get("text") or len(turns) != 2 or turns[0]["role"] != "user" or turns[1]["role"] != "assistant":
        return None
    return turns[0]["content"], turns[1]["content"]


def _remix_pools(seeds):
    """Index which sampled sources can support each remix without inventing targets."""
    pools = {name: [] for name in REMIX_FAMILIES}
    for index, row in enumerate(seeds):
        pair = _single_turn(row)
        turns = messages_of(row)
        if not row.get("text") and turns and turns[-1]["role"] == "assistant" and len(turns) <= 6:
            pools["multi_turn"].append(index)
        if pair is None:
            continue
        user, answer = pair
        if _FINAL.search(answer):
            pools["final_answer"].append(index)
            pools["clean_solution"].append(index)
        elif "\n" not in answer.strip() and 0 < len(answer.strip()) <= 120:
            pools["json_answer"].append(index)
        if _CONTEXT.match(user):
            pools["context_first"].append(index)
    return pools


def _remix_candidate(family, seeds, pool, rng):
    # Pools are shuffled once and consumed, so each source yields one variant per family.
    if not pool:
        return None
    index = pool.pop()
    row = seeds[index]
    source_ids = [_source_id(row, index)]
    if family == "multi_turn":
        # Chain 2–3 independent conversations; every assistant turn is copied verbatim.
        partners = [pool.pop() for _ in range(min(len(pool), rng.randint(1, 2)))]
        if not partners:
            return None
        picks = [index] + partners
        turns = [{"role": m["role"], "content": _CALC.sub("", m["content"])} for i in picks for m in messages_of(seeds[i]) if m["role"] != "system"]
        return {"messages": turns, "intent": "multi_turn", "computedVerified": True,
                "source_ids": [_source_id(seeds[i], i) for i in picks], "derivation": {"method": "conversation_chain", "parts": len(picks)}}
    user, answer = _single_turn(row)
    if family == "final_answer":
        final = _FINAL.search(answer).group(1).strip()
        return _example(f"{user}\n\nGive only the final answer, with no working.", final, family, True,
                        source_ids=source_ids, derivation={"method": "final_answer_extract", "final": final})
    if family == "clean_solution":
        final = _FINAL.search(answer).group(1).strip()
        steps = _CALC.sub("", _FINAL.sub("", answer)).strip()
        return _example(f"{user}\n\nSolve step by step, then finish with a line of the form 'Answer: <value>'.", f"{steps}\nAnswer: {final}", family, True,
                        source_ids=source_ids, derivation={"method": "solution_cleanup", "final": final})
    if family == "json_answer":
        value = answer.strip()
        return _example(f"{user}\n\nRespond with JSON only, in the form {{\"answer\": ...}}.", json.dumps({"answer": value}, ensure_ascii=False), family, True,
                        source_ids=source_ids, derivation={"method": "json_wrap", "answer": value})
    match = _CONTEXT.match(user)
    return _example(f"Read the context, then answer the question.\n\nContext:\n{match['context'].strip()}\n\nQuestion: {match['question'].strip()}", answer, family, False,
                    source_ids=source_ids, derivation={"method": "context_reorder"})


def _family_budgets(cfg):
    mode = cfg["mode"]
    families = cfg["families"] if mode in {"mixed", "reasoning", "structured", "remix"} else [mode]
    weights = [cfg["weights"].get(f, 1.) for f in families]
    total = sum(weights)
    shares = [cfg["count"] * w / total for w in weights]
    counts = [math.floor(x) for x in shares]
    order = sorted(range(len(families)), key=lambda i: (-(shares[i] - counts[i]), i))
    for index in order[:cfg["count"] - sum(counts)]:
        counts[index] += 1
    return dict(zip(families, counts))


def generate_batch(config, seeds=None, existing=None, stop=None, emit=None, teacher=None):
    """Generate a review batch. Cancellation returns the accepted partial batch.

    Teacher modes (topic/evolve/answer/paraphrase/dialogue) are delegated to
    studio_teacher_synth.generate_teacher_batch and return the same shape;
    `teacher` optionally supplies the chat object (default: OllamaTeacher).

    Events: {type:'progress', completed, total, attempted, text}. Output stats
    count every attempted candidate; rejected samples are capped at 100. Exact
    deduplication includes the supplied existing records; near matching examines
    a bounded recent index and is deliberately described as approximate.
    """
    started = time.monotonic()
    cfg = validate_config(config)
    if cfg["mode"] in TEACHER_MODES:
        from studio_teacher_synth import generate_teacher_batch
        return generate_teacher_batch(cfg, seeds=seeds, existing=existing, stop=stop, emit=emit, teacher=teacher)
    rng = random.Random(cfg["seed"])
    stop = stop if stop is not None else threading.Event()
    emit = emit or (lambda event: None)
    budgets = _family_budgets(cfg)
    needs_sources = any(f in budgets for f in ["grounded", "seeds", "model"] + REMIX_FAMILIES)
    seeds = [row for row in (seeds or []) if isinstance(row, dict) and (row.get("text") or row.get("messages") or row.get("user") or row.get("prompt"))] if needs_sources else []
    normalized_seeds = []
    for index, row in enumerate(seeds):
        if stop.is_set():
            break
        clean = normalize(row, source=row.get("source", "import"))
        if clean:
            clean["id"] = row.get("id", clean["id"])
            normalized_seeds.append(clean)
        if index and index % 10000 == 0:
            emit({"type": "progress", "completed": 0, "total": cfg["count"], "attempted": 0,
                  "text": f"Preparing source records: {index:,}/{len(seeds):,}…"})
    seeds = normalized_seeds
    batch_id = f"syn_{uuid.uuid4().hex[:12]}"
    emit({"type": "progress", "completed": 0, "total": cfg["count"], "attempted": 0, "text": "Indexing existing examples for duplicate and conflict checks…"})
    gate = QualityGate(cfg["quality"], existing, stop)
    rows, rejected, reasons, coverage = [], [], Counter(), Counter()
    difficulty_coverage = Counter()
    attempts_by_family = Counter()
    candidate_limit = cfg["count"] * cfg["attempt_multiplier"]
    resources = {}
    if any(f in budgets for f in ["grounded", "seeds"] + REMIX_FAMILIES) and not seeds and not stop.is_set():
        raise ValueError("Choose a source dataset with usable text or conversations for this recipe.")
    pools = _remix_pools(seeds) if cfg["mode"] == "remix" else {}
    for pool in pools.values():
        rng.shuffle(pool)
    if pools and not any(pools[f] for f in budgets):
        raise ValueError("The sampled source records do not support the selected remixes. Try another dataset or remix type.")
    exhausted = {f for f in budgets if f in pools and not pools[f]}
    if "templates" in budgets and not stop.is_set():
        resources["templates"] = iter(_template_candidates(cfg, rng, candidate_limit))
    if "seeds" in budgets and not stop.is_set():
        resources["seeds"] = iter(_seed_candidates(seeds, rng, candidate_limit))
    if "model" in budgets and not stop.is_set():
        if not cfg.get("checkpoint"):
            raise ValueError("Select a checkpoint trained in this studio.")
        prompts = cfg.get("model_prompts") or []
        if isinstance(prompts, str):
            prompts = [p.strip() for p in prompts.split("\n") if p.strip()]
        if not prompts:
            prompts = [next((m["content"] for m in reversed(messages_of(row)) if m["role"] == "user"), "") for row in seeds if not row.get("text")]
        prompts = list(dict.fromkeys(p for p in prompts if isinstance(p, str) and p.strip()))
        prompt_sources = {}
        for index, row in enumerate(seeds):
            prompt = next((m["content"] for m in reversed(messages_of(row)) if m["role"] == "user"), "")
            prompt_sources.setdefault(prompt, _source_id(row, index))
        if not prompts:
            raise ValueError("Provide model_prompts or a source dataset with user prompts.")
        from studio_model import LocalModel
        emit({"type": "progress", "completed": 0, "total": cfg["count"], "attempted": 0, "text": "Loading your local checkpoint…"})
        resources["model"] = LocalModel(cfg["checkpoint"], seed=cfg["seed"], device=cfg["device"])
        resources["model_prompts"] = prompts
        resources["prompt_sources"] = prompt_sources
    attempted, last_progress = 0, 0.
    while len(rows) < cfg["count"] and attempted < candidate_limit and not stop.is_set():
        available = [f for f, target in budgets.items() if coverage[f] < target and f not in exhausted and attempts_by_family[f] < max(1, target) * cfg["attempt_multiplier"]]
        if not available:
            break
        family = rng.choices(available, weights=[budgets[f] - coverage[f] for f in available], k=1)[0]
        fraction = len(rows) / cfg["count"]
        difficulty = cfg["difficulty"] if cfg["difficulty"] != "curriculum" else "easy" if fraction < 1 / 3 else "medium" if fraction < 2 / 3 else "hard"
        if family in resources and family != "model":
            try:
                raw = next(resources[family])
            except StopIteration:
                exhausted.add(family)
                continue
        elif family == "grounded":
            raw = _grounded_candidate(seeds, rng, difficulty)
        elif family in pools:
            raw = _remix_candidate(family, seeds, pools[family], rng)
            if raw is None:
                exhausted.add(family)
                continue
        elif family == "model":
            prompt = resources["model_prompts"][attempts_by_family[family] % len(resources["model_prompts"]) ]
            turns = []
            if cfg.get("system_prompt"):
                turns.append({"role": "system", "content": str(cfg["system_prompt"])})
            turns.append({"role": "user", "content": prompt})
            answer = resources["model"].generate(turns, max_tokens=cfg["max_tokens"], temperature=cfg["temperature"], top_k=cfg["top_k"], stop=stop)
            if stop.is_set():
                break  # An interrupted answer is not a completed draft.
            source = resources["prompt_sources"].get(prompt)
            raw = {"messages": turns + [{"role": "assistant", "content": answer}], "intent": "model", "computedVerified": False,
                   "source_ids": [source] if source else [],
                   "derivation": {"checkpoint": cfg["checkpoint"], "checkpoint_step": resources["model"].step}}
        else:
            raw = _procedural(family, rng, difficulty)
        attempted += 1
        attempts_by_family[family] += 1
        row = normalize(raw, source="synthetic")
        # A model may legitimately give several answers to one prompt, and the sampled source rows
        # carry the same prompts: exact duplicates are still removed, but differing answers are not conflicts.
        reason = gate.inspect(row, check_conflicts=False if family == "model" else None)
        if reason:
            reasons[reason] += 1
            if len(rejected) < 100:
                rejected.append({"reason": reason, "family": family, "user": (row or raw).get("user", "")[:1000], "assistant": (row or raw).get("assistant", "")[:1000]})
        else:
            row["quality"] = "Review"
            row["computedVerified"] = bool(raw.get("computedVerified"))
            row["qualityReport"] = {"status": "needs_review", "automated_checks": "passed", "computed_target": row["computedVerified"], "warnings": [] if row["computedVerified"] else ["Target meaning and correctness require human review."]}
            row["provenance"] = {"generator": "groundwork-synthesis-v2", "recipe": family,
                                 "seed": cfg["seed"], "batch_id": batch_id, "attempt": attempted,
                                 "difficulty": difficulty, "source_ids": raw.get("source_ids", []),
                                 "derivation": raw.get("derivation", {}), "synthetic": True}
            gate.remember(row)
            rows.append(row)
            coverage[family] += 1
            difficulty_coverage[difficulty] += 1
        tick = time.monotonic()
        if tick - last_progress >= .15 or len(rows) == cfg["count"]:
            emit({"type": "progress", "completed": len(rows), "total": cfg["count"], "attempted": attempted,
                  "text": f"{len(rows):,}/{cfg['count']:,} accepted · {family.replace('_', ' ')} · {difficulty}"})
            last_progress = tick
    stats = {"requested": cfg["count"], "attempted": attempted, "accepted": len(rows),
             "duplicates": reasons["duplicate"] + reasons["near_duplicate"], "conflicts": reasons["prompt_conflict"],
             "rejected": sum(reasons.values()), "reasons": dict(reasons),
             "elapsed": round(time.monotonic() - started, 3), "coverage": dict(coverage),
             "difficulty_coverage": dict(difficulty_coverage), "budgets": budgets,
             "attempts_by_family": dict(attempts_by_family), "cancelled": stop.is_set(),
             "shortfall": cfg["count"] - len(rows), "exhausted_families": sorted(exhausted),
             "near_duplicate_scope": "Approximate: recent 10,000 records; up to 256 candidates per record."}
    emit({"type": "progress", "completed": len(rows), "total": cfg["count"], "attempted": attempted,
          "text": ("Cancelled; partial batch preserved. " if stop.is_set() else "Batch ready for review. ") + f"{len(rows):,} accepted; {stats['rejected']:,} filtered."})
    return {"rows": rows, "rejected": rejected, "stats": stats, "config": cfg, "batch_id": batch_id}


_REASON_WORDS = {"duplicate": "duplicates", "near_duplicate": "near-duplicates", "prompt_conflict": "conflicting answers",
                 "answer_too_long": "too long", "answer_too_short": "too short", "prompt_too_long": "prompts too long",
                 "repetitive_answer": "repetitive", "invalid_json": "invalid JSON", "empty_answer": "empty",
                 "incomplete_conversation": "incomplete", "refusal": "refusals", "malformed_item": "malformed",
                 "low_score": "low teacher score", "meaning_changed": "changed the meaning", "unchanged": "unchanged",
                 "bad_shape": "wrong conversation shape", "echo": "answers that only repeat the prompt"}


def summarize_batch(result):
    """Plain-English one-liner for a generate_batch result, e.g. 'Made 186 examples. Skipped 14: 9 duplicates, 5 too long.'"""
    stats = result.get("stats", result) if isinstance(result, dict) else {}
    made = int(stats.get("accepted", len(result.get("rows", [])) if isinstance(result, dict) else 0))
    text = f"Made {made:,} example{'s' if made != 1 else ''}."
    reasons = {k: v for k, v in (stats.get("reasons") or {}).items() if v}
    skipped = sum(reasons.values())
    if skipped:
        parts = ", ".join(f"{count:,} {_REASON_WORDS.get(reason, reason.replace('_', ' '))}"
                          for reason, count in sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0])))
        text += f" Skipped {skipped:,}: {parts}."
    requested = stats.get("requested", made)
    if stats.get("cancelled"):
        text += " Stopped early; what was made so far is kept."
    elif made < requested:
        text += f" (You asked for {requested:,}.)"
    if stats.get("teacher_errors"):
        text += f" The teacher failed on {stats['teacher_errors']:,} request(s)."
    if stats.get("abort_reason"):
        text += f" {stats['abort_reason']}"
    return text


def apply_draft_edit(old, edited, others=(), quality=None):
    """Merge a hand-edited draft over its original and re-run the quality gates.

    Returns (row, reason). `row` keeps the original provenance, id and flags; a changed
    prompt/answer clears the computed-verified flag (a human changed the target) and is
    recorded in provenance. `reason` is the first failing gate or None.
    """
    row = dict(old)
    for key in ("messages", "text", "user", "assistant", "intent", "quality"):
        if key in edited:
            row[key] = edited[key]
    row["id"] = old["id"]
    changed = (edited.get("messages"), edited.get("text")) != (old.get("messages"), old.get("text"))
    if changed:
        report = dict(row.get("qualityReport") or {})
        was_computed = bool(row.get("computedVerified"))
        row["computedVerified"] = False
        report.update(status="needs_review", computed_target=False, edited=True,
                      warnings=list(report.get("warnings", [])) + (["Edited by hand: the computed verification no longer applies."] if was_computed else ["Edited by hand."]))
        row["qualityReport"] = report
        provenance = dict(row.get("provenance") or {})
        provenance["edited"] = True
        provenance["edited_at"] = now()
        row["provenance"] = provenance
    gate = QualityGate(quality, [r for r in others if r.get("id") != old["id"]])
    return row, gate.inspect(row)


_REASON_SENTENCES = {"duplicate": "it exactly duplicates another example",
                     "near_duplicate": "it is nearly identical to another example",
                     "prompt_conflict": "another example already answers this prompt differently",
                     "answer_too_long": "the answer is longer than the limit", "answer_too_short": "the answer is too short",
                     "empty_answer": "the answer is empty", "incomplete_conversation": "the conversation must end with an assistant answer",
                     "invalid_json": "the answer must be valid JSON", "repetitive_answer": "the answer repeats itself too much"}


def explain_reason(reason):
    """Plain-language version of a QualityGate reason code."""
    return _REASON_SENTENCES.get(reason, str(reason).replace("_", " "))
