"""Rules that decide if an answer is right: check(rule, prompt, answer, params) and generate(rule, rng, n).

For RULES the prompt text carries the problem (e.g. 'Reverse: groundwork'), so the rule needs only prompt and answer.
KEYED rules compare the answer with a key stored in params["answer"] (imported datasets such as GSM8K or ARC);
they have no generator. Copy and reverse are case-sensitive; surrounding whitespace is ignored everywhere.
"""
import re

WORDS = ["groundwork", "mirror", "harbor", "lantern", "meadow", "pebble", "thunder", "orchard", "ribbon", "window",
         "garden", "puzzle", "violin", "anchor", "bridge", "candle", "forest", "marble", "silver", "tomorrow"]
TERMINAL = ".!?…"
CLOSERS = "\"')]}”’"
PAIRS = {"(": ")", "[": "]", "{": "}", "“": "”"}
NUMBER = re.compile(r"-?\d+")
DECIMAL = re.compile(r"-?\d+(?:\.\d+)?")
LETTER = re.compile(r"^\W*([A-J])(?:\)|\.|:|\s|$)")


def _number(answer):
    found = NUMBER.findall(answer.replace(",", ""))
    return int(found[-1]) if found else None


def decimal(text):
    """The last number in text (commas ignored, decimals kept) as a float, else None."""
    found = DECIMAL.findall(str(text).replace(",", ""))
    return float(found[-1]) if found else None


def _tail(prompt, label):
    m = re.match(rf"\s*{label}\s*:\s*(.*)", prompt, re.S | re.I)
    return m.group(1).strip() if m else None


def _ends_cleanly(answer):
    text = answer.strip()
    if not text:
        return False
    stack = []
    for ch in text:
        if ch in PAIRS:
            stack.append(PAIRS[ch])
        elif ch in PAIRS.values() and ch not in "\"'" and stack and stack[-1] == ch:
            stack.pop()
    if stack or text.count('"') % 2 or text.count("“") != text.count("”"):
        return False
    core = text.rstrip(CLOSERS)
    # an apostrophe inside a word is fine; a lone opening single quote is not checked
    return bool(core) and core[-1] in TERMINAL


def _copy(prompt, answer):
    want = _tail(prompt, "Copy")
    return want is not None and answer.strip() == want


def _reverse(prompt, answer):
    want = _tail(prompt, "Reverse")
    return want is not None and answer.strip() == want[::-1]


def _sort(prompt, answer):
    want = _tail(prompt, r"Sort these (?:letters|words|numbers)")
    got = [t for t in re.split(r"[\s,]+", answer.strip()) if t]
    return want is not None and bool(got) and got == sorted(want.split())


def _count(prompt, answer):
    m = re.search(r"How many (.+?) in [\"“](.*)[\"”]\s*\?", prompt, re.S)
    return bool(m) and _number(answer) == m.group(2).count(m.group(1))


def _small_addition(prompt, answer):
    m = re.match(r"\s*(-?\d+)\s*\+\s*(-?\d+)\s*=?\s*$", prompt)
    return bool(m) and _number(answer) == int(m.group(1)) + int(m.group(2))


def _max_length(prompt, answer):
    m = re.search(r"under (\d+) characters", prompt)
    return bool(m) and 0 < len(answer.strip()) < int(m.group(1))


def _ends(prompt, answer):
    return _ends_cleanly(answer)


CHECKS = {"copy": _copy, "reverse": _reverse, "sort": _sort, "count": _count, "small_addition": _small_addition,
          "max_length": _max_length, "ends_cleanly": _ends}
def _final_number(prompt, answer, params):
    want, got = decimal(params.get("answer", "")), decimal(answer)
    return want is not None and got is not None and abs(got - want) <= 1e-6 * max(1., abs(want))


def _choice(prompt, answer, params):
    m = LETTER.match(answer.strip())
    return bool(m) and m.group(1) == str(params.get("answer", "")).strip().upper()


KEYED_CHECKS = {"final_number": _final_number, "choice": _choice}
LABELS = {"copy": "copy", "reverse": "reverse", "sort": "sort", "count": "count", "small_addition": "small addition",
          "max_length": "stay under N characters", "ends_cleanly": "end cleanly"}
LABELS.update({"final_number": "final number matches the key", "choice": "picks the keyed option"})
RULES = tuple(CHECKS)              # rules the app can make problems for
KEYED = tuple(KEYED_CHECKS)        # rules that need an answer key: imported problems only
CHECKABLE = RULES + KEYED

# Tokens an answer may use before it is cut off. Most rules check a short answer; the rest (a sentence, a length
# limit, an unknown rule or a free prompt) get room to finish, because a cut-off answer fails ends_cleanly and max_length.
DEFAULT_MAX_TOKENS = 96
MAX_TOKENS = {rule: 32 for rule in ("copy", "reverse", "sort", "count", "small_addition", "final_number", "choice")}


def max_tokens(rule=None):
    """The answer budget for problems checked by `rule` (the default when the rule is unknown or None)."""
    return MAX_TOKENS.get(rule, DEFAULT_MAX_TOKENS)


def check(rule, prompt, answer, params=None):
    if rule in KEYED_CHECKS:
        return bool(KEYED_CHECKS[rule](prompt or "", answer or "", params or {}))
    if rule not in CHECKS:
        raise ValueError(f"Unknown rule: {rule}")
    return bool(CHECKS[rule](prompt or "", answer or ""))


def _one(rule, rng):
    if rule == "copy":
        w = rng.choice(WORDS)
        return f"Copy: {w}", {"text": w}, w
    if rule == "reverse":
        w = rng.choice(WORDS)
        return f"Reverse: {w}", {"text": w}, w[::-1]
    if rule == "sort":
        letters = rng.sample("abcdefghijklmnopqrstuvwxyz", rng.randint(3, 6))
        return f"Sort these letters: {' '.join(letters)}", {"items": letters}, " ".join(sorted(letters))
    if rule == "count":
        w = rng.choice(WORDS)
        ch = rng.choice(sorted(set(w)))
        return f'How many {ch} in "{w}"?', {"text": w, "char": ch}, str(w.count(ch))
    if rule == "small_addition":
        a, b = rng.randint(1, 49), rng.randint(1, 49)
        return f"{a} + {b} =", {"a": a, "b": b}, str(a + b)
    if rule == "max_length":
        n = rng.choice([40, 60, 80, 120])
        return f"Describe {rng.choice(['rain', 'a bridge', 'the sea', 'a lantern'])} in under {n} characters.", {"limit": n}, None
    if rule == "ends_cleanly":
        return f"Write one sentence about {rng.choice(['tides', 'orchards', 'thunder', 'ribbons'])}.", {}, None
    raise ValueError(f"Unknown rule: {rule}")


def generate(rule, rng, n):
    """n problem examples; 'answer' is present only where a canonical answer exists."""
    out = []
    for _ in range(n):
        prompt, params, answer = _one(rule, rng)
        item = {"prompt": prompt, "rule": rule, "params": params}
        if answer is not None:
            item["answer"] = answer
        out.append(item)
    return out
