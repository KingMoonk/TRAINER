"""Cleaning before storage: strip personal data and secrets, and keep only text worth training on.

Standard library only. `strip_pii` replaces what it finds with a visible tag ([EMAIL], [PHONE], [SECRET]) so the sentence still
reads; a document that is mostly personal data (a contact list) is dropped by the pipeline instead (`PII_DROP`).
"""
from __future__ import annotations

import re
from collections import Counter

PII_DROP = 25            # redactions in one document: it is a contact list or a leaked config, not prose
EMAIL = re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b")
PHONE_INTL = re.compile(r"(?<![\w.])\+\d{1,3}[\s.-]?\(?\d{1,4}\)?(?:[\s.-]?\d{2,4}){2,4}(?![\w])")
PHONE_US = re.compile(r"(?<![\w.])\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?![\w])")
# well-known token shapes, then an assignment of a long value to a secret-looking name
TOKENS = re.compile("|".join([
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", r"\bAKIA[0-9A-Z]{16}\b", r"\bgh[pousr]_[A-Za-z0-9]{30,}\b",
    r"\bgithub_pat_[A-Za-z0-9_]{22,}\b", r"\bhf_[A-Za-z0-9]{30,}\b", r"\bsk-[A-Za-z0-9_-]{20,}\b", r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b",
    r"\bAIza[0-9A-Za-z_-]{35}\b", r"\b[sr]k_(?:live|test)_[A-Za-z0-9]{16,}\b", r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"]), re.S)
ASSIGNED = re.compile(r"(?i)\b((?:api[_-]?key|secret(?:[_-]?key)?|access[_-]?token|auth[_-]?token|token|passwd|password|client[_-]?secret)[\"']?\s*[:=]\s*[\"']?)"
                      r"([A-Za-z0-9/+_=.-]{12,})")
URL_AUTH = re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^\s/@:]+:[^\s/@]+@")


def _phones(text):
    count = 0

    def tag(match):
        nonlocal count
        digits = re.sub(r"\D", "", match.group())
        if not 9 <= len(digits) <= 15:
            return match.group()
        count += 1
        return "[PHONE]"
    for pattern in (PHONE_INTL, PHONE_US):
        text = pattern.sub(tag, text)
    return text, count


def strip_pii(text):
    """(cleaned text, number of redactions)."""
    total = 0
    text, n = TOKENS.subn("[SECRET]", text)
    total += n
    text, n = ASSIGNED.subn(lambda m: m.group(1) + "[SECRET]", text)
    total += n
    text, n = URL_AUTH.subn(lambda m: m.group(1) + "[SECRET]@", text)
    total += n
    text, n = EMAIL.subn("[EMAIL]", text)
    total += n
    text, n = _phones(text)
    return text, total + n


# ---- quality ------------------------------------------------------------------------------------------------

MIN_CHARS, MAX_CHARS = 200, 2_000_000
MIN_PAIR_CHARS = 15
BOILERPLATE = ("lorem ipsum", "javascript is disabled", "enable javascript", "all rights reserved. privacy policy", "403 forbidden",
               "404 not found", "access denied", "please enable cookies", "captcha", "this site requires javascript")


def _words(text):
    return re.findall(r"\w+", text)


def text_problem(text):
    """Why a document is not worth training on, or "" when it is. Gopher/C4-style rules, loose enough for code and maths."""
    n = len(text)
    if n < MIN_CHARS:
        return "too short"
    if n > MAX_CHARS:
        return "too long"
    if "\x00" in text or text.count("�") > 0.01 * n:
        return "binary or broken text"
    words = _words(text)
    if len(words) < 30:
        return "too few words"
    mean = sum(map(len, words)) / len(words)
    if not 2.5 <= mean <= 12:
        return "unusual word length"
    visible = [c for c in text if not c.isspace()]
    if sum(c.isalnum() for c in visible) / max(1, len(visible)) < 0.45:
        return "mostly symbols"
    low = text.lower()
    if any(phrase in low for phrase in BOILERPLATE) and n < 3000:
        return "boilerplate"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 8:
        counts = Counter(lines)
        repeated = sum(c for line, c in counts.items() if c > 1)
        if repeated / len(lines) > 0.3:
            return "repeated lines"
    grams = Counter(zip(words, words[1:], words[2:]))
    if grams and grams.most_common(1)[0][1] * 3 / max(1, len(words)) > 0.2:
        return "repeated phrases"
    if words and len(set(words)) / len(words) < 0.1:
        return "repeated words"
    return ""


def pair_problem(question, answer):
    """Why a question and answer are not worth training on, or ""."""
    if len(question.strip()) < MIN_PAIR_CHARS or len(answer.strip()) < MIN_PAIR_CHARS:
        return "too short"
    if len(question) + len(answer) > 200_000:
        return "too long"
    if question.strip().lower() == answer.strip().lower():
        return "answer repeats the question"
    return _pair_text(question, answer)


def _pair_text(question, answer):
    text = f"{question} {answer}"
    if "\x00" in text or text.count("�") > 0.01 * len(text):
        return "binary or broken text"
    if any(phrase in text.lower() for phrase in BOILERPLATE) and len(text) < 600:
        return "boilerplate"
    words = _words(answer)
    if len(set(words)) / max(1, len(words)) < 0.1 and len(words) > 20:
        return "repeated words"
    return ""
