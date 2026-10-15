"""The Builder's "your text, cut into pieces" card: the four word lists, fitted for real, and the example they cut.
No Tk here: services fit the lists off the Tk thread and the page draws what these functions return."""
from __future__ import annotations

import re

from studio_modules.tokenizer.bpe_custom import CustomBPEConfig, CustomBPETokenizer
from studio_modules.tokenizer.byte import Tokenizer
from studio_modules.tokenizer.word import PIECE, WordConfig, WordTokenizer
from studio_slots import OFFSET

SAMPLE_CHARS = 300_000  # characters of your text the two data-built lists are fitted on: enough for their frequent words
EXAMPLE = "The heron waited by the moonlit estuary."  # rare words, so word-piece splits show; used until there is text
SENTENCE = re.compile(r"[^\n]+?[.!?](?=\s|$)")  # "3.12" does not end one
PLAIN = re.compile(r"[A-Z][\w ,;:'’\"()-]+[.!?]")  # prose, not code or markup
VISIBLE = {" ": "␣", "\n": "↵", "\t": "⇥"}


def word_lists(texts, gpt2=None):
    """{"tokenizers": (letters, small, large, whole words), "sentence": example} from a sample of your text. The small
    and whole-word lists are None while there is no text, the large one (GPT-2's, fetched by the caller) when it is None."""
    sample, used = [], 0
    for text in texts:
        if used >= SAMPLE_CHARS:
            break
        sample.append(text[:SAMPLE_CHARS - used])
        used += len(sample[-1])
    examples = [{"text": t} for t in sample if t]
    small = CustomBPETokenizer.fit(examples, CustomBPEConfig().vocab_size) if examples else None
    words = WordTokenizer.fit(examples, WordConfig().vocab_size) if examples else None
    return {"tokenizers": (Tokenizer("byte"), small, gpt2, words), "sentence": example_sentence(sample, small, words)}


def example_sentence(texts, small=None, words=None):
    """A short plain sentence from your text, preferring one with a word the whole-word list loses and one the small list
    splits into pieces; EXAMPLE when the text has no such sentence of 4 to 12 words. A sentence that followed a space in
    the text keeps it, so its first word is cut the way it is in place."""
    found = []
    for text in texts:
        for match in SENTENCE.finditer(text):
            s = " ".join(match.group().split())
            if 4 <= len(s.split()) <= 12 and len(s) <= 80 and PLAIN.fullmatch(s):
                found.append(" " + s if match.group()[0] == " " else s)
        if len(found) >= 500:
            break
    if not found:
        return EXAMPLE

    def score(s):
        lost = words is not None and any(p not in words.lookup for p in PIECE.findall(s))
        split = small is not None and any(len(small.encode(w + " ")) > 1 for w in re.findall(r"\w+", s))
        return lost, split
    return max(found, key=score)  # max keeps the first of equals: the text's own order


def visible(piece):
    """A piece with its spaces and line breaks drawn, so GPT-2's leading space shows."""
    return "".join(VISIBLE.get(ch, ch) for ch in piece)


def _bytes(tok, token):
    i = int(token) - OFFSET
    if hasattr(tok, "byte_decoder"):  # GPT-2: its symbols stand for bytes
        return bytes(tok.byte_decoder[ch] for ch in tok.decoder[i])
    if hasattr(tok, "pieces"):
        return tok.pieces[i]
    return bytes([i])


def chips(tok, text):
    """[(label, known)] for `text` cut by `tok`, one per piece. A byte that is part of a longer character shows as hex;
    a word the whole-word list does not hold is known=False: the model sees only "unknown" there."""
    if isinstance(tok, WordTokenizer):
        return [(visible(p), p in tok.lookup) for p in PIECE.findall(text)]
    out = []
    for token in tok.encode(text):
        data = _bytes(tok, token)
        try:
            out.append((visible(data.decode("utf-8")), True))
        except UnicodeDecodeError:
            out.append((" ".join(f"{b:02x}" for b in data), True))
    return out


def counts(lists):
    """Piece count of the example under each of the four lists, None for a list that is not there."""
    sentence = lists["sentence"]
    return [len(chips(tok, sentence)) if tok is not None else None for tok in lists["tokenizers"]]
