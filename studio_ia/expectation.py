"""What a design can be expected to write: a short headline and a plain-English paragraph, from its size, word list and the text it sees.

Pure and window-free, so the Builder and Pretraining pages say the same thing. A rough guide, not a promise."""
from __future__ import annotations

from studio_ia.copy import EXPECTATION as T

LITTLE_TEXT = 5_000_000  # word pieces seen: below this a model only picks up letter patterns
ENOUGH_TEXT = 20_000_000  # about 80 MB of text as word pieces
SMALL, MEDIUM = 1_000_000, 10_000_000  # parameters that do the thinking
FIXED_WORD_LIST = {"byte": 256, "character": 256, "bpe_gpt2": 50257}  # custom word lists are sized by the data
WORD_PIECES = ("bpe_custom", "bpe_gpt2", "word")


def table_params(tokenizer, d_model, vocab=None):
    """Parameters in the word table (list size × width), or None when the list size is not known."""
    vocab = vocab or FIXED_WORD_LIST.get(tokenizer)
    return vocab * d_model if vocab else None


def expectation(params, tokenizer, tokens_seen=None, data_kind=None, embedding_params=None):
    """(headline, details): the headline names the tier; the details add what would move it up, the word list's effect and, for plain text, that it continues text.

    params: all parameters. tokenizer: "byte" | "character" | "bpe_custom" | "bpe_gpt2" | "word". tokens_seen: word pieces it
    will read (None = not known yet). data_kind: "text" for a plain-text run, which adds that answering comes from
    post-training. embedding_params: the word table's parameters, if known; when that is over half the total, only
    the rest counts as thinking capacity."""
    heavy = bool(embedding_params) and embedding_params * 2 > params
    thinking = params - embedding_params if heavy else params
    pieces = tokenizer in WORD_PIECES
    known = tokens_seen is not None
    little = known and tokens_seen < LITTLE_TEXT
    if thinking < SMALL or little:
        tier = 0
    elif thinking >= MEDIUM and pieces and not (known and tokens_seen < ENOUGH_TEXT):
        tier = 2
    else:
        tier = 1
    level = ("letters", "words", "sentences")[tier] + ("_text" if tier and not known else "")
    out = [T["guide"] + " " + T[level]]
    need = []  # the smallest changes that reach the next tier
    if tier < 2 and thinking < (SMALL if tier == 0 else MEDIUM):
        need.append(T["wider_deeper"])
    if tier == 1 and not pieces:
        need.append(T["custom_pieces"])
    if tier < 2 and known and tokens_seen < (LITTLE_TEXT if tier == 0 else ENOUGH_TEXT):
        need.append(T["more_text"])
    if need:
        out.append(T["move_up"].format(changes=" and ".join(need)))
    if tier == 2:
        out.append(T["chat_scale"])
    out.append(T[tokenizer if tokenizer in T else "byte"])
    if heavy:
        out.append(T["table_heavy"])
    if data_kind == "text":
        out.append(T["continues"])
    return T["short_" + level], " ".join(out)
