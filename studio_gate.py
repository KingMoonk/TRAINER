"""The one gate for every post-training switch: no check fails. Held-back text error did not rise (Round 1: stayed within a
drift budget, plus the chat checks) AND checked-problem pass rate did not fall.

before/after = {"bpc": float, "pass_k": int, "pass_n": int}, and for Round 1 also "conv_loss" and "eos_rate" (studio_rl.measure).
The judge's k of n is shown elsewhere and never gates.
"""
import math

DPO_ACCURACY = 0.55   # Round 4: the held-back pairs' reward accuracy must be above this (a coin flip is 0.5)
DRIFT = 0.05   # Round 1: how far raw-text bits per character may rise, as a share of the base's, before the round fails


def _finite(*values):
    return all(v is not None and math.isfinite(v) for v in values)


def _bpc_check(before, after, band, number, drift):
    b0, b1 = before.get("bpc"), after.get("bpc")
    if not _finite(b0, b1):
        name = "Raw text: error stayed within the drift budget" if number == 1 else "Held-back text: error did not rise"
        return {"name": name, "detail": "not measured: no held-back text", "ok": None}   # nothing to compare: never a failure
    if number == 1:   # SFT moves raw text a little by design: a budget on the base's error, not the bootstrap noise band
        limit = b0 * (1 + drift)
        ok = b1 <= limit
        detail = f"{b1:.3f} bits per character, was {b0:.3f}" + ("" if ok else ": rose") + f" (budget up to {limit:.3f}, +{drift:.0%})"
        return {"name": "Raw text: error stayed within the drift budget", "detail": detail, "ok": bool(ok)}
    limit = max(b0, band[1]) if band else b0   # a rise inside the noise band is noise
    ok = b1 <= limit
    detail = f"{b1:.3f} bits per character, was {b0:.3f}" + ("" if ok else ": rose") + (f" (noise band up to {band[1]:.3f})" if band else "")
    return {"name": "Held-back text: error did not rise", "detail": detail, "ok": bool(ok)}


def _chat_checks(before, after):
    """Round 1: the held-back conversations' answer loss fell and the greedy stop-with-EOS rate did not fall. Not measured is no
    evidence of an improvement, so it fails here (the version has no held-back conversations to judge by)."""
    none = "not measured: no held-back conversations"
    c0, c1 = before.get("conv_loss"), after.get("conv_loss")
    if _finite(c0, c1):
        conv = {"detail": f"{c1:.3f} per answer token, was {c0:.3f}" + ("" if c1 < c0 else ": did not fall"), "ok": bool(c1 < c0)}
    else:
        conv = {"detail": none, "ok": False}
    e0, e1 = before.get("eos_rate"), after.get("eos_rate")
    if _finite(e0, e1):
        eos = {"detail": f"{e1:.0%} of held-back answers end on their own, was {e0:.0%}" + ("" if e1 >= e0 else ": fell"), "ok": bool(e1 >= e0)}
    else:
        eos = {"detail": none, "ok": False}
    return [{"name": "Held-back conversations: answer error fell", **conv},
            {"name": "Answers still stop: end-of-answer rate did not fall", **eos}]


def _rose_check(before, after):
    """Round 2: the held-back pass rate rose by more than 1/sqrt(n), or, when n is below 4 (1/sqrt(n) is then coarser than the
    measurement), by at least one problem. Nothing to compare is no evidence, so it fails."""
    name = "Checked problems: pass rate rose"
    k0, n0, k1, n1 = before.get("pass_k", 0), before.get("pass_n", 0), after.get("pass_k", 0), after.get("pass_n", 0)
    if not (n0 and n1):
        return {"name": name, "detail": "not measured: no held-back problems", "ok": False}
    need = 1 / math.sqrt(n1)
    ok = k1 / n1 - k0 / n0 > need or (n1 < 4 and k1 * n0 > k0 * n1)
    detail = f"{k1} of {n1} held-back problems, was {k0} of {n0}" + ("" if ok else ": did not rise enough") + f" (needs a rise of more than {need:.0%})"
    return {"name": name, "detail": detail, "ok": bool(ok)}


def _kl_check(record):
    """Round 3: the student's gap to the teacher (reverse KL per word on the round's prompts) fell."""
    name = "Teacher gap: KL to the teacher fell"
    k0, k1 = (record or {}).get("kl_before"), (record or {}).get("kl_after")
    if not _finite(k0, k1):
        return {"name": name, "detail": "not measured: no teacher gap recorded", "ok": False}
    return {"name": name, "detail": f"{k1:.3f} per word, was {k0:.3f}" + ("" if k1 < k0 else ": did not fall"), "ok": bool(k1 < k0)}


def _dpo_check(record, threshold=DPO_ACCURACY):
    """Round 4: reward accuracy on held-back human pairs (dpo_train's final_validation) is above the threshold."""
    name = "Held-back pairs: preference accuracy above 55%"
    final, n = (record or {}).get("final_validation"), (record or {}).get("validation_pairs", 0)
    acc = (final or {}).get("reward_accuracy")
    if not n or not _finite(acc):
        return {"name": name, "detail": "not measured: too few pairs to hold any back", "ok": False}
    detail = f"{acc:.0%} of {n} held-back pairs rank the chosen answer higher" + ("" if acc > threshold else f": not above {threshold:.0%}")
    return {"name": name, "detail": detail, "ok": bool(acc > threshold)}


def passes(before, after, band=None, number=None, drift=DRIFT, record=None):
    """(ok, checks): checks are [{"name", "detail", "ok"}] in display order; a check's ok is True, False, or None (not measured).

    Round 1 (number=1): held-back conversation loss fell AND the stop-with-EOS rate did not fall AND raw-text bpc stayed within
    `drift` of the base's. Other rounds: raw-text bpc did not rise, beyond band = (lo, hi), the noise band before the round.
    Raw text that was not measured is ok=None and never fails a round; the round passes when no check is False.

    Rounds 2-4 also need a measured improvement, and a missing measurement fails them (ok=False, never None): round 2 the held-back
    pass rate rose (_rose_check); round 3 the pass rate did not fall and the teacher KL fell; round 4 the held-back DPO reward
    accuracy is above DPO_ACCURACY. `record` is the round's training record (round 3: kl_before/kl_after; round 4: final_validation
    and validation_pairs)."""
    k0, n0, k1, n1 = before.get("pass_k", 0), before.get("pass_n", 0), after.get("pass_k", 0), after.get("pass_n", 0)
    rate_ok = not (n0 and n1) or k1 * n0 >= k0 * n1   # no problems on either side: nothing can have fallen
    rate_detail = f"{k1} of {n1} held-back problems, was {k0} of {n0}" if n0 and n1 else "no held-back problems to check"
    rate = {"name": "Checked problems: pass rate did not fall", "detail": rate_detail, "ok": bool(rate_ok)}
    checks = [*(_chat_checks(before, after) if number == 1 else []), _bpc_check(before, after, band, number, drift),
              _rose_check(before, after) if number == 2 else rate]
    if number == 3:
        checks.append(_kl_check(record))
    elif number == 4:
        checks.insert(1, _dpo_check(record))
    return all(c["ok"] is not False for c in checks), checks


def moves_chat(new_bpc, best_bpc, paired=None):
    """Chat follows the best checkpoint only when the new version is clearly better than the previous best: paired is
    studio_eval.paired_gain on the same held-back documents, and its "better" decides. Without it (no held-back text to
    pair on) the new held-back error moves Chat only if it is lower, or if nothing was measured before."""
    if new_bpc is None or not math.isfinite(new_bpc):
        return False
    if best_bpc is None:
        return True
    return bool(paired["better"]) if paired is not None else new_bpc < best_bpc
