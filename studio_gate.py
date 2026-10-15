"""The one gate for every post-training switch: held-back error did not rise AND checked-problem pass rate did not fall.

before/after = {"bpc": float, "pass_k": int, "pass_n": int}. The judge's k of n is shown elsewhere and never gates.
"""
import math


def passes(before, after, band=None):
    """(ok, checks): checks are [{"name", "detail", "ok"}] in display order.

    band = (lo, hi), the noise band of the held-back error before the round: a rise that stays inside it is noise and
    does not fail the check; without a band any rise fails."""
    b0, b1 = before.get("bpc"), after.get("bpc")
    measured = all(v is not None and math.isfinite(v) for v in (b0, b1))
    limit = max(b0, band[1]) if measured and band else b0
    bpc_ok = measured and b1 <= limit
    bpc_detail = (f"{b1:.3f} bits per character, was {b0:.3f}" + ("" if bpc_ok else ": rose")
                  + (f" (noise band up to {band[1]:.3f})" if band and measured else "")) if measured else "not measured: no held-back text"
    k0, n0, k1, n1 = before.get("pass_k", 0), before.get("pass_n", 0), after.get("pass_k", 0), after.get("pass_n", 0)
    rate_ok = not (n0 and n1) or k1 * n0 >= k0 * n1   # no problems on either side: nothing can have fallen
    rate_detail = f"{k1} of {n1} held-back problems, was {k0} of {n0}" if n0 and n1 else "no held-back problems to check"
    checks = [{"name": "Held-back text: error did not rise", "detail": bpc_detail, "ok": bool(bpc_ok)},
              {"name": "Checked problems: pass rate did not fall", "detail": rate_detail, "ok": bool(rate_ok)}]
    return all(c["ok"] for c in checks), checks


def moves_chat(new_bpc, best_bpc, paired=None):
    """Chat follows the best checkpoint only when the new version is clearly better than the previous best: paired is
    studio_eval.paired_gain on the same held-back documents, and its "better" decides. Without it (no held-back text to
    pair on) the new held-back error moves Chat only if it is lower, or if nothing was measured before."""
    if new_bpc is None or not math.isfinite(new_bpc):
        return False
    if best_bpc is None:
        return True
    return bool(paired["better"]) if paired is not None else new_bpc < best_bpc
