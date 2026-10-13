"""The one gate for every post-training switch: held-back error did not rise AND checked-problem pass rate did not fall.

before/after = {"bpc": float, "pass_k": int, "pass_n": int}. The judge's k of n is shown elsewhere and never gates.
"""
import math


def passes(before, after):
    """(ok, checks): checks are [{"name", "detail", "ok"}] in display order."""
    b0, b1 = before.get("bpc"), after.get("bpc")
    measured = all(v is not None and math.isfinite(v) for v in (b0, b1))
    bpc_ok = measured and b1 <= b0
    bpc_detail = f"{b1:.3f} bits per character, was {b0:.3f}" + ("" if bpc_ok else ": rose") if measured else "not measured: no held-back text"
    k0, n0, k1, n1 = before.get("pass_k", 0), before.get("pass_n", 0), after.get("pass_k", 0), after.get("pass_n", 0)
    rate_ok = not (n0 and n1) or k1 * n0 >= k0 * n1   # no problems on either side: nothing can have fallen
    rate_detail = f"{k1} of {n1} held-back problems, was {k0} of {n0}" if n0 and n1 else "no held-back problems to check"
    checks = [{"name": "Held-back text: error did not rise", "detail": bpc_detail, "ok": bool(bpc_ok)},
              {"name": "Checked problems: pass rate did not fall", "detail": rate_detail, "ok": bool(rate_ok)}]
    return all(c["ok"] for c in checks), checks
