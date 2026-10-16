"""Training progress and verdict wording shared by the API and desktop, without Tk."""
import math
import studio_eval


def phase_at(phases, step):
    """Schedule phase (warm-up | steady | cool-down) of `step` from run_curve's [(start, end, name)]; '' without a schedule."""
    return next((name for _s, end, name in phases if step <= end), phases[-1][2]) if phases else ""


def history_of(steps, heldout):
    """Engine-style history rows ({"loss", "val_loss"}) from the curve's per-step and held-back points."""
    held = dict(heldout)
    return [{"loss": y, "val_loss": held.get(x)} for x, y in steps]


def switch_note(running, last_held, best, gate):
    """(copy key, {gap}) for the line about Chat switching. While the run goes it is only a first look at the held-back curve against
    the earlier best; once it ends and Chat has weighed it (services.gate_result), the paired test's verdict."""
    if not running and gate and gate.get("reason") == "few_docs":
        return "gate_few_docs", {"docs": gate["docs"], "need": studio_eval.MIN_DOCS}
    if not running and gate and math.isfinite(gate["diff"]):
        gap = f"{abs(gate['diff']):.3f}"
        return ("gate_better" if gate["better"] else "gate_not_sure" if gate["diff"] < 0 else "gate_not_better"), {"gap": gap}
    if last_held is None or best is None:
        return "no_band_yet", {}
    return ("held_back_below" if last_held < best else "held_back_inside"), {}
