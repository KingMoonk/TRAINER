"""Quick test descriptions shared by the API and desktop, without widget imports."""
from studio_ia.design import WIDTHS, read_spec, say, sizes


def describe(spec):
    """'4 blocks · 4 look-back heads · normal thinking room' for a ModelSpec."""
    s = read_spec(spec)
    return f"{s['L']} blocks · {s['H']} look-back heads · {WIDTHS[s['W']].lower()} thinking room"


def cost_sentence(current, draft):
    """Describe size and work per step using the Builder's cost estimates."""
    a, b = (sizes(spec, read_spec(spec)["V"]) for spec in (current, draft))
    size, work = say(a["params"], b["params"], "bigger", "smaller"), say(a["time"], b["time"], "slower", "faster")
    same = "about the same"
    return f"{'same size' if size == same else size} · {'same work per step' if work == same else work}"
