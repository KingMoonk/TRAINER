"""Actual Q·K attention matrix and causal probabilities for a selected head."""
from __future__ import annotations

import numpy as np

from studio_inspect import forward
from studio_net_common import to_numpy
from studio_net_probe import ProbeView


class InsideHeadView(ProbeView):
    title = "Inside a head"
    needed = ("attn.q", "attn.scores", "attn.pattern")

    def compute(self, source, ids, block, head):
        data = forward(source, ids)
        hooks = data["hooks"]
        q = to_numpy(hooks[f"block{block}.attn.q"])
        pattern = to_numpy(hooks[f"block{block}.attn.pattern"])
        if head >= pattern.shape[1]:
            raise ValueError(f"Block {block + 1} has {pattern.shape[1]} heads.")
        logits = hooks.get(f"block{block}.attn.logits")
        if logits is not None:  # what softmax actually sees: score scale (μP) and position bias included
            scores, note = to_numpy(logits)[0, head].T, "Scores are the pre-softmax logits (score scale and position bias applied)."
        else:
            raw = to_numpy(hooks[f"block{block}.attn.scores"])
            scores, note = raw[0, head].T / np.sqrt(q.shape[-1]), "Scores are q·k / √head_dim (this engine exposes no attn.logits; any μP scale or position bias is not shown)."
        return data["tokens"][0], scores, pattern[0, head], q.shape[-1], note

    def describe(self, result):
        tokens, scores, weights, depth, note = result
        count = min(len(tokens), 16)
        shown = range(len(tokens) - count, len(tokens))
        lines = [f"Block {self.block.get()} · head {self.head.get()} · head width {depth}",
                 f"Rows are keys; columns are queries. {note} Future keys are masked.",
                 "", f"Scaled scores (last {count} positions):"]
        for key in shown:
            values = ["   —  " if key > query else f"{scores[key, query]:6.2f}" for query in shown]
            lines.append(f"{key + 1:>2} {tokens[key]!r:>9} " + " ".join(values))
        lines += ["", "Softmax attention weights (each column sums to 1):"]
        for key in shown:
            values = [f"{weights[query, key]:5.2f}" for query in shown]
            lines.append(f"{key + 1:>2} {tokens[key]!r:>9} " + " ".join(values))
        if count < len(tokens):
            earlier = [f"{float(weights[query, :shown.start].sum()):5.2f}" for query in shown]
            lines.append(f"{'earlier positions':>12} " + " ".join(earlier))
        return "\n".join(lines)
