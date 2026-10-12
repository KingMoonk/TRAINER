"""Actual Q·K attention matrix and causal probabilities for a selected head."""
from __future__ import annotations

import numpy as np

from studio_inspect import forward
from studio_net_probe import ProbeView


class InsideHeadView(ProbeView):
    title = "Inside a head"
    needed = ("attn.q", "attn.scores", "attn.pattern")

    def compute(self, source, text, block, head):
        data = forward(source, text)
        hooks = data["hooks"]
        q = hooks[f"block{block}.attn.q"]
        raw = hooks[f"block{block}.attn.scores"]
        pattern = hooks[f"block{block}.attn.pattern"]
        if head >= pattern.shape[1]:
            raise ValueError(f"Block {block + 1} has {pattern.shape[1]} heads.")
        scores = raw[0, head].T / np.sqrt(q.shape[-1])
        return data["tokens"][0], scores, pattern[0, head], q.shape[-1]

    def describe(self, result):
        tokens, scores, weights, depth = result
        count = min(len(tokens), 16)
        lines = [f"Block {self.block.get()} · head {self.head.get()} · head width {depth}",
                 "Rows are keys; columns are queries. Scores are q·k / √head_dim. Future keys are masked.",
                 "", "Scaled scores (last 16 positions):"]
        for key in range(len(tokens) - count, len(tokens)):
            values = ["   —  " if key > query else f"{scores[key, query]:6.2f}"
                      for query in range(len(tokens) - count, len(tokens))]
            lines.append(f"{key + 1:>2} {tokens[key]!r:>9} " + " ".join(values))
        lines += ["", "Softmax attention weights:"]
        for key in range(len(tokens) - count, len(tokens)):
            values = [f"{weights[query, key]:5.2f}" for query in range(len(tokens) - count, len(tokens))]
            lines.append(f"{key + 1:>2} {tokens[key]!r:>9} " + " ".join(values))
        return "\n".join(lines)
