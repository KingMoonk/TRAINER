"""Trace attention and early readouts for one prompt."""
from __future__ import annotations

import numpy as np

from studio_inspect import forward, logit_lens
from studio_net_common import to_numpy
from studio_net_probe import ProbeView


class TraceView(ProbeView):
    title = "Trace a token"
    needed = ("attn.pattern",)

    def compute(self, source, ids, block, head):
        result = forward(source, ids)
        pattern = to_numpy(result["hooks"][f"block{block}.attn.pattern"])
        if head >= pattern.shape[1]:
            raise ValueError(f"Block {block + 1} has {pattern.shape[1]} heads.")
        return result, pattern[0, head, -1], logit_lens(source, ids)

    def describe(self, result):
        data, weights, lens = result
        tokens = data["tokens"][0]
        lines = [f"Last token attends to these earlier positions in block {self.block.get()}, head {self.head.get()}:"]
        lines += [f"  {index + 1:>2}. {token!r:>12}  {float(weight):.4f}" for index, (token, weight) in enumerate(zip(tokens, weights))]
        lines += ["", "Early readout through final norm and output head:"]
        for row in lens:
            guesses = ", ".join(f"{int(token)} ({float(prob):.3f})" for token, prob in
                                zip(to_numpy(row["top_ids"])[0, -1], to_numpy(row["top_probabilities"])[0, -1]))
            lines.append(f"  {row['hook']}: {guesses}")
        lines.append(f"Final prediction token id: {int(np.argmax(to_numpy(data['logits'])[0, -1]))}")
        return "\n".join(lines)
