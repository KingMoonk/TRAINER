"""One actual prediction's objective and backward signal."""
from __future__ import annotations

import numpy as np

from studio_inspect import backward, forward
from studio_net_probe import ProbeView


class LossView(ProbeView):
    title = "Loss"

    def compute(self, source, text, block, head):
        data = forward(source, text, hooks=False)
        if data["ids"].shape[1] < 2:
            raise ValueError("Use at least two tokens to inspect a next-token target.")
        position = data["ids"].shape[1] - 2
        result = backward(source, text, pos=position)
        target = int(result["targets"][0, position])
        probabilities = result["probabilities"][0, position]
        gradient = result["dlogits"][0, position]
        top = np.argsort(probabilities)[-8:][::-1]
        return position, target, float(probabilities[target]), float(result["loss"]), [
            (int(token), float(probabilities[token]), float(gradient[token])) for token in top
        ], result["groups"]

    def describe(self, result):
        position, target, probability, loss, top, groups = result
        objective = self.spec["train"]["objective"]["id"]
        lines = [f"Position {position + 1} predicts token id {target}",
                 f"Objective: {objective} · selected-position loss {loss:.5f} nats",
                 f"p(target) = {probability:.6f}",
                 "Cross-entropy component = −ln p(target)" if objective == "ce" else
                 "The configured objective supplies the displayed gradient.",
                 "", "Token ID       Probability       ∂loss/∂logit"]
        lines += [f"{token:>8}       {p:>11.6f}       {gradient:>14.6f}" for token, p, gradient in top]
        lines += ["", "Gradient norms by model part:"]
        lines += [f"{group}: {norm:.6f}" for group, norm in groups.items()]
        return "\n".join(lines)
