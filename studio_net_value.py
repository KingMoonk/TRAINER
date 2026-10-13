"""One head's V and output projection as actual checkpoint matrices."""
from __future__ import annotations

import numpy as np

from studio_inspect import probe_model
from studio_net_probe import ProbeView
from studio_spec import block_at


class ValueMapView(ProbeView):
    title = "Value map"
    needed = ("attn.v", "attn.head_out")

    def compute(self, source, text, block, head):
        model, _ = probe_model(source)
        spec = model.spec
        mixer = block_at(spec, block)["mixer"]
        d = spec["dims"]["d_model"]
        h = mixer["config"]["heads"]
        if head >= h:
            raise ValueError(f"Block {block + 1} has {h} heads.")
        depth = d // h
        if mixer["id"] == "attention":
            v = model.p[f"block{block}.qkv.w"][:, 2 * d + head * depth:2 * d + (head + 1) * depth]
            out = model.p[f"block{block}.proj.w"][head * depth:(head + 1) * depth]
        elif mixer["id"] == "gqa":
            kv = mixer["config"]["kv_heads"]
            vhead = head // (h // kv)
            v = model.p[f"block{block}.gqa.v.w"][:, vhead * depth:(vhead + 1) * depth]
            out = model.p[f"block{block}.gqa.proj.w"][head * depth:(head + 1) * depth]
        else:
            raise ValueError("This mixer does not expose a V and output projection pair.")
        return d, h, depth, float(np.linalg.norm(v)), float(np.linalg.norm(out)), int(np.linalg.matrix_rank(v @ out))

    def describe(self, result):
        d, heads, depth, vnorm, onorm, rank = result
        return (f"Block {self.block.get()} · head {self.head.get()}\n\n"
                f"V projection: {d} × {depth}; Frobenius norm {vnorm:.4f}\n"
                f"Output projection: {depth} × {d}; Frobenius norm {onorm:.4f}\n"
                f"Combined V·O rank: {rank} (at most {depth})\n\n"
                f"concat(o₁ … o_{heads}) · W_O = Σ_h o_h · W_O[h].\n"
                "The selected head writes one update into the residual stream.")
