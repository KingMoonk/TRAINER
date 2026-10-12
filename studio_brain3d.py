"""Standalone Panda3D viewer: the transformer drawn in the 3Blue1Brown (Manim) style.

The network is a line of translucent frosted-glass slabs receding into depth -- per block an "Attention" slab
then a "Multilayer Perceptron" slab. Each slab carries the residual stream as a row of bracketed column vectors
(one per token, cells coloured blue<->grey<->red by value); attention slabs draw gold query->key arcs (weight =
arc thickness) plus small per-head causal pattern grids, MLP slabs draw a neuron layer fed by thin grey-blue
connections. Tokens sit as boxed words at the far end.

Launched by studio_vizbridge.VizBridge with `--port P --authkey HEX`; without those it runs in DEMO mode
(sample spec, synthetic telemetry, edits applied locally). `--selftest N --screenshot f.png` renders N offscreen
frames in demo mode, applies synthetic edits, saves a screenshot and prints fps.
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import sys
import time
import uuid
import zlib

import numpy as np
from panda3d.core import (AntialiasAttrib, BitMask32, ClockObject, CollisionBox, CollisionHandlerQueue,
                          CollisionNode, CollisionRay, CollisionTraverser, ColorBlendAttrib, DynamicTextFont,
                          Filename, Geom, GeomLines, GeomNode, GeomTriangles, GeomVertexArrayFormat, GeomVertexData,
                          GeomVertexFormat, GeomVertexWriter, InternalName, OmniBoundingVolume, Point3, SamplerState, Shader,
                          TextNode, TextProperties, TextPropertiesManager, TransparencyAttrib, Vec3, loadPrcFileData)

MLP_SLOTS = 48        # telemetry neurons per block
FWD_SPEED = 3.0       # slabs per second the forward pulse climbs (two slabs per block)
MAX_HEADS = 16
MAX_BLOCKS = 24

# ---- Manim palette
def _hex(h: str) -> np.ndarray:
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (1, 3, 5)], np.float32)


BLUE, BLUE_E, TEAL = _hex("#58C4DD"), _hex("#1C758A"), _hex("#5CD0B3")
YELLOW, YELLOW_C, GOLD = _hex("#FFFF00"), _hex("#F4D345"), _hex("#C78D46")
FIRE_C = _hex("#8CF2FF")   # forward-pass spikes
RED, GREY, WHITE = _hex("#FC6255"), _hex("#888888"), np.ones(3, np.float32)
CELL0 = _hex("#23262d")
GREYBLUE = _hex("#4f6573")
NEUTRAL = _hex("#c9ced6")
BG_TOP, BG_BOT = _hex("#101217"), _hex("#06070a")

# ---- geometry (world units; slabs stack along -Y towards the camera, Z is up)
TOKENS = ("a", "fluffy", "blue", "creature", "roamed", "the", "verdant", "forest")
NT = len(TOKENS)
PX = 1.05                                   # token pitch
TOK_X = ((np.arange(NT) - (NT - 1) / 2) * PX).astype(np.float32)
NC = 10                                     # cells per column vector
CP = 0.30                                   # cell pitch
CELL_HW, CELL_HH = 0.24, 0.104
BRK = 0.335                                 # bracket half-width
COL_Z0 = 1.02                               # column bracket bottom
COL_Z1 = COL_Z0 + NC * CP + 0.10            # column bracket top
SW2 = 4.5                                   # slab half width (x)
SH = 5.8                                    # slab height (z)
TH = 0.30                                   # slab thickness (y)
DY = 3.4                                    # spacing between consecutive slabs
Y_TOK = DY * 1.15                           # token row sits behind the first slab
LABEL_STAGGER = 0.9                         # z step between alternate blocks' slab captions
FIT_X, FIT_Y = 0.70, 0.86                   # share of the half-screen the column may fill (x / y)
HUD_CX = -0.24                              # NDC x of the column centre: keeps it clear of the top-right edit card
LABEL_FILL = 0.9                            # a caption may fill this share of one slab pitch on screen


def stage_y(s: int) -> float:
    return -s * DY


# ---- shaders
_VS_QUAD = """#version 130
uniform mat4 p3d_ModelViewProjectionMatrix;
in vec4 p3d_Vertex;
in vec2 local;
in vec4 shape;
in vec4 fill;
in vec4 edge;
in float fx;
out vec2 lp;
out vec4 vshape;
out vec4 vfill;
out vec4 vedge;
out float vfx;
void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    lp = local; vshape = shape; vfill = fill; vedge = edge; vfx = fx;
}
"""
_FS_QUAD = """#version 130
in vec2 lp;
in vec4 vshape;
in vec4 vfill;
in vec4 vedge;
in float vfx;
out vec4 o_color;
float sdbox(vec2 p, vec2 b, float r) {
    vec2 q = abs(p) - b + vec2(r);
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
}
void main() {
    float d = sdbox(lp, vshape.xy, vshape.z);
    float aa = max(fwidth(d), 1e-5);
    if (vshape.w < 0.0) {                     // soft halo ring (additive)
        float s = -vshape.w;
        float o = max(d, 0.0) / s;
        float a = exp(-o * o * 2.2) * smoothstep(-2.0 * s, 0.0, d);
        a *= vfx * vfill.a;
        if (a < 0.002) discard;
        o_color = vec4(vfill.rgb * a, a);
        return;
    }
    float cov = 1.0 - smoothstep(-0.5 * aa, 0.5 * aa, d);
    if (cov <= 0.002) discard;
    vec4 c = vfill;
    if (vshape.w > 0.0) {                    // anti-aliased inner stroke, never thinner than ~1px
        float sw = max(vshape.w, aa);
        float band = 1.0 - smoothstep(sw - 0.5 * aa, sw + 0.5 * aa, -d);
        float ea = vedge.a * band * (vshape.w / sw);
        float oa = ea + c.a * (1.0 - ea);
        c = vec4((vedge.rgb * ea + c.rgb * c.a * (1.0 - ea)) / max(oa, 1e-4), oa);
    }
    o_color = vec4(c.rgb, c.a * cov);
}
"""
_VS_RIB = """#version 130
uniform mat4 p3d_ModelViewProjectionMatrix;
in vec4 p3d_Vertex;
in float side;
in vec4 p3d_Color;
out float vs;
out vec4 vc;
void main() { gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex; vs = side; vc = p3d_Color; }
"""
_FS_RIB = """#version 130
in float vs;
in vec4 vc;
out vec4 o_color;
void main() {
    float fw = max(fwidth(vs), 1e-5);
    float a = 1.0 - smoothstep(1.0 - 1.5 * fw, 1.0, abs(vs));
    float thin = clamp(1.0 / (fw * 1.2), 0.0, 1.0);          // sub-pixel ribbons fade instead of aliasing
    a *= mix(thin, 1.0, 0.35) * vc.a;
    if (a < 0.003) discard;
    o_color = vec4(vc.rgb, a);
}
"""


def _glsl(src):
    """macOS only offers GLSL 1.50+ (core profile); everywhere else keep 1.30."""
    return src.replace("#version 130", "#version 150") if sys.platform == "darwin" else src


# --------------------------------------------------------------------------- spec helpers
def _norm_block(b: dict, i: int = 0) -> dict:
    att = dict(b.get("attention") or {})
    mlp = dict(b.get("mlp") or {})
    return {"id": str(b.get("id") or f"blk_{i:06x}"), "label": str(b.get("label") or f"Block {i + 1}"),
            "attention": {"enabled": bool(att.get("enabled", True)), "heads": int(att.get("heads", 4))},
            "mlp": {"enabled": bool(mlp.get("enabled", True)), "expansion": int(mlp.get("expansion", 4)),
                    "layers": int(mlp.get("layers", 1)), "activation": mlp.get("activation", "gelu")}}


def normalize_spec(spec: dict) -> dict:
    """Light, forgiving normalisation (the GUI owns real validation)."""
    out = copy.deepcopy(spec)
    emb = dict(out.get("embedding") or {})
    emb.setdefault("width", 64)
    emb.setdefault("context", 128)
    out["embedding"] = emb
    out["blocks"] = [_norm_block(b, i) for i, b in enumerate((out.get("blocks") or [])[:MAX_BLOCKS])]
    if not out["blocks"]:
        out["blocks"] = [_norm_block({}, 0)]
    return out


def sample_spec(n: int = 6) -> dict:
    """Mixed sample architecture (width 64)."""
    combos = [(4, 4, 1, True, True), (8, 2, 2, True, True), (4, 4, 1, True, False), (4, 6, 1, False, True),
              (16, 8, 3, True, True), (2, 1, 1, True, True)]
    blocks = []
    for i in range(n):
        h, e, l, a, m = combos[i % len(combos)]
        blocks.append({"id": f"blk_{i:06x}", "label": f"Block {i + 1}",
                       "attention": {"enabled": a, "heads": h},
                       "mlp": {"enabled": m, "expansion": e, "layers": l, "activation": "gelu"}})
    return {"format": "groundwork-arch/1", "name": "Demo network",
            "embedding": {"tokenizer": "character", "width": 64, "context": 128, "dropout": 0.1},
            "blocks": blocks, "training": {"steps": 300}}


def valid_heads(width: int, current: int, direction: int):
    """Next head count in `direction` that divides width (1..16), or None."""
    h = current + direction
    while 1 <= h <= MAX_HEADS:
        if width % h == 0:
            return h
        h += direction
    return None


def synth_telemetry(spec: dict, step: int, total: int, t: float) -> dict:
    """Synthetic sinusoidal telemetry in the §2 shape (demo / selftest)."""
    ctx = float(spec.get("embedding", {}).get("context", 128))
    blocks = []
    for i, b in enumerate(spec["blocks"]):
        att, mlp = b["attention"], b["mlp"]
        a = m = None
        if att["enabled"]:
            ent = [max(0.05, math.log(ctx) * (0.5 + 0.45 * math.sin(t * 0.8 + i * 0.7 + h * 1.3)))
                   for h in range(att["heads"])]
            a = {"entropy": ent, "grad_norm": 0.12 + 0.1 * math.sin(t * 1.1 + i)}
        if mlp["enabled"]:
            n = [0.5 + 0.5 * math.sin(t * (1.2 + 0.05 * k) + k * 0.9 + i) for k in range(MLP_SLOTS)]
            m = {"neurons": n, "active_fraction": 0.4, "grad_norm": 0.1 + 0.08 * math.sin(t * 0.9 + i * 0.5)}
        blocks.append({"id": b["id"], "index": i, "residual_rms": 0.6 + 0.07 * i + 0.25 * math.sin(t + i * 0.4),
                       "attention": a, "mlp": m})
    return {"type": "telemetry", "step": step, "total": total,
            "loss": 3.4 * math.exp(-3.0 * step / max(1, total)) + 0.5 + 0.04 * math.sin(t * 2),
            "lr": 1e-3 * min(1.0, step / 20 + 0.05),
            "embedding": {"grad_norm": 0.2 + 0.1 * math.sin(t), "rms": 0.9},
            "blocks": blocks, "output": {"grad_norm": 0.25 + 0.1 * math.sin(t * 1.3)}}


def value_rgba(v) -> np.ndarray:
    """Cell colour: neutral values are faint frosted-white, |v| saturates to BLUE (-) or RED/YELLOW (+)."""
    v = np.clip(np.asarray(v, np.float32), -1.0, 1.0)
    m = np.clip(np.abs(v) * 1.35, 0.0, 1.0)[..., None]
    hue = np.where(v[..., None] < 0, BLUE, RED + (YELLOW_C - RED) * np.clip((v[..., None] - 0.72) / 0.28, 0, 1) * 0.6)
    out = np.empty(v.shape + (4,), np.float32)
    out[..., :3] = NEUTRAL + (hue - NEUTRAL) * m
    out[..., 3] = 0.10 + 0.82 * m[..., 0] ** 0.9
    return out


def diverging(v) -> np.ndarray:
    """Manim-style diverging map: -1 BLUE <- dark grey -> RED -> YELLOW +1. Returns (..., 3)."""
    v = np.clip(np.asarray(v, np.float32), -1.0, 1.0)[..., None]
    neg = CELL0 + (BLUE - CELL0) * np.clip(-v * 1.15, 0.0, 1.0)
    pos = CELL0 + (RED - CELL0) * np.clip(v * 1.25, 0.0, 1.0)
    pos = pos + (YELLOW_C - pos) * (np.clip((v - 0.7) / 0.3, 0.0, 1.0) * 0.65)
    return np.where(v < 0, neg, pos).astype(np.float32)


# --------------------------------------------------------------------------- layout builders (pure numpy)
class _Quads:
    """Rounded-rect / circle / halo quads in XZ planes; dyn row = fill(4) edge(4) fx(1)."""

    def __init__(self):
        self.st, self.dyn, self.grp, self.n = [], [], [], 0

    def add(self, group, centers, half, radius, stroke, fill, edge=(0, 0, 0, 0), margin=0.0, fx=0.0):
        c = np.atleast_2d(np.asarray(centers, np.float32))
        n = len(c)
        if not n:
            return np.zeros(0, np.int64)
        half = np.broadcast_to(np.asarray(half, np.float32), (n, 2))
        rad = np.broadcast_to(np.asarray(radius, np.float32), (n,))
        stk = np.broadcast_to(np.asarray(stroke, np.float32), (n,))
        mg = np.broadcast_to(np.asarray(margin, np.float32), (n,))
        ex, ez = half[:, 0] + mg, half[:, 1] + mg
        sx = np.array([-1, 1, 1, -1], np.float32)
        sz = np.array([-1, -1, 1, 1], np.float32)
        lx, lz = ex[:, None] * sx, ez[:, None] * sz
        st = np.zeros((n, 4, 9), np.float32)
        st[:, :, 0] = c[:, None, 0] + lx
        st[:, :, 1] = c[:, None, 1]
        st[:, :, 2] = c[:, None, 2] + lz
        st[:, :, 3], st[:, :, 4] = lx, lz
        st[:, :, 5], st[:, :, 6] = half[:, None, 0], half[:, None, 1]
        st[:, :, 7], st[:, :, 8] = rad[:, None], stk[:, None]
        d = np.zeros((n, 9), np.float32)
        d[:, 0:4] = np.broadcast_to(np.asarray(fill, np.float32), (n, 4))
        d[:, 4:8] = np.broadcast_to(np.asarray(edge, np.float32), (n, 4))
        d[:, 8] = fx
        self.st.append(st.reshape(-1, 9))
        self.dyn.append(d)
        self.grp.append(np.full(n, group, np.int64))
        idx = np.arange(self.n, self.n + n)
        self.n += n
        return idx

    def arrays(self):
        if not self.n:
            return np.zeros((0, 9), np.float32), np.zeros((0, 9), np.float32), np.zeros(0, np.int64)
        return np.concatenate(self.st), np.concatenate(self.dyn), np.concatenate(self.grp)


class _Lines:
    def __init__(self):
        self.p0, self.p1, self.col, self.grp, self.n = [], [], [], [], 0

    def add(self, group, p0, p1, col):
        p0 = np.atleast_2d(np.asarray(p0, np.float32))
        p1 = np.atleast_2d(np.asarray(p1, np.float32))
        n = len(p0)
        self.p0.append(p0)
        self.p1.append(p1)
        self.col.append(np.broadcast_to(np.asarray(col, np.float32), (n, 4)).copy())
        self.grp.append(np.full(n, group, np.int64))
        idx = np.arange(self.n, self.n + n)
        self.n += n
        return idx

    def polyline(self, group, pts, col, closed=False):
        pts = np.asarray(pts, np.float32)
        a, b = pts[:-1], pts[1:]
        if closed:
            a, b = np.concatenate([a, pts[-1:]]), np.concatenate([b, pts[:1]])
        return self.add(group, a, b, col)

    def arrays(self):
        if not self.n:
            z = np.zeros((0, 3), np.float32)
            return z, z, np.zeros((0, 4), np.float32), np.zeros(0, np.int64)
        return (np.concatenate(self.p0), np.concatenate(self.p1), np.concatenate(self.col),
                np.concatenate(self.grp))


def _bracket(L: _Lines, g, x, y, z0, z1, col, half=BRK, tick=0.09):
    for sgn in (-1, 1):
        xe = x + sgn * half
        L.add(g, [[xe, y, z0], [xe, y, z0], [xe, y, z1]],
              [[xe, y, z1], [xe - sgn * tick, y, z0], [xe - sgn * tick, y, z1]], col)


def build_layout(spec: dict) -> dict:
    """Procedural description of the whole scene for `spec` (no Panda objects)."""
    blocks = spec["blocks"]
    nb = len(blocks)
    ns = 2 * nb                                   # slabs: attention, mlp per block
    Q, H, L = _Quads(), _Quads(), _Lines()
    texts = []                                    # (group, text, pos, scale, rgba, align, font)
    G_TOK = 0                                     # group 0 = token row, slab s -> group s + 1
    yc = lambda s: stage_y(s) - TH * 0.5 - 0.02   # content plane just in front of slab s
    rows = []                                     # (cell idx, amp source: -1 embedding or block index)
    bracket_col = (0.86, 0.88, 0.92, 0.85)

    def add_row(g, y, amp_src):
        xs = np.repeat(TOK_X, NC)
        zs = np.tile(COL_Z0 + 0.05 + CP * (np.arange(NC) + 0.5), NT)
        idx = Q.add(g, np.column_stack([xs, np.full_like(xs, y), zs]), (CELL_HW, CELL_HH), 0.035, 0.011,
                    (0.2, 0.2, 0.2, 0.5), (0.92, 0.94, 0.97, 0.42))
        for x in TOK_X:
            _bracket(L, g, x, y, COL_Z0, COL_Z1, bracket_col)
        rows.append((idx, amp_src))

    # ---- token row (far end): embedding column vectors + boxed words with drop lines
    add_row(G_TOK, Y_TOK, -1)
    for k, w in enumerate(TOKENS):
        x = float(TOK_X[k])
        hw = min(0.052 * len(w) + 0.10, PX * 0.5 - 0.05)
        L.polyline(G_TOK, [[x - hw, Y_TOK, 4.5], [x + hw, Y_TOK, 4.5], [x + hw, Y_TOK, 5.0], [x - hw, Y_TOK, 5.0]],
                   (0.75, 0.8, 0.88, 0.8), closed=True)
        L.add(G_TOK, [[x, Y_TOK, 4.5]], [[x, Y_TOK, COL_Z1 + 0.04]], (0.75, 0.8, 0.88, 0.6))
        texts.append((G_TOK, w, (x, Y_TOK - 0.01, 4.66), 0.2,
                      (1, 1, 1, 1), "c", "serif"))
    # caption sits above the embedding column and reads rightwards, so it never runs off the left edge
    texts.append((G_TOK, "Embedding", (float(TOK_X[0]) - 0.45, Y_TOK - 0.01, 5.15), 0.3, (0.8, 0.82, 0.86, 1), "l",
                  "serif"))

    att_heads, att_cells, att_logits, att_blk = [], [], [], []
    arcs = []                                     # (block, q, k, center pts, normals)
    neur_idx, neur_slot, neur_halo = [], [], []
    fan_idx, fan_slot, fan_w = [], [], []
    edge_idx, edge_stage = [], []
    sel_halo_idx, sel_halo_stage = [], []
    boxes = []                                    # (block, stage)
    tril = np.tril_indices(NT)
    q_ar, k_ar = np.tril_indices(NT, -1)
    nseg = 22
    for i, blk in enumerate(blocks):
        rng = np.random.default_rng(zlib.crc32(blk["id"].encode()))
        for part in (0, 1):
            s = 2 * i + part
            g = s + 1
            ys, y = stage_y(s), yc(s)
            on = blk["attention"]["enabled"] if part == 0 else blk["mlp"]["enabled"]
            boxes.append((i, s))
            # glass edges (12 box edges) -- selection / shimmer driven
            x0, x1, y0, y1 = -SW2, SW2, ys - TH / 2, ys + TH / 2
            P = [(x0, y0, 0), (x1, y0, 0), (x1, y0, SH), (x0, y0, SH), (x0, y1, 0), (x1, y1, 0), (x1, y1, SH),
                 (x0, y1, SH)]
            E = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
            ei = L.add(g, [P[a] for a, _ in E], [P[b] for _, b in E], (0.85, 0.87, 0.9, 0.5 if on else 0.25))
            edge_idx.append(ei)
            edge_stage.append(np.full(len(ei), s))
            hi = H.add(g, [[0.0, y0 - 0.01, SH / 2]], (SW2, SH / 2), 0.02, -0.06, (*YELLOW_C, 0.45), margin=0.25)
            sel_halo_idx.append(hi)
            sel_halo_stage.append(np.full(len(hi), s))
            # residual stream at this stage, wired token-for-token to the previous stage's columns
            add_row(g, y, i)
            yp = Y_TOK if s == 0 else yc(s - 1)
            rail = np.array([[x + dx, 0.0, z] for x in TOK_X for dx in (-BRK, BRK) for z in (COL_Z0, COL_Z1)],
                            np.float32)
            L.add(g, rail + [0, yp, 0], rail + [0, y, 0], (0.75, 0.8, 0.88, 0.5))
            name = ("Attention" if part == 0 else "Multilayer\nPerceptron") if on else \
                ("Attention (off)" if part == 0 else "MLP (off)")
            # captions float clear of the token-word row (which sits deeper and so projects higher on screen);
            # odd blocks sit a step higher so neighbouring captions never share a height band
            tz = (SH + 1.65 if (part == 1 and on) else SH + 1.0) + LABEL_STAGGER * (i % 2)
            texts.append((g, name, (-SW2 + 0.6 if part == 0 else SW2 - 0.15, y, tz), 0.66 if part == 0 else 0.54,
                          (1, 1, 1, 1) if on else (0.55, 0.57, 0.6, 1), "l" if part == 0 else "r", "serif"))
            if not on:
                continue
            if part == 0:
                # ---- per-head causal pattern grids along the bottom strip
                Hn = int(np.clip(blk["attention"]["heads"], 1, MAX_HEADS))
                gs = min(0.8, (2 * SW2 - 0.6 - 0.08 * (Hn - 1)) / Hn)
                tot = Hn * gs + (Hn - 1) * 0.08
                cs = gs / NT
                for h in range(Hn):
                    gx = -tot / 2 + gs / 2 + h * (gs + 0.08)
                    gz = 0.12 + gs / 2
                    qq, kk = np.mgrid[0:NT, 0:NT]
                    cx = gx - gs / 2 + (kk + 0.5) * cs
                    cz = gz + gs / 2 - (qq + 0.5) * cs
                    cent = np.column_stack([cx.ravel(), np.full(NT * NT, y), cz.ravel()])
                    lower = (kk <= qq).ravel()
                    Q.add(g, cent[~lower], (cs * 0.4, cs * 0.4), cs * 0.1, 0.0, (0.12, 0.13, 0.15, 0.7))
                    ci = Q.add(g, cent[lower][np.argsort(np.ravel_multi_index(
                        (qq.ravel()[lower], kk.ravel()[lower]), (NT, NT)))], (cs * 0.4, cs * 0.4), cs * 0.1, 0.0,
                        (0.2, 0.2, 0.2, 1))
                    att_cells.append(ci)
                    L.polyline(g, [[gx - gs / 2 - 0.02, y, gz - gs / 2 - 0.02], [gx + gs / 2 + 0.02, y, gz - gs / 2 - 0.02],
                                   [gx + gs / 2 + 0.02, y, gz + gs / 2 + 0.02], [gx - gs / 2 - 0.02, y, gz + gs / 2 + 0.02]],
                               (0.6, 0.63, 0.68, 0.45), closed=True)
                    kind = (h + i) % 4
                    lg = rng.normal(0, 0.6, (NT, NT)).astype(np.float32)
                    r = np.arange(NT)
                    if kind == 0:
                        lg[r[1:], r[:-1]] += 3.0
                    elif kind == 1:
                        lg[r, np.maximum(r - 2, 0)] += 2.6
                    elif kind == 2:
                        lg[:, 0] += 2.4
                    else:
                        lg[r, rng.integers(0, r + 1)] += 3.0
                    lg[r, r] += 0.6
                    att_logits.append(lg)
                    att_blk.append(i)
                    att_heads.append(h)
                # ---- gold arcs above the columns (query q -> key k < q)
                zb = COL_Z1 + 0.10
                th = np.linspace(0, np.pi, nseg + 1, dtype=np.float32)
                for q, k in zip(q_ar, k_ar):
                    xa, xb = float(TOK_X[k]), float(TOK_X[q])
                    hh = 0.28 + 0.17 * (q - k)
                    mid, half = (xa + xb) / 2, (xb - xa) / 2
                    px = mid - half * np.cos(th)
                    pz = zb + hh * np.sin(th)
                    tx, tz_ = half * np.sin(th), hh * np.cos(th)
                    nrm = np.stack([-tz_, tx], 1)
                    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-6
                    # the arc leaves key k in the previous stage and lands on query q in this one
                    arcs.append((i, np.column_stack([px, np.linspace(yp, y - 0.01, nseg + 1, dtype=np.float32), pz]),
                                 np.column_stack([nrm[:, 0], np.zeros_like(px), nrm[:, 1]])))
            else:
                # ---- MLP: neuron rows above the columns, fans from column tops
                E = int(np.clip(blk["mlp"]["expansion"], 1, 8))
                Ly = int(np.clip(blk["mlp"]["layers"], 1, 4))
                N = int(min(MLP_SLOTS, 12 * E))
                nx = np.linspace(-SW2 + 0.55, SW2 - 0.55, N).astype(np.float32)
                rad = float(min(0.12, (nx[1] - nx[0]) * 0.38 if N > 1 else 0.12))
                layers = []
                for j in range(Ly):
                    nz = COL_Z1 + 0.62 + j * 0.36
                    cent = np.column_stack([nx, np.full(N, y - 0.01), np.full(N, nz)])
                    ni = Q.add(g, cent, (rad, rad), rad, rad * 0.22, (0.1, 0.1, 0.1, 1), (0.85, 0.9, 0.95, 0.85))
                    slot = np.arange(N) % MLP_SLOTS if j == Ly - 1 else (np.arange(N) * 7 + j * 13) % MLP_SLOTS
                    hi = H.add(g, cent, (rad, rad), rad, -rad * 1.4, (*BLUE, 1.0), margin=rad * 3.0)
                    neur_idx.append(ni)
                    neur_slot.append(i * MLP_SLOTS + slot)
                    neur_halo.append(hi)
                    layers.append((cent, slot))
                c0, s0 = layers[0]
                p0 = np.repeat(np.column_stack([TOK_X, np.full(NT, y), np.full(NT, COL_Z1 + 0.03)]), N, axis=0)
                p1 = np.tile(c0 - np.array([0, 0, rad], np.float32), (NT, 1))
                fan_idx.append(L.add(g, p0, p1, (0, 0, 0, 0)))
                fan_slot.append(np.tile(i * MLP_SLOTS + s0, NT))
                fan_w.append(rng.normal(0, 1, NT * N).astype(np.float32))
                for j in range(Ly - 1):
                    (ca, _), (cb, sb) = layers[j], layers[j + 1]
                    src = np.repeat(np.arange(N), 3)
                    dst = rng.integers(0, N, 3 * N)
                    fan_idx.append(L.add(g, ca[src] + [0, 0, rad], cb[dst] - [0, 0, rad], (0, 0, 0, 0)))
                    fan_slot.append(i * MLP_SLOTS + sb[dst])
                    fan_w.append(rng.normal(0, 1, 3 * N).astype(np.float32))
        texts.append((("blk", i), f"{i + 1}  {blk['label'][:16]}", (-SW2, yc(2 * i), -0.5), 0.34,
                      (1, 1, 1, 1), "l", "italic"))

    # residual row base values: each stage nudges the previous row
    R = len(rows)
    base = np.zeros((R, NT * NC), np.float32)
    rng0 = np.random.default_rng(7)
    base[0] = rng0.normal(0, 0.7, NT * NC)
    for r in range(1, R):
        rr = np.random.default_rng(1000 + r)
        base[r] = base[r - 1] * 0.86 + rr.normal(0, 0.38, NT * NC)
    # bounding points for camera framing
    corners = np.array([[x, y, z] for x in (-SW2, SW2) for y in (Y_TOK, stage_y(ns - 1) - TH)
                        for z in (-0.6, SH + 2.4)], np.float32)
    cat = lambda xs, dt=np.int64: np.concatenate(xs) if xs else np.zeros(0, dt)
    return {"nb": nb, "ns": ns, "Q": Q.arrays(), "H": H.arrays(), "L": L.arrays(), "texts": texts,
            "row_idx": np.stack([r[0] for r in rows]), "row_amp": np.array([r[1] for r in rows], np.int64),
            "row_base": base, "att_cells": np.stack(att_cells) if att_cells else np.zeros((0, len(tril[0])), np.int64),
            "att_logits": np.stack(att_logits) if att_logits else np.zeros((0, NT, NT), np.float32),
            "att_blk": np.array(att_blk, np.int64), "att_head": np.array(att_heads, np.int64), "arcs": arcs,
            "neur_idx": cat(neur_idx), "neur_slot": cat(neur_slot), "neur_halo": cat(neur_halo),
            "fan_idx": cat(fan_idx), "fan_slot": cat(fan_slot), "fan_w": cat(fan_w, np.float32),
            "edge_idx": cat(edge_idx), "edge_stage": cat(edge_stage),
            "sel_halo_idx": cat(sel_halo_idx), "sel_halo_stage": cat(sel_halo_stage), "boxes": boxes,
            "corners": corners, "ztop": SH + 1.6,
            "center": np.array([0.0, (Y_TOK + stage_y(ns - 1)) / 2, SH / 2], np.float32)}


# --------------------------------------------------------------------------- Panda geometry helpers
def _vfmt(columns_per_array):
    f = GeomVertexFormat()
    for cols in columns_per_array:
        a = GeomVertexArrayFormat()
        for name, n, nt, contents in cols:
            a.addColumn(name, n, nt, contents)
        f.addArray(a)
    return GeomVertexFormat.registerFormat(f)


_FMTS = {}


def _fmt(kind):
    if kind not in _FMTS:
        V, F, O = InternalName.getVertex(), Geom.NTFloat32, Geom.COther
        if kind == "quad":
            _FMTS[kind] = _vfmt([[(V, 3, F, Geom.CPoint), (InternalName.make("local"), 2, F, O),
                                  (InternalName.make("shape"), 4, F, O)],
                                 [(InternalName.make("fill"), 4, F, O), (InternalName.make("edge"), 4, F, O),
                                  (InternalName.make("fx"), 1, F, O)]])
        elif kind == "line":
            _FMTS[kind] = _vfmt([[(V, 3, F, Geom.CPoint)], [(InternalName.getColor(), 4, F, Geom.CColor)]])
        elif kind == "rib":
            _FMTS[kind] = _vfmt([[(V, 3, F, Geom.CPoint)], [(InternalName.make("side"), 1, F, O)],
                                 [(InternalName.getColor(), 4, F, Geom.CColor)]])
        elif kind == "glass":
            _FMTS[kind] = GeomVertexFormat.getV3c4()
    return _FMTS[kind]


def _put(vdata, array_index, arr):
    h = vdata.modifyArray(array_index).modifyHandle()
    h.copyDataFrom(np.ascontiguousarray(arr, np.float32))


def _prim(cls, idx):
    p = cls(Geom.UHStatic)
    p.setIndexType(Geom.NTUint32)
    if len(idx):
        v = p.modifyVertices(len(idx))
        h = v.modifyHandle()
        h.copyDataFrom(np.ascontiguousarray(idx, np.uint32))
        del h
    return p


def _grouped_nodes(name, vdata, cls, per_item, grp, verts_per_item):
    """One GeomNode per group, all sharing `vdata`; per_item = index pattern for one item."""
    out = {}
    if not len(grp):
        return out
    order = np.argsort(grp, kind="stable")
    cuts = np.flatnonzero(np.diff(grp[order])) + 1
    for chunk in np.split(order, cuts):
        idx = (chunk[:, None] * verts_per_item + per_item[None, :]).reshape(-1)
        geom = Geom(vdata)
        geom.addPrimitive(_prim(cls, idx))
        node = GeomNode(f"{name}{int(grp[chunk[0]])}")
        node.addGeom(geom)
        node.setBounds(OmniBoundingVolume())
        node.setFinal(True)
        out[int(grp[chunk[0]])] = node
    return out


class QuadMesh:
    def __init__(self, name, arrays):
        st, dyn, grp = arrays
        self.n = len(dyn)
        self.dyn = dyn.copy()
        self.vdata = GeomVertexData(name, _fmt("quad"), Geom.UHDynamic)
        self.vdata.setNumRows(4 * self.n)
        self.buf = np.zeros((self.n, 4, 9), np.float32)
        if self.n:
            _put(self.vdata, 0, st)
            self.upload()
        self.nodes = _grouped_nodes(name, self.vdata, GeomTriangles, np.array([0, 1, 2, 0, 2, 3], np.uint32), grp, 4)

    def upload(self):
        if self.n:
            self.buf[:] = self.dyn[:, None, :]
            _put(self.vdata, 1, self.buf)


class LineMesh:
    def __init__(self, name, arrays):
        p0, p1, col, grp = arrays
        self.n = len(col)
        self.col = col.copy()
        self.vdata = GeomVertexData(name, _fmt("line"), Geom.UHDynamic)
        self.vdata.setNumRows(2 * self.n)
        self.buf = np.zeros((self.n, 2, 4), np.float32)
        if self.n:
            _put(self.vdata, 0, np.stack([p0, p1], 1))
            self.upload()
        self.nodes = _grouped_nodes(name, self.vdata, GeomLines, np.array([0, 1], np.uint32), grp, 2)

    def upload(self):
        if self.n:
            self.buf[:] = self.col[:, None, :]
            _put(self.vdata, 1, self.buf)


class RibbonMesh:
    """Arcs as AA ribbons whose width is updated per frame (positions rewritten)."""

    def __init__(self, name, arcs, group_of):
        self.na = len(arcs)
        self.ns = len(arcs[0][1]) if arcs else 0
        self.pts = np.stack([a[1] for a in arcs]).astype(np.float32) if arcs else np.zeros((0, 0, 3), np.float32)
        self.nrm = np.stack([a[2] for a in arcs]).astype(np.float32) if arcs else np.zeros((0, 0, 3), np.float32)
        self.vdata = GeomVertexData(name, _fmt("rib"), Geom.UHDynamic)
        nv = self.na * self.ns * 2
        self.vdata.setNumRows(nv)
        self.pos = np.zeros((self.na, self.ns, 2, 3), np.float32)
        self.col = np.zeros((self.na, self.ns, 2, 4), np.float32)
        self.nodes = {}
        if not self.na:
            return
        side = np.tile(np.array([-1.0, 1.0], np.float32), self.na * self.ns)
        _put(self.vdata, 1, side)
        self.set(np.full(self.na, 0.05, np.float32), np.zeros((self.na, 4), np.float32))
        seg = np.arange(self.ns - 1, dtype=np.uint32) * 2
        per = (seg[:, None] + np.array([0, 2, 3, 0, 3, 1], np.uint32)[None, :]).reshape(-1)
        grp = np.array([group_of(a[0]) for a in arcs], np.int64)
        self.nodes = _grouped_nodes(name, self.vdata, GeomTriangles, per, grp, self.ns * 2)

    def set(self, width, rgba):
        if not self.na:
            return
        off = self.nrm * (width[:, None, None] * 0.5)
        self.pos[:, :, 0] = self.pts - off
        self.pos[:, :, 1] = self.pts + off
        self.col[:] = rgba[:, None, None, :]
        _put(self.vdata, 0, self.pos)
        _put(self.vdata, 2, self.col)


def make_glass(name, ys, on):
    """Thin frosted box (front, top, sides) with a slight vertical gradient."""
    vd = GeomVertexData(name, _fmt("glass"), Geom.UHStatic)
    x0, x1, y0, y1 = -SW2, SW2, ys - TH / 2, ys + TH / 2
    a = 0.40 if on else 0.18
    lo, hi_ = np.array([0.56, 0.58, 0.62, a]), np.array([0.70, 0.72, 0.76, a * 1.1])
    faces = [  # corners listed counter-clockwise seen from outside
        ([(x0, y0, 0), (x1, y0, 0), (x1, y0, SH), (x0, y0, SH)], "v"),     # front (-y)
        ([(x1, y1, 0), (x0, y1, 0), (x0, y1, SH), (x1, y1, SH)], "v"),     # back
        ([(x0, y0, SH), (x1, y0, SH), (x1, y1, SH), (x0, y1, SH)], "t"),   # top
        ([(x0, y1, 0), (x1, y1, 0), (x1, y0, 0), (x0, y0, 0)], "b"),       # bottom
        ([(x1, y0, 0), (x1, y1, 0), (x1, y1, SH), (x1, y0, SH)], "v"),     # right
        ([(x0, y1, 0), (x0, y0, 0), (x0, y0, SH), (x0, y1, SH)], "v"),     # left
    ]
    pos, col = [], []
    for quad, kind in faces:
        for p in quad:
            pos.append(p)
            if kind == "t":
                col.append(hi_ * [1.08, 1.08, 1.08, 1.25])
            elif kind == "b":
                col.append(lo * [0.8, 0.8, 0.8, 0.9])
            else:
                col.append(lo + (hi_ - lo) * (p[2] / SH))
    vd.setNumRows(len(pos))
    wv, wc = GeomVertexWriter(vd, "vertex"), GeomVertexWriter(vd, "color")
    for p, c in zip(pos, col):
        wv.addData3(*p)
        wc.addData4(*np.clip(c, 0, 1))
    idx = (np.arange(6, dtype=np.uint32)[:, None] * 4 + np.array([0, 1, 2, 0, 2, 3], np.uint32)[None, :]).reshape(-1)
    geom = Geom(vd)
    geom.addPrimitive(_prim(GeomTriangles, idx))
    node = GeomNode(name)
    node.addGeom(geom)
    return node


# --------------------------------------------------------------------------- the app
def _import_showbase():
    from direct.showbase.ShowBase import ShowBase
    return ShowBase


class BrainApp:
    """Owns the Panda3D ShowBase (created lazily so prc settings can be applied first)."""

    def __new__(cls, *a, **k):  # build a ShowBase subclass on demand
        Base = _import_showbase()
        impl = type("BrainShowBase", (_BrainMixin, Base), {})
        return impl(*a, **k)


def _load_font(cands, ppu=72):
    for path, face in cands:
        if os.path.exists(path):
            try:
                f = DynamicTextFont(Filename.fromOsSpecific(path), face)
                if f.isValid():
                    f.setPixelsPerUnit(ppu)
                    f.setPageSize(1024, 1024)
                    f.setMinfilter(SamplerState.FT_linear_mipmap_linear)
                    f.setMagfilter(SamplerState.FT_linear)
                    f.setAnisotropicDegree(4)
                    return f
            except Exception:
                pass
    return None


_FONTS = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")


class _BrainMixin:
    FRONT_H = -128.0         # heading of the cinematic 3/4 view (line recedes to the upper left)
    FRONT_P = 17.0

    def __init__(self, link=None, demo=True, size=(1280, 800), selftest=False):
        _import_showbase().__init__(self)
        from direct.gui import DirectGuiGlobals as DGG
        from direct.gui.DirectGui import DirectButton, DirectFrame
        from direct.gui.OnscreenText import OnscreenText
        self._G = (DGG, DirectButton, DirectFrame, OnscreenText)
        self.link = link
        self.demo = bool(demo)
        self.selftest = selftest
        self.spec = normalize_spec(sample_spec(6))
        self.sel = None
        self.sel_id = None
        self.world = None
        self.layout = None
        self.closed_sent = False
        self._ever_connected = bool(link is not None and getattr(link, "connected", False))
        self._lost_at = None
        self.auto_orbit = True
        self._sway = 0.0
        self._flipped = None
        self.disableMouse()
        self.setBackgroundColor(*BG_BOT, 1)
        self.camLens.setFov(40)
        self.camLens.setNearFar(0.3, 4000)
        self.render.setAntialias(AntialiasAttrib.MMultisample)
        self.quad_shader = Shader.make(Shader.SL_GLSL, _glsl(_VS_QUAD), _glsl(_FS_QUAD))
        self.rib_shader = Shader.make(Shader.SL_GLSL, _glsl(_VS_RIB), _glsl(_FS_RIB))
        F = _FONTS
        M = "/System/Library/Fonts/Supplemental"  # macOS fallbacks
        self.font_serif = _load_font([(os.path.join(F, "cambria.ttc"), 0), (os.path.join(F, "times.ttf"), 0),
                                      (os.path.join(M, "Times New Roman.ttf"), 0)], 80)
        self.font_ital = _load_font([(os.path.join(F, "cambriai.ttf"), 0), (os.path.join(F, "timesi.ttf"), 0),
                                     (os.path.join(M, "Times New Roman Italic.ttf"), 0)], 64) \
            or self.font_serif
        self.font_ui = _load_font([(os.path.join(F, "segoeui.ttf"), 0), (os.path.join(F, "cambria.ttc"), 0),
                                   (os.path.join(M, "Arial.ttf"), 0)], 48) \
            or self.font_serif
        self._make_background()
        # camera
        self.cam_h, self.cam_p, self.cam_d = self.FRONT_H, self.FRONT_P, 40.0
        self.cam_t = Point3(0, 0, 3)
        self._drag = None
        self._pan = None
        self._down_pos = None
        # picking
        self.traverser = CollisionTraverser("pick")
        self.cqueue = CollisionHandlerQueue()
        self.pick_node = CollisionNode("pickray")
        self.pick_ray = CollisionRay()
        self.pick_node.addSolid(self.pick_ray)
        self.pick_node.setFromCollideMask(BitMask32.bit(1))
        self.pick_node.setIntoCollideMask(BitMask32.allOff())
        self.traverser.addCollider(self.camera.attachNewNode(self.pick_node), self.cqueue)
        # state
        self.t_anim = 0.0
        self.fwd_t0 = -1e9       # when the latest forward pass (telemetry sample) arrived
        self.tele = None
        self.step_info = (0, 0, 0.0, 0.0)
        self.demo_step = 0
        self._fps_t = 0.0
        self._build_hud()
        self.set_spec(self.spec, fit=True)
        for ev, fn in (("mouse1", self._m1_down), ("mouse1-up", self._m1_up), ("mouse3", self._m3_down),
                       ("mouse3-up", self._m3_up), ("wheel_up", lambda: self.zoom(0.9)),
                       ("wheel_down", lambda: self.zoom(1.1)), ("space", self.toggle_orbit),
                       ("o", self.toggle_orbit), ("r", self.reset_camera), ("escape", self.quit_app)):
            self.accept(ev, fn)
        self.taskMgr.add(self._frame_task, "brain-frame")

    def _make_background(self):
        """Subtle vertical gradient card glued to the camera, drawn first."""
        vd = GeomVertexData("bg", _fmt("glass"), Geom.UHStatic)
        vd.setNumRows(4)

        wv, wc = GeomVertexWriter(vd, "vertex"), GeomVertexWriter(vd, "color")
        D, S = 3000.0, 3000.0 * 1.6
        for x, z, c in ((-S, -S, BG_BOT), (S, -S, BG_BOT), (S, S, BG_TOP), (-S, S, BG_TOP)):
            wv.addData3(x, D, z * 0.6)
            wc.addData4(float(c[0]), float(c[1]), float(c[2]), 1.0)
        geom = Geom(vd)
        geom.addPrimitive(_prim(GeomTriangles, np.array([0, 1, 2, 0, 2, 3], np.uint32)))
        node = GeomNode("bg")
        node.addGeom(geom)
        node.setBounds(OmniBoundingVolume())
        np_ = self.cam.attachNewNode(node)
        np_.setBin("background", 0)
        np_.setDepthWrite(False)
        np_.setDepthTest(False)
        np_.setLightOff()
        np_.setTwoSided(True)
        self.bg_np = np_

    # ------------------------------------------------------------------ messaging
    def _send(self, msg):
        if self.link is None:
            return
        try:
            self.link.send(msg)
        except Exception:
            pass

    def userExit(self):  # window close / escape both land here
        if not self.closed_sent:
            self.closed_sent = True
            self._send({"type": "closed"})
            close = getattr(self.link, "close", None)
            if close:
                try:
                    close()
                except Exception:
                    pass
        super().userExit()

    def quit_app(self):
        self.userExit()

    def handle_message(self, msg):
        kind = msg.get("type")
        if kind == "spec":
            sel = msg.get("selected")
            self.set_spec(msg["spec"], selected=sel if "selected" in msg else None, keep_selection="selected" not in msg)
        elif kind == "telemetry":
            self.apply_telemetry(msg)
        elif kind == "select":
            self.select_id(msg.get("block_id"), notify=False)
        elif kind == "quit":
            self.quit_app()

    # ------------------------------------------------------------------ scene
    STATE_KEYS = ("mlp", "head", "res", "agn", "mgn")

    def set_spec(self, spec, selected=None, keep_selection=True, fit=False):
        """Rebuild the scene graph for `spec`, carrying telemetry state over by block id; camera untouched."""
        spec = normalize_spec(spec)
        old_ids = [b["id"] for b in self.spec["blocks"]] if getattr(self, "ids", None) is not None else []
        prev = getattr(self, "st", None)
        prev_sel = self.sel_id
        self.spec = spec
        self.ids = [b["id"] for b in spec["blocks"]]
        self.index = {b: i for i, b in enumerate(self.ids)}
        nb = len(self.ids)
        if self.world is not None:
            self.world.removeNode()
        self.world = self.render.attachNewNode("world")
        lay = self.layout = build_layout(spec)
        self.ctx = float(spec["embedding"].get("context", 128))
        self.width = int(spec["embedding"].get("width", 64))

        def fresh():
            rs = np.random.default_rng(3)
            return {"mlp": (0.25 + 0.5 * rs.random((nb, MLP_SLOTS))).astype(np.float32),
                    "head": np.full((nb, MAX_HEADS), 0.5, np.float32), "res": np.full(nb, 0.5, np.float32),
                    "agn": np.full(nb, 0.3, np.float32), "mgn": np.full(nb, 0.3, np.float32),
                    "emb": np.array([0.3, 0.3], np.float32), "stepf": np.zeros(1, np.float32)}
        self.tgt, self.cur = fresh(), fresh()
        if prev is not None:
            oldmap = {b: i for i, b in enumerate(old_ids)}
            for i, bid in enumerate(self.ids):
                j = oldmap.get(bid)
                if j is None:
                    continue
                for store, src in ((self.tgt, prev[0]), (self.cur, prev[1])):
                    for k in self.STATE_KEYS:
                        store[k][i] = src[k][j]
            for store, src in ((self.tgt, prev[0]), (self.cur, prev[1])):
                store["emb"][:] = src["emb"]
                store["stepf"][:] = src["stepf"]
        self.st = (self.tgt, self.cur)
        # ---- meshes (shared vertex data, one node per slab group for painter's ordering)
        self.qm = QuadMesh("cells", lay["Q"])
        self.hm = QuadMesh("halo", lay["H"])
        self.lm = LineMesh("lines", lay["L"])
        self.rm = RibbonMesh("arcs", lay["arcs"], lambda i: 2 * i + 1)
        self.line_base = self.lm.col.copy()
        hst = lay["H"][1]
        self.halo_base = hst.copy()
        rq = lay["row_idx"]
        self.cell_y = lay["Q"][0][::4, 1]
        self.groups = {}
        ns = lay["ns"]
        for g in range(ns + 1):
            nps = {}
            if g >= 1:
                s = g - 1
                gl = self.world.attachNewNode(make_glass(f"glass{s}", stage_y(s), self._stage_on(s)))
                gl.setTransparency(TransparencyAttrib.MAlpha)
                nps["glass"] = gl
            for key, mesh, shader in (("halo", self.hm, self.quad_shader), ("lines", self.lm, None),
                                      ("cells", self.qm, self.quad_shader), ("arcs", self.rm, self.rib_shader)):
                node = mesh.nodes.get(g)
                if node is None:
                    continue
                npth = self.world.attachNewNode(node)
                if shader is not None:
                    npth.setShader(shader)
                if key == "halo":
                    npth.setAttrib(ColorBlendAttrib.make(ColorBlendAttrib.MAdd, ColorBlendAttrib.OOne,
                                                         ColorBlendAttrib.OOne))
                else:
                    npth.setTransparency(TransparencyAttrib.MAlpha)
                nps[key] = npth
            self.groups[g] = nps
        for nps in self.groups.values():
            for npth in nps.values():
                npth.setDepthWrite(False)
                npth.setDepthTest(False)
                npth.setTwoSided(npth.getName().startswith("glass") is False)
                npth.setLightOff()
        self.lines_np = [n["lines"] for n in self.groups.values() if "lines" in n]
        for npth in self.lines_np:
            npth.setRenderModeThickness(1.25)
        # ---- labels: static text flattened per group, block captions kept separate (selection colour)
        self.label_nps = [None] * nb
        tgroups = {}
        tpm = TextPropertiesManager.getGlobalPtr()
        if not tpm.hasProperties("sub"):
            tp = TextProperties()
            tp.setGlyphScale(0.7)
            tp.setGlyphShift(-0.25)
            tpm.setProperties("sub", tp)
        self.slab_labels = []                    # (nodepath, width at scale 1, base scale) for per-frame fitting
        for g, text, pos, scale, rgba, align, font in lay["texts"]:
            tn = TextNode("t")
            f = {"serif": self.font_serif, "italic": self.font_ital}.get(font)
            if f is not None:
                tn.setFont(f)
            tn.setText(text)
            tn.setAlign({"l": TextNode.ALeft, "c": TextNode.ACenter, "r": TextNode.ARight}[align])
            tn.setTextColor(*rgba)
            slab_cap = isinstance(g, int) and g >= 1
            if isinstance(g, tuple):
                tnp = self.world.attachNewNode(tn)
                self.label_nps[g[1]] = tnp
                grp_key = 2 * g[1] + 1
            elif slab_cap:
                # slab captions face the eye and are fitted to the slab pitch every frame (see _fit_slab_labels)
                tnp = self.world.attachNewNode(tn)
                tnp.setBillboardPointEye()
                width = max(tn.calcWidth(line) for line in text.split("\n"))
                self.slab_labels.append((tnp, float(width), float(scale)))
                self.groups[g]["slab"] = tnp
                grp_key = g
            else:
                parent = tgroups.get(g)
                if parent is None:
                    parent = tgroups[g] = self.world.attachNewNode(f"text{g}")
                tnp = parent.attachNewNode(tn.generate())
                grp_key = g
            tnp.setPos(*pos)
            tnp.setScale(scale)
            tnp.setTransparency(TransparencyAttrib.MAlpha)
            tnp.setDepthTest(False)
            tnp.setDepthWrite(False)
            tnp.setLightOff()
            if isinstance(g, tuple):
                self.groups[grp_key]["blabel"] = tnp
        for g, parent in tgroups.items():
            parent.flattenStrong()
            self.groups[g]["text"] = parent
        self._flipped = None
        self._order_bins()
        # ---- picking solids (one box per slab, tagged with its block)
        for i, s in lay["boxes"]:
            cn = CollisionNode(f"slab{s}")
            cn.addSolid(CollisionBox(Point3(0, stage_y(s), SH / 2), SW2, TH / 2 + 0.06, SH / 2))
            cn.setIntoCollideMask(BitMask32.bit(1))
            cn.setFromCollideMask(BitMask32.allOff())
            cnp = self.world.attachNewNode(cn)
            cnp.setPythonTag("blk", i)
        # dynamic index helpers
        self.row_idx = rq
        self.overlay = None
        want = selected if (selected is not None) else (prev_sel if keep_selection else None)
        self.sel = None
        self.sel_id = None
        self._set_selection(self.index[want] if want in self.index else None)
        if fit:
            self.reset_camera()
        elif nb != len(old_ids):  # block count changed: re-frame the column, keep the user's orbit angle/pitch
            self._frame_column()
        self.refresh_panel()
        self.animate(0.0)

    def _stage_on(self, s):
        b = self.spec["blocks"][s // 2]
        return b["attention"]["enabled"] if s % 2 == 0 else b["mlp"]["enabled"]

    _ORDER = ("glass", "halo", "lines", "cells", "arcs", "text", "slab", "blabel")

    def _order_bins(self):
        """Painter's order: far groups first. Flips when the camera moves behind the line."""
        cp = self.camera.getPos(self.render)
        flipped = bool(cp.getY() > (self.layout["center"][1] if self.layout else 0.0) + 2 * DY * 0 and
                       cp.getY() > Y_TOK)
        if flipped == self._flipped:
            return
        self._flipped = flipped
        n = len(self.groups)
        for g, nps in self.groups.items():
            o = (n - 1 - g) if flipped else g
            order = self._ORDER if not flipped else tuple(k for k in self._ORDER if k != "glass") + ("glass",)
            for j, key in enumerate(order):
                if key in nps:
                    nps[key].setBin("fixed", 100 + o * 10 + j)

    def _set_selection(self, idx):
        self.sel = idx
        self.sel_id = self.ids[idx] if idx is not None else None
        for i, lnp in enumerate(self.label_nps):
            if lnp is not None:
                if i == idx:
                    lnp.setColorScale(float(YELLOW_C[0]), float(YELLOW_C[1]), float(YELLOW_C[2]), 1.0)
                else:
                    lnp.setColorScale(0.62, 0.65, 0.7, 1.0)

    def select_idx(self, idx, notify=True):
        if idx is None or not (0 <= idx < len(self.ids)):
            return
        self._set_selection(idx)
        self.refresh_panel()
        if notify:
            self._send({"type": "select", "block_id": self.sel_id})

    def select_id(self, bid, notify=True):
        if bid in self.index:
            self.select_idx(self.index[bid], notify)
        elif bid is None:
            self._set_selection(None)
            self.refresh_panel()

    # ------------------------------------------------------------------ telemetry
    def apply_telemetry(self, msg):
        t = self.tgt
        self.step_info = (int(msg.get("step", 0)), int(msg.get("total", 0)), float(msg.get("loss", 0.0)),
                          float(msg.get("lr", 0.0)))
        self.tele = msg
        self.fwd_t0 = self.t_anim
        nb = len(self.ids)
        lnctx = math.log(max(self.ctx, 2.0))
        for pos, e in enumerate(msg.get("blocks") or []):
            i = self.index.get(e.get("id"))
            if i is None:
                i = e.get("index", pos)
            if not isinstance(i, int) or not (0 <= i < nb):
                continue
            t["res"][i] = math.tanh(float(e.get("residual_rms", 0.5)) / 1.2)
            a = e.get("attention")
            if a:
                ent = np.asarray(a.get("entropy") or [], np.float32)[:MAX_HEADS]
                t["head"][i, :] = 0.5
                t["head"][i, :len(ent)] = np.clip(1.0 - ent / lnctx, 0.0, 1.0)   # focus = 1 - normalised entropy
                t["agn"][i] = math.tanh(float(a.get("grad_norm", 0.0)) * 5)
            m = e.get("mlp")
            if m:
                nr = np.asarray(m.get("neurons") or [0.3], np.float32)
                t["mlp"][i] = np.resize(np.clip(nr, 0, 1), MLP_SLOTS)
                t["mgn"][i] = math.tanh(float(m.get("grad_norm", 0.0)) * 5)
        emb, out = msg.get("embedding") or {}, msg.get("output") or {}
        t["emb"][0] = math.tanh(float(emb.get("grad_norm", 0.1)) * 5)
        t["emb"][1] = math.tanh(float(out.get("grad_norm", 0.1)) * 5)
        t["stepf"][0] = float(msg.get("step", 0))
        self._update_hud_text()

    def _update_hud_text(self):
        step, total, loss, lr = self.step_info
        mode = "DEMO" if self.demo else "LIVE"
        self.hud_text.setText(f"{mode}  \u00b7  step {step} / {total}\nloss {loss:.4f}    lr {lr:.2e}\n"
                              f"{len(self.ids)} blocks  \u00b7  {self._fps:.0f} fps")

    _fps = 0.0

    # ------------------------------------------------------------------ per-frame
    def _frame_task(self, task):
        dt = min(float(ClockObject.getGlobalClock().getDt()), 0.1)
        self.t_anim += dt
        if self.link is not None:
            try:
                for m in self.link.poll():
                    self.handle_message(m)
            except Exception:
                pass
            if self._ever_connected and not getattr(self.link, "connected", True):
                self._lost_at = self._lost_at or time.time()
                if time.time() - self._lost_at > 1.5:
                    self.closed_sent = True
                    self.userExit()
            elif getattr(self.link, "connected", False):
                self._ever_connected = True
                self._lost_at = None
        elif self.demo and not self.selftest:
            if int(self.t_anim * 8) != int((self.t_anim - dt) * 8):
                self.demo_step = (self.demo_step + 1) % 301
                self.apply_telemetry(synth_telemetry(self.spec, self.demo_step, 300, self.t_anim))
        self._camera_input(dt)
        self._order_bins()
        self.animate(dt)
        self._fps_t += dt
        if self._fps_t > 0.5:
            self._fps_t = 0.0
            self._fps = ClockObject.getGlobalClock().getAverageFrameRate()
            self._update_hud_text()
        return task.cont

    def animate(self, dt):
        k = 1.0 - math.exp(-dt * 5.0)
        for key in self.cur:
            if key == "stepf" and abs(self.tgt[key][0] - self.cur[key][0]) > 50:
                self.cur[key][:] = self.tgt[key]
            self.cur[key] += (self.tgt[key] - self.cur[key]) * k
        c, lay, t = self.cur, self.layout, self.t_anim
        ns = lay["ns"]
        dyn = self.qm.dyn
        # ---- backprop shimmer: a soft gold wave travelling from the last slab back to the tokens
        gstage = np.repeat(0.5 * (c["agn"] + c["mgn"]), 2)
        period = 3.5 + 0.06 * ns
        wave = (ns + 1.5) - ((t / period) % 1.0) * (ns + 4.0)
        shim = (0.25 + 0.75 * gstage) * np.exp(-((np.arange(ns) - wave) / 0.9) ** 2)
        shim_tok = (0.25 + 0.75 * c["emb"][0]) * math.exp(-((-1.0 - wave) / 0.9) ** 2)
        # ---- residual column vectors
        amp = np.where(lay["row_amp"] < 0, 0.7, c["res"][np.maximum(lay["row_amp"], 0)])
        R = len(amp)
        if not hasattr(self, "_row_ph") or self._row_ph.shape != lay["row_base"].shape:
            rs = np.random.default_rng(5)
            self._row_ph = rs.uniform(0, 6.283, lay["row_base"].shape).astype(np.float32)
            self._row_w = rs.uniform(0.4, 1.3, lay["row_base"].shape).astype(np.float32)
        sp = float(c["stepf"][0])
        v = np.tanh(1.2 * (lay["row_base"] * (0.55 + 0.8 * amp[:, None]) + 0.22 * np.sin(t * self._row_w + self._row_ph)
                           + 0.15 * np.sin(sp * 0.02 + self._row_ph * 1.7)))
        col = value_rgba(v)
        rshim = np.concatenate([[shim_tok], shim])[:R] if R == ns + 1 else np.zeros(R, np.float32)
        col[..., :3] += GOLD * (rshim[:, None, None] * 0.35)
        col[..., 3] += rshim[:, None] * 0.2
        dyn[self.row_idx, 0:4] = np.clip(col, 0, 1)
        # ---- attention: per-head softmax patterns, sharpened by focus (1 - normalised entropy)
        if len(lay["att_blk"]):
            foc = c["head"][lay["att_blk"], lay["att_head"]]
            sharp = 0.45 + 5.5 * foc ** 1.3
            if not hasattr(self, "_att_ph") or self._att_ph.shape != lay["att_logits"].shape:
                self._att_ph = np.random.default_rng(9).uniform(0, 6.283, lay["att_logits"].shape).astype(np.float32)
            lg = (lay["att_logits"] + 0.45 * np.sin(t * 0.55 + self._att_ph)) * sharp[:, None, None]
            mask = np.tril(np.ones((NT, NT), bool))
            lg = np.where(mask, lg, -np.inf)
            lg -= lg.max(axis=2, keepdims=True)
            w = np.exp(lg)
            w /= w.sum(axis=2, keepdims=True)
            q_, k_ = np.tril_indices(NT)
            wl = w[:, q_, k_]
            b = np.clip(wl * 1.1, 0, 1) ** 0.65
            gc = CELL0 + (BLUE - CELL0) * np.clip(b * 1.35, 0, 1)[..., None]
            gc = gc + (WHITE - gc) * (np.clip((b - 0.6) / 0.4, 0, 1) * 0.85)[..., None]
            dyn[lay["att_cells"], 0:3] = gc
            # arcs: block-mean weights for q -> k < q
            blk = lay["att_blk"]
            starts = np.flatnonzero(np.r_[True, blk[1:] != blk[:-1]])
            wm = np.add.reduceat(w, starts, axis=0) / np.diff(np.r_[starts, len(blk)])[:, None, None]
            qa, ka = np.tril_indices(NT, -1)
            aw = wm[:, qa, ka].ravel()
            width = 0.018 + 0.20 * aw ** 0.85
            ac = np.empty((len(aw), 4), np.float32)
            hot = np.clip((aw - 0.35) / 0.5, 0, 1)[:, None]
            ac[:, :3] = YELLOW_C * (1 - hot) + np.array([1.0, 0.97, 0.75], np.float32) * hot
            ac[:, 3] = np.clip(0.05 + 1.6 * aw, 0, 1)
            self.rm.set(width.astype(np.float32), ac)
        # ---- forward pass: a cyan front climbs the slabs once per telemetry sample; neurons flash by their real activity
        front = (t - self.fwd_t0) * FWD_SPEED - 1.0
        sl = np.arange(ns, dtype=np.float32)
        fire = np.exp(-((sl - front) / 1.1) ** 2) + np.where(sl < front, 0.3 * np.exp(-(front - sl) / 3.0), 0.0)
        fire = np.clip(fire, 0.0, 1.0).astype(np.float32)
        bf = fire[1::2]                                   # per block: the MLP slab fires with the front
        # ---- MLP neurons + halos
        mflat = c["mlp"].ravel()
        if len(lay["neur_idx"]):
            ff = bf[lay["neur_slot"] // MLP_SLOTS]
            a = mflat[lay["neur_slot"]]
            a = np.clip(a + ff * (0.25 + 0.75 * a), 0, 1)
            dyn[lay["neur_idx"], 0:3] = (np.array([0.09, 0.1, 0.12], np.float32)
                                         + (np.array([0.96, 0.98, 1.0], np.float32) - [0.09, 0.1, 0.12]) * (a ** 1.2)[:, None])
            self.hm.dyn[lay["neur_halo"], 8] = np.clip((a - 0.62) * 2.6, 0, 1) * 0.55
        self.qm.upload()
        # ---- lines: fans brighten with their neuron, slab edges carry selection + shimmer
        lc = self.lm.col
        lc[:] = self.line_base
        if len(lay["fan_idx"]):
            a = mflat[lay["fan_slot"]]
            a = np.clip(a + bf[lay["fan_slot"] // MLP_SLOTS] * (0.25 + 0.75 * a), 0, 1)
            fw = np.abs(lay["fan_w"])
            hue = np.where((lay["fan_w"] > 0)[:, None], BLUE, RED)
            m = (a ** 2 * np.minimum(fw, 1.5) / 1.5)[:, None]
            lc[lay["fan_idx"], 0:3] = GREYBLUE + (hue - GREYBLUE) * m
            lc[lay["fan_idx"], 3] = 0.05 + 0.5 * m[:, 0]
        ei, es = lay["edge_idx"], lay["edge_stage"]
        if len(ei):
            sh = shim[es][:, None]
            lc[ei, 0:3] += (GOLD - lc[ei, 0:3]) * np.clip(sh * 0.8, 0, 1)
            lc[ei, 3] = np.clip(lc[ei, 3] + sh[:, 0] * 0.4, 0, 1)
            fe = fire[es][:, None]
            lc[ei, 0:3] += (FIRE_C - lc[ei, 0:3]) * np.clip(fe * 0.9, 0, 1)
            lc[ei, 3] = np.clip(lc[ei, 3] + fe[:, 0] * 0.5, 0, 1)
            if self.sel is not None:
                m = (es // 2) == self.sel
                lc[ei[m], 0:3] = YELLOW_C
                lc[ei[m], 3] = 0.75 + 0.2 * math.sin(t * 2.4)
        self.lm.upload()
        hd = self.hm.dyn
        if len(lay["sel_halo_idx"]):
            on = ((lay["sel_halo_stage"] // 2) == self.sel) if self.sel is not None else np.zeros(
                len(lay["sel_halo_idx"]), bool)
            hd[lay["sel_halo_idx"], 8] = on * (0.75 + 0.25 * math.sin(t * 2.4))
        self.hm.upload()
        self._fit_slab_labels()

    def _fit_slab_labels(self):
        """Shrink slab captions so none is wider than LABEL_FILL of one slab pitch as seen from the current view.
        Captions are billboards, so their on-screen width equals their world width along the camera's right axis.
        Same-kind captions repeat every block (two slabs, DY each along -Y) and alternate height with the block
        parity, so the block step is the pitch they must fit in; it projects onto the right axis as 2*DY*|right.y|."""
        if not self.slab_labels:
            return
        right = self.render.getRelativeVector(self.camera, Vec3(1, 0, 0))
        pitch = 2 * DY * abs(float(right.y))
        for tnp, width, base in self.slab_labels:
            f = 1.0 if width <= 0 else min(1.0, max(0.3, LABEL_FILL * pitch / (width * base)))
            tnp.setScale(base * f)

    # ------------------------------------------------------------------ camera
    def reset_camera(self):
        self.cam_p = self.FRONT_P  # pitch first, so the framing below uses the view it will be shown from
        self._frame_column()

    def _frame_column(self):
        """Aim at the middle of the slab row (tokens .. last slab) and pull back until all of it fits.
        The column is then slid so its projected centre sits at HUD_CX (left of the top-right edit card)."""
        lay = self.layout
        ctr = lay["center"] if lay else np.array([0, 0, 3], np.float32)
        self.cam_t = Point3(float(ctr[0]), float(ctr[1]), float(ctr[2]))
        if lay is None:
            return
        hf, vf = (math.radians(a * 0.5) for a in self.camLens.getFov())
        tx, tz = math.tan(hf) * FIT_X, math.tan(vf) * FIT_Y
        h, p = math.radians(self.cam_h), math.radians(self.cam_p)
        back = np.array([math.cos(p) * math.cos(h), math.cos(p) * math.sin(h), math.sin(p)])
        fwd = -back
        right = np.cross(fwd, [0, 0, 1.0])
        right /= np.linalg.norm(right) + 1e-9
        up = np.cross(right, fwd)
        rel = lay["corners"] - ctr
        x0, y, z = rel @ right, rel @ up, rel @ fwd
        shift, d = 0.0, 8.0
        for _ in range(4):  # the fit depends on the sideways shift and vice versa; two or three passes converge
            x = x0 - shift
            d = float(np.clip(np.max(np.maximum(np.abs(x) / tx - z, np.abs(y) / tz - z)), 8.0, 1500.0))
            nx = x / (tx * (d + z))
            centre = 0.5 * (float(nx.max()) + float(nx.min()))
            shift += (centre - HUD_CX) * tx * d
        self.cam_d = d
        sh = right * shift
        self.cam_t = Point3(self.cam_t.getX() + float(sh[0]), self.cam_t.getY() + float(sh[1]), self.cam_t.getZ())

    def zoom(self, f):
        self.cam_d = float(np.clip(self.cam_d * f, 3.0, 1500.0))

    def toggle_orbit(self):
        self.auto_orbit = not self.auto_orbit
        self.refresh_panel()

    def _m1_down(self):
        mw = self.mouseWatcherNode
        if mw is not None and mw.hasMouse():
            self._drag = (mw.getMouseX(), mw.getMouseY())
            self._down_pos = self._drag

    def _m1_up(self):
        mw = self.mouseWatcherNode
        was, start = self._drag, self._down_pos
        self._drag = self._down_pos = None
        if was is None or mw is None or not mw.hasMouse():
            return
        x, y = mw.getMouseX(), mw.getMouseY()
        if abs(x - start[0]) + abs(y - start[1]) < 0.012:
            self.pick(x, y)

    def _m3_down(self):
        mw = self.mouseWatcherNode
        if mw is not None and mw.hasMouse():
            self._pan = (mw.getMouseX(), mw.getMouseY())

    def _m3_up(self):
        self._pan = None

    def _camera_input(self, dt):
        mw = self.mouseWatcherNode
        if mw is not None and mw.hasMouse():
            x, y = mw.getMouseX(), mw.getMouseY()
            if self._drag is not None:
                self.cam_h -= (x - self._drag[0]) * 160.0
                self.cam_p = float(np.clip(self.cam_p + (y - self._drag[1]) * 90.0, -80, 85))
                self._drag = (x, y)
            if self._pan is not None:
                # pan along the slab row (screen-vertical drag moves along depth)
                h = math.radians(self.cam_h)
                dy = (y - self._pan[1]) * self.cam_d * 0.6
                dx = (x - self._pan[0]) * self.cam_d * 0.6
                rx, ry = -math.sin(h), math.cos(h)          # camera right vector in the ground plane
                self.cam_t = Point3(self.cam_t.getX() - dx * rx, self.cam_t.getY() - dx * ry, self.cam_t.getZ() - dy)
                self._pan = (x, y)
        if self.auto_orbit and self._drag is None:
            # gentle cinematic sway around the current heading (never edge-on)
            self._sway += dt
            self.cam_h += 14.0 * 0.16 * math.cos(self._sway * 0.16) * dt
        h, p, d = math.radians(self.cam_h), math.radians(self.cam_p), self.cam_d
        t = self.cam_t
        self.camera.setPos(t.getX() + d * math.cos(p) * math.cos(h), t.getY() + d * math.cos(p) * math.sin(h),
                           t.getZ() + d * math.sin(p))
        self.camera.lookAt(t)

    def pick(self, mx, my):
        """Ray-pick a slab at normalized screen coords; selects its block. Returns index or None."""
        self.pick_ray.setFromLens(self.camNode, mx, my)
        self.traverser.traverse(self.world)
        if self.cqueue.getNumEntries() == 0:
            return None
        self.cqueue.sortEntries()
        idx = self.cqueue.getEntry(0).getIntoNodePath().getPythonTag("blk")
        if idx is not None:
            self.select_idx(int(idx))
        return idx

    def screen_of_block(self, i):
        """Normalized screen position of a visible spot (top strip of the attention slab) of block i (tests)."""
        pt = Point3(0.0, stage_y(2 * i) - TH / 2, SH - 0.3)
        pc = self.camera.getRelativePoint(self.render, pt)
        from panda3d.core import Point2
        p2 = Point2()
        if self.camLens.project(pc, p2):
            return float(p2.getX()), float(p2.getY())
        return None

    # ------------------------------------------------------------------ HUD
    def _build_hud(self):
        DGG, DirectButton, DirectFrame, OnscreenText = self._G
        fs, fu = self.font_serif, self.font_ui
        self.hud_text = OnscreenText(text="", parent=self.a2dTopLeft, pos=(0.06, -0.09), scale=0.048,
                                     fg=(0.92, 0.93, 0.95, 1), align=TextNode.ALeft, mayChange=True, font=fs)
        self.help_text = OnscreenText(
            text="drag: orbit    wheel: zoom    right-drag: pan    click: select    space: auto-orbit    r: reset",
            parent=self.a2dBottomLeft, pos=(0.06, 0.05), scale=0.034, fg=(0.53, 0.55, 0.6, 1),
            align=TextNode.ALeft, font=fu)
        # diverging colour legend
        leg = DirectFrame(parent=self.a2dBottomRight, frameColor=(0, 0, 0, 0), frameSize=(0, 0.01, 0, 0.01),
                          pos=(-0.62, 0, 0.11))
        vals = np.linspace(-1, 1, 24)
        rg = value_rgba(vals)
        cols = rg[:, :3] * rg[:, 3:] + BG_BOT * (1 - rg[:, 3:])
        for j, cc in enumerate(cols):
            DirectFrame(parent=leg, frameColor=(float(cc[0]), float(cc[1]), float(cc[2]), 1),
                        frameSize=(j * 0.022, j * 0.022 + 0.019, 0, 0.03), relief=DGG.FLAT)
        for x, s in ((0.0, "\u2212"), (0.262, "0"), (0.51, "+")):
            OnscreenText(text=s, parent=leg, pos=(x + 0.01, -0.045), scale=0.035, fg=(0.7, 0.72, 0.76, 1), font=fs)
        OnscreenText(text="value", parent=leg, pos=(0.27, 0.05), scale=0.032, fg=(0.6, 0.62, 0.66, 1), font=fs)
        # edit panel: dark frosted card with a thin light rim
        card = self.a2dTopRight.attachNewNode("card")
        card.setScale(0.8)
        card.setPos(-0.01, 0, -0.01)
        DirectFrame(parent=card, frameColor=(0.55, 0.58, 0.64, 0.35),
                    frameSize=(-0.905, -0.025, -1.365, -0.025), relief=DGG.FLAT)
        self.panel = DirectFrame(parent=card, frameColor=(0.055, 0.06, 0.075, 0.92),
                                 frameSize=(-0.90, -0.03, -1.36, -0.03), relief=DGG.FLAT)
        P = self.panel
        self.p_title = OnscreenText(text="", parent=P, pos=(-0.86, -0.125), scale=0.06, fg=(1, 1, 1, 1),
                                    align=TextNode.ALeft, mayChange=True, font=fs)
        self.p_sub = OnscreenText(text="", parent=P, pos=(-0.86, -0.195), scale=0.036, fg=(0.6, 0.63, 0.68, 1),
                                  align=TextNode.ALeft, mayChange=True, font=fs)
        self.p_vals = {}
        self.p_btns = {}

        def btn(text, x, y, cmd, args=(), w=1.0, scale=0.045):
            half = 0.5 * w / scale
            return DirectButton(parent=P, text=text, scale=scale, pos=(x, 0, y), command=cmd, extraArgs=list(args),
                                frameSize=(-half, half, -0.6, 1.0), relief=DGG.FLAT, text_fg=(0.95, 0.96, 0.98, 1),
                                text_font=fu,
                                frameColor=((0.13, 0.14, 0.17, 1), (0.36, 0.77, 0.87, 1), (0.11, 0.46, 0.54, 1),
                                            (0.1, 0.1, 0.12, 0.5)),
                                rolloverSound=None, clickSound=None, pressEffect=0)

        rows = (("heads", "Heads", -0.33), ("expansion", "Expansion", -0.46), ("layers", "MLP layers", -0.59))
        for key, label, y in rows:
            OnscreenText(text=label, parent=P, pos=(-0.86, y), scale=0.046, fg=(0.85, 0.87, 0.9, 1),
                         align=TextNode.ALeft, font=fs)
            self.p_vals[key] = OnscreenText(text="-", parent=P, pos=(-0.38, y), scale=0.052,
                                            fg=(float(YELLOW_C[0]), float(YELLOW_C[1]), float(YELLOW_C[2]), 1),
                                            mayChange=True, font=fs)
            btn("\u2212", -0.27, y, self.cmd_step, (key, -1), 0.1)
            btn("+", -0.13, y, self.cmd_step, (key, 1), 0.1)
        self.p_btns["attn"] = btn("Attention", -0.465, -0.75, self.cmd_toggle, ("attention",), 0.78)
        self.p_btns["mlp"] = btn("MLP", -0.465, -0.88, self.cmd_toggle, ("mlp",), 0.78)
        btn("+ Add block after", -0.465, -1.01, self.cmd_add, (), 0.78)
        self.p_btns["remove"] = btn("Remove block", -0.465, -1.14, self.cmd_remove, (), 0.78)
        self.p_btns["orbit"] = btn("Auto-orbit", -0.465, -1.27, self.toggle_orbit, (), 0.78)

    def refresh_panel(self):
        i = self.sel
        P_title, P_sub = self.p_title, self.p_sub
        orbit = f"Auto-orbit: {'ON' if self.auto_orbit else 'OFF'}"
        self.p_btns["orbit"]["text"] = orbit
        if i is None:
            P_title.setText("No block selected")
            P_sub.setText("click a slab in the 3D view")
            for v in self.p_vals.values():
                v.setText("-")
            self.p_btns["attn"]["text"] = "Attention"
            self.p_btns["mlp"]["text"] = "MLP"
            return
        b = self.spec["blocks"][i]
        P_title.setText(f"{i + 1}. {b['label'][:20]}")
        P_sub.setText(f"width {self.width}   context {int(self.ctx)}")
        a, m = b["attention"], b["mlp"]
        self.p_vals["heads"].setText(str(a["heads"]) if a["enabled"] else "off")
        self.p_vals["expansion"].setText(f"\u00d7{m['expansion']}" if m["enabled"] else "off")
        self.p_vals["layers"].setText(str(m["layers"]) if m["enabled"] else "off")
        self.p_btns["attn"]["text"] = f"Attention: {'ON' if a['enabled'] else 'OFF'}"
        self.p_btns["mlp"]["text"] = f"MLP: {'ON' if m['enabled'] else 'OFF'}"

    # ------------------------------------------------------------------ edits (viewer -> GUI, or local in demo)
    def _emit(self, msg):
        if self.demo and self.link is None:
            self._apply_local(msg)
        else:
            self._send(msg)

    def cmd_step(self, key, d):
        i = self.sel
        if i is None:
            return
        b = self.spec["blocks"][i]
        if key == "heads":
            if not b["attention"]["enabled"]:
                return
            v = valid_heads(self.width, b["attention"]["heads"], d)
            path = "attention.heads"
        elif key == "expansion":
            if not b["mlp"]["enabled"]:
                return
            v = b["mlp"]["expansion"] + d
            v = v if 1 <= v <= 8 else None
            path = "mlp.expansion"
        else:
            if not b["mlp"]["enabled"]:
                return
            v = b["mlp"]["layers"] + d
            v = v if 1 <= v <= 4 else None
            path = "mlp.layers"
        if v is None:
            return
        self._emit({"type": "edit", "block_id": b["id"], "path": path, "value": v})

    def cmd_toggle(self, which):
        i = self.sel
        if i is None:
            return
        b = self.spec["blocks"][i]
        other = "mlp" if which == "attention" else "attention"
        new = not b[which]["enabled"]
        if not new and not b[other]["enabled"]:
            return  # a block needs attention or mlp
        self._emit({"type": "edit", "block_id": b["id"], "path": f"{which}.enabled", "value": new})

    def cmd_add(self):
        if len(self.ids) >= MAX_BLOCKS:
            return
        self._emit({"type": "add_block", "after": self.sel_id if self.sel_id else (self.ids[-1] if self.ids else None)})

    def cmd_remove(self):
        if self.sel is None or len(self.ids) <= 1:
            return
        self._emit({"type": "remove_block", "block_id": self.sel_id})

    def _apply_local(self, msg):
        spec = copy.deepcopy(self.spec)
        blocks = spec["blocks"]
        ids = [b["id"] for b in blocks]
        kind = msg["type"]
        select = self.sel_id
        if kind == "edit" and msg["block_id"] in ids:
            sect, field = msg["path"].split(".")
            blocks[ids.index(msg["block_id"])][sect][field] = msg["value"]
        elif kind == "add_block" and len(blocks) < MAX_BLOCKS:
            pos = ids.index(msg["after"]) + 1 if msg.get("after") in ids else len(blocks)
            ref = blocks[pos - 1] if pos > 0 else blocks[0]
            nb_ = copy.deepcopy(ref)
            nb_["id"] = "blk_" + uuid.uuid4().hex[:6]
            nb_["label"] = f"Block {len(blocks) + 1}"
            blocks.insert(pos, nb_)
            select = nb_["id"]
        elif kind == "remove_block" and msg["block_id"] in ids and len(blocks) > 1:
            j = ids.index(msg["block_id"])
            blocks.pop(j)
            select = blocks[min(j, len(blocks) - 1)]["id"]
        else:
            return
        self.set_spec(spec, selected=select, keep_selection=False)


# --------------------------------------------------------------------------- selftest
def run_selftest(app, frames: int, shot: str) -> dict:
    """Drive the app offscreen: edits + telemetry + picking, screenshot, then a 24-block fps benchmark."""
    clock = ClockObject.getGlobalClock()
    clock.setMode(ClockObject.MNonRealTime)
    clock.setDt(1 / 30)
    app.demo = True
    app.auto_orbit = False
    app.cam_h = _BrainMixin.FRONT_H
    app.set_spec(sample_spec(6), fit=True)
    sent = []
    app.link = None
    real_send = app._send
    app._send = lambda m: sent.append(m)
    N = max(30, frames)
    script = {
        0.15: lambda: app.pick(*(app.screen_of_block(1) or (0, 0))),
        0.30: lambda: app.cmd_step("heads", 1),
        0.40: lambda: app.cmd_step("expansion", 1),
        0.50: lambda: app.cmd_step("layers", 1),
        0.60: lambda: (app.select_idx(3), app.cmd_toggle("attention")),
        0.70: lambda: app.cmd_add(),
        0.80: lambda: (app.select_idx(len(app.ids) - 1), app.cmd_remove()),
        0.85: lambda: app.select_idx(1),
    }
    marks = {int(N * k): fn for k, fn in script.items()}
    t0 = time.perf_counter()
    for f in range(N):
        if f in marks:
            marks[f]()
        if f % 3 == 0:
            app.apply_telemetry(synth_telemetry(app.spec, f, N, f / 30.0))
        app.taskMgr.step()
    fps6 = N / max(time.perf_counter() - t0, 1e-6)
    ok = app.win.saveScreenshot(Filename.fromOsSpecific(os.path.abspath(shot)))
    result = {"fps6": fps6, "blocks_after_edits": len(app.ids), "screenshot_ok": bool(ok), "selected": app.sel_id}
    # 24-block benchmark
    big = sample_spec(24)
    for i, b in enumerate(big["blocks"]):
        b["id"] = f"blk_{i + 100:06x}"
    app.cam_h = _BrainMixin.FRONT_H
    app.set_spec(big, fit=True)
    app.select_idx(5)
    for f in range(20):  # warmup
        app.apply_telemetry(synth_telemetry(app.spec, f, 100, f / 30.0))
        app.taskMgr.step()
    M = max(60, N // 2)
    t0 = time.perf_counter()
    for f in range(M):
        if f % 3 == 0:
            app.apply_telemetry(synth_telemetry(app.spec, f, M, f / 30.0))
        app.taskMgr.step()
    result["fps24"] = M / max(time.perf_counter() - t0, 1e-6)
    stem, ext = os.path.splitext(shot)
    app.win.saveScreenshot(Filename.fromOsSpecific(os.path.abspath(stem + "_24" + (ext or ".png"))))
    app._send = real_send
    result["messages"] = [m["type"] for m in sent]
    return result


# --------------------------------------------------------------------------- entry
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Network brain viewer")
    ap.add_argument("--port", type=int)
    ap.add_argument("--authkey")
    ap.add_argument("--selftest", type=int, default=0, metavar="N")
    ap.add_argument("--screenshot", default="brain.png")
    ap.add_argument("--size", default="1280x800")
    ap.add_argument("--offscreen", action="store_true", help="no visible window (tests)")
    ap.add_argument("--no-msaa", action="store_true", help="disable multisample anti-aliasing")
    args, _ = ap.parse_known_args(sys.argv[1:] if argv is None else argv)
    w, h = (int(x) for x in args.size.lower().split("x"))
    link = None
    if not args.selftest:
        port, key = args.port, args.authkey
        try:
            from studio_vizbridge import parse_viewer_args
            port, key = parse_viewer_args(sys.argv[1:] if argv is None else argv)
        except ImportError:
            pass
        if port and key:
            try:
                from studio_vizbridge import ViewerLink
                link = ViewerLink(port, key)
            except ImportError:
                link = None
    prc = ["audio-library-name null", "sync-video 0", f"win-size {w} {h}", "window-title Transformer",
           "textures-power-2 none"]
    if sys.platform == "darwin":
        prc.append("gl-version 3 2")
    if not args.no_msaa:
        prc += ["framebuffer-multisample 1", "multisamples 4"]
    if args.selftest or args.offscreen:
        prc += ["window-type offscreen", "framebuffer-hardware 0"]
    loadPrcFileData("", "\n".join(prc))
    app = BrainApp(link=link, demo=link is None, size=(w, h), selftest=bool(args.selftest))
    if args.selftest:
        res = run_selftest(app, args.selftest, args.screenshot)
        print(f"selftest ok: fps(6 blocks)={res['fps6']:.1f} fps(24 blocks)={res['fps24']:.1f} "
              f"blocks={res['blocks_after_edits']} screenshot={args.screenshot}")
        print("messages:", ",".join(res["messages"]))
        app.destroy()
        return 0 if res["screenshot_ok"] else 1
    try:
        app.run()
    except SystemExit:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
