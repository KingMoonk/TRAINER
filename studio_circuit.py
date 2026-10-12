"""Circuit canvas: draws an architecture spec as a glowing PCB of chips, pads and traces.

The canvas is a pure view. It never edits a spec; it reports user intent (select, move, toggle) to its
owner. Telemetry updates only recolour existing items, so streaming events never rebuild the scene.
"""
from __future__ import annotations

import math
import re
import time
import tkinter as tk

from studio_widgets import FONT

BG, GRID, CHIP, BOARD, EDGE = "#060a12", "#0c1526", "#0c1626", "#09111e", "#2a3d5a"
BUS, TEXT, MUTED, ERR, SEL = "#2ee6d0", "#e8f1ff", "#8fa3c0", "#ff4d5e", "#ffffff"
GRAD, HOT, PAD_OFF = "#ff4fa3", "#ff7a45", "#0f1a2c"
ACCENT = {"emb": "#5df2b0", "att": "#5aa9ff", "mlp": "#c383ff", "final": "#ffd166",
          "out": "#7cf2ff", "train": "#ff9f5a", "board": "#2ee6d0", "tied": "#9d7bff"}

BW, GAP, SUB = 212, 52, 36          # block width, gap between chips, bus-to-subchip distance
EMB = (0, -100, 132, 100)
HEAD_COLS, HEAD_STEP, HEAD_SIZE = 6, 24, 19
MLP_COLS, MLP_STEP, MLP_SIZE = 8, 18, 14

_RGB: dict[str, tuple] = {}


def _rgb(color):
    if color not in _RGB:
        _RGB[color] = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
    return _RGB[color]


def mix(a, b, t):
    t = 0.0 if t < 0 else 1.0 if t > 1 else t
    return "#%02x%02x%02x" % tuple(int(round(x + (y - x) * t)) for x, y in zip(_rgb(a), _rgb(b)))


def fmt_params(n):
    return f"{n:,}" if n < 10000 else f"{n / 1000:.1f}K" if n < 1e6 else f"{n / 1e6:.2f}M"


def _num(value, default, lo, hi, cast=int):
    try:
        number = cast(value)
    except (TypeError, ValueError):
        number = default
    return max(lo, min(hi, number))


def lenient(spec):
    """Display-safe copy of a possibly invalid spec (the editor shows drafts while they are being fixed)."""
    spec = spec if isinstance(spec, dict) else {}
    emb = spec.get("embedding") if isinstance(spec.get("embedding"), dict) else {}
    blocks = []
    for i, raw in enumerate(spec.get("blocks") or []):
        if not isinstance(raw, dict):
            continue
        att = raw.get("attention") if isinstance(raw.get("attention"), dict) else {}
        mlp = raw.get("mlp") if isinstance(raw.get("mlp"), dict) else {}
        blocks.append({"id": str(raw.get("id") or f"blk_{i:06x}"), "label": str(raw.get("label") or f"Block {i + 1}"),
                       "att": bool(att.get("enabled", True)), "heads": _num(att.get("heads"), 4, 1, 16),
                       "mlp": bool(mlp.get("enabled", True)), "exp": _num(mlp.get("expansion"), 4, 1, 8),
                       "layers": _num(mlp.get("layers"), 1, 1, 4), "act": str(mlp.get("activation", "gelu"))})
    return {"name": str(spec.get("name") or "My network"), "tok": str(emb.get("tokenizer", "character")),
            "width": _num(emb.get("width"), 64, 1, 4096), "ctx": _num(emb.get("context"), 128, 1, 4096),
            "drop": _num(emb.get("dropout"), 0.1, 0, 1, float), "blocks": blocks,
            "train": spec.get("training") if isinstance(spec.get("training"), dict) else {}}


def _ceil(a, b):
    return -(-a // b)


def layout(s):
    """World-space geometry (bus at y=0). `s` is a lenient spec."""
    x = EMB[2] + GAP
    blocks, top, bottom = [], -130, 130
    for i, b in enumerate(s["blocks"]):
        ah = 44 + HEAD_STEP * _ceil(b["heads"], HEAD_COLS) + 6 if b["att"] else 46
        rows = _ceil(6 * b["exp"], MLP_COLS)
        mh = 44 + b["layers"] * (rows * MLP_STEP + 9) + 2 if b["mlp"] else 46
        att, mlp = (x + 20, -(SUB + ah), x + BW - 20, -SUB), (x + 20, SUB, x + BW - 20, SUB + mh)
        board = (x, att[1] - 34, x + BW, mlp[3] + 14)
        blocks.append({"i": i, "id": b["id"], "x0": x, "x1": x + BW, "att": att, "mlp": mlp, "board": board,
                       "rows": rows, "tin": x + 50, "tout": x + BW - 50})
        top, bottom = min(top, board[1]), max(bottom, board[3])
        x += BW + GAP
    final = (x, -40, x + 120, 40)
    out = (final[2] + 44, -72, final[2] + 44 + 140, 72)
    y_fb = bottom + 62
    cx = (out[0] + out[2]) / 2
    train = (cx - 118, y_fb - 84, cx + 118, y_fb + 84)
    y_tied = top - 40
    bounds = (-70, y_tied - 26, out[2] + 76, y_fb + 110)
    return {"blocks": blocks, "final": final, "out": out, "train": train, "y_fb": y_fb, "y_tied": y_tied,
            "bounds": bounds, "top": top, "bottom": bottom}


def _path(points):
    cum, total = [0.0], 0.0
    for (ax, ay), (bx, by) in zip(points, points[1:]):
        total += math.hypot(bx - ax, by - ay)
        cum.append(total)
    return {"pts": points, "cum": cum, "total": max(total, 1.0)}


def _along(path, s):
    s %= path["total"]
    cum, pts = path["cum"], path["pts"]
    for i in range(1, len(cum)):
        if s <= cum[i]:
            span = cum[i] - cum[i - 1] or 1.0
            t = (s - cum[i - 1]) / span
            (ax, ay), (bx, by) = pts[i - 1], pts[i]
            return ax + (bx - ax) * t, ay + (by - ay) * t
    return pts[-1]


class CircuitCanvas(tk.Canvas):
    def __init__(self, parent, owner):
        super().__init__(parent, bg=BG, highlightthickness=0, takefocus=True)
        self.owner = owner
        self.spec = self.s = self.geo = None
        self.parts, self.stale, self.error, self.err_keys = {}, False, "", set()
        self.selected = None
        self.z, self.ox, self.oy = 0.6, 40.0, 300.0
        self.tele, self.heat = None, {}
        self.borders, self.pads, self.bus_items, self.dot_items, self.paths, self.cfg_cache = {}, {}, {}, [], [], {}
        self.train_widgets = {}
        self.fitted = False
        self.phase, self.last_tick = 0.0, time.monotonic()
        self.grad_visible = None
        self._down = self._last = None
        self._mode = self._press_key = self._drag_id = None
        self._moved = False
        self._hover = None
        self._redraw_job = None
        self.bind("<Configure>", self._on_resize)
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._motion)
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<Double-Button-1>", self._double)
        for button in (2, 3):
            self.bind(f"<ButtonPress-{button}>", self._pan_start)
            self.bind(f"<B{button}-Motion>", self._pan_move)
            self.bind(f"<ButtonRelease-{button}>", self._pan_end)
        self.bind("<MouseWheel>", self._wheel)
        self.bind("<Motion>", self._hovering)
        self.bind("<Leave>", lambda _: self.owner.net_hover(""))
        self.anim = self.after(60, self._tick)

    # ------------------------------------------------------------------ model in
    def set_spec(self, spec, error="", err_keys=(), parts=None, stale=False):
        self.spec, self.error, self.err_keys, self.stale = spec, error, set(err_keys), stale
        if parts is not None:
            self.parts = parts
        self.s = lenient(spec)
        self.geo = layout(self.s)
        if not self.fitted and self.winfo_width() > 100:
            self.fit()
        else:
            self.request_redraw()

    def set_selected(self, key):
        self.selected = key
        self.restyle()

    def set_telemetry(self, tele):
        self.tele = tele
        self.paint_live()

    def clear_telemetry(self):
        self.tele, self.heat = None, {}
        self.paint_live()

    def request_redraw(self):
        if self._redraw_job is None:
            self._redraw_job = self.after_idle(self._do_redraw)

    def _do_redraw(self):
        self._redraw_job = None
        try:
            self.redraw()
        except tk.TclError:
            pass

    # ------------------------------------------------------------------ view
    def _on_resize(self, event):
        if not self.fitted and self.geo and event.width > 100:
            self.fit()

    def fit(self):
        if not self.geo:
            return
        w, h = self.winfo_width(), self.winfo_height()
        if w < 100 or h < 80:
            return
        x0, y0, x1, y1 = self.geo["bounds"]
        z = max(0.36, min(1.3, min((w - 30) / (x1 - x0), (h - 24) / (y1 - y0))))
        self.z = z
        self.ox = (w - (x1 - x0) * z) / 2 - x0 * z if (x1 - x0) * z < w else 18 - x0 * z
        self.oy = (h - (y1 - y0) * z) / 2 - y0 * z if (y1 - y0) * z < h else 12 - y0 * z
        self.fitted = True
        self.request_redraw()

    def reveal(self, block_id):
        """Pan horizontally just enough to bring a block into view."""
        d = next((b for b in (self.geo or {}).get("blocks", []) if b["id"] == block_id), None)
        if not d:
            return
        w = self.winfo_width()
        left, right = self._p(d["x0"], 0)[0], self._p(d["x1"], 0)[0]
        if left < 16:
            self.ox += 16 - left
        elif right > w - 16 and w > 100:
            self.ox -= right - (w - 16)
        else:
            return
        self.fitted = True
        self.request_redraw()

    def zoom(self, factor, cx=None, cy=None):
        cx = self.winfo_width() / 2 if cx is None else cx
        cy = self.winfo_height() / 2 if cy is None else cy
        z = max(0.15, min(3.0, self.z * factor))
        factor = z / self.z
        self.ox, self.oy, self.z = cx - (cx - self.ox) * factor, cy - (cy - self.oy) * factor, z
        self.fitted = True
        self.request_redraw()

    def _wheel(self, event):
        cx, cy = self.winfo_pointerx() - self.winfo_rootx(), self.winfo_pointery() - self.winfo_rooty()
        self.zoom(1.12 ** (event.delta / 120), cx, cy)
        return "break"

    def world(self, x, y):
        return (x - self.ox) / self.z, (y - self.oy) / self.z

    def _p(self, x, y):
        return x * self.z + self.ox, y * self.z + self.oy

    def _flat(self, pts):
        return [v for x, y in pts for v in self._p(x, y)]

    # ------------------------------------------------------------------ interaction
    def key_at(self, x, y):
        for item in reversed(self.find_overlapping(x - 2, y - 2, x + 2, y + 2)):
            for tag in self.gettags(item):
                if tag.startswith("k:"):
                    return tag[2:]
        return None

    def _block_of(self, key):
        return key.split(":")[1] if key and key.startswith("b:") else None

    def _press(self, event):
        self.focus_set()
        self._down = self._last = (event.x, event.y)
        self._moved = False
        self._press_key = self.key_at(event.x, event.y)
        self._drag_id = self._block_of(self._press_key)
        self._mode = "block" if self._drag_id else "pan"

    def _motion(self, event):
        if self._down is None:
            return
        if not self._moved and abs(event.x - self._down[0]) + abs(event.y - self._down[1]) < 7:
            return
        self._moved = True
        if self._mode == "pan":
            dx, dy = event.x - self._last[0], event.y - self._last[1]
            self.move("all", dx, dy)
            self.ox += dx
            self.oy += dy
            self.fitted = True
        elif self._mode == "block":
            slot = self.slot_at(event.x)
            self.show_marker(slot)
            self.show_ghost(event.x, event.y, "BLOCK")
            self.configure(cursor="fleur")
        self._last = (event.x, event.y)

    def _release(self, event):
        mode, moved = self._mode, self._moved
        self._mode = None
        self._down = None
        self.configure(cursor="")
        if mode == "block" and moved:
            slot = self.slot_at(event.x)
            self.hide_marker()
            self.hide_ghost()
            ids = [b["id"] for b in self.geo["blocks"]]
            if self._drag_id in ids:
                index = ids.index(self._drag_id)
                self.owner.net_move_block(self._drag_id, slot - 1 if slot > index else slot)
        elif mode == "pan" and moved:
            self.request_redraw()
        elif mode in ("block", "pan"):
            self.owner.net_canvas_select(self._press_key)

    def _double(self, event):
        key = self.key_at(event.x, event.y)
        if key:
            self.owner.net_canvas_activate(key)

    def _pan_start(self, event):
        self._pan = (event.x, event.y)
        self.configure(cursor="fleur")

    def _pan_move(self, event):
        dx, dy = event.x - self._pan[0], event.y - self._pan[1]
        self._pan = (event.x, event.y)
        self.move("all", dx, dy)
        self.ox += dx
        self.oy += dy
        self.fitted = True

    def _pan_end(self, event):
        self.configure(cursor="")
        self.request_redraw()

    def slot_at(self, sx):
        """Insertion slot (0..n) for a screen x: the number of blocks whose centre is left of the pointer."""
        wx = self.world(sx, 0)[0]
        return sum(1 for b in self.geo["blocks"] if (b["x0"] + b["x1"]) / 2 < wx) if self.geo else 0

    def show_marker(self, slot):
        self.hide_marker()
        blocks = self.geo["blocks"]
        if not blocks:
            x = EMB[2] + GAP / 2
        elif slot <= 0:
            x = blocks[0]["x0"] - GAP / 2
        elif slot >= len(blocks):
            x = blocks[-1]["x1"] + GAP / 2
        else:
            x = (blocks[slot - 1]["x1"] + blocks[slot]["x0"]) / 2
        y0, y1 = self.geo["top"] - 14, self.geo["bottom"] + 14
        pts = self._flat([(x, y0), (x, y1)])
        self.create_line(*pts, fill="#2ee6d0", width=max(2, 5 * self.z), tags=("marker",), capstyle="round")
        self.create_line(*pts, fill="#ffffff", width=max(1, 1.5 * self.z), tags=("marker",), capstyle="round")

    def hide_marker(self):
        self.delete("marker")

    def show_ghost(self, x, y, text):
        self.hide_ghost()
        w, h = 120, 34
        self.create_rectangle(x - w / 2, y - h / 2, x + w / 2, y + h / 2, fill=CHIP, outline=BUS, dash=(4, 3), width=2, tags=("ghost",))
        self.create_text(x, y, text=text, fill=TEXT, font=(FONT, -12, "bold"), tags=("ghost",))

    def hide_ghost(self):
        self.delete("ghost")

    def _hovering(self, event):
        if self._down is not None:
            return
        text = ""
        for item in reversed(self.find_overlapping(event.x - 1, event.y - 1, event.x + 1, event.y + 1)):
            tags = [t for t in self.gettags(item) if t.startswith("p:")]
            if tags:
                text = self.describe(tags[0])
                break
        if text != self._hover:
            self._hover = text
            self.owner.net_hover(text)

    def describe(self, tag):
        parts = tag.split(":")
        kind, ident = parts[1], parts[2]
        index = next((i for i, b in enumerate(self.s["blocks"]) if b["id"] == ident), 0)
        block = next((b for b in (self.tele or {}).get("blocks", []) if b["id"] == ident), None)
        if kind == "h":
            head = int(parts[3])
            entropy = ((block or {}).get("attention") or {}).get("entropy")
            live = f" · entropy {entropy[head]:.2f} nats" if entropy and head < len(entropy) else ""
            return f"Block {index + 1} · attention head {head + 1}{live}"
        if kind == "n":
            layer, unit = int(parts[3]), int(parts[4])
            neurons = ((block or {}).get("mlp") or {}).get("neurons")
            last = layer == self.s["blocks"][index]["layers"] - 1
            live = f" · activity {neurons[min(len(neurons) - 1, unit * len(neurons) // max(1, self._unit_count(index)))]:.2f}" if neurons and last else ""
            return f"Block {index + 1} · MLP layer {layer + 1} neuron pad {unit + 1}{live}"
        return "Embedding dimension sample"

    def _unit_count(self, index):
        return 6 * self.s["blocks"][index]["exp"]

    # ------------------------------------------------------------------ drawing helpers
    def _font(self, size, bold=False):
        px = max(1, int(round(size * self.z)))
        return (FONT, -px, "bold") if bold else (FONT, -px)

    def _text(self, x, y, text, size, color, anchor="w", bold=False, tags=()):
        if size * self.z < 4.4:
            return None
        sx, sy = self._p(x, y)
        return self.create_text(sx, sy, text=text, fill=color, anchor=anchor, font=self._font(size, bold), tags=tags)

    def _rrect(self, x0, y0, x1, y1, r, **kw):
        r = min(r, (x1 - x0) / 2, (y1 - y0) / 2)
        pts = [x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1, x1 - r, y1, x0 + r, y1,
               x0, y1, x0, y1 - r, x0, y0 + r, x0, y0]
        screen = [v for i in range(0, len(pts), 2) for v in self._p(pts[i], pts[i + 1])]
        if "width" in kw:
            kw["width"] = max(1, kw["width"] * self.z)
        return self.create_polygon(*screen, smooth=True, **kw)

    def _line(self, pts, color, width, tags=(), dash=None, glow=True):
        ids = []
        flat = self._flat(pts)
        if glow:
            ids.append(self.create_line(*flat, fill=mix(BG, color, 0.28), width=max(2, (width + 7) * self.z),
                                        capstyle="round", joinstyle="round", tags=tags, dash=dash))
        ids.append(self.create_line(*flat, fill=color, width=max(1, width * self.z), capstyle="round",
                                    joinstyle="round", tags=tags, dash=dash))
        return ids

    def _pad(self, x, y, size, accent, tags, level=0.1):
        sx0, sy0 = self._p(x, y)
        sx1, sy1 = self._p(x + size, y + size)
        item = self.create_rectangle(sx0, sy0, sx1, sy1, fill=mix(PAD_OFF, accent, 0.1 + 0.9 * level),
                                     outline=mix(EDGE, accent, 0.25 + 0.6 * level), width=1, tags=tags)
        return item

    def _cfg(self, item, **kw):
        if item is None:
            return
        key = tuple(sorted(kw.items()))
        if self.cfg_cache.get(item) != key:
            self.cfg_cache[item] = key
            self.itemconfigure(item, **kw)

    def _legs(self, box, color, sides="lr", count=5, key=""):
        x0, y0, x1, y1 = box
        tags = ("k:" + key,)
        for i in range(count):
            f = (i + 1) / (count + 1)
            if "l" in sides or "r" in sides:
                y = y0 + (y1 - y0) * f
                if abs(y) < 9:
                    continue
                for side, edge, d in (("l", x0, -8), ("r", x1, 8)):
                    if side in sides:
                        a, b = self._p(edge, y - 2.5), self._p(edge + d, y + 2.5)
                        self.create_rectangle(a[0], a[1], b[0], b[1], fill=color, outline="", tags=tags)
            if "t" in sides or "b" in sides:
                x = x0 + (x1 - x0) * f
                for side, edge, d in (("t", y0, -8), ("b", y1, 8)):
                    if side in sides:
                        a, b = self._p(x - 2.5, edge), self._p(x + 2.5, edge + d)
                        self.create_rectangle(a[0], a[1], b[0], b[1], fill=color, outline="", tags=tags)

    def _chip(self, box, accent, key, legs="lr", radius=10, count=5):
        x0, y0, x1, y1 = box
        tags = ("k:" + key,)
        glow = self._rrect(x0 - 3, y0 - 3, x1 + 3, y1 + 3, radius + 3, fill="", outline=mix(BG, accent, .3), width=5, tags=tags)
        body = self._rrect(x0, y0, x1, y1, radius, fill=CHIP, outline=mix(EDGE, accent, .5), width=2, tags=tags)
        self._legs(box, mix(EDGE, accent, .55), legs, count, key)
        pin = self._p(x0 + 10, y0 + 10)
        r = max(1.5, 2.6 * self.z)
        self.create_oval(pin[0] - r, pin[1] - r, pin[0] + r, pin[1] + r, fill=mix(CHIP, accent, .6), outline="", tags=tags)
        self.borders[key] = (body, glow, accent)
        return body

    def _params(self, component):
        value = self.parts.get(component)
        return None if value is None else fmt_params(value)

    # ------------------------------------------------------------------ the scene
    def redraw(self):
        self.delete("all")
        self.cfg_cache.clear()
        self.borders, self.pads, self.bus_items, self.dot_items, self.paths = {}, {}, {}, [], []
        self.train_widgets = {}
        self.grad_visible = None
        if not self.geo:
            return
        s, g = self.s, self.geo
        self._grid()
        blocks = g["blocks"]
        bus_y = 0
        # block boards sit under the bus
        for b, d in zip(s["blocks"], blocks):
            key = "b:" + d["id"]
            x0, y0, x1, y1 = d["board"]
            board = self._rrect(x0, y0, x1, y1, 14, fill=BOARD, outline=mix(BG, EDGE, .8), width=1.5, tags=("k:" + key,))
            self.borders[key] = (board, None, ACCENT["board"])
            custom = not re.fullmatch(r"Block( \d+)?", b["label"])
            title = f"BLOCK {d['i'] + 1}" + (f"  ·  {b['label']}" if custom else "")
            self._text(x0 + 16, y0 + 18, title, 14.5, ERR if key in self.err_keys else TEXT, bold=True, tags=("k:" + key,))
            total = sum(self.parts.get(f"{d['id']}:{k}", 0) for k in ("attention", "mlp"))
            self._text(x1 - 16, y0 + 18, fmt_params(total) + " params" if total else "", 11.5, MUTED, "e", tags=("k:" + key,))
        # residual stream (bus)
        left = -58
        self._line([(left, 0), (EMB[0], 0)], BUS, 3, ("k:emb",))
        self._text(left, -14, "TOKENS", 10.5, MUTED, tags=("k:emb",))
        stops = [(EMB[2], blocks[0]["x0"] if blocks else g["final"][0], "emb")]
        for d in blocks:
            stops.append((d["x0"], d["x1"], d["id"]))
            nxt = next((n for n in blocks if n["i"] == d["i"] + 1), None)
            stops.append((d["x1"], nxt["x0"] if nxt else g["final"][0], d["id"]))
        for a, b, who in stops:
            self.bus_items.setdefault(who, []).extend(self._line([(a, bus_y), (b, bus_y)], BUS, 3.2, ("k:" + ("b:" + who if who != "emb" else "emb"),)))
        self._line([(g["final"][2], 0), (g["out"][0], 0)], BUS, 3.2, ("k:final",))
        self._line([(g["out"][2], 0), (g["out"][2] + 56, 0)], BUS, 3.2, ("k:out",))
        self._text(g["out"][2] + 10, -14, "LOGITS", 10.5, MUTED, tags=("k:out",))
        ax, ay = self._p(g["out"][2] + 62, 0)
        r = 7 * self.z
        self.create_polygon(ax, ay, ax - 1.6 * r, ay - r * .8, ax - 1.6 * r, ay + r * .8, fill=BUS, outline="")
        # tied weights + loss + gradient traces
        e_cx, o_cx = (EMB[0] + EMB[2]) / 2, (g["out"][0] + g["out"][2]) / 2
        tied = [(o_cx, g["out"][1]), (o_cx, g["y_tied"]), (e_cx, g["y_tied"]), (e_cx, EMB[1])]
        self._line(tied, ACCENT["tied"], 2, ("k:out",), dash=(7, 5))
        self._text((e_cx + o_cx) / 2, g["y_tied"] - 14, "TIED WEIGHTS  ·  output head shares the embedding table", 11.5, ACCENT["tied"], "center", tags=("k:out",))
        loss = [(o_cx, g["out"][3]), (o_cx, g["train"][1])]
        self._line(loss, ACCENT["train"], 2.4, ("k:train",))
        self._text(o_cx + 12, (g["out"][3] + g["train"][1]) / 2, "loss", 11.5, ACCENT["train"], tags=("k:train",))
        grad = [(g["train"][0], g["y_fb"]), (e_cx, g["y_fb"]), (e_cx, EMB[3])]
        self._line(grad, GRAD, 2.2, ("k:train",), dash=(3, 5))
        self._text((g["train"][0] + e_cx) / 2, g["y_fb"] + 18, "BACKPROP  ·  gradients flow back to every chip", 11.5, GRAD, "center", tags=("k:train",))
        # branch taps first, then the signal dots, then the chips so dots slip underneath them
        later = []
        for b, d in zip(s["blocks"], blocks):
            for kind, sign, box, on in (("att", -1, d["att"], b["att"]), ("mlp", 1, d["mlp"], b["mlp"])):
                key = f"b:{d['id']}:{kind}"
                accent = ACCENT[kind]
                if on:
                    near = box[3] if sign < 0 else box[1]
                    self._line([(d["tin"], 0), (d["tin"], near)], accent, 2.2, ("k:" + key,))
                    self._line([(d["tout"], near), (d["tout"], 0)], accent, 2.2, ("k:" + key,))
                    ln = (d["tin"] - 15, sign * SUB / 2 - 7, d["tin"] + 15, sign * SUB / 2 + 7)
                    self._rrect(*ln, 4, fill=CHIP, outline=ACCENT["final"], width=1.2, tags=("k:" + key,))
                    self._text(d["tin"], sign * SUB / 2, "LN", 9, ACCENT["final"], "center", True, ("k:" + key,))
                    px, py = self._p(d["tout"], 0)
                    r = 8 * self.z
                    self.create_oval(px - r, py - r, px + r, py + r, fill=CHIP, outline=accent, width=max(1, 2 * self.z), tags=("k:" + key,))
                    self.create_line(px - r * .55, py, px + r * .55, py, fill=accent, width=max(1, 1.6 * self.z), tags=("k:" + key,))
                    self.create_line(px, py - r * .55, px, py + r * .55, fill=accent, width=max(1, 1.6 * self.z), tags=("k:" + key,))
                    later.append(lambda b=b, d=d, box=box, key=key, kind=kind: (self._attention_chip if kind == "att" else self._mlp_chip)(b, d, box, key))
                else:
                    later.append(lambda box=box, kind=kind, key=key: self._ghost(box, kind, key))
        self._build_dots()
        for draw in later:
            draw()
        self._embedding_chip(key="emb")
        self._simple_chip(g["final"], "final", "FINAL NORM", "LayerNorm", self._params("final"), "lr")
        self._output_chip(g["out"])
        self._training_chip(g["train"])
        self.restyle()
        self.paint_live()
        self.tag_raise("marker")
        self.tag_raise("ghost")

    def _grid(self):
        w, h = max(1, self.winfo_width()), max(1, self.winfo_height())
        step = 60.0
        while step * self.z < 26:
            step *= 2
        x0, y0 = self.world(0, 0)
        x1, y1 = self.world(w, h)
        n = math.floor(x0 / step) * step
        while n <= x1:
            sx = self._p(n, 0)[0]
            self.create_line(sx, 0, sx, h, fill=GRID)
            n += step
        n = math.floor(y0 / step) * step
        while n <= y1:
            sy = self._p(0, n)[1]
            self.create_line(0, sy, w, sy, fill=GRID)
            n += step

    def _ghost(self, box, kind, key):
        x0, y0, x1, y1 = box
        self._rrect(x0, y0, x1, y1, 10, fill=BOARD, outline=mix(BG, ACCENT[kind], .45), width=1.5, tags=("k:" + key,), dash=(5, 4))
        name = "ATTENTION" if kind == "att" else "MLP"
        self._text((x0 + x1) / 2, (y0 + y1) / 2 - 7, f"+ {name}", 13, mix(BG, ACCENT[kind], .7), "center", True, ("k:" + key,))
        self._text((x0 + x1) / 2, (y0 + y1) / 2 + 9, "off · double-click to add", 10.5, MUTED, "center", tags=("k:" + key,))
        self.borders[key] = (None, None, ACCENT[kind])

    def _title(self, x, y, title, sub, key, accent, bad=False):
        self._text(x + 16, y + 17, title, 14.5, ERR if bad else TEXT, bold=True, tags=("k:" + key,))
        self._text(x + 16, y + 34, sub, 11.5, MUTED, tags=("k:" + key,))

    def _attention_chip(self, b, d, box, key):
        x0, y0, x1, y1 = box
        self._chip(box, ACCENT["att"], key)
        params = self._params(f"{d['id']}:attention")
        self._title(x0, y0, "ATTENTION", f"{b['heads']} head{'s' * (b['heads'] != 1)}" + (f"  ·  {params}" if params else ""), key, ACCENT["att"])
        cols = min(HEAD_COLS, b["heads"])
        gx = x0 + ((x1 - x0) - (cols * HEAD_STEP - (HEAD_STEP - HEAD_SIZE))) / 2
        for h in range(b["heads"]):
            px, py = gx + (h % HEAD_COLS) * HEAD_STEP, y0 + 46 + (h // HEAD_COLS) * HEAD_STEP
            level = 0.08 + 0.1 * (((h * 37 + d["i"] * 11) % 7) / 7)
            self.pads[("h", d["id"], h)] = self._pad(px, py, HEAD_SIZE, ACCENT["att"], ("k:" + key, f"p:h:{d['id']}:{h}"), level)
            if self.z > 0.9:
                self._text(px + HEAD_SIZE / 2, py + HEAD_SIZE / 2, str(h + 1), 8, MUTED, "center", tags=("k:" + key, f"p:h:{d['id']}:{h}"))

    def _mlp_chip(self, b, d, box, key):
        x0, y0, x1, y1 = box
        self._chip(box, ACCENT["mlp"], key)
        params = self._params(f"{d['id']}:mlp")
        sub = f"×{b['exp']}  ·  {b['layers']}L  ·  {b['act']}" + (f"  ·  {params}" if params else "")
        self._title(x0, y0, "MLP", sub, key, ACCENT["mlp"])
        units = 6 * b["exp"]
        cols = min(MLP_COLS, units)
        gx = x0 + ((x1 - x0) - (cols * MLP_STEP - (MLP_STEP - MLP_SIZE))) / 2
        group = d["rows"] * MLP_STEP + 9
        for layer in range(b["layers"]):
            top = y0 + 46 + layer * group
            for u in range(units):
                px, py = gx + (u % MLP_COLS) * MLP_STEP, top + (u // MLP_COLS) * MLP_STEP
                level = 0.06 + 0.12 * (((u * 53 + layer * 7 + d["i"] * 3) % 9) / 9)
                self.pads[("n", d["id"], layer, u)] = self._pad(px, py, MLP_SIZE, ACCENT["mlp"], ("k:" + key, f"p:n:{d['id']}:{layer}:{u}"), level)
            if layer < b["layers"] - 1:
                ly = top + d["rows"] * MLP_STEP + 1
                self._line([(x0 + 18, ly), (x1 - 18, ly)], mix(BG, ACCENT["mlp"], .35), 1, ("k:" + key,), glow=False)

    def _embedding_chip(self, key):
        s, box = self.s, EMB
        x0, y0, x1, y1 = box
        self._chip(box, ACCENT["emb"], key, "lrtb", 12, 6)
        bad = key in self.err_keys
        self._text(x0 + 14, y0 + 20, "TOKEN", 14, ERR if bad else TEXT, bold=True, tags=("k:emb",))
        self._text(x0 + 14, y0 + 36, "EMBEDDING", 14, ERR if bad else TEXT, bold=True, tags=("k:emb",))
        for i, line in enumerate((f"width  {s['width']}", f"context  {s['ctx']}", f"{s['tok']}", f"dropout  {s['drop']:g}")):
            self._text(x0 + 14, y0 + 60 + i * 15, line, 11.5, MUTED, tags=("k:emb",))
        params = self._params("embedding")
        self._text(x0 + 14, y1 - 52, (params + " params") if params else "", 11.5, ACCENT["emb"], bold=True, tags=("k:emb",))
        for i in range(24):
            px, py = x0 + 17 + (i % 8) * 15.5, y1 - 40 + (i // 8) * 10.5
            self.pads[("e", "emb", i)] = self._pad(px, py, 9, ACCENT["emb"], ("k:emb", f"p:e:emb:{i}"), 0.1 + 0.1 * ((i * 5 % 7) / 7))

    def _simple_chip(self, box, key, title, sub, params, legs):
        x0, y0, x1, y1 = box
        self._chip(box, ACCENT[key], key, legs, 10, 3)
        self._text(x0 + 14, y0 + 20, title, 14, TEXT, bold=True, tags=("k:" + key,))
        self._text(x0 + 14, y0 + 40, sub, 11.5, MUTED, tags=("k:" + key,))
        self._text(x0 + 14, y0 + 58, (params + " params") if params else "", 11.5, ACCENT[key], tags=("k:" + key,))

    def _output_chip(self, box):
        x0, y0, x1, y1 = box
        self._chip(box, ACCENT["out"], "out", "lrtb", 12, 5)
        self._text(x0 + 14, y0 + 20, "OUTPUT HEAD", 14, TEXT, bold=True, tags=("k:out",))
        self._text(x0 + 14, y0 + 40, "tied to embedding", 11.5, MUTED, tags=("k:out",))
        self._text(x0 + 14, y0 + 58, "x · Eᵀ → logits", 11.5, ACCENT["out"], tags=("k:out",))
        self._text(x0 + 14, y0 + 78, "next-token probs", 11.5, MUTED, tags=("k:out",))
        for i in range(10):
            px, py = x0 + 14 + i * 13.2, y1 - 28
            self.pads[("o", "out", i)] = self._pad(px, py, 9, ACCENT["out"], ("k:out", f"p:e:out:{i}"), 0.1)

    def _training_chip(self, box):
        t = self.s["train"]
        x0, y0, x1, y1 = box
        self._chip(box, ACCENT["train"], "train", "lrt", 12, 5)
        bad = "train" in self.err_keys
        self._text(x0 + 16, y0 + 20, "TRAINING", 14, ERR if bad else TEXT, bold=True, tags=("k:train",))
        lines = (f"steps {t.get('steps', '?')}   batch {t.get('batch_size', '?')} × {t.get('accumulation', '?')}",
                 f"lr {t.get('learning_rate', '?')}   warmup {t.get('warmup', '?')}",
                 f"eval / {t.get('eval_every', '?')}   val {t.get('validation', '?')}",
                 f"objective {t.get('objective', '?')}   {t.get('device', '?')}")
        for i, line in enumerate(lines):
            self._text(x0 + 16, y0 + 44 + i * 16, line, 11.5, MUTED, tags=("k:train",))
        bx0, by0, bx1, by1 = x0 + 16, y1 - 40, x1 - 16, y1 - 30
        self._rrect(bx0, by0, bx1, by1, 4, fill=PAD_OFF, outline=mix(EDGE, ACCENT["train"], .4), width=1, tags=("k:train",))
        sx0, sy0 = self._p(bx0, by0)
        sx1, sy1 = self._p(bx1, by1)
        fill = self.create_rectangle(sx0 + 1, sy0 + 1, sx0 + 1, sy1 - 1, fill=ACCENT["train"], outline="", tags=("k:train",))
        info = self._text(x0 + 16, y1 - 14, "idle", 11.5, ACCENT["train"], bold=True, tags=("k:train",))
        self.train_widgets = {"fill": fill, "x0": sx0 + 1, "x1": sx1 - 1, "info": info}

    # ------------------------------------------------------------------ animated signals
    def _build_dots(self):
        g = self.geo
        end = g["out"][2] + 56
        main = _path([(-58, 0), (end, 0)])
        self.paths.append(main)
        for k in range(max(5, int(main["total"] / 240))):
            self.dot_items.append(self._dot(0, k * main["total"] / max(5, int(main["total"] / 240)), BUS, False))
        for b, d in zip(self.s["blocks"], g["blocks"]):
            for kind, box, on, sign in (("att", d["att"], b["att"], -1), ("mlp", d["mlp"], b["mlp"], 1)):
                if on:
                    near = box[3] if sign < 0 else box[1]
                    self.paths.append(_path([(d["tin"], 0), (d["tin"], near), (d["tout"], near), (d["tout"], 0)]))
                    self.dot_items.append(self._dot(len(self.paths) - 1, 0.0, ACCENT[kind], False))
        e_cx = (EMB[0] + EMB[2]) / 2
        back = _path([(g["train"][0], g["y_fb"]), (e_cx, g["y_fb"]), (e_cx, EMB[3])])
        self.paths.append(back)
        index = len(self.paths) - 1
        for k in range(max(3, int(back["total"] / 300))):
            self.dot_items.append(self._dot(index, k * back["total"] / max(3, int(back["total"] / 300)), GRAD, True))

    def _dot(self, path_index, offset, color, grad):
        glow = self.create_oval(0, 0, 0, 0, fill=mix(BG, color, .35), outline="", state="hidden" if grad else "normal")
        core = self.create_oval(0, 0, 0, 0, fill=mix(color, "#ffffff", .55), outline="", state="hidden" if grad else "normal")
        return {"path": path_index, "off": offset, "glow": glow, "core": core, "grad": grad, "color": color}

    def _tick(self):
        self.anim = None
        try:
            if not self.winfo_exists():
                return
            now = time.monotonic()
            dt = min(0.1, now - self.last_tick)
            self.last_tick = now
            if self.owner.net_visible() and self.dot_items:
                training = self.owner.net_is_training()
                self.phase += dt * (260 if training else 80)
                self._move_dots(training)
            self.anim = self.after(40, self._tick)
        except tk.TclError:
            pass

    def _move_dots(self, training):
        w, h = self.winfo_width(), self.winfo_height()
        show = bool(training)
        if show != self.grad_visible:
            self.grad_visible = show
            for dot in self.dot_items:
                if dot["grad"]:
                    state = "normal" if show else "hidden"
                    self.itemconfigure(dot["glow"], state=state)
                    self.itemconfigure(dot["core"], state=state)
        r = max(2.5, 4.2 * self.z)
        for dot in self.dot_items:
            if dot["grad"] and not show:
                continue
            path = self.paths[dot["path"]]
            speed = 1.0 if dot["path"] == 0 else 0.7
            s = dot["off"] + self.phase * (speed if not dot["grad"] else 1.3)
            if dot["path"] != 0 and not dot["grad"]:
                s += dot["path"] * 90
            x, y = _along(path, s)
            sx, sy = self._p(x, y)
            if -20 < sx < w + 20 and -20 < sy < h + 20:
                self.coords(dot["glow"], sx - r * 1.9, sy - r * 1.9, sx + r * 1.9, sy + r * 1.9)
                self.coords(dot["core"], sx - r * .8, sy - r * .8, sx + r * .8, sy + r * .8)

    # ------------------------------------------------------------------ styling and live state
    def _border_style(self, key, accent):
        color = mix(EDGE, accent, .5)
        width = 2.0
        heat = self.heat.get(key, 0.0)
        if heat:
            color = mix(color, HOT, min(1, heat) * .75)
            width = 2 + 1.6 * heat
        if key in self.err_keys:
            return ERR, 3.0, mix(BG, ERR, .55), 8
        if self.selected == key:
            return SEL, 3.0, mix(BG, SEL, .35), 9
        return color, width, mix(BG, color, .4 + .3 * heat), 5 + 3 * heat

    def restyle(self):
        for key, (body, glow, accent) in self.borders.items():
            if body is None:
                continue
            if glow is None:  # block boards: quiet unless selected, chosen through a sub-chip, or invalid
                inside = bool(self.selected) and (self.selected == key or self.selected.startswith(key + ":"))
                color = ERR if key in self.err_keys else SEL if self.selected == key else mix(EDGE, SEL, .5) if inside else mix(BG, EDGE, .8)
                self._cfg(body, outline=color, width=max(1, (2.5 if inside or key in self.err_keys else 1.5) * self.z))
                continue
            color, width, gcolor, gwidth = self._border_style(key, accent)
            self._cfg(body, outline=color, width=max(1, width * self.z))
            self._cfg(glow, outline=gcolor, width=max(1, gwidth * self.z))

    def paint_live(self):
        """Recolour pads, traces and borders from the latest telemetry. Never creates or deletes items."""
        t = self.tele
        by_id = {b["id"]: b for b in (t or {}).get("blocks", [])}
        ctx_max = math.log(max(2, self.s["ctx"])) if self.s else 4.0
        grads = {}
        if t:
            grads["emb"] = (t.get("embedding") or {}).get("grad_norm")
            grads["out"] = (t.get("output") or {}).get("grad_norm")
            grads["final"] = grads["out"]
            for blk in t.get("blocks", []):
                for kind, name in (("att", "attention"), ("mlp", "mlp")):
                    if blk.get(name):
                        grads[f"b:{blk['id']}:{kind}"] = blk[name].get("grad_norm")
        values = [v for v in grads.values() if isinstance(v, (int, float)) and v > 0]
        top = max(values) if values else 0.0
        self.heat = {}
        for key, v in grads.items():
            if isinstance(v, (int, float)) and v > 0 and top > 0:
                self.heat[key] = max(0.0, min(1.0, 1 + math.log10(v / top) / 2.2))
        for (kind, bid, *rest), item in self.pads.items():
            if kind == "h":
                blk = by_id.get(bid)
                entropy = ((blk or {}).get("attention") or {}).get("entropy")
                if entropy and rest[0] < len(entropy):
                    level = max(0.0, min(1.0, (1 - entropy[rest[0]] / ctx_max) * 1.7))
                else:
                    level = 0.08 + 0.1 * (((rest[0] * 37 + self._index(bid) * 11) % 7) / 7)
                self._pad_level(item, ACCENT["att"], level)
            elif kind == "n":
                layer, unit = rest
                blk = by_id.get(bid)
                mlp = (blk or {}).get("mlp")
                spec_block = next((b for b in self.s["blocks"] if b["id"] == bid), None)
                if mlp and spec_block:
                    if layer == spec_block["layers"] - 1 and mlp.get("neurons"):
                        n = mlp["neurons"]
                        level = n[min(len(n) - 1, unit * len(n) // (6 * spec_block["exp"]))]
                    else:
                        level = float(mlp.get("active_fraction", .3)) * (.55 + .45 * (((unit * 31 + layer * 17) % 9) / 9))
                else:
                    level = 0.06 + 0.12 * (((unit * 53 + layer * 7 + self._index(bid) * 3) % 9) / 9)
                self._pad_level(item, ACCENT["mlp"], max(0.0, min(1.0, level)))
            elif kind in ("e", "o"):
                idx = rest[-1]
                rms = (t.get("embedding") or {}).get("rms") if t else None
                shade = .5 + .5 * ((idx * 7 % 5) / 5)
                level = 0.1 + 0.1 * ((idx * 5 % 7) / 7)
                if kind == "e" and rms is not None:
                    level = max(0.0, min(1.0, rms / 1.6 * shade))
                elif kind == "o" and t:
                    level = max(0.0, min(1.0, 0.25 + 0.7 * self.heat.get("out", 0) * shade))
                self._pad_level(item, ACCENT["emb"] if kind == "e" else ACCENT["out"], level)
        # residual stream brightness
        base = (t.get("embedding") or {}).get("rms") if t else None
        for who, items in self.bus_items.items():
            rms = (by_id.get(who) or {}).get("residual_rms") if who != "emb" else base
            level = .35 if rms is None else .25 + .75 * max(0.0, min(1.0, rms / 1.8))
            core = mix(mix(BG, BUS, .45), BUS, level)
            for i, item in enumerate(items):
                self._cfg(item, fill=mix(BG, BUS, .12 + .3 * level) if i % 2 == 0 else core)
        self.restyle()
        widgets = self.train_widgets
        if widgets and widgets.get("info"):
            if t:
                step, total = t.get("step", 0), max(1, t.get("total", 1))
                x1 = widgets["x0"] + (widgets["x1"] - widgets["x0"]) * min(1.0, step / total)
                coords = self.coords(widgets["fill"])
                self.coords(widgets["fill"], coords[0], coords[1], max(coords[0], x1), coords[3])
                self._cfg(widgets["info"], text=f"step {step}/{total}  ·  loss {t.get('loss', 0):.3f}")
            else:
                self._cfg(widgets["info"], text="idle")
                coords = self.coords(widgets["fill"])
                self.coords(widgets["fill"], coords[0], coords[1], coords[0], coords[3])

    def _index(self, bid):
        return next((i for i, b in enumerate(self.s["blocks"]) if b["id"] == bid), 0)

    def _pad_level(self, item, accent, level):
        level = round(level, 2)
        self._cfg(item, fill=mix(PAD_OFF, accent, .1 + .9 * level), outline=mix(EDGE, accent, .25 + .7 * level))

    def destroy(self):
        try:
            if self.anim:
                self.after_cancel(self.anim)
        except tk.TclError:
            pass
        super().destroy()
