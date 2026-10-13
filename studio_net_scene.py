"""Bounded in-app 2D scene controls plus the separate 3D viewer link."""
from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import ttk

import numpy as np

from studio_net_common import NetworkView, to_numpy
from studio_scene import LEVELS, MODES, SceneModel
from studio_widgets import label


SUMMARY_PROMPT = "The model reads this prompt."


class SceneView(NetworkView):
    title = "Inside the network · 2D / 3D"

    def __init__(self, parent, host):
        super().__init__(parent, host)
        self.scene = None
        line = ttk.Frame(self)
        line.pack(fill="x")
        self.level = tk.StringVar(value="Model")
        self.mode = tk.StringVar(value="Build")
        self.weight_kind = tk.StringVar(value="weight")
        for title, variable, values in (("Level", self.level, LEVELS), ("Mode", self.mode, MODES),
                                        ("Weight map", self.weight_kind, ("weight", "gradient", "update"))):
            label(line, title).pack(side="left")
            box = ttk.Combobox(line, textvariable=variable, values=values, state="readonly", width=14)
            box.pack(side="left", padx=(6, 14))
            box.bind("<<ComboboxSelected>>", lambda _event: self.changed())
        ttk.Button(line, text="Open separate 3D viewer", command=host.net_open_viewer).pack(side="left")
        self.details = tk.StringVar()
        label(self, textvariable=self.details, style="Muted.TLabel", wraplength=900, justify="left").pack(anchor="w", pady=(8, 4))
        self.canvas = tk.Canvas(self, height=320, background="#0c1626", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self.draw())
        self._data_key = None

    def refresh(self):
        if not self.spec:
            return
        if self.scene is None:
            self.scene = SceneModel(self.spec)
        else:
            self.scene.update_spec(self.spec)
        self.changed()
        try:
            row = self.checkpoint_row()
            checkpoint = str(self.checkpoint(row))
        except ValueError:
            row = checkpoint = None
        if checkpoint and checkpoint != self._data_key:
            self._data_key = checkpoint
            weights = self.host.arch_store.weights(self.host.arch_id)
            run_dirs = [item["run_dir"] for item in weights if item.get("run_dir")]
            # token snapshots only from the picked checkpoint's own run, never from another version's
            replay = [path for path in self.host.net_replay_paths()
                      if row.get("run_dir") and Path(path).parent.parent == Path(row["run_dir"])]
            header = f"{self.provenance(row)} · prompt {SUMMARY_PROMPT!r}"
            self.work(lambda: (header, self._load_data(checkpoint, run_dirs, replay)), self._data_loaded)
        elif not checkpoint:
            self.scene.snapshot = None
            self.scene.paths, self.scene.summaries, self.scene.stream_norms = [], {}, {}
            self.scene.summary_edges = None
            self.draw()
            self.message.set("Showing the current design. Train or register a checkpoint to add run, attention, and weight data.")

    @staticmethod
    def _load_data(checkpoint, run_dirs, replay):
        from studio_snapshot import load
        from studio_summary import binned_attention, run_paths
        path = Path(checkpoint)
        summary = binned_attention(path, SUMMARY_PROMPT, n_bins=16)
        paths = run_paths(list(dict.fromkeys(map(str, run_dirs))))
        snapshots = sorted(map(Path, replay), key=lambda item: item.stat().st_mtime)
        snapshot = load(snapshots[-1]) if snapshots else None
        return {"attention": {key: to_numpy(value) for key, value in summary["attention"].items()},
                "stream_norms": {key: to_numpy(value) for key, value in summary["stream_norms"].items()},
                "edges": to_numpy(summary["edges"]).tolist(), "paths": paths, "snapshot": snapshot}

    def _data_loaded(self, ok, result):
        if not ok:
            self.message.set(f"Checkpoint scene data unavailable: {result}")
            return
        header, result = result
        self.scene.summaries = result["attention"]
        self.scene.stream_norms = result["stream_norms"]
        self.scene.summary_edges = result["edges"]
        self.scene.paths = result["paths"]
        if result["snapshot"]:
            self.scene.load_snapshot(result["snapshot"])
        self.changed()
        self.message.set(header + ("" if result["snapshot"] else " · no token snapshot recorded in this run"))

    def changed(self):
        if self.scene:
            self.scene.level, self.scene.mode = self.level.get(), self.mode.get()
            frame = self.scene.frame()
            self.details.set(f"{frame['level']} · {frame['mode']} · {len(frame.get('gates', []))} blocks · "
                             f"selected block {frame['selection']['layer'] + 1}. 2D view is bounded; separate viewer uses the same design.")
            self.host.net_send({"type": "scene", "level": self.level.get(), "mode": self.mode.get(),
                                "selected": frame["selection"]})
            self.draw(frame)

    def draw(self, frame=None):
        if not self.scene:
            return
        frame = frame or self.scene.frame()
        c = self.canvas
        c.delete("all")
        width, height = max(c.winfo_width(), 300), max(c.winfo_height(), 220)
        fg, muted, accent = "#d7e2f1", "#93a4ba", "#62d6b5"
        c.create_text(18, 14, anchor="nw", text=f"{frame['level']} · {frame['mode']}", fill=fg,
                      font=("TkDefaultFont", 12, "bold"))
        if frame["level"] == "Run":
            paths = frame.get("paths", [])
            points = [point for path in paths for point in path.get("points", [])]
            if not points:
                c.create_text(width / 2, height / 2, text="No recorded validation-loss paths yet", fill=muted)
            else:
                xs = [float(p[0]) for p in points]; ys = [float(p[1]) for p in points]
                bounds = (min(xs), max(xs), min(ys), max(ys)); x0, x1, y0, y1 = bounds
                def xy(point):
                    return (44 + (point[0] - x0) / max(x1 - x0, 1e-9) * (width - 70),
                            height - 28 - (point[1] - y0) / max(y1 - y0, 1e-9) * (height - 70))
                for idx, path in enumerate(paths):
                    line = [coord for p in path.get("points", []) for coord in xy(p)]
                    if len(line) >= 4: c.create_line(*line, fill=("#62d6b5", "#f4c76b", "#8aa8ff")[idx % 3], width=2)
                c.create_text(44, height - 10, anchor="w", text="step →", fill=muted)
        elif frame["level"] == "Model":
            gates = frame.get("gates", [])
            if gates:
                top, bottom = 70, height - 56
                for i, gate in enumerate(gates):
                    x = 36 + (i + .5) * (width - 72) / len(gates)
                    color = accent if i == frame["selection"]["layer"] else "#34445c"
                    c.create_rectangle(x - 12, top, x + 12, bottom, fill=color, outline="")
                    c.create_text(x, bottom + 18, text=str(i + 1), fill=fg)
                    c.create_text(x, top - 10, text=gate["id"][:10], fill=muted, angle=90)
                c.create_text(36, height - 14, anchor="w", text="blocks · brighter block is selected", fill=muted)
            mats = frame.get("attention", {})
            layer, head = frame["selection"]["layer"], frame["selection"]["head"]
            data = mats.get(f"block{layer}.attn.pattern")
            if data is not None:
                data = np.asarray(data)
                self._heatmap(data[min(head, data.shape[0] - 1)] if data.ndim == 3 else data, width - 210, 62, 170, 130,
                              probabilities=True, axes=("query bins", "key bins"))
                c.create_text(width - 205, 48, anchor="nw", text=f"attention · block {layer + 1} · head {head + 1}", fill=muted)
            elif mats:
                c.create_text(width - 205, 48, anchor="nw", text=f"no attention summary for block {layer + 1}", fill=muted)
        elif frame["level"] in ("Window", "Head"):
            key = f"block{frame['selection']['layer']}.attn.pattern"
            matrix = frame.get("pattern", []) if frame["level"] == "Head" else frame.get("patterns", {}).get(key, [])
            arr = np.asarray(matrix)
            if arr.ndim == 3: arr = arr[min(frame["selection"]["head"], arr.shape[0] - 1)]
            if arr.ndim == 2 and arr.size:
                box = min(height - 110, 230)
                self._heatmap(arr, 52, 50, min(width - 90, 520), box, probabilities=True)
                c.create_text(52, 50 + box + 22, anchor="nw", text=f"token window {frame.get('start', 0)}–{frame.get('end', 0)} · head {frame['selection']['head'] + 1}", fill=muted)
            else:
                c.create_text(width / 2, height / 2, text="No recorded token snapshot; attention summary is available at Model level", fill=muted)
        else:
            data = frame.get(self.weight_kind.get(), [])
            arr = np.asarray(data)
            if arr.ndim == 2 and arr.size:
                box = min(height - 110, 230)
                self._heatmap(arr, 52, 50, min(width - 90, 520), box, axes=("rows", "columns"))
                (rows, cols), (top, left) = frame.get("shape", arr.shape), frame.get("origin", (0, 0))
                c.create_text(52, 50 + box + 22, anchor="nw", fill=muted,
                              text=f"{self.weight_kind.get()} · {frame.get('weight_name')} · rows {top}–{top + arr.shape[0] - 1} of {rows}, "
                                   f"columns {left}–{left + arr.shape[1] - 1} of {cols}")
            else:
                c.create_text(width / 2, height / 2, text="No recorded weight snapshot", fill=muted)

    def _heatmap(self, values, x, y, w, h, probabilities=False, axes=("queries", "keys")):
        """Probabilities use a fixed 0–1 scale; signed data a diverging one (blue negative, orange positive)."""
        full = np.asarray(values, dtype=float)
        if full.ndim != 2 or not full.size:
            return
        arr = full[:32, :32]
        low, high = float(full.min()), float(full.max())
        scale = max(abs(low), abs(high), 1e-9)
        cellw, cellh = w / arr.shape[1], h / arr.shape[0]
        for r in range(arr.shape[0]):
            for col in range(arr.shape[1]):
                value = float(arr[r, col])
                if probabilities:
                    t = min(1.0, max(0.0, value))
                    rgb = (12 + t * 86, 22 + t * 192, 38 + t * 143)
                else:
                    t = max(-1.0, min(1.0, value / scale))
                    tip = (255, 153, 51) if t > 0 else (51, 153, 255)
                    rgb = tuple(26 + abs(t) * (channel - 26) for channel in tip)
                color = "#%02x%02x%02x" % tuple(int(channel) for channel in rgb)
                self.canvas.create_rectangle(x + col * cellw, y + r * cellh,
                                             x + (col + 1) * cellw, y + (r + 1) * cellh,
                                             fill=color, outline="#0c1626", width=.5)
        shown = "" if arr.shape == full.shape else f" · showing the first {arr.shape[0]}×{arr.shape[1]} of {full.shape[0]}×{full.shape[1]}"
        colours = "colour 0 to 1" if probabilities else f"colour ±{scale:.3g}, blue negative, orange positive"
        self.canvas.create_text(x, y + h + 4, anchor="nw", width=max(w, 160), fill="#93a4ba",
                                text=f"rows = {axes[0]}, columns = {axes[1]} · range {low:.3g} to {high:.3g} · {colours}{shown}")
