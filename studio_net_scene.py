"""Bounded in-app 2D scene controls plus the separate 3D viewer link."""
from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import ttk

import numpy as np

from studio_net_common import NetworkView
from studio_scene import LEVELS, MODES, SceneModel
from studio_widgets import label


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
            checkpoint = str(self.checkpoint())
        except ValueError:
            checkpoint = None
        if checkpoint and checkpoint != self._data_key:
            self._data_key = checkpoint
            weights = self.host.arch_store.weights(self.host.arch_id)
            run_dirs = [row["run_dir"] for row in weights if row.get("run_dir")]
            replay = self.host.net_replay_paths()
            self.work(lambda: self._load_data(checkpoint, run_dirs, replay), self._data_loaded)
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
        summary = binned_attention(path, "The model reads this prompt.", n_bins=16)
        paths = run_paths(list(dict.fromkeys(map(str, run_dirs))))
        snapshots = sorted(map(Path, replay), key=lambda item: item.stat().st_mtime)
        snapshot = load(snapshots[-1]) if snapshots else None
        return {"attention": summary["attention"], "stream_norms": summary["stream_norms"],
                "edges": summary["edges"].tolist(), "paths": paths, "snapshot": snapshot}

    def _data_loaded(self, ok, result):
        if not ok:
            self.message.set(f"Checkpoint scene data unavailable: {result}")
            return
        self.scene.summaries = result["attention"]
        self.scene.stream_norms = result["stream_norms"]
        self.scene.summary_edges = result["edges"]
        self.scene.paths = result["paths"]
        if result["snapshot"]:
            self.scene.load_snapshot(result["snapshot"])
        self.changed()
        self.message.set("Scene data loaded from the selected checkpoint; arrays and replay windows are bounded.")

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
            if mats:
                key = sorted(mats)[min(frame["selection"]["layer"], len(mats) - 1)]
                data = np.asarray(mats[key])
                self._heatmap(data[0] if data.ndim == 3 else data, width - 210, 62, 170, 130)
                c.create_text(width - 205, 48, anchor="nw", text="attention · selected layer", fill=muted)
        elif frame["level"] in ("Window", "Head"):
            key = f"block{frame['selection']['layer']}.attn.pattern"
            matrix = frame.get("pattern", []) if frame["level"] == "Head" else frame.get("patterns", {}).get(key, [])
            arr = np.asarray(matrix)
            if arr.ndim == 3: arr = arr[min(frame["selection"]["head"], arr.shape[0] - 1)]
            if arr.ndim == 2 and arr.size:
                self._heatmap(arr, 52, 50, min(width - 90, 520), min(height - 90, 250))
                c.create_text(52, height - 20, anchor="w", text=f"token window {frame.get('start', 0)}–{frame.get('end', 0)} · head {frame['selection']['head'] + 1}", fill=muted)
            else:
                c.create_text(width / 2, height / 2, text="No recorded token snapshot; attention summary is available at Model level", fill=muted)
        else:
            data = frame.get(self.weight_kind.get(), [])
            arr = np.asarray(data)
            if arr.ndim == 2 and arr.size:
                self._heatmap(arr, 52, 50, min(width - 90, 520), min(height - 90, 250))
                c.create_text(52, height - 20, anchor="w", text=f"{self.weight_kind.get()} · {frame.get('weight_name') or 'no weight snapshot'} · 16×16 maximum", fill=muted)
            else:
                c.create_text(width / 2, height / 2, text="No recorded weight snapshot", fill=muted)

    def _heatmap(self, values, x, y, w, h):
        arr = np.asarray(values, dtype=float)
        if arr.ndim != 2 or not arr.size:
            return
        arr = arr[:32, :32]
        maxv = max(float(np.max(np.abs(arr))), 1e-9)
        cellw, cellh = w / arr.shape[1], h / arr.shape[0]
        for r in range(arr.shape[0]):
            for col in range(arr.shape[1]):
                value = min(1.0, abs(float(arr[r, col])) / maxv)
                shade = int(24 + value * 190)
                color = f"#{14:02x}{shade:02x}{min(80 + shade // 2, 255):02x}"
                self.canvas.create_rectangle(x + col * cellw, y + r * cellh,
                                             x + (col + 1) * cellw, y + (r + 1) * cellh,
                                             fill=color, outline="#0c1626", width=.5)
