"""3D scene controls and bounded scene summary for the separate viewer process."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from studio_net_common import NetworkView
from studio_scene import LEVELS, MODES, SceneModel
from studio_widgets import label


class SceneView(NetworkView):
    title = "Inside the network · 3D"

    def __init__(self, parent, host):
        super().__init__(parent, host)
        self.scene = None
        line = ttk.Frame(self)
        line.pack(fill="x")
        self.level = tk.StringVar(value="Model")
        self.mode = tk.StringVar(value="Build")
        for title, variable, values in (("Level", self.level, LEVELS), ("Mode", self.mode, MODES)):
            label(line, title).pack(side="left")
            box = ttk.Combobox(line, textvariable=variable, values=values, state="readonly", width=16)
            box.pack(side="left", padx=(6, 18))
            box.bind("<<ComboboxSelected>>", lambda _event: self.changed())
        ttk.Button(line, text="Open separate 3D viewer", command=host.net_open_viewer).pack(side="left")
        self.details = tk.StringVar()
        label(self, textvariable=self.details, style="Muted.TLabel", wraplength=740, justify="left").pack(anchor="w", pady=18)

    def changed(self):
        if self.scene:
            self.scene.level, self.scene.mode = self.level.get(), self.mode.get()
            frame = self.scene.frame()
            self.details.set(f"{frame['level']} · {frame['mode']} · {len(frame.get('gates', []))} gates · "
                             f"selected layer {frame['selection']['layer'] + 1}. "
                             "The separate viewer uses the same design and selection.")
            self.host.net_send({"type": "scene", "level": self.level.get(), "mode": self.mode.get(),
                                "selected": frame["selection"]})

    def refresh(self):
        if not self.spec:
            return
        if self.scene is None:
            self.scene = SceneModel(self.spec)
        else:
            self.scene.update_spec(self.spec)
        self.changed()
        self.message.set("Use Build to swap modules in the Sandbox; Train and Ride a token use recorded snapshots.")
