"""Inside the network (3D): the Tk control surface around the separate Panda3D viewer (studio_brain3d.py).

All decisions (modes, zoom levels, what is sent to the viewer, labels, enabled state) are plain data and functions
at the top so tests can use them without Tk. The page only draws them."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import tkinter as tk
from tkinter import ttk

from studio_ia.copy import INSIDE3D as T
from studio_ia.pages.base import Page
from studio_ia.pages.trace import LI
from studio_ia.theme import C, GUTTER, tk_label
from studio_ia.widgets import Card, Kicker, Segmented, button
from studio_scene import SceneModel
from studio_spec import upgrade

# (id, label on this page, name the viewer uses): the viewer's MODES are 'Build', 'Train', 'Ride a token'
MODES = (("build", T["build_mode"], "Build"), ("train", T["train_mode"], "Train"), ("ride", T["ride_mode"], "Ride a token"))
# the board shows three of the viewer's five levels (Run and Model are not on the board)
LEVELS = (("window", T["window"], "Window", T["window_scale"]), ("head", T["head"], "Head", T["head_scale"]),
          ("weights", T["weights"], "Weights", T["weights_scale"]))
MODE_LABELS = [m[1] for m in MODES]
LEVEL_LABELS = [lv[1] for lv in LEVELS]
DEFAULT_TOKEN = "is"

# per mode: (hud kicker, title, body, tip kicker, tip text, link label, link route)
_MODE_TEXT = {
    "build": ("build_hud", "build_title", "build_body", "build_tip", "build_tip_text", "try_sandbox", "quicktest"),
    "train": ("train_hud", "train_title", "train_body", "train_tip", "train_tip_text", "measure_heads", "head_lab"),
    "ride": ("ride_hud", "ride_title", "ride_body", "ride_tip", "ride_tip_text", "same_data", "trace"),
}
_LEVEL_TEXT = {
    "head": ("zoom_head", "head_title", "head_body", "head_tip", "head_tip_text", "same_head", "head_zoom"),
    "weights": ("zoom_weights", "weights_title", "weights_body", "weights_tip", "weights_tip_text", "what_matrix", "value_map"),
}
_LEVEL_LEGEND = {
    "head": (("pillars", "pillars_text"), ("amber_col", "amber_col_text"), ("dark_tiles", "dark_text")),
    "weights": (("amber", "amber_weight_text"), ("blue", "blue_text")),
}
_WINDOW_LEGEND = (("lanes", "lanes_text"), ("bars", "bars_text"), ("gates", "gates_text"), ("towers", "towers_text"), ("orange_waves", "orange_text"))


def label_to_id(table, label, col=1):
    return next(row[0] for row in table if row[col] == label)


@dataclass
class SceneState:
    level: str = "window"
    mode: str = "build"
    block: int = 1  # 1-based, for the sentences only: the viewer owns the real selection
    token: str = DEFAULT_TOKEN

    def message(self):
        """What the viewer is told: its own names for the level and mode (handled by its 'scene' message)."""
        return {"level": next(v[2] for v in LEVELS if v[0] == self.level), "mode": next(m[2] for m in MODES if m[0] == self.mode)}

    def hud(self):
        """(kicker, title, body): at Head and Weights the zoom text replaces the mode text, as on the board."""
        if self.level != "window":
            k, ti, b = _LEVEL_TEXT[self.level][:3]
            return T[k], T[ti].format(block=self.block, head=1, token=self.token), T[b].format(token=self.token)
        k, ti, b = _MODE_TEXT[self.mode][:3]
        return T[k], T[ti].format(block=self.block, token=self.token), T[b]

    def tip(self):
        """(kicker, text, link label, link route)"""
        k, text, link, route = (_LEVEL_TEXT[self.level] if self.level != "window" else _MODE_TEXT[self.mode])[3:]
        return T[k], T[text], T[link], route

    def legend(self):
        pairs = _WINDOW_LEGEND if self.level == "window" else _LEVEL_LEGEND[self.level]
        return [(T[a], T[b]) for a, b in pairs]

    def scale(self):
        return next(v[3] for v in LEVELS if v[0] == self.level)


def viewer_view(status, has_version):
    """Everything the control row shows, from services.viewer_status() and whether a saved version exists:
    {button, enabled, scene_enabled, note, notes}. The scene controls only do something while the viewer runs."""
    status = status or {}
    if not status.get("available"):
        return {"button": T["open_viewer"], "enabled": False, "scene_enabled": False, "note": T["viewer_missing"], "notes": []}
    running = bool(status.get("running"))
    notes = [T["viewer_no_replay"], T["values_note"]]
    if not has_version:
        notes.insert(0, T["viewer_demo"])
    return {"button": T["close_viewer"] if running else T["open_viewer"], "enabled": True, "scene_enabled": running,
            "note": T["viewer_on"] if running else T["viewer_off"], "notes": notes}


def swap_spec(spec, message):
    """(new spec, block index, slot, entry) for the viewer's 'module_swap' message, checked the way the designer checks it.
    Raises ValueError when the block id is not one of the spec's or the swap is invalid."""
    try:
        index = int(str(message["block_id"]).removeprefix("blk_"), 16)  # block ids are blk_<index in hex>
        slot, entry = message["slot"], message["entry"]
    except (KeyError, ValueError):
        raise ValueError("the viewer sent an unreadable change") from None
    scene = SceneModel(copy.deepcopy(upgrade(spec)))
    scene.swap(index, slot, entry)
    return scene.spec, index, slot, entry


def save_swap(services, message):
    """Save the viewer's module swap as a new version. Returns (the sentence to show, the spec to send back or None)."""
    version = services.current_version()
    if version is None:
        return T["swap_no_version"], None
    try:
        spec, index, slot, entry = swap_spec(version.spec, message)
    except ValueError as error:
        return T["swap_failed"].format(reason=error), None
    args = {"slot": slot, "block": index + 1, "module": entry["id"]}
    saved = services.save_version(spec, T["swap_change"].format(**args))
    return T["swap_saved"].format(n=saved.number, **args), spec


def stats_message(payload):
    """What the viewer gets from a 'train' queue event: compact stats, or a saved snapshot to load; None for other events."""
    if not isinstance(payload, dict):
        return None
    if payload.get("type") == "viz_snapshot":  # a saved training snapshot: the viewer loads it and the scrubber follows
        return {"type": "viz_snapshot", "path": payload["path"]}
    if payload.get("step") is None or payload.get("loss") is None:
        return None
    return {"type": "stats", **{k: payload.get(k) for k in ("step", "total", "loss", "lr", "tokens_per_second")}}


def status_of(services):
    try:
        return services.viewer_status()
    except NotImplementedError:
        return None


class Inside3DPage(Page):
    title = T["h1"]
    lead = T["lead"]
    crumb = (*LI, "3D")

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.scene = SceneState()
        self._poll = None
        self._scrubbing = False  # True while the viewer's own position moves the slider
        self._build()
        self.bind("<Destroy>", self._cancel_poll, add="+")

    # layout -----------------------------------------------------------------
    def _build(self):
        row = Card(self.body)
        row.pack(fill="x")
        Kicker(row, T["viewer_label"], bg="panel").pack(anchor="w", pady=(0, 6))
        self.note = tk_label(row, "", 14, "text2", "panel", wraplength=760)
        self.note.pack(anchor="w")
        self.launch = button(row, T["open_viewer"], self.toggle_viewer, primary=True)
        self.launch.pack(anchor="w", pady=(8, 0))
        self.notes_box = tk_label(row, "", 13, "muted", "panel", wraplength=760)
        self.notes_box.pack(anchor="w", pady=(8, 0))

        pick = Card(self.body)
        pick.pack(fill="x", pady=(GUTTER, 0))
        Kicker(pick, T["mode_label"], bg="panel").pack(anchor="w", pady=(0, 6))
        self.modes = Segmented(pick, MODE_LABELS, T["build_mode"], command=self.pick_mode)
        self.modes.pack(anchor="w")
        Kicker(pick, T["zoom_label"], bg="panel").pack(anchor="w", pady=(GUTTER, 6))
        self.levels = Segmented(pick, [f"{i + 1} · {lv[1]}" for i, lv in enumerate(LEVELS)], f"1 · {T['window']}", command=self.pick_level)
        self.levels.pack(anchor="w")
        self.scale_label = tk_label(pick, "", 13, "muted", "panel")
        self.scale_label.pack(anchor="w", pady=(6, 0))

        self.moves = Card(self.body)
        self.moves.pack(fill="x", pady=(GUTTER, 0))
        Kicker(self.moves, T["step_through"], bg="panel").pack(anchor="w", pady=(0, 6))
        steps, weights = (tk.Frame(self.moves, bg=C["panel"]) for _ in range(2))
        steps.pack(anchor="w")
        Kicker(self.moves, T["weights"], bg="panel").pack(anchor="w", pady=(GUTTER, 6))
        weights.pack(anchor="w")
        self.move_buttons = [
            button(steps, T["step_back"], lambda: self.send_view({"type": "step", "delta": -1})),
            button(steps, T["step_next"], lambda: self.send_view({"type": "step", "delta": 1})),
            button(steps, T["step_action_1"], lambda: self.send_view({"type": "step", "stage": "scores"})),
            button(steps, T["step_action_2"], lambda: self.send_view({"type": "step", "stage": "values"})),
            button(weights, T["show_weights"], lambda: self.send_view({"type": "show", "what": "weight"})),
            button(weights, T["weights_action_1"], lambda: self.send_view({"type": "show", "what": "gradient"})),
            button(weights, T["weights_action_2"], lambda: self.send_view({"type": "show", "what": "update"})),
        ]
        for i, b in enumerate(self.move_buttons):
            b.grid(row=(i % 4) // 2, column=i % 2, sticky="w", padx=(0, 8), pady=2)
        self.scrub_label = tk_label(self.moves, "", 13, "muted", "panel")
        self.scrub_label.pack(anchor="w", pady=(8, 0))
        self.scrubber = ttk.Scale(self.moves, from_=0, to=0, command=self._scrub)
        self.scrubber.pack(fill="x")
        self.move_note = tk_label(self.moves, "", 13, "muted", "panel", wraplength=760)
        self.move_note.pack(anchor="w", pady=(6, 0))

        self.hud_card = Card(self.body)
        self.hud_card.pack(fill="x", pady=(GUTTER, 0))
        self.hud_kicker = Kicker(self.hud_card, "", color="violet", bg="panel")
        self.hud_kicker.pack(anchor="w")
        self.hud_title = tk_label(self.hud_card, "", 18, "text", "panel", wraplength=760)
        self.hud_title.pack(anchor="w", pady=(4, 0))
        self.hud_body = tk_label(self.hud_card, "", 14, "text2", "panel", wraplength=760)
        self.hud_body.pack(anchor="w", pady=(6, 0))

        self.legend_card = Card(self.body)
        self.legend_card.pack(fill="x", pady=(GUTTER, 0))
        self.tip_card = Card(self.body)
        self.tip_card.pack(fill="x", pady=(GUTTER, 0))
        tk_label(self.body, T["controls"], 13, "muted", wraplength=760).pack(anchor="w", pady=(GUTTER, 0))
        self._paint()

    # behaviour --------------------------------------------------------------
    def has_version(self):
        try:
            return self.services.current_version() is not None
        except NotImplementedError:
            return False

    def refresh(self):
        self._paint()
        self._tick()

    def _paint(self):
        view = viewer_view(status_of(self.services), self.has_version())
        self.launch.configure(text=view["button"], state="normal" if view["enabled"] else "disabled")
        for b in self.move_buttons:
            b.configure(state="normal" if view["scene_enabled"] else "disabled")
        self.note.configure(text=view["note"])
        self.notes_box.configure(text="\n".join(view["notes"]))
        sc = self.scene
        self.scale_label.configure(text=sc.scale())
        kicker, title, body = sc.hud()
        self.hud_kicker.configure(text=kicker.upper())
        self.hud_title.configure(text=title)
        self.hud_body.configure(text=body)
        for card in (self.legend_card, self.tip_card):
            for child in card.winfo_children():
                child.destroy()
        Kicker(self.legend_card, T["what_seeing"], bg="panel").pack(anchor="w", pady=(0, 6))
        for name, text in sc.legend():
            tk_label(self.legend_card, f"{name}: {text}", 13, "text2", "panel", wraplength=760).pack(anchor="w", pady=2)
        kicker, text, link, route = sc.tip()
        Kicker(self.tip_card, kicker, color="amber", bg="panel").pack(anchor="w")
        tk_label(self.tip_card, text, 14, "text2", "panel", wraplength=760).pack(anchor="w", pady=(4, 6))
        button(self.tip_card, link, lambda: self.navigate(route)).pack(anchor="w")

    def toggle_viewer(self):
        try:
            if (status_of(self.services) or {}).get("running"):
                self.services.viewer_stop()
            else:
                self.services.viewer_start(self.scene.message())
        except (NotImplementedError, OSError) as error:
            self.note.configure(text=str(error) or T["viewer_missing"])
            return
        self._paint()
        self._tick()

    def send_view(self, message):
        if (status_of(self.services) or {}).get("running"):
            self.services.viewer_send(message)

    def _send_scene(self):
        self.send_view({"type": "scene", **self.scene.message()})

    def _scrub(self, value):
        if not self._scrubbing:
            self.send_view({"type": "scrub", "index": round(float(value))})

    def on_event(self, kind, payload):
        """Training progress goes to the viewer: compact 'stats' (the link keeps only the latest) and each saved snapshot."""
        message = stats_message(payload) if kind == "train" else None
        if message:
            self.send_view(message)

    def _from_viewer(self, message):
        kind = message.get("type") if isinstance(message, dict) else None
        if kind == "module_swap":
            text, spec = save_swap(self.services, message)
            self.move_note.configure(text=text)
            if spec is not None:
                self.send_view({"type": "spec", "spec": spec, "selected": message.get("block_id")})
        elif kind == "replay_state":
            self._scrubbing = True
            self.scrubber.configure(to=max(0, message["count"] - 1))
            self.scrubber.set(message["index"])
            self.scrub_label.configure(text=T["scrub_label"].format(n=message["index"] + 1, count=message["count"]))
            self._scrubbing = False

    def pick_mode(self, label):
        self.scene.mode = label_to_id(MODES, label)
        self.modes.set(label)
        self._paint()
        self._send_scene()

    def pick_level(self, label):
        name = label.split(" · ", 1)[1]
        self.scene.level = label_to_id(LEVELS, name)
        self.levels.set(label)
        self._paint()
        self._send_scene()

    def _tick(self):
        """While the viewer runs: drain what it says (so the link never backs up), notice when it closes, and repaint."""
        self._cancel_poll()
        status = status_of(self.services) or {}
        if status.get("running"):
            try:
                messages = self.services.viewer_poll()  # the viewer's own selection stays its own; only swaps and the replay position are read
            except NotImplementedError:
                return
            for message in messages:
                self._from_viewer(message)
            self._poll = self.after(1000, self._tick)
        elif self.launch.cget("text") == T["close_viewer"]:
            self._paint()  # the viewer window was closed from its own side

    def _cancel_poll(self, _event=None):
        if self._poll is not None:
            try:
                self.after_cancel(self._poll)
            except Exception:  # noqa: BLE001 - the widget may already be gone
                pass
            self._poll = None
