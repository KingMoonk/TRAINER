"""Design tokens and ttk styling for the Groundwork Studio shell."""
from __future__ import annotations

import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk

C = dict(
    bg="#0e1014", panel="#15181e", panel2="#1b1f27", border="#2a2f3a", border2="#20242c",
    text="#e6e8ec", text2="#c3c8d1", muted="#8f97a6", dim="#5b6271",
    violet="#9f8cff", amber="#e5b567", coral="#ef8a7a", blue="#6cb4ee", pink="#e58fb0", sage="#9fd36b",
    amber_bg="#211c12", coral_bg="#26171a", blue_bg="#121a22", violet_bg="#1a1730",
)

# State -> (foreground, background): not saved, saved/needs training, training, live.
STATES = {
    "unsaved": (C["amber"], C["amber_bg"]),
    "needs_training": (C["coral"], C["coral_bg"]),
    "training": (C["violet"], C["violet_bg"]),
    "live": (C["blue"], C["blue_bg"]),
    "attention": (C["pink"], C["panel2"]),
    "pieces": (C["sage"], C["panel2"]),
    "neutral": (C["muted"], C["panel2"]),
}

GUTTER = 16
SIDEBAR_W = 220
BUTTON_H = 44


def font(size=15, bold=False, mono=False, serif=False):
    """Font tuple with the size in pixels (negative size = pixels in Tk)."""
    if mono:
        family = tkfont.nametofont("TkFixedFont").actual("family")
    elif serif:
        family = "Georgia"
    else:
        family = tkfont.nametofont("TkDefaultFont").actual("family")
    return (family, -size, "bold") if bold else (family, -size)


def apply(root):
    """Dark theme for the whole app: root background, option database and ttk styles."""
    root.configure(bg=C["bg"])
    root.option_add("*Background", C["bg"])
    root.option_add("*Foreground", C["text"])
    style = ttk.Style(root)
    style.theme_use("clam")
    base = font(15)
    style.configure(".", background=C["bg"], foreground=C["text"], fieldbackground=C["panel2"], bordercolor=C["border"],
                    lightcolor=C["border"], darkcolor=C["border"], troughcolor=C["panel"], font=base, focuscolor=C["violet"])
    style.configure("TLabel", background=C["bg"], foreground=C["text"])
    style.configure("Panel.TFrame", background=C["panel"])
    style.configure("H1.TLabel", font=font(32, bold=True))
    style.configure("H2.TLabel", font=font(22, bold=True))
    style.configure("Lead.TLabel", foreground=C["text2"])
    style.configure("Muted.TLabel", foreground=C["muted"], font=font(13))
    style.configure("TButton", background=C["panel2"], foreground=C["text"], padding=(14, 10), borderwidth=1, relief="flat")
    style.map("TButton", background=[("active", C["border"]), ("focus", C["border"])], bordercolor=[("focus", C["violet"])])
    # Primary buttons are at least 44px tall: 15px text + 2 * 14px padding + border.
    style.configure("Primary.TButton", background=C["violet"], foreground=C["bg"], padding=(20, 14), font=font(15, bold=True))
    style.map("Primary.TButton", background=[("active", "#b3a4ff"), ("focus", "#b3a4ff")], foreground=[("disabled", C["dim"])])
    style.configure("Vertical.TScrollbar", background=C["panel2"], arrowcolor=C["muted"])
    style.configure("TProgressbar", background=C["violet"], troughcolor=C["panel2"])
    return style


def tk_label(parent, text="", size=15, color="text", bg="bg", **kw):
    """Plain tk.Label in token colours (use where per-widget background matters, e.g. inside cards)."""
    return tk.Label(parent, text=text, font=kw.pop("font", font(size)), fg=C.get(color, color), bg=C.get(bg, bg),
                    anchor=kw.pop("anchor", "w"), justify=kw.pop("justify", "left"), **kw)
