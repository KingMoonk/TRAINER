"""Run the studio with hot reload: code changes are applied inside the open window, with no relaunch."""
import importlib
import os
import sys
import tkinter as tk
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SKIP = {"tests", "__pycache__", "runs", "data", "exports", "artifacts"}
RELOAD_KEY = "<Command-r>" if sys.platform == "darwin" else "<Control-r>"


def snapshot():
    files = {}
    for folder, dirs, names in os.walk(ROOT):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in SKIP]
        for name in names:
            if name.endswith(".py"):
                path = Path(folder, name)
                try:
                    files[path] = path.stat().st_mtime_ns
                except FileNotFoundError:
                    pass
    return files


def install():
    """Let one Tk window be constructed and torn down more than once, so the app's own __init__ and destroy do the rebuild."""
    if getattr(tk.Tk.__init__, "_dev", False):
        return
    tk_init, tk_destroy = tk.Tk.__init__, tk.Tk.destroy

    def init(self, *args, **kwargs):
        if "tk" in vars(self):  # rebuild: the window already exists
            return
        tk_init(self, *args, **kwargs)
        self._dev_binds = {seq: self.bind_class("all", seq) for seq in self.bind_class("all")}
        self._dev_keep = set(vars(self)) | {"_dev_keep"}

    def destroy(self):
        if not vars(self).get("_dev_soft"):
            return tk_destroy(self)
        for child in list(self.children.values()):  # rebuild: empty the window, keep it open
            child.destroy()

    init._dev = True
    tk.Tk.__init__, tk.Tk.destroy = init, destroy


def rebuild(app, cls, *args):
    """Strip the window back to a bare Tk root, then run cls's constructor on it."""
    app._dev_soft = True
    app.destroy()  # the app's own cleanup: saves pending edits, stops the 3D viewer
    for job in app.tk.splitlist(app.tk.call("after", "info")):
        app.after_cancel(job)
    for seq in app.bind_class("all"):
        app.unbind_class("all", seq)
    for seq, script in app._dev_binds.items():
        app.bind_class("all", seq, script)
    for seq in app.bind():
        app.unbind(seq)
    for key in set(vars(app)) - app._dev_keep:
        del vars(app)[key]
    app.__class__ = cls
    cls.__init__(app, *args)


def project_modules():
    return {name: mod for name, mod in sys.modules.items()
            if name not in ("__main__", "dev") and getattr(mod, "__file__", None) and Path(mod.__file__).parent == ROOT}


def reload(app):
    """Import the project fresh and rebuild the window with it. On any error the old code keeps running."""
    old_cls, old_modules = type(app), project_modules()
    geometry, page = app.geometry(), getattr(app, "active_page", None)
    store, folder = type(app.workspace), app.workspace.root
    app.workspace.save()
    for name in old_modules:
        del sys.modules[name]
    error, torn = None, False
    try:
        cls = getattr(importlib.import_module(old_cls.__module__), old_cls.__name__)
        fresh = getattr(sys.modules[store.__module__], store.__name__)(folder)
        torn = True
        rebuild(app, cls, fresh)
    except Exception:
        error = traceback.format_exc()
        for name in project_modules():
            del sys.modules[name]
        sys.modules.update(old_modules)
        if torn:  # the new code broke while building: put the old window back
            rebuild(app, old_cls, store(folder))
    app.geometry(geometry)
    if page in getattr(app, "pages", {}):
        app.show_page(page)
    if error:
        print(error, file=sys.stderr, flush=True)
        app.status_var.set("Reload failed, still running the old code · " + error.strip().splitlines()[-1][:120])
    else:
        print("reloaded", flush=True)
    return not error


def watch(app):
    """Reload after an edit once the files settle, but only while the app is in the background; RELOAD_KEY reloads at once."""
    state = {"applied": snapshot(), "last": None, "force": False, "hinted": None}

    def arm():
        app.bind_all(RELOAD_KEY, lambda _e: state.update(force=True), add="+")

    def tick():
        current = snapshot()
        changed, settled = current != state["applied"], current == state["last"]
        state["last"] = current
        busy = app.job_kind or app.tk.call("grab", "current")  # never reload over a running job or an open dialog
        focused = app.tk.call("focus", "-displayof", ".")
        if not busy and (state["force"] or (changed and settled and not focused)):
            state.update(applied=current, force=False)
            reload(app)
            arm()
        elif changed and state["hinted"] != current:
            state["hinted"] = current
            app.status_var.set("Code changed · reloads when you switch away" + (" and the job ends" if app.job_kind else "") + f", or press {RELOAD_KEY[1:-1]}")
        app.after(500, tick)

    arm()
    app.after(500, tick)


def main():
    install()
    from studio_gui import Studio
    app = Studio()
    watch(app)
    app.mainloop()


if __name__ == "__main__":
    main()
