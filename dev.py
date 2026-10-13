"""Run the studio with hot reload: code changes are applied inside the open window, with no relaunch."""
import importlib
import sys
import tkinter as tk
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RELOAD_KEY = "<Command-r>" if sys.platform == "darwin" else "<Control-r>"


def snapshot():
    """The files a reload re-imports: top-level modules and the studio_modules package."""
    files = {}
    for path in [*ROOT.glob("*.py"), *(ROOT / "studio_modules").rglob("*.py")]:
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
        self._dev_cmds = set(self._tclCommands or ())  # Tcl commands that belong to the bare window
        self._dev_keep = set(vars(self)) | {"_dev_keep", "_dev_cmds"}

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
    for name in set(app._tclCommands or ()) - app._dev_cmds:  # callbacks the old app registered on the root
        try:
            app.deletecommand(name)
        except tk.TclError:
            pass
    for key in set(vars(app)) - app._dev_keep:
        del vars(app)[key]
    app.__class__ = cls
    cls.__init__(app, *args)


def project_modules():
    # decided by name, never by touching the module: a lazily loaded third party module (cupyx.scipy) imports itself on any attribute read
    # the studio_modules package goes too, or its @register decorators never re-run against the fresh registry
    return {name: mod for name, mod in list(sys.modules.items())
            if name == "studio_modules" or name.startswith("studio_modules.")
            or (name not in ("__main__", "dev") and "." not in name and (ROOT / f"{name}.py").is_file())}


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
            try:
                rebuild(app, old_cls, store(folder))
            except Exception:
                error += "\nRollback failed too:\n" + traceback.format_exc()
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
        app.bind_all(RELOAD_KEY, lambda _e: state.update(force=True))  # replace, never stack: a failed reload keeps the old binding

    def tick():
        try:
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
        except Exception:
            traceback.print_exc()
        finally:
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
