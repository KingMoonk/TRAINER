"""Launch the Groundwork Python AI Studio desktop application."""
import sys

if len(sys.argv) > 2 and sys.argv[1] == "--run-module":
    # A frozen build has no `python -m`: its own executable runs the passive trainer, watchdog and 3D viewer (see studio_core.module_command).
    import runpy
    for stream in ("stdout", "stderr"):
        if getattr(sys, stream) is None:  # windowed builds have no console
            setattr(sys, stream, open("nul" if sys.platform == "win32" else "/dev/null", "w"))
    sys.argv = sys.argv[2:]
    runpy.run_module(sys.argv[0], run_name="__main__")
    raise SystemExit(0)

try:
    from studio_ia.shell import main
except ImportError as exc:
    raise SystemExit("Install the native studio dependencies with: python -m pip install -r requirements.txt\n" + str(exc)) from exc


if __name__ == "__main__":
    raise SystemExit(main())
