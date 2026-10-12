"""Launch the Groundwork Python AI Studio desktop application."""

try:
    from studio_gui import main
except ImportError as exc:
    raise SystemExit("Install the native studio dependencies with: python -m pip install -r requirements.txt\n" + str(exc)) from exc


if __name__ == "__main__":
    raise SystemExit(main())
