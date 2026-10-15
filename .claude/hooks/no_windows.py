"""PreToolUse hook for Bash: refuse commands that open an app window or run the CI-only tests (exit 2 blocks the call).

Window tests (tests/gui_*.py), the e2e runs, tasks.py test:ci and test:ci:e2e, the app itself and the CI / GROUNDWORK_GUI_TESTS
switches that unlock them are CI-only; locally the one test command is `python tasks.py test`. Patterns match within a single
command line, and heredoc bodies are ignored, so grep, ls and commit messages that merely mention these names are not blocked.
The gap: a python script fed through a heredoc that opens a window is not caught; CLAUDE.md still forbids it.
"""
import json
import re
import sys

PY = r"(?:python[\w.]*|py)(?:\.exe)?"
LINE = r"[^\n;&|]*"
BLOCKED = (
    (r"(?:^|[;&|]\s*|\bexport\s+|\benv\s+)(?:CI|GROUNDWORK_GUI_TESTS)=", "setting CI or GROUNDWORK_GUI_TESTS unlocks the window tests"),
    (rf"\b{PY}\s+{LINE}(?:\bgui_\w*|\btests\.gui|\be2e_\w+|\btest:ci)", "window tests and e2e runs are CI-only"),
    (rf"\bunittest\b{LINE}(?:\bgui_\w*|\btests\.gui|\be2e_\w+)", "window tests and e2e runs are CI-only"),
    (r"\btasks\.py\s+test:ci", "test:ci and test:ci:e2e are CI-only"),
    (rf"\b{PY}\s+(?:\./)?app\.py\b|\b{PY}\s+-m\s+studio_ia\b|\bstart_studio\b", "do not open the app locally"),
    (rf"\b{PY}\s+-c\s+[^\n]*\bTk\(", "do not open a Tk window locally"),
)


def blocked(command):
    command = re.sub(r"(<<-?\s*['\"]?(\w+)['\"]?[^\n]*\n).*?\n\s*\2\b", r"\1", command, flags=re.S)  # heredoc bodies are text, not commands
    return next((why for pattern, why in BLOCKED if re.search(pattern, command)), None)


if __name__ == "__main__":
    why = blocked(json.load(sys.stdin).get("tool_input", {}).get("command", ""))
    if why:
        print(f"Blocked: {why}. Local tests: `.venv/bin/python tasks.py test` (see .claude/rules/testing.md).", file=sys.stderr)
        sys.exit(2)
