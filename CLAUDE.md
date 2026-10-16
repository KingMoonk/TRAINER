# Agent rules

- Test with `.venv/bin/python tasks.py test` (all window-free tests and ruff) or `.venv/bin/python tasks.py test tests.test_x tests.test_y` (just those modules). That is the only test command you run. Details: `.claude/rules/testing.md`.
- Only run `tasks.py test:ci` or `test:ci:e2e`, `tests/gui_*.py`, or `tests.e2e_train` if trying to debug CI issues. 
- A new test that opens a window goes in `gui_*.py` with that import as its first import; `test_*.py` must stay window-free.
- While iterating, run just the affected tests (see `.claude/rules/affected-tests.md`).
- Git: every task in its own worktree, cut from origin/master; the sync-master hook keeps master current. Details: `.claude/rules/git-workflow.md`.
- No placeholders, fake features or dead ends: leave unfinished work out and open a GitHub issue for it, and for any bug you cannot fix. Details: `.claude/rules/no-placeholders.md`.
- Work from GitHub issues: every feature, bug and open question is an issue with acceptance criteria. Format and labels: `.claude/rules/github-issues.md`.
- Delegate non-trivial, well-defined work to a subagent: `implementer` (Sonnet) for moderate or difficult work, `grunt` (Haiku) for repetitive or read-heavy work. Details: `.claude/rules/subagents.md`.
