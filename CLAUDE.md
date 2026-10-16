# Agent rules

- Test with `.venv/bin/python tasks.py test` (all window-free tests and ruff) or `.venv/bin/python tasks.py test tests.test_x tests.test_y` (just those modules). That is the only test command you run. Details: `.claude/rules/testing.md`.
- Only run `tasks.py test:ci` or `test:ci:e2e`, `tests/gui_*.py`, or `tests.e2e_train` if trying to debug CI issues. 
- A new test that opens a window goes in `gui_*.py` with that import as its first import; `test_*.py` must stay window-free.
- While iterating, run just the affected tests (see `.claude/rules/affected-tests.md`).
- Git: every task in its own worktree, a new branch off origin/wedd-compute-cluster in a new worktree (the base branch; PRs target it; never check out wedd-compute-cluster itself). `master` is legacy: only when the user asks. The sync-base hook keeps wedd-compute-cluster current. Details: `.claude/rules/git-workflow.md`.
- No placeholders, fake features or dead ends: leave unfinished work out and open a GitHub issue for it, and for any bug you cannot fix. Details: `.claude/rules/no-placeholders.md`.
- Work from GitHub issues: every feature, bug and open question is an issue with acceptance criteria. Format and labels: `.claude/rules/github-issues.md`.
