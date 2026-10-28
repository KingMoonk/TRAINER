# Git workflow

- `main` is the base branch: the upcoming main branch. PRs target it and are squash-merged into it. Merge, never rebase, never force-push.
- Issue-based dev: each issue or task gets a new branch off `main` in its own new worktree.
  - Subagents (`implementer`): Claude Code makes the worktree (`isolation: worktree` in `.claude/agents/*.md`) under `.claude/worktrees/`, links `.venv` and `web/node_modules` into it (`worktree.symlinkDirectories` in `.claude/settings.json`) and removes it when the agent ends without changes. It is cut from the default branch (`main`); the agent's first command still branches from the latest: `git fetch -q origin main && git switch -c ai/<slug> origin/main`.
  - The main session, by hand, under `/Users/wedd/dev/TRAINER-worktrees/<slug>`: `cd /Users/wedd/dev/TRAINER && git fetch -q origin main && git worktree add /Users/wedd/dev/TRAINER-worktrees/<slug> -b <branch> origin/main && ln -s /Users/wedd/dev/TRAINER/.venv /Users/wedd/dev/TRAINER-worktrees/<slug>/.venv`. Open its PR with `--base main`.
- Never check out `main` itself (no `git checkout`/`switch` to it, no `git worktree add <path> main`): git allows a branch in only one worktree, so a checkout anywhere blocks everyone else with "already used by worktree". Always branch from `origin/main` with `-b`.
- The main checkout `/Users/wedd/dev/TRAINER` is the user's: it has `main` checked out for the live dev server (`make dev`), which is why nothing else may check that branch out. Never switch, detach, edit or commit in it; only fast-forward it with `git pull --ff-only` after a merge.
- `git worktree add` does not copy `.venv`, so the manual command symlinks the main checkout's into the new worktree; `make test` then works there unchanged. If a worktree lacks it, run `ln -s /Users/wedd/dev/TRAINER/.venv .venv` in it.
- Parallel tasks never edit the same files. Split file ownership up front.
- Staying current: the `sync-base.sh` hook (`.claude/hooks/sync-base.sh`, wired in `.claude/settings.json`) fetches `main` at session start and at most every two minutes per prompt. It fast-forwards the local `main` ref, and reports when `origin/main` moved and whether a merge into your branch would be clean.
  - It reports a clean merge and your tree is clean: merge now with `git merge --no-edit origin/main`.
  - It reports conflicts: finish and commit your current edit, then merge and resolve.
  - Always merge the PR's base branch before opening the PR.
  - In app-made worktrees, `mcp__ccd_host__sync_with_base_branch` does the same.
- Pushing: the SSH remote may lack keys in a sandbox, so push with `git -c credential.helper= -c "credential.helper=!gh auth git-credential" push -q https://github.com/KingMoonk/TRAINER.git HEAD:<branch>`.
- Push straight to `main` only when the user asks for it. Otherwise open a PR and let the user merge.
- Never run bare `git stash`: the stack is shared across worktrees. Prefer a WIP commit to set work aside. If you must stash, use `git stash push -u -m <tag>`, take the entry's SHA from `git stash list --format='%H %gs'`, restore with `git stash apply <sha>` (not pop), then drop the entry by its current `stash@{n}`, found by tag.
- After the PR merges: `git worktree remove <path>` and delete the branch.
- Why a hook and not a GitHub webhook: a webhook needs a public endpoint, and this machine is behind CGNAT on Starlink. A throttled fetch in a hook gives the same awareness within a prompt.
