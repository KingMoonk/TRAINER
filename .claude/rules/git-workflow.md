# Git workflow

- `wedd-compute-cluster` is the base branch: the upcoming main branch. PRs target it and are squash-merged into it. Merge, never rebase, never force-push.
- `master` is legacy. Do not fetch, compare against, merge from or open PRs to `master` unless the user asks for it in that task.
- Issue-based dev: each issue or task gets a new branch off `wedd-compute-cluster` in its own new worktree under `/Users/wedd/dev/TRAINER-worktrees/<slug>`, then work starts there: `cd /Users/wedd/dev/TRAINER && git fetch -q origin wedd-compute-cluster && git worktree add /Users/wedd/dev/TRAINER-worktrees/<slug> -b <branch> origin/wedd-compute-cluster && ln -s /Users/wedd/dev/TRAINER/.venv /Users/wedd/dev/TRAINER-worktrees/<slug>/.venv`. Open its PR with `--base wedd-compute-cluster`.
- Never check out `wedd-compute-cluster` itself (no `git checkout`/`switch` to it, no `git worktree add <path> wedd-compute-cluster`): git allows a branch in only one worktree, so a checkout anywhere blocks everyone else with "already used by worktree". Always branch from `origin/wedd-compute-cluster` with `-b`.
- The main checkout `/Users/wedd/dev/TRAINER` stays on `master`. Never switch or detach it.
- `git worktree add` does not copy `.venv`, so the command above symlinks the main checkout's into the new worktree; `make test` then works there unchanged. (`worktree.symlinkDirectories` in `.claude/settings.json` does the same for worktrees Claude Code makes itself.) If a worktree lacks it, run `ln -s /Users/wedd/dev/TRAINER/.venv .venv` in it.
- Parallel tasks never edit the same files. Split file ownership up front.
- Staying current: the `sync-base.sh` hook (`.claude/hooks/sync-base.sh`, wired in `.claude/settings.json`) fetches `wedd-compute-cluster` at session start and at most every two minutes per prompt. It fast-forwards the local `wedd-compute-cluster` ref, and reports when `origin/wedd-compute-cluster` moved and whether a merge into your branch would be clean. It never touches `master`, and stays quiet on branches forked from `master`.
  - It reports a clean merge and your tree is clean: merge now with `git merge --no-edit origin/wedd-compute-cluster`.
  - It reports conflicts: finish and commit your current edit, then merge and resolve.
  - Always merge the PR's base branch before opening the PR.
  - In app-made worktrees, `mcp__ccd_host__sync_with_base_branch` does the same.
- Pushing: the SSH remote may lack keys in a sandbox, so push with `git -c credential.helper= -c "credential.helper=!gh auth git-credential" push -q https://github.com/KingMoonk/TRAINER.git HEAD:<branch>`.
- Push straight to `wedd-compute-cluster` or `master` only when the user asks for it. Otherwise open a PR and let the user merge.
- Never run bare `git stash`: the stack is shared across worktrees. Prefer a WIP commit to set work aside. If you must stash, use `git stash push -u -m <tag>`, take the entry's SHA from `git stash list --format='%H %gs'`, restore with `git stash apply <sha>` (not pop), then drop the entry by its current `stash@{n}`, found by tag.
- After the PR merges: `git worktree remove <path>` and delete the branch.
- Why a hook and not a GitHub webhook: a webhook needs a public endpoint, and this machine is behind CGNAT on Starlink. A throttled fetch in a hook gives the same awareness within a prompt.
