# Git workflow

- `master` is the baseline. PRs are squash-merged into it. Long-lived integration branches (currently `wedd-compute-cluster`) merge `master` in regularly. Merge, never rebase, never force-push.
- Every task runs in its own worktree under `/Users/wedd/dev/TRAINER-worktrees/<slug>`, on a branch cut from `origin/master` (or from the integration branch the task targets): `cd /Users/wedd/dev/TRAINER && git fetch -q origin master && git worktree add /Users/wedd/dev/TRAINER-worktrees/<slug> -b <branch> origin/master`.
- The main checkout `/Users/wedd/dev/TRAINER` stays on `master`. Never switch or detach it.
- A worktree made with `git worktree add` has no `.venv`. Run tests with `/Users/wedd/dev/TRAINER/.venv/bin/python tasks.py test ...`.
- Parallel tasks never edit the same files. Split file ownership up front.
- Staying current: the `sync-master.sh` hook (`.claude/hooks/sync-master.sh`, wired in `.claude/settings.json`) fetches at session start and at most every two minutes per prompt. It fast-forwards `master` when its checkout is clean, and reports when `origin/master` moved and whether a merge into your branch would be clean.
  - It reports a clean merge and your tree is clean: merge now with `git merge --no-edit origin/master`.
  - It reports conflicts: finish and commit your current edit, then merge and resolve.
  - Always merge the PR's base branch before opening the PR.
  - In app-made worktrees, `mcp__ccd_host__sync_with_base_branch` does the same.
- Pushing: the SSH remote may lack keys in a sandbox, so push with `git -c credential.helper= -c "credential.helper=!gh auth git-credential" push -q https://github.com/KingMoonk/TRAINER.git HEAD:<branch>`.
- Push straight to `master` only when the user asks for it. Otherwise open a PR and let the user merge.
- Never run bare `git stash`: the stack is shared across worktrees. Prefer a WIP commit to set work aside. If you must stash, use `git stash push -u -m <tag>`, take the entry's SHA from `git stash list --format='%H %gs'`, restore with `git stash apply <sha>` (not pop), then drop the entry by its current `stash@{n}`, found by tag.
- After the PR merges: `git worktree remove <path>` and delete the branch.
- Why a hook and not a GitHub webhook: a webhook needs a public endpoint, and this machine is behind CGNAT on Starlink. A throttled fetch in a hook gives the same awareness within a prompt.
