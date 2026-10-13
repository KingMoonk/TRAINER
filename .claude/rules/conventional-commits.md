# Conventional Commits

- Commit messages and PR titles are Conventional Commits: `type(scope)!: subject`, with type one of feat, fix, perf, revert, docs, style, refactor, test, build, ci, chore. The scope is optional.
- PRs are squash-merged, so the PR title becomes the commit on master. `.github/workflows/commitlint.yml` rejects a PR whose title does not match.
- The type picks the release (`.github/workflows/auto-release.yml`): feat = minor; fix, perf, revert, refactor, style = patch; docs, chore, test, ci, build = no release; `!` or `BREAKING CHANGE` = major.
- Never mark a change breaking (`!` or `BREAKING CHANGE`) on your own judgement. This is an experimental app: use it only when the user asks for a major release, however large the change is.
