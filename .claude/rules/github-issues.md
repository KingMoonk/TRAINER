# GitHub issues

- Work comes from GitHub issues on `KingMoonk/TRAINER`. mgwedd and KingMoonk each run their own agent fleet; the issue is the hand-off between them, so it must make sense to an agent with no other context.
- Features and bug fixes are tracked in an issue, whether the main session or a subagent does them: find or open it before starting, claim it, and close it through the PR. Small fixes and tweaks (copy, styling, a quick obvious fix, config or rule changes) don't need one.
- Track every TODO, bug and open question as an issue, not in code comments, docs or chat.
- One issue is one PR-sized piece of work. Larger work is a parent issue whose checklist links its child issues.
- Labels: exactly one type, `enhancement` (feature), `bug` or `discovery` (a question to investigate; done means the answer is written in the issue and follow-ups are filed). Plus one area: `api`, `web`, `cluster`, `viewer`, `training` or `ci`.
- Claim before work: an agent starts on an issue only once it is assigned to the authenticated gh user (`gh api user --jq .login`). Assign it first with `gh issue edit N --repo KingMoonk/TRAINER --add-assignee @me`, then re-read the issue to confirm no one else is on it. If it is assigned to someone else, do not start: ask them. This keeps the two fleets from colliding.
- Title: `scope: imperative summary`, for example `web: Chat page with streamed replies`.
- Body, short and concrete, no filler:

  ```
  Part of #N. Needs #M.            (only the lines that apply)

  ## What
  - the change, naming the files, routes or functions involved

  ## Acceptance criteria
  - [ ] an outcome a reviewer can check by a test or a stated manual step
  ```

- The PR body says `Closes #N`; the `close-issues` workflow closes the issue when the PR merges into `wedd-compute-cluster` (GitHub itself only does that for the default branch). Tick acceptance criteria as they are met. A finding that changes the scope goes in an issue comment, or a new issue.
- Create issues with `gh issue create --repo KingMoonk/TRAINER --title ... --label ... --assignee ... --body-file ...`.
