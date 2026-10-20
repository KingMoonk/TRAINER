# GitHub issues

- Work comes from GitHub issues on `KingMoonk/TRAINER`. mgwedd and KingMoonk each run their own agent fleet; the issue is the hand-off between them, so it must make sense to an agent with no other context.
- Features and bug fixes are tracked in an issue, whether the main session or a subagent does them: find or open it before starting, claim it, and close it through the PR. Small fixes and tweaks (copy, styling, a quick obvious fix, config or rule changes) don't need one.
- Track every TODO, bug and open question as an issue, not in code comments, docs or chat.
- One issue is one PR-sized piece of work. Larger work is a parent issue whose checklist links its child issues.
- Labels: exactly one type, `enhancement` (feature), `bug` or `discovery` (a question to investigate; done means the answer is written in the issue and follow-ups are filed). Plus one area: `api`, `web`, `cluster`, `viewer`, `training` or `ci`.
- Check before claiming: before any work on an issue, including choosing it for a subagent, read its assignees (`gh issue view N --repo KingMoonk/TRAINER --json assignees --jq '[.assignees[].login]'`). If anyone other than the authenticated gh user (`gh api user --jq .login`) is assigned, the issue is taken, even when you are assigned as well: do not start, do not add yourself, and ask them or the user. Only an unassigned issue, or one assigned to you alone, may be worked.
- Claim before work: assign it with `gh issue edit N --repo KingMoonk/TRAINER --add-assignee @me`, then re-read the assignees; if someone else appeared, stop. This keeps the two fleets from colliding.
- Title: `scope: imperative summary`, for example `web: Chat page with streamed replies`.
- Body, short and concrete, no filler:

  ```
  Part of #N. Needs #M.            (only the lines that apply)

  ## What
  - the change, naming the files, routes or functions involved

  ## Acceptance criteria
  - [ ] an outcome a reviewer can check by a test or a stated manual step
  ```

- The PR body says `Closes #N`; GitHub closes the issue when the PR merges into `main`, the default branch. Tick acceptance criteria as they are met. A finding that changes the scope goes in an issue comment, or a new issue.
- Create issues with `gh issue create --repo KingMoonk/TRAINER --title ... --label ... --assignee ... --body-file ...`.
