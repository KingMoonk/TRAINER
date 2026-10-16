# GitHub issues

- Work comes from GitHub issues on `KingMoonk/TRAINER`. mgwedd and KingMoonk each run their own agent fleet; the issue is the hand-off between them, so it must make sense to an agent with no other context.
- Find or open the issue before starting work. Track every TODO, bug and open question as an issue, not in code comments, docs or chat.
- One issue is one PR-sized piece of work. Larger work is a parent issue whose checklist links its child issues.
- Labels: exactly one type, `enhancement` (feature), `bug` or `discovery` (a question to investigate; done means the answer is written in the issue and follow-ups are filed). Plus one area: `api`, `web`, `cluster`, `viewer`, `training` or `ci`.
- The assignee owns the issue. Assign yourself before starting an unassigned one. Do not work on an issue assigned to the other person without asking them.
- Title: `scope: imperative summary`, for example `web: Chat page with streamed replies`.
- Body, short and concrete, no filler:

  ```
  Part of #N. Needs #M.            (only the lines that apply)

  ## What
  - the change, naming the files, routes or functions involved

  ## Acceptance criteria
  - [ ] an outcome a reviewer can check by a test or a stated manual step
  ```

- The PR body says `Closes #N`. Tick acceptance criteria as they are met. A finding that changes the scope goes in an issue comment, or a new issue.
- Create issues with `gh issue create --repo KingMoonk/TRAINER --title ... --label ... --assignee ... --body-file ...`.
