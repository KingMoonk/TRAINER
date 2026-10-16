# Delegating to subagents

- Delegate only when it pays: the cost of briefing, waiting and reviewing the PR must be smaller than doing the work here. That means work that is both significant (many files or steps, a feature, a hard bug, a sweep across the codebase, or pieces that can run in parallel) and well defined (you can name the files and what done looks like).
- Otherwise do it on the main thread. That includes:
  - small work: an edit or a few, a question, a handful of tool calls;
  - work whose context is already loaded here, so the brief would restate it;
  - work that is not yet well defined: settle it first (read, ask the user), then decide;
  - anything that needs a decision only the user can make.
- When you do delegate, the main session scopes the task, writes the brief and reviews the PR that comes back.
- Pick the agent by the kind of work:
  - `implementer` (Sonnet) for moderate or difficult work: features, bug fixes that need diagnosis, refactors, anything with design choices.
  - `grunt` (Haiku) for repetitive, mechanical or read-heavy work: sweeping searches, collecting facts across many files, migrating N similar call sites, renames, running tests and summarising failures, reading logs.
- Never Opus for a subagent unless the user asks for it. When spawning a general-purpose agent instead of these two, still pass `model: "sonnet"` or `model: "haiku"`.

## Every task that changes files is an issue, a branch and a PR

1. Before delegating, find or open the GitHub issue (`.claude/rules/github-issues.md`). Its acceptance criteria are the subagent's definition of done.
2. The subagent claims it: `gh issue edit N --repo KingMoonk/TRAINER --add-assignee @me`, then re-reads it to confirm no one else holds it.
3. It works on its own branch in its own worktree, off `origin/wedd-compute-cluster` (`.claude/rules/git-workflow.md`). Two subagents never share a worktree or edit the same files.
4. It commits (Conventional Commits), pushes, and opens a PR with `--base wedd-compute-cluster`.
5. The PR description is precise and short: what changed and why, how it was tested, known limitations, and `Closes #N` for each issue it finishes (`Part of #N` for a partial step). The `close-issues` workflow closes those issues when the PR merges into `wedd-compute-cluster`.

Read-only work (searches, investigations) needs no branch or PR. If it answers a `discovery` issue, the subagent writes the answer in that issue.

## The brief

A subagent has none of this conversation. The brief states:

- the goal and the issue number;
- scope: the files to read and change, and what is out of scope;
- expectations: the acceptance criteria, the test command, and anything it must not do;
- the report you want back.

Ask for a concise report: the PR and issue links, what changed, the tests run and their results, and an honest account of limitations, risks and walls it hit: what it could not do, what it guessed, what it skipped. A report without that section is incomplete; ask for it.

## After it reports

- A report is a claim. Read the PR diff and check its CI before telling the user it is done.
- Fan out in the same turn when pieces are independent.
