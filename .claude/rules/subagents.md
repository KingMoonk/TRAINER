# Delegating to subagents

- Default: delegate non-trivial, well-defined work to a subagent. The main session scopes the task, writes the brief, reviews the result and owns the commit, push and PR.
- Well defined means you can write down the files involved and what done looks like. If you cannot, settle that first (read, ask the user), then delegate.
- Do it yourself when:
  - the work is trivial: one edit, a question, a few tool calls;
  - writing the brief would take longer than doing the work, because the context is already loaded here;
  - it needs a decision only the user can make.
- Pick the agent by the kind of work:
  - `implementer` (Sonnet) for moderate or difficult work: features, bug fixes that need diagnosis, refactors, anything with design choices.
  - `grunt` (Haiku) for repetitive, mechanical or read-heavy work: sweeping searches, collecting facts across many files, migrating N similar call sites, renames, running tests and summarising failures, reading logs.
- Never Opus for a subagent unless the user asks for it. When spawning a general-purpose agent instead of these two, still pass `model: "sonnet"` or `model: "haiku"`.
- The brief names the goal, the worktree path, the files to read and change, the acceptance criteria, the test command, and what to report back. A subagent has none of this conversation.
- Fan out in the same turn when pieces are independent. Two subagents that write code never touch the same files.
- A subagent's report is a claim. Check its diff and run the affected tests before telling the user it is done.
