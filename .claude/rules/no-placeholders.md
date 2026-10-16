# No placeholders, fake features or dead ends

- Ship only what works. Nothing is better than a misleading placeholder.
- Never add a button, menu item, route, setting or API endpoint that does nothing, says "coming soon", or pretends to work.
- Never add a stub that returns canned or invented data where real data belongs. `FakeServices` and test fixtures are the only exception, and they stay out of the real app.
- Never leave a code path that ends in `pass`, `TODO`, `NotImplementedError` or a silent no-op where the user expects an effect.
- Never show made-up numbers, example charts or sample text as if they were the user's results.
- If part of a task must wait, leave it out entirely and open a GitHub issue for it: `gh issue create --repo KingMoonk/TRAINER --title "..." --body "..."`. Say what is missing, why, and what done looks like.
- If you hit a bug you cannot fix within the task, open a GitHub issue with the evidence (error text, failing test, steps) instead of working around it quietly.
- Link the issues in the PR body and in your report, so the gap is visible rather than hidden in the code.
