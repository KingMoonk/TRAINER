// The Test the cluster card's strings (GET and POST /cluster/selftest, studio_selftest.py).

export const SELFTEST = {
  title: "Test the cluster",
  lead: "Runs a tiny training run, three short rounds on a built-in text, across the machines you tick and this computer, and checks every step on the way. It takes about a minute.",
  noPeers: "Add another machine to the list first: the test needs at least one besides this computer.",
  pick: "Machines to test with",
  start: "Run the test",
  running: "Testing…",
  never: "Not run yet.",
  passed: "Passed: a run across these machines works, step by step.",
  failed: "Failed: the first red step says what to change.",
  inProgress: "Running now.",
  ranAt: "Last run {when}, with {machines}.",
  fix: "Fix",
  seconds: "{s} s",
  couldNotLoad: "Could not read the test: {why}",
  couldNotStart: "Could not start the test: {why}",
  states: { pass: "Passed", fail: "Failed", skipped: "Not reached", running: "Running", pending: "Waiting" },
} as const;
