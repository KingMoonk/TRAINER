// The strings the Storage page shows.
export const STORAGE = {
  kicker: "STORAGE",
  desc: "History is run folders, checkpoints, saved versions, logs, exports and backups. Settings and chat ratings are never deleted.",
  olderThan: "Older than {days} days",
  all: "All history",
  data: "Also delete my datasets and downloads",
  free: "{size} free on this computer's disk",
  frees: "This frees {size} in {n} items",
  freesOne: "This frees {size} in 1 item",
  nothing: "Nothing to delete",
  freed: "Freed {size}",
  delete: "Delete…",
  deleting: "Deleting",
  confirm: "Delete {n} items and free {size}? This cannot be undone.",
  confirmOne: "Delete 1 item and free {size}? This cannot be undone.",
  confirmYes: "Delete",
  confirmNo: "Cancel",
  choices: "What to delete",
  loadFailed: "Could not read what is on disk.",
};

export const LEFTOVERS = {
  kicker: "LEFTOVERS",
  desc:
    "Attempts the app moved aside when it restarted a job, runs that failed, copies of training data no run uses, and temporary files a crash left. " +
    "A running job, the checkpoint a waiting job will continue from and every saved version are never listed.",
  none: "Nothing left over.",
  kinds: { interrupted: "Interrupted attempt", failed: "Failed run", snapshot: "Unused training data", temporary: "Leftover temporary file" },
  shared: "{size} is shared with other runs and stays",
  select: "Select {name}",
  remove: "Remove selected…",
  removing: "Removing",
  confirm: "Remove {n} items and free {size}? This cannot be undone.",
  confirmOne: "Remove 1 item and free {size}? This cannot be undone.",
  removed: "Freed {size}",
  loadFailed: "Could not read what training left behind.",
};
