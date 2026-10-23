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
    "Attempts the app moved aside when it restarted a job, runs that failed, and copies of training data no run uses. " +
    "A running job, the checkpoint a waiting job will continue from and every saved version are never listed.",
  none: "Nothing left over.",
  kinds: { interrupted: "Interrupted attempt", failed: "Failed run", snapshot: "Unused training data" },
  shared: "{size} is shared with other runs and stays",
  select: "Select {name}",
  remove: "Remove selected…",
  removing: "Removing",
  confirm: "Remove {n} items and free {size}? This cannot be undone.",
  confirmOne: "Remove 1 item and free {size}? This cannot be undone.",
  removed: "Freed {size}",
  loadFailed: "Could not read what training left behind.",
};

export const EXPORT = {
  kicker: "EXPORT",
  desc:
    "Write a trained version to a file other tools can open. Hugging Face: a folder with the weights (safetensors), config, word list and a README, downloaded as a zip. " +
    "GGUF: one file for llama.cpp and Ollama. A design that is not GPT-2's architecture exports to Hugging Face as raw tensors that Hugging Face " +
    "cannot load directly, and not to GGUF at all.",
  none: "Nothing to export yet. A version appears here once it has trained.",
  hf: "Hugging Face",
  gguf: "GGUF",
  working: "Exporting",
  done: "Exported {name}. The file is in your downloads.",
  loadFailed: "Could not read the versions.",
};
