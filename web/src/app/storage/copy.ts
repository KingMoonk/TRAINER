// The strings the Storage page shows, moved from the STORAGE and EXPORT tables in studio_ia/copy.py (which stay while the Tk page uses them).
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
