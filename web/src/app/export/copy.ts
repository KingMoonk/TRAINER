// The strings the Export page shows.
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
