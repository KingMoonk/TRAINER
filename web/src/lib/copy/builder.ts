// The Builder's strings, moved here from studio_ia/copy.py BUILDER (the Python copies stay until Phase 5 removes the Tk
// page). Only the design half (issue #148) is here; the learning loop's strings (#83) come with that half.

export const BUILDER = {
  h1: "Build your model",
  lead: "Change the picture. See what it costs and what it does.",
  next_word: "Next word",
  next_word_desc: "Scores every piece in the word list, then picks one.",
  blocks: "Blocks",
  blocks_desc: "Blocks: rounds of processing per word.",
  your_text_title: "Your text, cut into pieces",
  look_back_heads: "Look-back heads",
  look_back_heads_tip:
    "How many things each block looks for at once. Each look-back head looks for something different in earlier words (attention).",
  thinking_room: "Thinking room",
  thinking_room_tip: "Space to work on each word. The thinking room works on each word by itself (MLP).",
  what_costs: "What your changes cost",
  training_memory: "Training memory:",
  training_memory_fits: "unknown (memory size not reported)",
  training_memory_desc:
    "on this computer at 16 rows per step. The bars compare weights, gradients and optimizer state (16 bytes per parameter at 32-bit); the engine adds activations, which grow with rows per step.",
  what_expect: "What to expect",
  no_changes: "No changes",
  no_changes_text: "Version {version} is the current design. Chat answers with the best checkpoint on held-back text.",
  not_saved: "Not saved · only this page sees these",
  not_saved_text: "Nothing else in the app sees these yet. Saving makes {version}.",
  first_save: "Not saved yet",
  first_save_text:
    "Save this design as {version} so Pretraining can train it. Change it first if you like; you can always save a new version later.",
  save_as: "Save as {version}",
  discard: "Discard",
  saved: "Saved {version}.",
  cost_bigger: "bigger",
  cost_smaller: "smaller",
  cost_slower: "slower",
  cost_faster: "faster",
  cost_more: "more",
  cost_less: "less",
  about_same: "about the same",
  about_multiplier: "about {mult}×",
  size_label: "Size",
  time_per_step: "Time per training step",
  memory_while_training: "Memory while training",
  carry_over: "Can the old weights carry over?",
  carry_over_none: "nothing to carry over: no change yet",
  part_embedding: "Word table",
  part_position: "Position",
  part_norm: "Even out",
  part_mixer: "Look back",
  part_channel: "Thinking room",
  part_head: "Word scores",
  final_norm: "Even out",
  final_norm_tip: "One more evening-out after the last block. It is the same kind the blocks use.",
  same_as_blocks: "same as the blocks",
  add_back: "add back",
  scope_all: "All blocks",
  scope_one: "Only block {n}",
  own_parts: "own parts",
  only_here: "only this block",
  wont_fit: "doesn't fit",
  no_heads: "The look-back you picked has no heads.",
  no_room: "The thinking room you picked sets its own size.",
  block_parts_tip: "Open the block to see and change its parts.",
  was: "was {name}",
  changed: "changed",
  block_new: "new",
  block_removed: "removed",
  start_from: "Start from",
  start_from_pick: "A ready-made design…",
  start_from_tip: "Load a known model's shape into your draft. Nothing is saved or trained until you save it.",
  start_from_version: "An earlier version…",
  start_from_version_tip: "Load a saved version's design into your draft. The current version stays the comparison.",
  preset_load: "Load into draft",
  import_gpt2: "Start from GPT-2 (124M)",
  import_gpt2_tip:
    "Download OpenAI's small GPT-2 once and start from its trained weights instead of a blank model. It becomes a new version that continues text; it does not answer as a chat until you fine-tune it.",
  import_gpt2_ask:
    "This downloads about 548 MB from Hugging Face once and keeps it. Each time you start from GPT-2 the studio adds about 0.5 GB for the new version, and a full fine-tune saves checkpoints with their training state of about 1.5 GB each. Continue?",
  import_gpt2_started:
    "Downloading GPT-2 in the background. Its version is listed now but gets its weights only when the queue finishes it; if the download fails or you stop it, that version is removed or marked.",
  preset_starter: "This studio's small default: trains in minutes on any computer.",
  preset_tiny_llama_style: "A small model built from Llama's newer parts: RMSNorm, rotary positions, shared look-back heads, SwiGLU.",
  preset_gpt2_small_style: "The shape of OpenAI's smallest GPT-2: 12 blocks, 768 numbers per word, a 1,024-piece window.",
  preset_gpt3_small_style: "GPT-2 small's shape with GPT-3's 2,048-piece window and training recipe.",
  preset_gpt3_medium_style: "GPT-3 Medium: 24 blocks, 1,024 numbers per word, a 2,048-piece window, GPT-3's recipe.",
  preset_recipe: "Recipe: {recipe}",
  compute_recount:
    "A different word list cuts your text into a different number of pieces. Save this design to see how much text it needs.",
  compute_counting: "Counting your text…",
  versions: "Versions",
  versions_empty: "No version saved yet.",
  cannot_write: "This design cannot be written:",
  loading_failed: "The Builder could not load its design maths:",
} as const;

/** `BUILDER.x` with its `{name}` holes filled. */
export function fill(text: string, values: Record<string, string | number>): string {
  return text.replace(/\{(\w+)\}/g, (hole, key: string) => (key in values ? String(values[key]) : hole));
}

/** The blurb under a preset's name; the server names each preset by `key` (studio_ia/design.py PRESETS). */
export function presetAbout(key: string): string {
  return (BUILDER as Record<string, string>)[`preset_${key}`] ?? "";
}
