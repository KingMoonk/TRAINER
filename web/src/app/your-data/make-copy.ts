export const MAKE = {
  title: "Make extra data", lead: "Optional. Generate more, then keep what passes review.",
  count: "How many", rule: "Rule", allRules: "All rules", preview: "Try 5 first",
  previewTitle: "Preview, nothing saved", generate: "Generate problems", drafts: "Make drafts",
  generating: "Making examples…", yourData: "Your data", review: "Review drafts",
  empty: "No drafts waiting for review in this category.", read: "Read draft", keep: "Mark Good", discard: "Discard",
  next: "Load more drafts", retry: "Try again", limits: (max: number) => `Enter a whole number from 1 to ${max.toLocaleString()}.`,
  cards: {
    problems: { title: "Problems a rule can check", note: "Sort, reverse, count, copy, small addition, length limits, ending cleanly.",
      safety: "Safe: the rule decides what is right", where: "Goes to Your data → Problems. Used by Post-training round 2." },
    conversations: { title: "Rewrite with your local model", note: "Rephrases prompts and answers you already have, so round 1 sees more variety. Uses your local model (Ollama).",
      safety: "For review only: can introduce mistakes", where: "Goes to Your data → Conversations as drafts. Nothing is used until you mark it Good." },
    text: { title: "Generate prose with your local model", note: "Short passages written by your local model (Ollama).",
      safety: "For review only: can introduce mistakes", where: "Goes to Your data → Text as drafts. Kept small next to real text." },
  },
} as const;
