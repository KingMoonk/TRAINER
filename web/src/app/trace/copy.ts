// The Trace page's strings, moved from studio_ia/copy.py (TRACE). The Python copy stays while the Tk page uses it.

export const TRACE = {
  example_prompt: "EXAMPLE PROMPT · FOLLOWING THE LAST POSITION",
  example_text: "The largest planet in our solar system",
  last_position: "is",
  stage: "Stage",
  size_norm: "Size ‖x‖",
  relative_change: "Relative change ‖Δx‖ / ‖x‖",
  direction_kept: "Direction kept · cosine with before",
  top_guess: "Top guess if read here",
  embedding: "Embedding",
  embedding_what: "token vector (positions are applied inside attention)",
  mixer_what: "mixer adds",
  channel_what: "MLP adds",
  output: "Output",
  output_what: "final norm, then unembedding to logits",
  approx_legend:
    "≈ marks a guess read before the last layer: approximate, because those layers are not trained to be read this way. Only the output row is the model's real prediction.",
  relative_note:
    "Relative change says how much a stage added compared with what was already there; the cosine says whether it kept the direction (near 1) or turned it. Size alone grows steadily through most models and says little. The guess column is the logit lens: the final norm and unembedding applied early. Layers before the last are not trained to be read this way, so early guesses can look odd.",
  // Page chrome with no Tk text of its own: the Tk page follows one fixed prompt, the web page also lets you type another.
  prompt_label: "Prompt",
  run: "Trace",
  running: "Tracing…",
  no_numbers: "Numbers appear when the engine has run your prompt.",
  server_error: "The server did not answer the request",
} as const;
