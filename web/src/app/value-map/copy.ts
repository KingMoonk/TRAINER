// The Value map page's strings, moved from studio_ia/copy.py (VALUE_MAP). The copy is written for d_model 256; `fit` swaps in
// the model's own d_model. The Python copy stays while the Tk page uses it.

export const VALUE_MAP = {
  d_model: "d_model 256 · heads",
  head_dim: "head_dim = 256 ÷ {heads} =",
  one_head: "ONE HEAD: WHAT IT CAN WRITE INTO THE STREAM",
  x_label: "x · 256",
  w_v: "W_V",
  v_label: "v · {hd}",
  w_o: "W_O",
  delta_x: "Δx · 256",
  together: "Together the two matrices form one 256 × 256 map, but its rank is at most",
  each_head: "Each head can only write into a subspace of at most {hd} dimensions: a set of directions, not a block of coordinates.",
  weighting: "Weighting by attention happens on the {hd}-number vectors in the middle, before W_O.",
  params: "Per head W_V and W_O hold {per_head} numbers; all {heads} heads together hold {total} = 2 · {d}², whatever the head count.",
  measured: "Measured on this model, block {block} head {head}: the rank of W_V · W_O is {rank}, at most {hd}. Singular values, largest first:",
  measured_none: "Real singular values appear when a trained checkpoint is available.",
  measured_other: "Pick {heads} heads to see this model's measured singular values; it has {heads} heads per block.",
  // Page chrome with no Tk text of its own.
  no_numbers: "Numbers appear when the engine has run your prompt.",
  server_error: "The server did not answer the request",
  heads_label: "Heads",
  not_equal: "{d} does not split into {heads} equal heads.",
} as const;

/** Fill `{name}` slots. */
export function fill(template: string, values: Record<string, string | number>): string {
  return template.replace(/\{(\w+)\}/g, (whole, name: string) => (name in values ? String(values[name]) : whole));
}
