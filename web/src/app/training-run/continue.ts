// "Continue this run": the choices, the strings and the request body POST /versions/{n}/continue takes.

export type ContinueKind = "same" | "new_data" | "grow";
export type Continued = { job_id: string; label: string; version: number; kind: ContinueKind; from_step: number };

export const CONTINUE = {
  title: "Continue this run",
  lead: "It stopped at step {step}. Go on from there instead of starting over: nothing it learned is lost.",
  kinds: {
    same: { title: "More steps on the same data", note: "Takes up exactly where it stopped, with the optimizer, sampler and random state it had." },
    new_data: { title: "Continue on other text", note: "The same weights, optimizer and step, read on the text you tick below." },
    grow: { title: "Grow into a bigger design", note: "A wider, deeper design that starts from these weights. It is saved as the next version." },
  },
  stepsAdd: "Steps to add",
  stepsGrow: "Steps to train the bigger design",
  width: "Width",
  layers: "Layers",
  queue: "Add to the queue",
  working: "Adding…",
  queued: "Queued: {label}. It runs when the queue gets to it.",
} as const;

const whole = (text: string) => (/^\d+$/.test(text.trim()) ? Number(text.trim()) : null);

/** The request for these choices, or null while something is missing (steps, the text for other data, the size of the new design). */
export function continueBody(c: { kind: ContinueKind; steps: string; ids: string[]; width: string; layers: string }) {
  const steps = whole(c.steps);
  if (!steps || steps < 1) return null;
  const plain = { kind: c.kind, steps, dataset_ids: c.kind === "same" ? [] : c.ids, rows: 16, hold: 0.05 };
  if (c.kind === "same") return plain;
  if (c.kind === "new_data") return c.ids.length ? plain : null;
  const [width, layers] = [whole(c.width), whole(c.layers)];
  return width && layers ? { ...plain, width, layers } : null;
}
