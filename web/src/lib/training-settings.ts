// The training settings on the model setup page: how each is shown, which values it takes and the sentence for a bad one.
// The ranges are the ones studio_spec (RUN_INTS, dropout) and the optimizer and schedule modules enforce; the server checks
// them again when the draft is written (design.write_spec), so a value that gets past here still gets a plain 422.

import type { Choices } from "@/lib/builder";

export type TrainingKey = "batch_size" | "accumulation" | "learning_rate" | "weight_decay" | "warmup" | "dropout" | "eval_every" | "seed";

export type TrainingField = {
  key: TrainingKey;
  label: string;
  help: string;
  min: number;
  max: number;
  whole: boolean;
};

export const TRAINING_FIELDS: TrainingField[] = [
  { key: "batch_size", label: "Batch size", help: "Rows read per step. More rows: steadier learning, more memory.", min: 1, max: 128, whole: true },
  { key: "accumulation", label: "Gradient accumulation", help: "Steps read before each update, which acts like a bigger batch without the memory. 1 means off.", min: 1, max: 64, whole: true },
  { key: "learning_rate", label: "Learning rate", help: "How far each update moves the weights. The schedule scales it.", min: 1e-7, max: 0.1, whole: false },
  { key: "weight_decay", label: "Weight decay", help: "Shrinks the weights a little every step.", min: 0, max: 1, whole: false },
  { key: "warmup", label: "Warm-up steps", help: "Steps over which the learning rate climbs from zero to its peak.", min: 0, max: 100000, whole: true },
  { key: "dropout", label: "Dropout", help: "The share of values switched off while training, against memorising. 0 is off.", min: 0, max: 0.5, whole: false },
  { key: "eval_every", label: "Evaluate every (steps)", help: "How often to test on the text held back.", min: 1, max: 100000, whole: true },
  { key: "seed", label: "Seed", help: "Starts the random numbers: the same seed repeats a run.", min: 0, max: 2147483647, whole: true },
];

/** A plain number for a field, without float noise. */
export function showNumber(value: number): string {
  return value.toLocaleString("en-US", { maximumFractionDigits: 12, useGrouping: false });
}

/** The number the text means for `field`, or the sentence saying why it is not one. */
export function checkSetting(field: TrainingField, text: string): { value: number } | { error: string } {
  const trimmed = text.trim();
  const value = Number(trimmed);
  if (trimmed === "" || !Number.isFinite(value)) return { error: `${field.label} must be a number.` };
  if (field.whole && !Number.isInteger(value)) return { error: `${field.label} must be a whole number.` };
  if (value < field.min || value > field.max) return { error: `${field.label} must be between ${showNumber(field.min)} and ${showNumber(field.max)}.` };
  return { value };
}

/** The value of a field in the choices; null when the design has none (a schedule without warm-up). */
export function settingValue(choices: Choices, key: TrainingKey): number | null {
  const value = choices[key];
  return typeof value === "number" ? value : null;
}
