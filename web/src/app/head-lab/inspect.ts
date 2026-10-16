// What the Head lab and Inside a head share: running an /inspect job to its result, the version they read, and the plain
// maths over an attention pattern (ported from studio_inspect.head_tests and studio_ia/pages/head_zoom.py).

import { api, ApiError, unwrap } from "@/lib/api";
import type { components } from "@/lib/api-schema";

/** The part of a job these pages read: every job-starting route answers with at least this. */
export type Job = { id: string; status: string; result?: unknown; error?: string };
export type Version = components["schemas"]["Version"];

/** The prompt Inside a head reads and the Head lab answers (TRACE example_text + last_position in studio_ia/copy.py). */
export const EXAMPLE_PROMPT = "The largest planet in our solar system is";

const POLL_MS = 400;

const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/** A sentence for an error from the API or a job: the job's own message ("ValueError: ...") loses its class name. */
export function sentence(error: unknown): string {
  if (error instanceof ApiError) {
    try {
      const body = JSON.parse(error.body) as { detail?: unknown };
      if (typeof body.detail === "string") return body.detail;
    } catch {
      // not JSON: fall through to the status line
    }
    return error.status === 404 ? "The server does not have this yet." : "The server could not do that.";
  }
  const message = error instanceof Error ? error.message : String(error);
  return message.replace(/^[A-Za-z]*Error: /, "");
}

/**
 * Wait for a job that was just submitted and return its result. Polls GET /jobs/{id}: these pages need one result each, so the
 * shared event stream is not worth a subscription. Rejects with the job's error when it fails or is cancelled.
 */
export async function runJob<T>(submit: Promise<Job>, isStale: () => boolean = () => false): Promise<T> {
  let job = (await submit) as Job;
  while (job.status === "queued" || job.status === "running") {
    await wait(POLL_MS);
    if (isStale()) throw new Stale();
    job = (await unwrap(api.GET("/jobs/{job_id}", { params: { path: { job_id: job.id } } }))) as Job;
  }
  if (job.status !== "done") throw new Error(job.error || `The job ${job.status}.`);
  return job.result as T;
}

/** Thrown when the page moved on (another selection, another mode) before a job finished; its result is dropped. */
export class Stale extends Error {
  constructor() {
    super("stale");
  }
}

/**
 * The version the look-inside pages read: the one Chat follows, else the newest (the Tk pages' current()). Null when there is
 * none, which the pages say in a sentence.
 */
export async function readVersion(isStale?: () => boolean): Promise<Version | null> {
  const best = await runJob<Version | null>(unwrap(api.POST("/versions/best")), isStale);
  return best ?? (await unwrap(api.GET("/versions/current")));
}

export type Forward = {
  ids: number[][];
  tokens: string[][];
  logits: number[][][];
  hooks: Record<string, number[][][][] | number[][][]>;
};

/** Pieces made visible: a space shows as a middle dot and a new line as a return mark. */
export function showToken(token: string): string {
  return token.replace(/ /g, "·").replace(/\n/g, "⏎") || "·";
}

/** Fill {name} in a copy string. */
export function fill(template: string, values: Record<string, string | number>): string {
  return template.replace(/\{(\w+)\}/g, (_, key: string) => String(values[key] ?? `{${key}}`));
}

/** An attention pattern [batch][head][query][key]. */
export type Pattern = number[][][][];

/** {block: head count} for the blocks of a forward result that have an attention pattern, in block order. */
export function attentionBlocks(hooks: Forward["hooks"]): Record<number, number> {
  const found: [number, number][] = [];
  for (const [name, value] of Object.entries(hooks)) {
    const match = /^block(\d+)\.attn\.pattern$/.exec(name);
    if (match) found.push([Number(match[1]), (value as Pattern)[0].length]);
  }
  return Object.fromEntries(found.sort((a, b) => a[0] - b[0]));
}

export const TESTS = [
  { key: "previous", name: "previous_word", what: "previous_desc" },
  { key: "first", name: "parks_first", what: "parks_desc" },
  { key: "induction", name: "copying_pattern", what: "copying_desc" },
] as const;
export type TestKey = (typeof TESTS)[number]["key"];

/** A share of attention above this counts as "passes": a description of where a head looks, not a verdict. */
export const THRESHOLD = 0.5;

/**
 * studio_inspect.head_tests for one block: per head, the mean share on the previous piece, on the first piece (an idle-head
 * sign), and, on the repeated half of BOS + A + A, on the piece after the earlier copy of the current piece (induction).
 */
export function headTests(pattern: Pattern, ids: number[]): Record<TestKey, number[]> {
  const heads = pattern[0].length;
  const length = ids.length;
  const previous = new Array<number>(heads).fill(0);
  const first = new Array<number>(heads).fill(0);
  const induction = new Array<number>(heads).fill(0);
  if (length > 1) {
    for (let h = 0; h < heads; h++) {
      const rows = pattern[0][h];
      for (let q = 1; q < length; q++) {
        previous[h] += rows[q][q - 1] / (length - 1);
        first[h] += rows[q][0] / (length - 1);
      }
    }
  }
  let count = 0;
  for (let q = Math.floor((length - 1) / 2) + 1; q < length; q++) {
    const keys: number[] = [];
    for (let k = 0; k < q - 1; k++) if (ids[k] === ids[q]) keys.push(k + 1);
    if (keys.length === 0) continue;
    for (let h = 0; h < heads; h++) for (const k of keys) induction[h] += pattern[0][h][q][k];
    count += 1;
  }
  return { previous, first, induction: induction.map((total) => (count ? total / count : 0)) };
}

const PATTERN_WORDS: Record<TestKey, string> = {
  previous: "looks at the previous word",
  first: "parks on the first word",
  induction: "copying pattern",
};
export const NO_PATTERN = "no clear pattern";

/** Words for where one head looks: the test it scores highest on, or NO_PATTERN when none reaches THRESHOLD. */
export function headPattern(tests: Record<TestKey, number[]>, head: number): string {
  const best = TESTS.reduce((a, b) => (tests[b.key][head] > tests[a.key][head] ? b : a));
  return tests[best.key][head] >= THRESHOLD ? PATTERN_WORDS[best.key] : NO_PATTERN;
}

/** The k heads whose switch-off raises the error most, largest first. */
export function topRelied(rises: Map<string, number>, k = 3): [string, number][] {
  return [...rises.entries()].sort((a, b) => b[1] - a[1]).slice(0, k);
}

export const headKey = (block: number, head: number) => `${block}.${head}`;
export const parseKey = (key: string): [number, number] => {
  const [block, head] = key.split(".").map(Number);
  return [block, head];
};

/** The steps of Inside a head, from one head's hooks, in the reference orientation [key i, query j]. */
export type Derived = {
  tokens: string[];
  raw: number[][];
  scaled: number[][];
  weights: number[][];
  error: number;
  dHead: number;
  plainScale: boolean;
};

const transpose = (m: number[][]) => m[0].map((_, j) => m.map((row) => row[j]));

/**
 * studio_ia/pages/head_zoom.derive: raw = attn.scores, scaled = attn.logits (post-scale and post-bias, so muP and ALiBi are
 * in), weights = softmax down each column with the causal mask. `error` = max |derived softmax - attn.pattern|.
 */
export function derive(forward: Forward, block: number, head: number): Derived {
  const part = (name: string) => (forward.hooks[`block${block}.attn.${name}`] as number[][][][])[0];
  const raw = transpose(part("scores")[head]);
  const scaled = transpose(part("logits")[head]);
  const pattern = transpose(part("pattern")[head]);
  const dHead = part("q")[0][0].length;
  const T = raw.length;
  const weights = scaled.map((row) => row.map(() => 0));
  let error = 0;
  for (let j = 0; j < T; j++) {
    let max = -Infinity;
    for (let i = 0; i <= j; i++) max = Math.max(max, scaled[i][j]);
    let sum = 0;
    for (let i = 0; i <= j; i++) sum += Math.exp(scaled[i][j] - max);
    for (let i = 0; i < T; i++) {
      weights[i][j] = i > j ? 0 : Math.exp(scaled[i][j] - max) / sum;
      error = Math.max(error, Math.abs(weights[i][j] - pattern[i][j]));
    }
  }
  const plainScale = raw.every((row, i) => row.every((value, j) => Math.abs(scaled[i][j] - value / Math.sqrt(dHead)) <= 1e-6 + 1e-5 * Math.abs(scaled[i][j])));
  return { tokens: forward.tokens[0], raw, scaled, weights, error, dHead, plainScale };
}

/** Derived numbers disagree with the model's own pattern by more than this: the picture is not to be trusted. */
export const TOLERANCE = 1e-4;
