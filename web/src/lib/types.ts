// The wire types the web UI reads. They live in one place so Phase 1 (docs/rewrite.md) can replace them with the types
// generated from the API's OpenAPI schema without touching a component: components import from here, never from a page.

/** Whether the event stream is up. `connecting` is the first attempt, `offline` is any error after that. */
export type Connection = "connecting" | "online" | "offline";

export type QueueState = "idle" | "running" | "paused";

/** What the status strip shows. The `status` event replaces it whole. */
export type StudioStatus = {
  /** Current saved version and its state; null before the first design is saved. */
  version: string | null;
  queue: { state: QueueState; running: number; waiting: number };
  resources: { device: string | null; diskFreeGb: number | null };
};

/** studio_ia/services.py problem_report(): the result of generating or importing rule-made problems. */
export type ProblemReport = {
  items: unknown[];
  added: number;
  duplicates: number;
  checked: number;
  passed: number;
};

/** The events the shell reacts to. Other kinds from `subscribe()` are for the pages that own them. */
export type StudioEvent =
  | { kind: "status"; payload: StudioStatus }
  | { kind: "problem_report"; payload: ProblemReport };

/** Before the first `status` event: nothing is trained, queued or measured. */
export const PLACEHOLDER_STATUS: StudioStatus = {
  version: null,
  queue: { state: "idle", running: 0, waiting: 0 },
  resources: { device: null, diskFreeGb: null },
};
