// Builders for the Training tab's tests: a member, a finished round, a run (with its live fields at rest), a timeline round and an event.
import type { Lane, RoundTimeline, RunEvent } from "@/app/runs/view/live";
import type { Member, Round, Run } from "./logic";

export const member = (name: string, over: Partial<Member> = {}): Member => ({
  node: name, address: name, local: false, device: "cuda", device_name: "RTX 4090", state: "training", note: "", reason: "", blocking_job: null, missed: 0, sent: false, rounds_done: 0, steps: 0,
  tokens_per_second: null, last_delta_bytes: null, last_delta_dtype: "", can_stop: true,
  phase: null, step_in_round: 0, round_steps: 100, loss: null, live_tokens_per_second: null, reported_ago: null, shards: 0, samples: [], ...over,
});

export const round = (n: number, over: Partial<Round> = {}): Round => ({
  round: n, step: (n + 1) * 100, loss: 4 - n, val_loss: 4.5 - n, bytes_up: 2 ** 20, bytes_down: 2 * 2 ** 20, seconds: 2, members: 2, contributed: 2, dropped: [], ...over,
});

export const lane = (node: string, over: Partial<Lane> = {}): Lane => ({
  node, theta_taken: 0.2, first_step: 0.6, delta_sent: 3, delta_received: 3.4, counted: true, steps: 100, tokens: 1000, delta_bytes: 1_000_000, ...over,
});

export const line = (n: number, over: Partial<RoundTimeline> = {}): RoundTimeline => ({
  round: n, open: false, deadline: 5, window: 5, closed: 6, lanes: [lane("mac-mini"), lane("gpu-box")], ...over,
});

export const event = (seq: number, over: Partial<RunEvent> = {}): RunEvent => ({ seq, t: seq, kind: "joined", node: "gpu-box", round: 0, text: "gpu-box joined.", ...over });

export const run = (over: Partial<Run> = {}): Run => ({
  id: "r1", label: "Version 2 across machines", version: 2, saved_version: null, state: "running", note: "", started_at: 100, owner: "mac-mini", round: 2, rounds: 10, inner_steps: 100,
  outer_lr: 0.7, outer_momentum: 0.9, delta_dtype: "float16", term: 1, coordinator: "", handoff: "", origin: "", age_s: 0, origin_down: false, can_stop: true, history: [round(0), round(1)],
  phase: "training", deadline_s: null, grace_s: 2, window_s: null, contributed: 0, expected: 2, theta_bytes: 0, now_s: 20, timeline: [], events: [],
  members: [
    member("mac-mini", { address: "local", local: true, device: "mps", device_name: "Apple M2", rounds_done: 2, steps: 200, tokens_per_second: 2048, last_delta_bytes: 4_000_000, last_delta_dtype: "float16", can_stop: false }),
    member("gpu-box", { rounds_done: 2, steps: 200, tokens_per_second: 5120.4, last_delta_bytes: 4_000_000, last_delta_dtype: "float16" }),
  ],
  ...over,
});
