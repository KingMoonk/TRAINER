// Builders for the join flow's tests: an invite (a run this machine was added to) and a pause request.
import type { Invite, PauseRequest } from "@/lib/join";

export const invite = (over: Partial<Invite> = {}): Invite => ({
  job_id: "j1", run_id: "r7", run_label: "Version 7", owner: "Michael's Mac", node: "BEAST", state: "invited", reason: "", blocking_job: null,
  round: null, rounds: null, tokens_per_second: null, can_pause_and_join: false, ...over,
});

export const pauseRequest = (over: Partial<PauseRequest> = {}): PauseRequest => ({
  id: "q1", at: 1, from_node: "Michael's Mac", run_label: "Version 7", blocking_job: { label: "Version 1", eta_s: 600 }, ...over,
});
