// Builders for the Auto Run tests: a plan, a run in a given state, and its summary row.
import type { PhaseState, Plan, RunSummary, RunView } from "./logic";

const KINDS = ["data", "pretrain", "midtrain", "sft", "rlvr", "distill", "dpo", "rlhf"];
const OPT_IN = ["midtrain", "rlhf"];

export const plan = (over: Partial<Plan> = {}): Plan => ({
  id: "",
  name: "Auto Run",
  version: 1,
  budget: { hours: 12 },
  phases: KINDS.map((kind) => ({ kind, enabled: !OPT_IN.includes(kind), settings: {}, on_pass: "continue", on_fail: "pause" })),
  controller: {
    enabled: true,
    bounds: { learning_rate: { min: 1e-6, max: 1e-3 }, steps: { min: 10, max: 20000 }, beta: { min: 0.02, max: 0.5 }, kl: { min: 0, max: 0.2 } },
  },
  ...over,
});

/** A run whose phases have the given statuses (by kind); every other enabled phase is pending. */
export function run(
  status: RunView["state"]["status"],
  current: string | null,
  phases: Record<string, Partial<PhaseState>> = {},
  over: Partial<RunView> & { reason?: string } = {},
): RunView {
  const { reason = "", ...rest } = over;
  const base = plan({ id: "run1" });
  return {
    id: "run1",
    plan: base,
    state: {
      id: "run1",
      status,
      current,
      reason,
      choices: [],
      attempts: [],
      phases: KINDS.map((kind) => ({ kind, status: OPT_IN.includes(kind) ? "skipped" : "pending", attempt: 1, error: "", ...phases[kind] }) as PhaseState),
    },
    journal: [],
    progress: null,
    ...rest,
  };
}

export const summary = (view: RunView): RunSummary => ({
  id: view.id,
  name: view.plan.name,
  version: view.plan.version,
  status: view.state.status,
  current: view.state.current,
  reason: view.state.reason,
  phases: view.state.phases,
});
