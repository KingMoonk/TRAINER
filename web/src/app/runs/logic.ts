// The Runs page's rules, apart from the page so a test can check them without a browser.
// One row per run, merged on the client by job id from three reads: the local queue (GET /queue), the jobs placed on or run for
// peers (GET /cluster/jobs) and the DiLoCo runs this computer coordinates (GET /distributed/runs). A DiLoCo run's owner job and
// its Run record share an id, so they are one row.
import type { components } from "@/lib/api-schema";
import { nodeName } from "@/lib/node-name";
import type { ClusterNode, Run } from "@/app/cluster/training/logic";
import type { Job, Queue } from "@/app/queue/logic";
import { T, fill } from "./copy";

export type ClusterJob = components["schemas"]["ClusterJob"];
export type Section = "active" | "scheduled" | "completed";

export type RunRow = {
  id: string;
  section: Section;
  name: string;
  /** "pretrain", "pretrain · DiLoCo", ... */
  kindText: string;
  distributed: boolean;
  version: number | null;
  /** Machine names. For a job that is not a DiLoCo run, the one machine it runs on (or "any idle node"). */
  workers: string[];
  coordinator: string;
  /** DiLoCo: round done, rounds in all, steps per round (0 when not a DiLoCo run). */
  round: number;
  rounds: number;
  innerSteps: number;
  /** Any other job: step and total, 0 when not known. */
  step: number;
  total: number;
  trainLoss: number | null;
  valLoss: number | null;
  etaS: number | null;
  /** Position among the local queue's waiting jobs, or -1 when the job is not in it. */
  queueIndex: number;
  held: boolean;
  /** Why it waits, or why it ended. */
  note: string;
  /** The machine that queued a job this computer runs for it, else "". */
  forOwner: string;
  /** Why the machine it runs on was picked (it measured the fastest step for this design), else "": the rule picked it. */
  basis: string;
  state: string;
  /** A failed job still in the queue's failed list: it can be restarted. */
  restartable: boolean;
  /** A running DiLoCo run another computer coordinates and this one is not in: it may ask to join from the next round. */
  canJoin: boolean;
};

export type Inputs = { queue: Queue | undefined; runs: Run[] | undefined; clusterJobs: ClusterJob[] | undefined; nodes: ClusterNode[] | undefined };

const LIVE_RUN = ["starting", "waiting", "running"];
const DISTRIBUTED = "pretrain_distributed";
const MEMBER = "diloco_member";

/** This computer's name for a page: the local node's hostname, else a plain fallback. */
export function thisComputer(nodes: ClusterNode[] | undefined): string {
  const local = (nodes ?? []).find((n) => n.local);
  return local ? nodeName(local, nodes ?? []) : "This computer";
}

/** The name a machine goes by on this page, from whatever a read called it (an address, "local", a hostname). */
export function machineName(raw: string, nodes: ClusterNode[] | undefined): string {
  const list = nodes ?? [];
  if (raw === "local") return thisComputer(nodes);
  const lower = raw.trim().toLowerCase();
  const found = list.find((n) => n.address.toLowerCase() === lower || (n.hostname !== "" && n.hostname.toLowerCase() === lower));
  return found ? nodeName(found, list) : raw;
}

/** "distributed" kinds read as "pretrain · DiLoCo"; the rest are their kind with spaces. */
export function kindText(kind: string): string {
  return kind === DISTRIBUTED ? "pretrain · DiLoCo" : kind.replace(/_/g, " ");
}

function versionOf(job: Job | undefined): number | null {
  return typeof job?.params?.version === "number" ? job.params.version : null;
}

/** The DiLoCo run a `diloco_member` job joins (the invite posts it as params.run), else "". */
function memberRunOf(job: Job): string {
  return job.kind === MEMBER && typeof job.params?.run === "string" ? job.params.run : "";
}

function pinnedOf(job: Job | undefined): string {
  return typeof job?.params?.pinned === "string" ? job.params.pinned : "";
}

function machinesOf(job: Job | undefined): string[] {
  const list = job?.params?.machines;
  if (!Array.isArray(list)) return [];
  return list.map((m) => (typeof m === "object" && m !== null ? String((m as { node?: unknown }).node ?? "") : "")).filter(Boolean);
}

function last<T>(items: T[], pick: (item: T) => number | null): number | null {
  for (let i = items.length - 1; i >= 0; i -= 1) {
    const value = pick(items[i]);
    if (value != null) return value;
  }
  return null;
}

/** Seconds left in a run, from the pace of its last three rounds; null before any round has finished. */
export function runEta(run: Run): number | null {
  const recent = run.history.slice(-3).filter((r) => r.seconds > 0);
  if (recent.length === 0 || run.rounds <= run.round) return null;
  return (recent.reduce((sum, r) => sum + r.seconds, 0) / recent.length) * (run.rounds - run.round);
}

function runSection(run: Run): Section {
  if (run.state === "queued") return "scheduled";
  return LIVE_RUN.includes(run.state) ? "active" : "completed";
}

function jobSection(status: string): Section {
  if (status === "running") return "active";
  return status === "waiting" || status === "held" ? "scheduled" : "completed";
}

function blank(id: string): RunRow {
  return {
    id, section: "scheduled", name: "", kindText: "", distributed: false, version: null, workers: [], coordinator: "", round: 0, rounds: 0,
    innerSteps: 0, step: 0, total: 0, trainLoss: null, valLoss: null, etaS: null, queueIndex: -1, held: false, note: "", forOwner: "", basis: "", state: "",
    restartable: false, canJoin: false,
  };
}

/** Every run the page lists, merged by id. Active first as they came, scheduled in queue order, completed newest first. */
export function buildRows({ queue, runs, clusterJobs, nodes }: Inputs): RunRow[] {
  const here = thisComputer(nodes);
  const name = (raw: string) => machineName(raw, nodes);
  const rows = new Map<string, RunRow>();
  const queued = queue?.up_next ?? [];
  const failed = queue?.failed ?? [];
  // Failed jobs are listed too, under Completed with their reason, so they can be restarted from here.
  const jobs = new Map<string, Job>([...(queue?.job ? [queue.job] : []), ...queued, ...failed].map((j) => [j.id, j]));

  for (const run of runs ?? []) {
    const job = jobs.get(run.id);
    const row = blank(run.id);
    // `local` is relative to the coordinator, so members are named by node name, resolved against this machine's survey.
    const members = run.members.map((m) => name(m.node || m.address));
    const planned = machinesOf(job).map(name);
    Object.assign(row, {
      section: runSection(run),
      name: run.label || job?.label || kindText(DISTRIBUTED),
      kindText: kindText(DISTRIBUTED),
      distributed: true,
      version: run.version,
      workers: members.length ? members : planned.length ? planned : [here],
      coordinator: run.owner ? name(run.owner) : here,
      round: run.round,
      rounds: run.rounds,
      innerSteps: run.inner_steps,
      trainLoss: last(run.history, (r) => r.loss),
      valLoss: last(run.history, (r) => r.val_loss),
      etaS: runEta(run) ?? job?.eta_s ?? null,
      queueIndex: queued.findIndex((j) => j.id === run.id),
      held: job?.status === "held",
      note: run.note,
      state: run.state,
      canJoin: run.can_join,
    });
    rows.set(run.id, row);
  }

  const place = (job: Job): string => {
    if (job.node) return name(job.node);
    const pinned = pinnedOf(job);
    if (job.status === "running") return here;
    return pinned ? name(pinned) : T.anyIdleNode;
  };

  for (const job of jobs.values()) {
    if (rows.has(job.id)) continue; // a DiLoCo run already read from its Run record
    if (rows.has(memberRunOf(job))) continue; // this machine's member job for a listed run: that run's row already shows it
    const section = jobSection(job.status);
    const running = job.id === queue?.job?.id;
    const where = place(job);
    rows.set(job.id, {
      ...blank(job.id),
      section,
      name: job.label || kindText(job.kind),
      kindText: kindText(job.kind),
      distributed: job.kind === DISTRIBUTED,
      version: versionOf(job),
      workers: [where],
      coordinator: job.kind === MEMBER ? "" : where, // a member never coordinates
      step: running ? queue?.step ?? 0 : 0,
      total: running ? queue?.total ?? 0 : 0,
      etaS: (running ? queue?.eta_s : job.eta_s) ?? null,
      queueIndex: queued.findIndex((j) => j.id === job.id),
      held: job.status === "held",
      note: job.error || job.note,
      state: job.status,
    });
  }

  for (const cj of clusterJobs ?? []) {
    const existing = rows.get(cj.id);
    if (cj.role === "owner") {
      // Queued here and running (or waiting) on a peer: the machine it is on replaces "this computer" or "any idle node".
      const where = cj.node ? name(cj.node) : "";
      if (existing && !existing.distributed && where) Object.assign(existing, { workers: [where], coordinator: where });
      if (existing) {
        if (cj.waiting_reason && !existing.note) existing.note = cj.waiting_reason;
        if (cj.basis) existing.basis = cj.basis;
        continue;
      }
      rows.set(cj.id, {
        ...blank(cj.id), section: jobSection(cj.status), name: cj.label || kindText(cj.kind), kindText: kindText(cj.kind),
        distributed: cj.kind === DISTRIBUTED, workers: [where || T.anyIdleNode], coordinator: where || T.anyIdleNode, note: cj.waiting_reason, basis: cj.basis ?? "", state: cj.status,
      });
    } else if (!existing) {
      // Queued by another machine and running here.
      rows.set(cj.id, {
        ...blank(cj.id), section: jobSection(cj.status), name: cj.label || kindText(cj.kind), kindText: kindText(cj.kind),
        distributed: cj.kind === DISTRIBUTED, workers: [here], coordinator: here, forOwner: cj.node ? name(cj.node) : "", state: cj.status,
      });
    }
  }

  const all = [...rows.values()];
  const failedIds = new Set(failed.map((j) => j.id));
  for (const row of all) row.restartable = failedIds.has(row.id);
  const byIndex = (a: RunRow, b: RunRow) => (a.queueIndex < 0 ? Infinity : a.queueIndex) - (b.queueIndex < 0 ? Infinity : b.queueIndex);
  const startedAt = new Map((runs ?? []).map((r) => [r.id, r.started_at]));
  return [
    ...all.filter((r) => r.section === "active"),
    ...all.filter((r) => r.section === "scheduled").sort(byIndex),
    ...all.filter((r) => r.section === "completed").sort((a, b) => (startedAt.get(b.id) ?? 0) - (startedAt.get(a.id) ?? 0)),
  ];
}

export function inSection(rows: RunRow[], section: Section): RunRow[] {
  return rows.filter((r) => r.section === section);
}

/** The roles a machine has in the active runs: "c, w v5" style lines, one per run it takes part in. */
export function rolesOn(machine: string, active: RunRow[]): { roles: string; run: string; runName: string }[] {
  const out: { roles: string; run: string; runName: string }[] = [];
  for (const row of active) {
    const roles = [row.coordinator === machine ? T.coordinatorRole : "", row.workers.includes(machine) ? T.workerRole : ""].filter(Boolean);
    if (roles.length) out.push({ roles: roles.join(", "), run: row.version != null ? `v${row.version}` : row.name, runName: row.name });
  }
  return out;
}

/** "5h 10m", "42 min", "under a minute". */
export function etaText(seconds: number): string {
  const minutes = Math.round(seconds / 60);
  if (minutes < 1) return "under a minute";
  if (minutes < 60) return `${minutes} min`;
  const h = Math.floor(minutes / 60);
  const m = minutes % 60;
  return m ? `${h}h ${m}m` : `${h}h`;
}

export function lossText(value: number | null): string {
  return value == null ? "" : value.toFixed(2);
}

/** 0-100 for the card's bar: rounds for a DiLoCo run, steps for another job; 0 when neither is known. */
export function progressPercent(row: RunRow): number {
  const [done, all] = row.distributed ? [row.round, row.rounds] : [row.step, row.total];
  return all > 0 ? Math.max(0, Math.min(100, (done / all) * 100)) : 0;
}

/** The `v5 · from scratch` line: the version and the run's own name. */
export function titleText(row: RunRow): { version: string; name: string } {
  return { version: row.version != null ? `v${row.version}` : "", name: row.name };
}

/** What a finished row says on the right: val loss and a ckpt note, or the reason it failed or stopped. */
export function completedText(row: RunRow): string {
  if (row.state === "failed") return fill(`${T.failed}: {why}`, { why: row.note || T.noReason });
  const parts = [row.valLoss != null ? fill(T.val, { value: lossText(row.valLoss) }) : "", row.rounds ? fill(T.roundsDone, { round: row.round, rounds: row.rounds }) : ""];
  const state = row.state === "stopped" ? T.stopped : "";
  return [state, ...parts, state && row.note ? row.note : ""].filter(Boolean).join(" · ") || (row.note || T.finished);
}

export function gb(bytes: number): string {
  return (bytes / 1024 ** 3).toFixed(bytes >= 10 * 1024 ** 3 ? 0 : 1);
}
