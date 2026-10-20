// The New run form's rules, apart from the page so a test can check them without a browser.
import type { components } from "@/lib/api-schema";
import type { QueueRow } from "@/lib/cluster";
import { nodeName } from "@/lib/node-name";
import type { Plan } from "../../pretraining/logic";
import { fill, NEW_RUN as T } from "./copy";

export type Node = components["schemas"]["ClusterNode"];
export type Stage = (typeof T.stages)[number]["id"];
type Version = components["schemas"]["Version"];
export type Source = components["schemas"]["ContinueSource"];

/** The one phrase `GET /queue/outlook` uses for a job that would start at once. */
const STARTS_NOW = "Starts now";

/**
 * The machines that train. A run over several machines always includes this computer (it is the coordinator and a member),
 * so two or more picked peers bring it in even when it is not ticked. SFT and Ablation run on one machine: the first picked.
 */
export function workersOf(stage: Stage, picked: string[], nodes: Node[]): string[] {
  const known = picked.filter((address) => nodes.some((n) => n.address === address));
  if (stage !== "pretrain") return known.slice(0, 1);
  const local = nodes.find((n) => n.local)?.address;
  const peers = known.filter((address) => address !== local);
  return local && peers.length >= 2 && !known.includes(local) ? [local, ...peers] : known;
}

/** Whether this computer is only in W because a run over several machines needs it. */
export function localForced(stage: Stage, picked: string[], nodes: Node[]): boolean {
  const local = nodes.find((n) => n.local)?.address;
  return !!local && !picked.includes(local) && workersOf(stage, picked, nodes).includes(local);
}

/** The peers of W: the addresses `machines` and `params.pinned` take. This computer is never listed there. */
export function peersOf(workers: string[], nodes: Node[]): string[] {
  return workers.filter((address) => nodes.find((n) => n.address === address)?.local === false);
}

/** The peers among `workers` that are busy: the scheduler gives a busy machine no slot, so the run starts without it and it joins a later round if it frees up. */
export function busyPeers(workers: string[], nodes: Node[]): Node[] {
  return peersOf(workers, nodes).flatMap((address) => nodes.filter((n) => n.address === address && n.online && n.busy));
}

/** "A", "A and B", "A, B and C". */
export function nameList(names: string[]): string {
  return names.length < 2 ? names.join("") : `${names.slice(0, -1).join(", ")} and ${names.at(-1)}`;
}

/** What a busy machine is running, from the merged queue's rows: "Version 3 post-training · about 4.7 days left"; "" when no row says. */
export function runningText(address: string, rows: QueueRow[]): string {
  const row = rows.find((r) => r.status === "running" && r.node === address);
  if (!row) return "";
  const s = row.eta_s;
  const left = s == null ? "" : s < 5400 ? `${Math.max(1, Math.round(s / 60))} min` : s < 172_800 ? `${Math.round(s / 3600)} hours` : `${(s / 86_400).toFixed(1)} days`;
  return [row.label || row.kind, left && fill(T.etaLeft, { time: left })].filter(Boolean).join(" · ");
}

/** The topology chip: its word and the sentence in its tooltip. */
export function shapeOf(workers: string[], nodes: Node[]): { label: string; tip: string } {
  const name = (address: string) => {
    const node = nodes.find((n) => n.address === address);
    return node?.local ? T.coordinatorValue : node ? nodeName(node, nodes) : address;
  };
  if (workers.length === 0) return { label: T.shapeInvalid, tip: T.shapeInvalidTip };
  if (workers.length > 1) return { label: fill(T.shapeDiloco, { count: workers.length }), tip: T.shapeDilocoTip };
  const [only] = workers;
  return nodes.find((n) => n.address === only)?.local
    ? { label: T.shapeLocal, tip: fill(T.shapeLocalTip, { node: name(only) }) }
    : { label: T.shapeRemote, tip: fill(T.shapeRemoteTip, { node: name(only) }) };
}

/** "Start" when the queue would start the job at once, else "Add to queue". */
export function ctaLabel(outlook: string | undefined): string {
  return outlook === STARTS_NOW ? T.startNow : T.addToQueue;
}

/** The pretrain job the plan built, kept on `address` when one peer is the only worker ("it runs there or waits"). */
export function pinnedJob(job: NonNullable<Plan["job"]>, address: string | null) {
  return { ...job, note: "", params: address ? { ...job.params, pinned: address } : job.params };
}

/** The version SFT starts from when none is chosen: the best trained one, else the latest (as the Post-training page does). */
export function defaultSftVersion(versions: Version[]): Version | undefined {
  return [...versions].reverse().find((v) => v.best_checkpoint) ?? versions.at(-1);
}

/** A wall time in plain words: "45 s", "12 min", "1 h 30 min", "2 d 3 h". */
export function durationText(seconds: number): string {
  const s = Math.round(seconds);
  if (s < 60) return `${s} s`;
  const minutes = Math.round(s / 60);
  if (minutes < 60) return `${minutes} min`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return minutes % 60 ? `${hours} h ${minutes % 60} min` : `${hours} h`;
  const days = Math.floor(hours / 24);
  return hours % 24 ? `${days} d ${hours % 24} h` : `${days} d`;
}

const COMPACT = new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 });
const DAY = new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short", year: "numeric" });

/** What one earlier run did, for its row under "Start from": a title, then the facts it recorded (each left out when it is unknown). */
export function sourceFacts(s: Source): { title: string; when: string; facts: string[]; design: string } {
  const title = [fill(T.sourceTitle, { number: s.version }), s.name, s.parent ? T.sourceParent : ""].filter(Boolean).join(" · ");
  const status = s.status ? (T.statuses[s.status] ?? s.status) : "";
  const date = s.finished_at && !Number.isNaN(Date.parse(s.finished_at)) ? DAY.format(new Date(s.finished_at)) : "";
  const when = status || date ? fill(T.sourceFinished, { status: status || T.statuses.completed, date }).trim() : "";
  const loss = (x: number) => x.toFixed(3);
  const facts = [
    s.steps == null ? "" : s.steps_planned ? fill(T.sourceSteps, { done: s.steps.toLocaleString("en-US"), planned: s.steps_planned.toLocaleString("en-US") }) : fill(T.sourceStepsDone, { done: s.steps.toLocaleString("en-US") }),
    s.rounds && s.inner_steps ? fill(T.sourceRounds, { rounds: s.rounds, inner: s.inner_steps }) : "",
    s.seconds ? durationText(s.seconds) : "",
    s.best_val_loss == null ? "" : fill(T.sourceBest, { loss: loss(s.best_val_loss) }),
    s.train_loss == null ? "" : fill(T.sourceTrain, { loss: loss(s.train_loss) }),
    s.params ? fill(T.sourceParams, { count: COMPACT.format(s.params) }) : "",
    s.tokens ? fill(T.sourceTokens, { count: COMPACT.format(s.tokens) }) : "",
    s.device ?? "",
  ].filter(Boolean);
  return { title, when, facts: when || facts.length ? facts : [T.sourceNoRun], design: s.same_design ? T.sourceSame : T.sourceOther };
}

/** The route of a run's page: a query parameter, since the app is a static export. */
export function runHref(id: string): string {
  return `/runs/view/?id=${encodeURIComponent(id)}`;
}
