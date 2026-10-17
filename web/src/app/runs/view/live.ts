// The Training tab's live view of a run, apart from the page so a test can check it without a browser: the phase of the round, what each
// machine is doing, the edges between the coordinator and them, the swimlanes, the per-machine series and the event log.
import type { components } from "@/lib/api-schema";
import type { Series } from "@/components/series-chart";
import { TRAINING as T } from "@/lib/copy/training";
import { bytesText, deltaText, isActive, type Member, type Run } from "@/app/cluster/training/logic";

export type Lane = components["schemas"]["Lane"];
export type RoundTimeline = components["schemas"]["RoundTimeline"];
export type RunEvent = components["schemas"]["RunEvent"];
export type Sample = components["schemas"]["studio_api__routers__distributed__Sample"];

/** The phases of a round, in order. */
export const STAGES = ["training", "collecting", "outer_step", "validating", "next_round"] as const;

/** The rounds the timeline shows: the open one and the last few before it. */
export const TIMELINE_ROUNDS = 4;
/** The events the log lists. */
export const EVENTS_SHOWN = 50;
/** The other runs listed under the live one. */
export const OTHER_RUNS_SHOWN = 8;

/** The run to watch: the one chosen if it is still listed, else the newest that is going, else the newest. */
export function liveRun(runs: Run[], chosen: string | null): Run | undefined {
  return runs.find((run) => run.id === chosen) ?? runs.find(isActive) ?? runs[0];
}

/** The runs under the live one, newest first, and how many did not fit. */
export function otherRuns(runs: Run[], live: Run | undefined): { shown: Run[]; hidden: number } {
  const rest = runs.filter((run) => run.id !== live?.id);
  return { shown: rest.slice(0, OTHER_RUNS_SHOWN), hidden: Math.max(0, rest.length - OTHER_RUNS_SHOWN) };
}

/** The round being trained, counted from one and never past the last. */
export function roundNumber(run: Run): number {
  return Math.min(run.round + 1, run.rounds);
}

/** Where the run is among the phases of a round, or -1 while it is waiting, finished or stopped. */
export function stageIndex(run: Run): number {
  return (STAGES as readonly string[]).indexOf(run.phase);
}

/** "1 of 2 in": how many changes have arrived of how many are expected, or "" outside the collecting phase. */
export function collectingText(run: Run): string {
  return run.phase === "collecting" ? T.collectingIn.replace("{n}", String(run.contributed)).replace("{m}", String(run.expected)) : "";
}

/** The seconds left to the round's deadline in words, "" when no round is open. */
export function countdownText(run: Run): string {
  if (run.deadline_s == null || (run.phase !== "training" && run.phase !== "collecting")) return "";
  if (run.deadline_s <= 0) return T.deadlinePassed;
  return T.countdown.replace("{n}", String(Math.ceil(run.deadline_s)));
}

/** How far a member is through its steps of the round, from 0 to 1. */
export function progressFraction(member: Member, run: Run): number {
  const total = member.round_steps || run.inner_steps;
  return total > 0 ? Math.min(1, Math.max(0, member.step_in_round / total)) : 0;
}

/** "40 of 100 steps". */
export function stepsText(member: Member, run: Run): string {
  return T.stepsOf.replace("{k}", String(member.step_in_round)).replace("{h}", String(member.round_steps || run.inner_steps));
}

/** The speed right now, else the last round's, in words; "" before either is known. */
export function liveSpeedText(member: Member): string {
  const speed = member.live_tokens_per_second ?? member.tokens_per_second;
  return speed == null ? "" : T.tokensPerSecond.replace("{n}", Math.round(speed).toLocaleString("en-US"));
}

/** "Loss 4.213", or "" before the first step is reported. */
export function liveLossText(member: Member): string {
  return member.loss == null ? "" : T.liveLoss.replace("{n}", member.loss.toFixed(3));
}

/** "Reported 3 s ago", or "No report yet". */
export function reportedText(member: Member): string {
  return member.reported_ago == null ? T.noReportYet : T.reportedAgo.replace("{n}", String(Math.max(0, Math.round(member.reported_ago))));
}

/** "2 shards" of the text this member holds, "" when none are held. */
export function shardsText(member: Member): string {
  return member.shards <= 0 ? "" : member.shards === 1 ? T.shard : T.shards.replace("{n}", String(member.shards));
}

/** The device kind as a badge ("CUDA", "MPS"), "" before the machine said. Any kind is shown as it is. */
export function deviceBadge(member: Member): string {
  return member.device.toUpperCase();
}

/** The colour a member's node wears for its state (a theme token, so it follows light and dark). */
export function stateColor(member: Member): string {
  switch (member.state) {
    case "training": return "var(--chart-1)";
    case "finished": return "var(--chart-2)";
    case "behind": return "var(--chart-4)";
    case "dropped":
    case "unreachable": return "var(--destructive)";
    default: return "var(--border)";
  }
}

export type Edge = { direction: "down" | "up"; label: string; active: boolean };

/** The two edges between the coordinator and a member: θ down (while the member downloads it) and Δ up (while it uploads its change). */
export function edges(run: Run, member: Member): [Edge, Edge] {
  const theta = [run.theta_bytes > 0 ? bytesText(run.theta_bytes) : "", "float32"].filter(Boolean).join(", ");
  return [
    { direction: "down", label: theta, active: member.phase === "downloading" },
    { direction: "up", label: deltaText(member) || run.delta_dtype, active: member.phase === "uploading" },
  ];
}

export type Segment = { kind: "download" | "train" | "wait" | "upload"; from: number; to: number };

/**
 * One lane's segments in seconds from the round opening, up to `end` (the round's close, or now while it is open): θ downloaded, trained,
 * Δ uploaded, then waiting for the others. A moment the member never reported ends the lane there.
 */
export function laneSegments(lane: Lane, end: number): Segment[] {
  const out: Segment[] = [];
  const add = (kind: Segment["kind"], from: number, to: number | null) => {
    const stop = Math.min(to ?? end, end);
    if (stop > from) out.push({ kind, from, to: stop });
  };
  const start = lane.theta_taken ?? lane.first_step;
  if (start === null) return out;
  add("download", start, lane.first_step);
  if (lane.first_step === null) return out;
  add("train", lane.first_step, lane.delta_sent);
  if (lane.delta_sent === null) return out;
  add("upload", lane.delta_sent, lane.delta_received);
  if (lane.delta_received !== null) add("wait", lane.delta_received, end);
  return out;
}

/** The last rounds of the timeline, oldest first. */
export function recentTimeline(run: Run): RoundTimeline[] {
  return run.timeline.slice(-TIMELINE_ROUNDS);
}

/** Seconds into the open round on the coordinator's clock, from the window and what is left of it. */
export function elapsedSeconds(run: Run): number {
  return run.window_s != null && run.deadline_s != null ? Math.max(0, run.window_s - run.deadline_s) : 0;
}

/** Where a round's lanes end: its close, or for the open round the seconds elapsed (never before the latest moment a member reported). */
export function timelineEnd(line: RoundTimeline, run: Run): number {
  if (line.closed != null) return line.closed;
  const moments = line.lanes.flatMap((lane) => [lane.theta_taken, lane.first_step, lane.delta_sent, lane.delta_received]).filter((t): t is number => t != null);
  return Math.max(elapsedSeconds(run), ...moments, 0);
}

/** The seconds the timeline's shared axis spans, so the rounds line up and their deadline marks compare. */
export function timelineExtent(lines: RoundTimeline[], run: Run): number {
  return Math.max(1, ...lines.flatMap((line) => [line.window, line.deadline, timelineEnd(line, run)]));
}

/** A position along the axis as a percentage. */
export function percent(seconds: number, extent: number): number {
  return Math.min(100, Math.max(0, (seconds / extent) * 100));
}

/** One line per member over the open round's reported samples, x being the step in the round. Ids are positional so a hostname never reaches a CSS name. */
function memberSeries(run: Run, prefix: string, pick: (sample: Sample) => number | null): Series[] {
  return run.members
    .map((member, index) => ({
      id: `${prefix}${index}`,
      label: member.node,
      points: member.samples.flatMap((sample) => {
        const y = pick(sample);
        return y == null ? [] : [{ x: sample.step, y }];
      }),
    }))
    .filter((series) => series.points.length > 0);
}

/** Each member's inner loss within the open round. */
export function liveLossSeries(run: Run): Series[] {
  return memberSeries(run, "loss", (sample) => sample.loss);
}

/** Each member's tokens per second within the open round. */
export function tokensSeries(run: Run): Series[] {
  return memberSeries(run, "tps", (sample) => sample.tokens_per_second);
}

/** The newest events first, as many as the log shows. */
export function eventsNewestFirst(run: Run): RunEvent[] {
  return [...run.events].sort((a, b) => b.seq - a.seq).slice(0, EVENTS_SHOWN);
}

/** How long ago an event happened on the coordinator's clock. */
export function eventAge(run: Run, event: RunEvent): string {
  return T.secondsAgo.replace("{n}", String(Math.max(0, Math.round(run.now_s - event.t))));
}
