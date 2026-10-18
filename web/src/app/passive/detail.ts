import type { Series } from "@/components/series-chart";
import type { components } from "@/lib/api-schema";
import { formatAgo } from "./session";

export type Detail = components["schemas"]["PassiveDetail"];
export type Progress = components["schemas"]["PassiveProgress"];
export type JournalEvent = components["schemas"]["PassiveEvent"];

/** How the trainer's phase reads. "" is a session whose trainer has not set up yet. */
export const PHASES: Record<string, string> = {
  "": "Getting ready",
  idle: "Between rounds",
  train: "Training",
  save: "Saving a checkpoint",
  teach: "Learning from the teacher",
};

/** Why a session ended (the trainer's stop reason). */
export const STOP_REASONS: Record<string, string> = {
  user: "you stopped it",
  duration: "the time limit was reached",
  patience: "it ran out of rounds without a better checkpoint",
  target: "the test error reached the target",
  win_rate: "it won enough head-to-head replies",
  pivots: "it ran out of changes to try when it got worse",
};

/** The test error of each checkpoint, in the order they were saved, for the chart. */
export function errorSeries(progress: Progress): Series[] {
  const points = progress.checkpoints.map((c) => ({ x: c.n, y: c.error }));
  return points.length === 0 ? [] : [{ id: "error", label: "Test error", points, dots: true }];
}

const num = (value: unknown, digits = 4): string => (typeof value === "number" && Number.isFinite(value) ? Number(value.toPrecision(digits)).toString() : "?");
const text = (value: unknown): string => (value === null || value === undefined ? "" : String(value));
const plural = (n: unknown, word: string): string => `${num(n, 10)} ${word}${n === 1 ? "" : "s"}`;

const POWER: Record<string, string> = { battery_paused: "Paused because the computer is on battery power.", battery_resumed: "Resumed because the computer is back on mains power." };
const RESTART: Record<string, string> = { silent: "it had gone quiet", stopped: "it had stopped" };

/** One plain line for a journal event; the reasons for a rollback or a restart are spelled out. */
export function eventLine(event: JournalEvent): string {
  const e = event as Record<string, unknown>;
  switch (event.type) {
    case "start":
      return `Started: ${plural(e.train, "example")} to train on, ${num(e.held_out, 10)} held back to test.`;
    case "round":
      return `Round ${text(e.round)} began${e.recovered ? " from the checkpoint the last run left" : ""} (learning rate ${num(e.learning_rate)}, batch ${text(e.batch_size)}).`;
    case "check":
      return `Round ${text(e.round)}, step ${text(e.step)}: test error ${num(e.error)}.`;
    case "checkpoint":
      return `Checkpoint ${text(e.n)} saved: test error ${num(e.error)}${e.best ? ", the best so far" : ""}.`;
    case "rollback": {
      const discarded = Array.isArray(e.discarded) ? e.discarded.length : 0;
      return `It got worse (checkpoint ${text(e.from)}, test error ${num(e.error)}), so it went back to checkpoint ${text(e.to)}${e.pivot ? ` and tried: ${text(e.pivot).replaceAll("_", " ")}` : ""}${discarded ? `; ${plural(discarded, "checkpoint")} discarded` : ""}.`;
    }
    case "teacher":
      if (e.skipped) return `The teacher was skipped in round ${text(e.round)}: ${text(e.skipped)}`;
      return `Teacher, round ${text(e.round)}: ${e.polished ? "polished the checkpoint" : "left the checkpoint as it was"}${e.win_rate == null ? "" : `, won ${num(Number(e.win_rate) * 100, 3)}% head to head`}${typeof e.lessons === "number" ? `, ${plural(e.lessons, "lesson")}` : ""}.`;
    case "lesson":
      return `${plural(e.count, "lesson")} kept from checkpoint ${text(e.checkpoint)}.`;
    case "restart":
      return e.reason ? `The watchdog started the trainer again because ${RESTART[text(e.reason)] ?? text(e.reason)}.` : `The trainer restarted at round ${text(e.round)} and carried on from its last checkpoint.`;
    case "resume":
      return e.reason === "requeued" ? `You resumed it (it had ${text(e.from)}).` : "Resumed.";
    case "power":
      return POWER[text(e.reason)] ?? `Power: ${text(e.reason)}.`;
    case "pause":
      return "Paused.";
    case "stop":
      return `Stopped: ${STOP_REASONS[text(e.reason)] ?? text(e.reason)}.`;
    case "memory":
      return e.restart ? `It restarted after running out of memory, so the batch is now ${text(e.batch_size)}.` : `Memory is tight (${num(e.used, 3)} of ${num(e.limit, 3)} bytes), so the batch is now ${text(e.batch_size)}.`;
    case "error":
      return text(e.message) || "An error.";
    default:
      return event.type;
  }
}

/** Whether a journal event is a problem the person should notice (an error, a rollback, a restart or a memory squeeze). */
export function isTrouble(event: JournalEvent): boolean {
  return ["error", "rollback", "restart", "memory"].includes(event.type);
}

/** Older journal pages plus the newest page, de-duplicated by index and oldest first. */
export function mergeEvents(older: JournalEvent[], newest: JournalEvent[]): JournalEvent[] {
  const byIndex = new Map<number, JournalEvent>();
  for (const event of [...older, ...newest]) byIndex.set(event.index, event);
  return [...byIndex.values()].sort((a, b) => a.index - b.index);
}

/** The progress facts as label/value pairs: phase, round, best error, how long it has run. */
export function facts(progress: Progress): { label: string; value: string }[] {
  const used = progress.active_seconds;
  return [
    { label: "Phase", value: PHASES[progress.phase] ?? progress.phase },
    { label: "Round", value: String(progress.round) },
    { label: "Best test error", value: progress.best_error == null ? "none yet" : `${num(progress.best_error)} (checkpoint ${progress.best})` },
    { label: "Latest test error", value: progress.current_error == null ? "none yet" : num(progress.current_error) },
    { label: "Learning rate", value: progress.learning_rate == null ? "" : num(progress.learning_rate) },
    { label: "Batch size", value: progress.batch_size == null ? "" : String(progress.batch_size) },
    { label: "Trained for", value: `${formatAgo(used)} of ${formatAgo(progress.limit_seconds)}` },
    { label: "Rounds without a better checkpoint", value: String(progress.rounds_since_best) },
  ].filter((fact) => fact.value !== "");
}
