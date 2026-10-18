import { describe, expect, test } from "vitest";
import { errorSeries, eventLine, facts, isTrouble, mergeEvents, type JournalEvent, type Progress } from "./detail";

const event = (index: number, type: string, fields: Record<string, unknown> = {}): JournalEvent => ({ index, type, ...fields });

const progress: Progress = {
  phase: "train",
  round: 3,
  best: 2,
  best_error: 1.23456789,
  current: 3,
  current_error: 2.5,
  learning_rate: 0.001,
  batch_size: 4,
  win_rate: null,
  stop_reason: null,
  active_seconds: 600,
  limit_seconds: 28800,
  worse_streak: 1,
  rounds_since_best: 1,
  restarts: 0,
  teacher: null,
  checkpoints: [
    { n: 1, round: 1, error: 3, best: false },
    { n: 2, round: 2, error: 1.23456789, best: true },
    { n: 3, round: 3, error: 2.5, best: false },
  ] as Progress["checkpoints"],
};

describe("eventLine", () => {
  test.each([
    [event(0, "start", { train: 90, held_out: 10 }), "Started: 90 examples to train on, 10 held back to test."],
    [event(1, "round", { round: 2, learning_rate: 0.001, batch_size: 4 }), "Round 2 began (learning rate 0.001, batch 4)."],
    [event(1, "round", { round: 2, learning_rate: 0.001, batch_size: 4, recovered: true }), "Round 2 began from the checkpoint the last run left (learning rate 0.001, batch 4)."],
    [event(2, "check", { round: 2, step: 40, error: 1.5 }), "Round 2, step 40: test error 1.5."],
    [event(3, "checkpoint", { n: 2, error: 1.2345678, best: true }), "Checkpoint 2 saved: test error 1.235, the best so far."],
    [event(3, "checkpoint", { n: 3, error: 2.5, best: false }), "Checkpoint 3 saved: test error 2.5."],
    [
      event(4, "rollback", { from: 3, to: 2, pivot: "halve_lr", discarded: [3], error: 2.5 }),
      "It got worse (checkpoint 3, test error 2.5), so it went back to checkpoint 2 and tried: halve lr; 1 checkpoint discarded.",
    ],
    [event(5, "restart", { reason: "silent", pid: 7 }), "The watchdog started the trainer again because it had gone quiet."],
    [event(5, "restart", { reason: "stopped", pid: 7 }), "The watchdog started the trainer again because it had stopped."],
    [event(6, "restart", { round: 4, best: 2 }), "The trainer restarted at round 4 and carried on from its last checkpoint."],
    [event(7, "power", { reason: "battery_paused" }), "Paused because the computer is on battery power."],
    [event(8, "teacher", { round: 2, skipped: "no teacher" }), "The teacher was skipped in round 2: no teacher"],
    [event(8, "teacher", { round: 2, polished: true, win_rate: 0.75, lessons: 3 }), "Teacher, round 2: polished the checkpoint, won 75% head to head, 3 lessons."],
    [event(9, "lesson", { checkpoint: 2, count: 1 }), "1 lesson kept from checkpoint 2."],
    [event(10, "stop", { reason: "target" }), "Stopped: the test error reached the target."],
    [event(11, "error", { message: "Training in round 2 went wrong (boom), so it tried again with a smaller learning rate." }), "Training in round 2 went wrong (boom), so it tried again with a smaller learning rate."],
    [event(12, "resume", { reason: "requeued", from: "failed" }), "You resumed it (it had failed)."],
    [event(13, "memory", { restart: true, batch_size: 2 }), "It restarted after running out of memory, so the batch is now 2."],
    [event(14, "something_new"), "something_new"],
  ])("%j", (e, line) => {
    expect(eventLine(e)).toBe(line);
  });

  test("only errors, rollbacks, restarts and memory squeezes are trouble", () => {
    expect(["error", "rollback", "restart", "memory", "checkpoint", "pause"].map((type) => isTrouble(event(0, type)))).toEqual([true, true, true, true, false, false]);
  });
});

describe("mergeEvents", () => {
  test("older pages and the newest page join without repeats, oldest first", () => {
    const merged = mergeEvents([event(0, "start"), event(1, "round")], [event(1, "round"), event(2, "check")]);
    expect(merged.map((e) => e.index)).toEqual([0, 1, 2]);
  });

  test("a newer copy of an event replaces the old one", () => {
    expect(mergeEvents([event(1, "check", { error: 1 })], [event(1, "check", { error: 2 })])[0]).toMatchObject({ error: 2 });
  });
});

describe("progress", () => {
  test("the chart has one point per checkpoint", () => {
    expect(errorSeries(progress)).toEqual([{ id: "error", label: "Test error", points: [{ x: 1, y: 3 }, { x: 2, y: 1.23456789 }, { x: 3, y: 2.5 }], dots: true }]);
    expect(errorSeries({ ...progress, checkpoints: [] })).toEqual([]);
  });

  test("the facts name the phase, round, best error and time", () => {
    const list = Object.fromEntries(facts(progress).map((f) => [f.label, f.value]));
    expect(list).toMatchObject({ Phase: "Training", Round: "3", "Best test error": "1.235 (checkpoint 2)", "Latest test error": "2.5", "Learning rate": "0.001", "Batch size": "4" });
    expect(list["Trained for"]).toBe("10 min of 8.0 hours");
  });

  test("a session that has not started says none yet", () => {
    const list = Object.fromEntries(facts({ ...progress, phase: "", round: 0, best: null, best_error: null, current_error: null, learning_rate: null, batch_size: null, checkpoints: [] }).map((f) => [f.label, f.value]));
    expect(list).toMatchObject({ Phase: "Getting ready", "Best test error": "none yet" });
    expect(list["Learning rate"]).toBeUndefined();
  });
});
