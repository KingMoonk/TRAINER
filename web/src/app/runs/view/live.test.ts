import { describe, expect, test } from "vitest";
import {
  ageText, collectingText, countdownText, deviceBadge, edges, eventAge, eventsNewestFirst, EVENTS_SHOWN, laneSegments, liveLossSeries, liveLossText, liveRun, liveSpeedText,
  factAge, otherRuns, OTHER_RUNS_SHOWN, percent, progressFraction, recentTimeline, reportedText, roundNumber, shardsText, stageIndex, stateColor, stepsText, timelineEnd,
  memberPhaseText, timelineExtent, tokensSeries, TIMELINE_ROUNDS,
} from "./live";
import { event, lane, line, member, run } from "@/app/cluster/training/test-data";

describe("which run is live", () => {
  const going = run({ id: "going", state: "running" });
  const done = run({ id: "done", state: "finished" });

  test("the one chosen, else the newest that is going, else the newest", () => {
    expect(liveRun([done, going], "done")?.id).toBe("done");
    expect(liveRun([done, going], null)?.id).toBe("going");
    expect(liveRun([done, going], "gone")?.id).toBe("going"); // a chosen run that is no longer listed
    expect(liveRun([done, run({ id: "older", state: "stopped" })], null)?.id).toBe("done");
    expect(liveRun([], null)).toBeUndefined();
  });

  test("the others are the rest, capped, with the count that did not fit", () => {
    const runs = Array.from({ length: OTHER_RUNS_SHOWN + 3 }, (_, i) => run({ id: `r${i}` }));
    const { shown, hidden } = otherRuns(runs, runs[0]);
    expect(shown).toHaveLength(OTHER_RUNS_SHOWN);
    expect(shown.map((r) => r.id)).not.toContain("r0");
    expect(hidden).toBe(2);
    expect(otherRuns([going], going)).toEqual({ shown: [], hidden: 0 });
  });
});

describe("the round and its phase", () => {
  test("the round in progress is counted from one and stops at the last", () => {
    expect(roundNumber(run({ round: 0, rounds: 5 }))).toBe(1);
    expect(roundNumber(run({ round: 4, rounds: 5 }))).toBe(5);
    expect(roundNumber(run({ round: 5, rounds: 5 }))).toBe(5);
  });

  test("a phase is a stage of the round, or none while waiting, finished or stopped", () => {
    expect(stageIndex(run({ phase: "training" }))).toBe(0);
    expect(stageIndex(run({ phase: "validating" }))).toBe(3);
    for (const phase of ["waiting", "finished", "stopped"] as const) expect(stageIndex(run({ phase }))).toBe(-1);
  });

  test("changes in are only counted while collecting", () => {
    expect(collectingText(run({ phase: "collecting", contributed: 1, expected: 3 }))).toBe("1 of 3 in");
    expect(collectingText(run({ phase: "training", contributed: 1, expected: 3 }))).toBe("");
  });

  test("the countdown rounds up, says when the deadline is reached, and is blank when no round is open", () => {
    expect(countdownText(run({ phase: "training", deadline_s: 3.2 }))).toBe("4 s to the deadline");
    expect(countdownText(run({ phase: "collecting", deadline_s: 0 }))).toBe("Deadline reached");
    expect(countdownText(run({ phase: "outer_step", deadline_s: 1 }))).toBe("");
    expect(countdownText(run({ phase: "training", deadline_s: null }))).toBe("");
  });
});

describe("a machine's node", () => {
  test("progress is the steps done of the round, never past the whole", () => {
    expect(progressFraction(member("a", { step_in_round: 25, round_steps: 100 }), run())).toBe(0.25);
    expect(progressFraction(member("a", { step_in_round: 130, round_steps: 100 }), run())).toBe(1);
    expect(progressFraction(member("a", { step_in_round: 5, round_steps: 0 }), run({ inner_steps: 50 }))).toBe(0.1);
    expect(stepsText(member("a", { step_in_round: 25, round_steps: 100 }), run())).toBe("25 of 100 steps");
  });

  test("the speed now, else the last round's, else nothing", () => {
    expect(liveSpeedText(member("a", { live_tokens_per_second: 2380.4, tokens_per_second: 100 }))).toBe("2,380 tokens/s");
    expect(liveSpeedText(member("a", { live_tokens_per_second: null, tokens_per_second: 100 }))).toBe("100 tokens/s");
    expect(liveSpeedText(member("a"))).toBe("");
  });

  test("loss, report age and shards in words", () => {
    expect(liveLossText(member("a", { loss: 4.2134 }))).toBe("Loss 4.213");
    expect(liveLossText(member("a"))).toBe("");
    expect(reportedText(member("a", { reported_ago: 2.6 }))).toBe("Reported 3 s ago");
    expect(reportedText(member("a"))).toBe("No report yet");
    expect(shardsText(member("a", { shards: 0 }))).toBe("");
    expect(shardsText(member("a", { shards: 1 }))).toBe("1 shard");
    expect(shardsText(member("a", { shards: 3 }))).toBe("3 shards");
  });

  test("the device badge is whatever kind the machine said, in capitals", () => {
    expect(deviceBadge(member("a", { device: "cuda" }))).toBe("CUDA");
    expect(deviceBadge(member("a", { device: "mps" }))).toBe("MPS");
    expect(deviceBadge(member("a", { device: "" }))).toBe("");
  });

  test("a state has a colour from the theme, and trouble is not the colour of training", () => {
    expect(stateColor(member("a", { state: "training" }))).toBe("var(--chart-1)");
    expect(stateColor(member("a", { state: "dropped" }))).toBe("var(--destructive)");
    expect(stateColor(member("a", { state: "unreachable" }))).toBe("var(--destructive)");
    expect(stateColor(member("a", { state: "behind" }))).not.toBe(stateColor(member("a", { state: "training" })));
  });
});

describe("the edges", () => {
  test("θ moves down while a machine downloads it, Δ up while it uploads its change", () => {
    const r = run({ theta_bytes: 6_000_000, delta_dtype: "int8" });
    const [down, up] = edges(r, member("a", { phase: "downloading", last_delta_bytes: 1_000_000, last_delta_dtype: "int8" }));
    expect(down).toEqual({ direction: "down", label: "5.7 MB, float32", active: true });
    expect(up).toEqual({ direction: "up", label: "977 KB, int8", active: false });
    const [, sending] = edges(r, member("a", { phase: "uploading" }));
    expect(sending).toEqual({ direction: "up", label: "int8", active: true }); // nothing sent yet: the format it will be sent in
  });

  test("an unknown size is left out of the θ label", () => {
    expect(edges(run({ theta_bytes: 0 }), member("a"))[0].label).toBe("float32");
  });
});

describe("the timeline", () => {
  test("a lane is download, train, upload and wait, each ending where the next begins", () => {
    expect(laneSegments(lane("a", { theta_taken: 0.2, first_step: 0.6, delta_sent: 3, delta_received: 3.4 }), 6)).toEqual([
      { kind: "download", from: 0.2, to: 0.6 },
      { kind: "train", from: 0.6, to: 3 },
      { kind: "upload", from: 3, to: 3.4 },
      { kind: "wait", from: 3.4, to: 6 },
    ]);
  });

  test("a lane stops where the member stopped reporting, and an open one runs up to now", () => {
    expect(laneSegments(lane("a", { first_step: 0.6, delta_sent: null, delta_received: null }), 2)).toEqual([
      { kind: "download", from: 0.2, to: 0.6 },
      { kind: "train", from: 0.6, to: 2 },
    ]);
    expect(laneSegments(lane("a", { theta_taken: null, first_step: null, delta_sent: null, delta_received: null }), 2)).toEqual([]);
    expect(laneSegments(lane("a", { theta_taken: 0.5, first_step: null, delta_sent: null, delta_received: null }), 2)).toEqual([{ kind: "download", from: 0.5, to: 2 }]);
    expect(laneSegments(lane("a", { delta_sent: 3, delta_received: null }), 4)).toContainEqual({ kind: "upload", from: 3, to: 4 });
  });

  test("a segment never runs past the end, and an empty one is dropped", () => {
    const segments = laneSegments(lane("a", { theta_taken: 0, first_step: 0, delta_sent: 3, delta_received: 3.4 }), 3.2);
    expect(segments.map((s) => s.kind)).toEqual(["train", "upload"]);
    expect(segments[1].to).toBe(3.2);
  });

  test("the last few rounds, oldest first", () => {
    const r = run({ timeline: Array.from({ length: TIMELINE_ROUNDS + 3 }, (_, i) => line(i)) });
    expect(recentTimeline(r).map((l) => l.round)).toEqual([3, 4, 5, 6]);
  });

  test("a closed round ends at its close; the open one at the seconds elapsed, never before a reported moment", () => {
    const open = run({ window_s: 8, deadline_s: 3 });
    expect(timelineEnd(line(0, { closed: 6 }), open)).toBe(6);
    expect(timelineEnd(line(1, { open: true, closed: null, lanes: [lane("a", { delta_received: 3.4 })] }), open)).toBe(5);
    expect(timelineEnd(line(1, { open: true, closed: null, lanes: [lane("a", { delta_received: 7 })] }), open)).toBe(7);
  });

  test("the axis spans the longest round or window so the rounds line up", () => {
    const r = run({ window_s: 8, deadline_s: 3 });
    expect(timelineExtent([line(0, { closed: 6, window: 5, deadline: 5 }), line(1, { open: true, closed: null, window: 8, deadline: 8 })], r)).toBe(8);
    expect(timelineExtent([], run())).toBe(1);
    expect(percent(2, 8)).toBe(25);
    expect(percent(20, 8)).toBe(100);
    expect(percent(-1, 8)).toBe(0);
  });
});

describe("the per-machine series", () => {
  const sampled = run({
    members: [
      member("mac-mini", { samples: [{ step: 0, t: 1, loss: 4.4, tokens_per_second: 2300 }, { step: 20, t: 2, loss: 4.3, tokens_per_second: null }] }),
      member("gpu-box", { samples: [] }),
      member("127.0.0.1:9", { samples: [{ step: 10, t: 1, loss: null, tokens_per_second: 5000 }] }),
    ],
  });

  test("a line for each machine that has a sample, x being the step in the round, with an id safe for a CSS name", () => {
    const loss = liveLossSeries(sampled);
    expect(loss.map((s) => [s.id, s.label])).toEqual([["loss0", "mac-mini"]]);
    expect(loss[0].points).toEqual([{ x: 0, y: 4.4 }, { x: 20, y: 4.3 }]);
    const speed = tokensSeries(sampled);
    expect(speed.map((s) => [s.id, s.label])).toEqual([["tps0", "mac-mini"], ["tps2", "127.0.0.1:9"]]);
    expect(speed[0].points).toEqual([{ x: 0, y: 2300 }]);
  });
});

describe("the event log", () => {
  test("newest first, and no more than the log shows", () => {
    const r = run({ events: Array.from({ length: EVENTS_SHOWN + 5 }, (_, i) => event(i + 1)) });
    const shown = eventsNewestFirst(r);
    expect(shown).toHaveLength(EVENTS_SHOWN);
    expect(shown[0].seq).toBe(EVENTS_SHOWN + 5);
    expect(shown[1].seq).toBe(EVENTS_SHOWN + 4);
  });

  test("an event's age is on the coordinator's clock and never negative", () => {
    const r = run({ now_s: 30 });
    expect(eventAge(r, event(1, { t: 12 }))).toBe("18 s ago");
    expect(eventAge(r, event(2, { t: 31 }))).toBe("0 s ago");
  });
});

describe("whose account of a run this is", () => {
  test("a span of seconds reads in the unit that fits", () => {
    expect([0, 12.4, 89, 90, 600, 5399, 5400, 7200].map(ageText)).toEqual(["0 s", "12 s", "89 s", "2 min", "10 min", "90 min", "2 h", "2 h"]);
  });

  test("a run read live says nothing; one another computer wrote says whose it is and how old", () => {
    expect(factAge(run({ origin: "", age_s: 0 }))).toBe("");
    expect(factAge(run({ origin: "mac", age_s: 0, origin_down: false }))).toBe("");
    expect(factAge(run({ origin: "mac", age_s: 12 }))).toBe("from mac, 12 s ago");
  });

  test("a run whose origin is not answering says so, with the age of its last account", () => {
    expect(factAge(run({ origin: "mac", age_s: 300, origin_down: true }))).toBe("mac is unreachable; last heard 5 min ago");
    expect(factAge(run({ origin: "mac", age_s: 0, origin_down: true }))).toContain("unreachable");
  });
});

describe("what a member is doing before its first step", () => {
  const training = run({ theta_bytes: 3_800_000 });

  test("downloading names the size of θ, loading is plain", () => {
    expect(memberPhaseText(member("a", { phase: "downloading" }), training)).toBe("Downloading θ (3.6 MB)");
    expect(memberPhaseText(member("a", { phase: "downloading" }), run({ theta_bytes: 0 }))).toBe("Downloading θ");
    expect(memberPhaseText(member("a", { phase: "loading" }), training)).toBe("Loading shards");
  });

  test("training before its first step of the round is warming up", () => {
    expect(memberPhaseText(member("a", { phase: "training", step_in_round: 0 }), training)).toBe("Warming up");
    expect(memberPhaseText(member("a", { phase: "training", step_in_round: 3 }), training)).toBe("Training");
  });

  test("a member that has not reported has no phase text", () => {
    expect(memberPhaseText(member("a", { phase: null }), training)).toBe("");
  });
});
