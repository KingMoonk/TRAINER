import { describe, expect, test } from "vitest";
import {
  bytesText, deltaText, lessonsText, holdButtons, defaultInnerSteps, deltaDtypeOf, initialForm, isActive, isProblem, memberDevice, needsReason, neverJoined, reasonText, speedText, sparsityProblem, startBody, stepsProblem, stopReasonText, toggled, trouble,
  type Form,
} from "./logic";
import { initialRules } from "./rules";
import { TRAINING as T } from "@/lib/copy/training";
import { member as build, run } from "./test-data";

// the shared member, with the numbers these tests were written around
const member = (over: Partial<Parameters<typeof build>[1]> = {}) => build("gpu-box", { rounds_done: 2, steps: 200, tokens_per_second: 1234.4, last_delta_bytes: 4_000_000, last_delta_dtype: "float16", ...over });

const form = (over: Partial<Form> = {}): Form => ({ rounds: "10", innerSteps: "100", outerLr: "0.7", outerMomentum: "0.9", version: "", datasets: ["a"], chats: [], machines: [], minMachines: "", maxMachines: "", overlap: true, timeBox: true, deltaDtype: "float16", deltaSparsity: "", reviewedOnly: false, coordinator: "", teacher: "", polishEvery: "", targetWinRate: "", rules: initialRules(), maxSteps: 1_000_000, ...over });

describe("the form", () => {
  test("starts from the defaults, the newest design and the text sets that are on", () => {
    const sets = [{ id: "a", name: "A", count: 3, on: true, held: false, library: false }, { id: "b", name: "B", count: 1, on: false, held: false, library: false }];
    expect(initialForm({ rounds: 10, inner_steps: 100, remote_inner_steps: 1500, outer_lr: 0.7, outer_momentum: 0.9, delta_dtype: "float16", max_steps: 1_000_000 }, sets)).toEqual(form());
  });

  test("steps per round: the usual on this computer alone, the larger with another machine", () => {
    const defaults = { rounds: 10, inner_steps: 100, remote_inner_steps: 1500, outer_lr: 0.7, outer_momentum: 0.9, delta_dtype: "float16", max_steps: 1_000_000 };
    expect(defaultInnerSteps(defaults, [])).toBe("100");
    expect(defaultInnerSteps(defaults, ["gpu-box"])).toBe("1500");
  });

  test("keeps Start off, saying why, while rounds × steps per round is more than a run trains", () => {
    expect(stepsProblem(form({ rounds: "3000", innerSteps: "1000" }))).toBe("3,000 rounds of 1,000 steps is 3,000,000 steps; a run trains at most 1,000,000. Use fewer rounds or fewer steps per round.");
    expect(startBody(form({ rounds: "3000", innerSteps: "1000" }))).toBeNull();
    expect(stepsProblem(form({ rounds: "1000", innerSteps: "1000" }))).toBe("");
    expect(startBody(form({ rounds: "1000", innerSteps: "1000" }))).not.toBeNull();
    expect(stepsProblem(form({ rounds: "", innerSteps: "1000" }))).toBe(""); // not filled in yet
  });

  test("makes the request, with the latest design as null and the chosen one as a number", () => {
    expect(startBody(form({ machines: ["gpu-box"] }))).toEqual({
      version: null, dataset_ids: ["a"], machines: ["gpu-box"], rounds: 10, inner_steps: 100, outer_lr: 0.7, outer_momentum: 0.9, overlap: null, time_box: null, delta_dtype: "float16", delta_sparsity: 0, reviewed_only: false, rows: 16, hold: 0.05, chat_share: 0, chat_datasets: [],
    });
    expect(startBody(form({ version: "3" }))?.version).toBe(3);
  });

  test("sends the teacher only with a use for it, the win-rate as a share", () => {
    expect(startBody(form())).not.toHaveProperty("teacher"); // no teacher, nothing sent
    expect(startBody(form({ teacher: "judge:1b", polishEvery: "3" }))).toMatchObject({ teacher: "judge:1b", polish_every: 3 });
    expect(startBody(form({ teacher: "judge:1b", polishEvery: "3" }))).not.toHaveProperty("target_win_rate");
    expect(startBody(form({ teacher: "judge:1b", targetWinRate: "60" }))).toMatchObject({ teacher: "judge:1b", target_win_rate: 0.6 });
    expect(startBody(form({ teacher: "judge:1b", targetWinRate: "60" }))).not.toHaveProperty("polish_every");
    expect(startBody(form({ teacher: "judge:1b", polishEvery: "9", targetWinRate: "100" }))).toMatchObject({ polish_every: 9, target_win_rate: 1 });
  });

  test("the ticked conversation sets go with the teacher only", () => {
    expect(startBody(form({ chats: ["c"] }))?.dataset_ids).toEqual(["a"]);
    expect(startBody(form({ chats: ["c", "a"], teacher: "t", polishEvery: "2" }))?.dataset_ids).toEqual(["a", "c"]);
  });

  test.each([
    ["a teacher with neither a polish nor a target", { teacher: "t" }],
    ["a teacher with a coordinator that is another computer", { teacher: "t", polishEvery: "2", machines: ["gpu-box"], coordinator: "gpu-box" }],
    ["a polish with no teacher", { polishEvery: "2" }],
    ["a win-rate target with no teacher", { targetWinRate: "50" }],
    ["a polish every 0 rounds", { teacher: "t", polishEvery: "0" }],
    ["a polish every part of a round", { teacher: "t", polishEvery: "1.5" }],
    ["a polish as often as there are rounds", { teacher: "t", polishEvery: "10" }],
    ["a win-rate of zero", { teacher: "t", targetWinRate: "0" }],
    ["a win-rate under 1", { teacher: "t", targetWinRate: "0.5" }],
    ["a win-rate above 100", { teacher: "t", targetWinRate: "101" }],
    ["a win-rate that is a word", { teacher: "t", targetWinRate: "most" }],
  ] as [string, Partial<Form>][])("is not complete with %s", (_, over) => expect(startBody(form(over))).toBeNull());

  test.each([
    ["no text ticked", { datasets: [] }],
    ["zero rounds", { rounds: "0" }],
    ["a fraction of a step", { innerSteps: "1.5" }],
    ["a blank learning rate", { outerLr: "" }],
    ["a momentum of one", { outerMomentum: "1" }],
    ["a settings word", { rounds: "ten" }],
    ["a fewest machines of zero", { minMachines: "0" }],
    ["a most machines above 64", { maxMachines: "65" }],
    ["a fewest above the most", { minMachines: "3", maxMachines: "2" }],
  ] as [string, Partial<Form>][])("is not complete with %s", (_, over) => expect(startBody(form(over))).toBeNull());

  test("the machine counts are left out while blank and sent as numbers when typed", () => {
    expect(startBody(form())).not.toHaveProperty("min_members");
    expect(startBody(form({ minMachines: "2", maxMachines: "4" }))).toMatchObject({ min_members: 2, max_members: 4 });
    expect(startBody(form({ maxMachines: "2" }))).not.toHaveProperty("min_members");
  });

  test("a coordinator is sent only when it is one of the ticked machines", () => {
    expect(startBody(form({ machines: ["gpu-box"] }))).not.toHaveProperty("coordinator");
    expect(startBody(form({ machines: ["gpu-box"], coordinator: "gpu-box" }))).toMatchObject({ coordinator: "gpu-box" });
    expect(startBody(form({ machines: ["other"], coordinator: "gpu-box" }))).not.toHaveProperty("coordinator");
  });

  test("overlap ticked is left to the server, unticked is sent as off", () => {
    expect(startBody(form({ overlap: true }))?.overlap).toBeNull();
    expect(startBody(form({ overlap: false }))?.overlap).toBe(false);
  });
  test("sending only the largest changes is off when empty, sent as a fraction, and needs int8 and a share up to the limit", () => {
    expect(startBody(form({ deltaDtype: "int8" }))?.delta_sparsity).toBe(0);
    expect(startBody(form({ deltaDtype: "int8", deltaSparsity: "5" }))?.delta_sparsity).toBe(0.05);
    expect(startBody(form({ deltaDtype: "int8", deltaSparsity: "20" }))?.delta_sparsity).toBe(0.2);
    for (const bad of ["0", "-1", "21", "x"]) expect(startBody(form({ deltaDtype: "int8", deltaSparsity: bad }))).toBeNull();
    expect(startBody(form({ deltaDtype: "float16", deltaSparsity: "5" }))).toBeNull();
    expect(sparsityProblem(form({ deltaDtype: "float16", deltaSparsity: "5" }))).not.toBe("");
    expect(sparsityProblem(form({ deltaDtype: "float16" }))).toBe("");
  });
  test("the number format of the traffic is sent, and an unknown default falls back to int8", () => {
    expect(startBody(form({ deltaDtype: "int8" }))?.delta_dtype).toBe("int8");
    expect(deltaDtypeOf("fp16")).toBe("int8");
    expect(deltaDtypeOf("float32")).toBe("float32");
  });

  test("reviewed only is sent as ticked", () => expect(startBody(form({ reviewedOnly: true }))?.reviewed_only).toBe(true));

  test("momentum may be zero", () => expect(startBody(form({ outerMomentum: "0" }))?.outer_momentum).toBe(0));

  test("a ticked item is added once, and removed when unticked", () => {
    expect(toggled(["a"], "b", true)).toEqual(["a", "b"]);
    expect(toggled(["a", "b"], "b", true)).toEqual(["a", "b"]);
    expect(toggled(["a", "b"], "a", false)).toEqual(["b"]);
  });
});

describe("words and numbers", () => {
  test("sizes", () => {
    expect(bytesText(512)).toBe("512 B");
    expect(bytesText(4_000_000)).toBe("3.8 MB");
    expect(bytesText(3 * 2 ** 30)).toBe("3.0 GB");
    expect(bytesText(null)).toBe("");
  });

  test("a member's speed, change and device, never assuming the kind of device", () => {
    expect(speedText(member())).toBe("1,234 tokens/s");
    expect(speedText(member({ tokens_per_second: null }))).toBe("");
    expect(deltaText(member())).toBe("3.8 MB, float16");
    expect(deltaText(member({ last_delta_bytes: null }))).toBe("");
    expect(memberDevice(member())).toBe("RTX 4090 (cuda)");
    expect(memberDevice(member({ device: "mps", device_name: "Apple M2" }))).toBe("Apple M2 (mps)");
    expect(memberDevice(member({ device: "", device_name: "" }))).toBe("");
  });

  test("a run is active until it ends", () => {
    for (const state of ["queued", "starting", "waiting", "running"] as const) expect(isActive(run({ state }))).toBe(true);
    for (const state of ["finished", "stopped", "failed"] as const) expect(isActive(run({ state }))).toBe(false);
  });
});

describe("a member that is out", () => {
  test("unreachable, dropped and behind members need a reason; training ones do not", () => {
    expect(needsReason(member())).toBe(false);
    for (const state of ["unreachable", "dropped", "behind"] as const) expect(needsReason(member({ state }))).toBe(true);
  });

  test("the reason is the server's sentence, else a plain one", () => {
    expect(reasonText(member({ state: "unreachable", note: "gpu-box refused this computer (401)." }))).toBe("gpu-box refused this computer (401).");
    expect(reasonText(member({ state: "dropped" }))).toMatch(/Dropped/);
    expect(reasonText(member({ state: "behind" }))).toMatch(/Late/);
    expect(reasonText(member({ state: "unreachable" }))).toMatch(/could not be reached/i);
  });

  test("a machine that never joined shows its reason but is not a problem; one that left at the finish is not either", () => {
    expect(needsReason(member({ state: "never_joined" }))).toBe(true);
    expect(isProblem(member({ state: "never_joined" }))).toBe(false);
    expect(neverJoined(member({ state: "never_joined" }))).toBe(true);
    expect(reasonText(member({ state: "never_joined", note: "It was busy.", reason: "It was busy." }))).toBe("It was busy.");
    expect(needsReason(member({ state: "waiting", reason: "Busy with 'Version 1'." }))).toBe(true);
    expect(neverJoined(member({ state: "waiting" }))).toBe(false);
    expect(reasonText(member({ state: "never_joined" }))).toMatch(/never joined/i);
    expect(needsReason(member({ state: "finished", note: "" }))).toBe(false);
    expect(isProblem(member({ state: "left", note: "Left the run." }))).toBe(true);
  });

  test("trouble lists them", () => {
    const r = run({ members: [member(), member({ node: "old", state: "unreachable" })] });
    expect(trouble(r).map((m) => m.node)).toEqual(["old"]);
  });
});

describe("training labels", () => {
  test("the journal and the phase line have words for the teacher's polish and lessons", () => {
    expect(T.phases.polishing).toBe("Teacher polishing");
    expect([T.eventKinds.polish, T.eventKinds.polish_off, T.eventKinds.lessons]).toEqual(["Teacher polish", "Polish off", "Lessons"]);
    expect(T.pivots.drop_lessons).toBe("Lessons left out");
    expect(T.phases.judging).toBe("Teacher comparing weights");
    expect([T.eventKinds.judging, T.eventKinds.win_rate, T.eventKinds.win_rate_off]).toEqual(["Comparing weights", "Win rate", "Win rate off"]);
  });

  test("the run says how many lessons the teacher added, in the singular too", () => {
    expect(lessonsText(run())).toBe("");
    expect(lessonsText(run({ lessons: 1 }))).toBe("1 lesson from the teacher in the training text");
    expect(lessonsText(run({ lessons: 4 }))).toBe("4 lessons from the teacher in the training text");
    expect(lessonsText(run({ lessons: 4, lessons_left_out: 2 }))).toBe("4 lessons from the teacher in the training text, 2 left out");
  });

  test("Pause shows while the run can be paused, Resume while it is paused, Restart once it ended", () => {
    expect(holdButtons(run({ can_pause: true }))).toEqual({ pause: true, resume: false, restart: false });
    expect(holdButtons(run({ can_pause: true, can_resume: true, paused: true }))).toEqual({ pause: false, resume: true, restart: false });
    expect(holdButtons(run({ state: "stopped", can_restart: true }))).toEqual({ pause: false, resume: false, restart: true });
    expect(holdButtons(run())).toEqual({ pause: false, resume: false, restart: false });
  });

  test("the stop reason is a sentence, or nothing while the run goes", () => {
    expect(stopReasonText(run({ stop_reason: "The loss stopped improving." }))).toBe("Stopped because: The loss stopped improving.");
    expect(stopReasonText(run())).toBe("");
  });
});
