import { describe, expect, test } from "vitest";
import {
  bytesText, deltaText, errorMarks, holdButtons, initialForm, isActive, isProblem, lossSeries, memberDevice, needsReason, neverJoined, reasonText, speedText, startBody, stopReasonText, toggled, trafficSeries, trouble,
  type Form,
} from "./logic";
import { member as build, round, run } from "./test-data";

// the shared member, with the numbers these tests were written around
const member = (over: Partial<Parameters<typeof build>[1]> = {}) => build("gpu-box", { rounds_done: 2, steps: 200, tokens_per_second: 1234.4, last_delta_bytes: 4_000_000, last_delta_dtype: "float16", ...over });

const form = (over: Partial<Form> = {}): Form => ({ rounds: "10", innerSteps: "100", outerLr: "0.7", outerMomentum: "0.9", version: "", datasets: ["a"], machines: [], minMachines: "", maxMachines: "", overlap: false, coordinator: "", ...over });

describe("the form", () => {
  test("starts from the defaults, the newest design and the text sets that are on", () => {
    const sets = [{ id: "a", name: "A", count: 3, on: true, held: false, library: false }, { id: "b", name: "B", count: 1, on: false, held: false, library: false }];
    expect(initialForm({ rounds: 10, inner_steps: 100, outer_lr: 0.7, outer_momentum: 0.9, delta_dtype: "float16" }, sets)).toEqual(form());
  });

  test("makes the request, with the latest design as null and the chosen one as a number", () => {
    expect(startBody(form({ machines: ["gpu-box"] }))).toEqual({
      version: null, dataset_ids: ["a"], machines: ["gpu-box"], rounds: 10, inner_steps: 100, outer_lr: 0.7, outer_momentum: 0.9, overlap: false,
    });
    expect(startBody(form({ version: "3" }))?.version).toBe(3);
  });

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

  test("overlap is sent as ticked", () => expect(startBody(form({ overlap: true }))?.overlap).toBe(true));

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

describe("the charts", () => {
  test("loss has a line for each kind the rounds carry, numbered from one", () => {
    const series = lossSeries([round(0), round(1, { val_loss: null })]);
    expect(series.map((s) => s.id)).toEqual(["loss", "val_loss"]);
    expect(series[0].points).toEqual([{ x: 1, y: 4 }, { x: 2, y: 3 }]);
    expect(series[1].points).toEqual([{ x: 1, y: 4.5 }]);
    expect(lossSeries([round(0, { loss: null, val_loss: null })])).toEqual([]);
  });

  test("traffic is in megabytes, up and down", () => {
    const [up, down] = trafficSeries([round(0)]);
    expect(up.points).toEqual([{ x: 1, y: 1 }]);
    expect(down.points).toEqual([{ x: 1, y: 2 }]);
  });

  test("the error curve marks the best round and each pivot round, counting rounds from one", () => {
    const r = run({ best_round: 1, history: [round(0), round(1), round(2, { pivot: "halve_lr" }), round(3, { pivot: "next_dataset" }), round(4, { pivot: "something_new" })] });
    expect(errorMarks(r).map((m) => [m.id, m.x, m.label])).toEqual([
      ["pivot-2", 3, "Learning rate halved"], ["pivot-3", 4, "Next text"], ["pivot-4", 5, "Changed course"], ["best", 2, "Best"],
    ]);
    expect(errorMarks(run({ history: [round(0)] }))).toEqual([]);
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
