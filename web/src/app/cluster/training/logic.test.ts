import { describe, expect, test } from "vitest";
import {
  bytesText, deltaText, initialForm, isActive, lossSeries, memberDevice, needsReason, reasonText, speedText, startBody, toggled, trafficSeries, trouble,
  type Form, type Member, type Round, type Run,
} from "./logic";

const member = (over: Partial<Member> = {}): Member => ({
  node: "gpu-box", address: "gpu-box", local: false, device: "cuda", device_name: "RTX 4090", state: "training", note: "", missed: 0, sent: false,
  rounds_done: 2, steps: 200, tokens_per_second: 1234.4, last_delta_bytes: 4_000_000, last_delta_dtype: "float16", can_stop: true, ...over,
});
const round = (n: number, over: Partial<Round> = {}): Round => ({
  round: n, step: (n + 1) * 100, loss: 4 - n, val_loss: 4.5 - n, bytes_up: 2 ** 20, bytes_down: 2 * 2 ** 20, seconds: 2, members: 2, contributed: 2, dropped: [], ...over,
});
const run = (over: Partial<Run> = {}): Run => ({
  id: "r1", label: "Run", version: 1, state: "running", note: "", started_at: 1, owner: "mac-mini", round: 1, rounds: 10, inner_steps: 100, outer_lr: 0.7,
  outer_momentum: 0.9, delta_dtype: "float16", members: [], history: [], can_stop: true, ...over,
});
const form = (over: Partial<Form> = {}): Form => ({ rounds: "10", innerSteps: "100", outerLr: "0.7", outerMomentum: "0.9", version: "", datasets: ["a"], machines: [], ...over });

describe("the form", () => {
  test("starts from the defaults, the newest design and the text sets that are on", () => {
    const sets = [{ id: "a", name: "A", count: 3, on: true, held: false, library: false }, { id: "b", name: "B", count: 1, on: false, held: false, library: false }];
    expect(initialForm({ rounds: 10, inner_steps: 100, outer_lr: 0.7, outer_momentum: 0.9, delta_dtype: "float16" }, sets)).toEqual(form());
  });

  test("makes the request, with the latest design as null and the chosen one as a number", () => {
    expect(startBody(form({ machines: ["gpu-box"] }))).toEqual({
      version: null, dataset_ids: ["a"], machines: ["gpu-box"], rounds: 10, inner_steps: 100, outer_lr: 0.7, outer_momentum: 0.9,
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
  ] as [string, Partial<Form>][])("is not complete with %s", (_, over) => expect(startBody(form(over))).toBeNull());

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
});
