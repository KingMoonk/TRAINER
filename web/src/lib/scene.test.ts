import { describe, expect, test } from "vitest";
import {
  applyFrame,
  HEAD_STAGES,
  isMatrixGroup,
  LEVELS,
  MODES,
  pulsePose,
  row,
  TILE_KINDS,
  type GateGroup,
  type MatrixGroup,
  type SceneFrame,
  type ScenePayload,
  type SceneGroup,
} from "./scene";

const gate = (index: number, selected = false): GateGroup => ({ index, selected, origin: [0, index, 0], boxes: [], lines: [], labels: [] });
const matrix = (): MatrixGroup => ({
  level: "Head", kind: "scores", weight_name: null, shape: [2, 2], origin_cell: [0, 0], pitch: 1.5, caption: "c",
  origin: [0, 0, 0], boxes: [], lines: [], labels: [],
});

const scene = (groups: Record<string, SceneGroup>, over: Partial<ScenePayload> = {}) =>
  ({
    schema: 1, full: true, up: "z", palette: {}, camera: { target: [0, 0, 0], distance: 5, level: "Model" },
    view: { level: "Model" }, selection: { layer: 0 }, stats: { step: 1 }, design: { name: "d" }, playback: { show: false },
    groups, removed_groups: ["stale"], cursor: { view: "v1", "groups/gate:0": "g0" },
    ...over,
  }) as unknown as ScenePayload;

const frame = (over: Partial<SceneFrame>): SceneFrame => ({ schema: 1, full: false, groups: {}, removed_groups: [], cursor: {}, ...over });

describe("the control lists", () => {
  test("list the viewer's levels, modes, head stages and tile kinds in order", () => {
    expect([...LEVELS]).toEqual(["Model", "Window", "Head", "Weights"]);
    expect([...MODES]).toEqual(["Build", "Train", "Ride a token"]);
    expect([...HEAD_STAGES]).toEqual(["scores", "scale", "mask", "pattern", "values"]);
    expect([...TILE_KINDS]).toEqual(["weight", "gradient", "update"]);
  });
});

describe("isMatrixGroup", () => {
  test("a matrix group is told from a gate group by its pitch", () => {
    expect(isMatrixGroup(matrix())).toBe(true);
    expect(isMatrixGroup(gate(0))).toBe(false);
  });
});

describe("applyFrame", () => {
  test("replaces the groups the frame carries and keeps the others", () => {
    const before = scene({ "gate:0": gate(0), "gate:1": gate(1) });
    const next = applyFrame(before, frame({ groups: { "gate:1": gate(1, true), "gate:2": gate(2) } }));
    expect(Object.keys(next.groups).sort()).toEqual(["gate:0", "gate:1", "gate:2"]);
    expect(next.groups["gate:0"]).toBe(before.groups["gate:0"]);
    expect((next.groups["gate:1"] as GateGroup).selected).toBe(true);
  });

  test("drops the groups the frame removes", () => {
    const next = applyFrame(scene({ "gate:0": gate(0), "gate:1": gate(1) }), frame({ removed_groups: ["gate:0"] }));
    expect(Object.keys(next.groups)).toEqual(["gate:1"]);
  });

  test("a group removed and re-sent in one frame is the new one", () => {
    const replacement = gate(0, true);
    const next = applyFrame(scene({ "gate:0": gate(0) }), frame({ removed_groups: ["gate:0"], groups: { "gate:0": replacement } }));
    expect(next.groups["gate:0"]).toBe(replacement);
  });

  test("replaces only the sections the frame carries", () => {
    const before = scene({});
    const stats = { step: 9, loss: 0.5, info: "", training: {}, training_line: "", note: "" };
    const next = applyFrame(before, frame({ stats }));
    expect(next.stats).toBe(stats);
    expect(next.view).toBe(before.view);
    expect(next.selection).toBe(before.selection);
    expect(next.design).toBe(before.design);
    expect(next.playback).toBe(before.playback);
  });

  test("keeps the new cursor and schema, clears removed_groups, and stays a full scene", () => {
    const next = applyFrame(scene({}), frame({ schema: 2, cursor: { view: "v2" } }));
    expect(next.cursor).toEqual({ view: "v2" });
    expect(next.schema).toBe(2);
    expect(next.removed_groups).toEqual([]);
    expect(next.full).toBe(true);
  });

  test("keeps what a frame never carries: palette, camera and up", () => {
    const before = scene({});
    const next = applyFrame(before, frame({}));
    expect(next.palette).toBe(before.palette);
    expect(next.camera).toBe(before.camera);
    expect(next.up).toBe("z");
  });

  test("does not change the scene it was given", () => {
    const before = scene({ "gate:0": gate(0) });
    const snapshot = JSON.stringify(before);
    applyFrame(before, frame({ removed_groups: ["gate:0"], groups: { "gate:5": gate(5) }, cursor: { view: "x" } }));
    expect(JSON.stringify(before)).toBe(snapshot);
  });
});

describe("pulsePose", () => {
  test("starts at the first block, moving forward, at depth zero", () => {
    expect(pulsePose(4, 0, false)).toEqual({ block: 0, depth: 0, forward: true });
  });

  test("goes forward through the blocks at two blocks per second", () => {
    expect(pulsePose(4, 0.5, false)).toEqual({ block: 1, depth: 1, forward: true });
    expect(pulsePose(4, 1.5, false)).toEqual({ block: 3, depth: 3, forward: true });
  });

  test("turns round after the last block and comes back down", () => {
    expect(pulsePose(4, 2, false)).toEqual({ block: 3, depth: 4, forward: false });
    expect(pulsePose(4, 2.5, false)).toEqual({ block: 2, depth: 3, forward: false });
    expect(pulsePose(4, 3.5, false)).toEqual({ block: 0, depth: 1, forward: false });
  });

  test("never names a block outside 0..blocks-1, over a whole cycle", () => {
    for (let t = 0; t < 8; t += 1 / 64) { // binary steps: exact in floating point
      const { block } = pulsePose(4, t, false);
      expect(block).toBeGreaterThanOrEqual(0);
      expect(block).toBeLessThanOrEqual(3);
    }
  });

  test("repeats every `blocks` seconds", () => {
    expect(pulsePose(4, 4, false)).toEqual(pulsePose(4, 0, false));
    expect(pulsePose(4, 5.25, false)).toEqual(pulsePose(4, 1.25, false));
  });

  test("a negative time wraps to the end of the cycle", () => {
    expect(pulsePose(4, -0.25, false)).toEqual({ block: 0, depth: 0.5, forward: false });
  });

  test("with reduced motion the pulse sits on a block instead of gliding between them", () => {
    expect(pulsePose(4, 0.75, true)).toEqual({ block: 1, depth: 1, forward: true });
    expect(pulsePose(4, 2, true)).toEqual({ block: 3, depth: 3, forward: false });
  });

  test("a single block still works in both directions", () => {
    expect(pulsePose(1, 0.25, false)).toEqual({ block: 0, depth: 0.5, forward: true });
    expect(pulsePose(1, 0.75, false)).toEqual({ block: 0, depth: 0.5, forward: false });
  });
});

describe("row", () => {
  const array = { shape: [3, 2], data: [1, 2, 3, 4, 5, 6] };

  test("reads one row of a flat row-major array", () => {
    expect(row(array, 0, 2)).toEqual([1, 2]);
    expect(row(array, 2, 2)).toEqual([5, 6]);
  });

  test("a row past the end is empty", () => {
    expect(row(array, 3, 2)).toEqual([]);
  });

  test("a stride of one reads single values", () => {
    expect(row({ shape: [3], data: [7, 8, 9] }, 1, 1)).toEqual([8]);
  });

  test("a final partial row is returned as far as it goes", () => {
    expect(row({ shape: [2, 2], data: [1, 2, 3] }, 1, 2)).toEqual([3]);
  });
});
