// Where the Inside 3D page gets its scene. Until the Phase 1 `viewer` router exists (docs/rewrite.md, issue #96) both
// functions read src/fixtures/scene.json, which `studio_scene_json.py` wrote. Phase 1 replaces the bodies with
// `fetchJson` calls to the router; nothing else in the page changes.

import type { HeadStage, Level, Mode, ScenePayload, SceneFrame, TileKind } from "@/lib/scene";
import { applyFrame } from "@/lib/scene";

/** Everything the page asks the viewer for: the controls' state. */
export type SceneQuery = {
  level: Level;
  mode: Mode;
  headStage: HeadStage;
  tileKind: TileKind;
  /** Which snapshot of the run to show, from 0. */
  snapshot: number;
};

type FixtureScene = { payload: ScenePayload; frames: Partial<Record<Mode, SceneFrame>> };
type Fixture = { snapshots: number; scenes: Record<string, FixtureScene> };

let fixture: Promise<Fixture> | undefined;

function loadFixture(): Promise<Fixture> {
  // A dynamic import keeps the fixture out of every other page's bundle.
  fixture ??= import("@/fixtures/scene.json").then((module) => module.default as unknown as Fixture);
  return fixture;
}

/** The fixture holds one payload per level and, at Head and Weights, per step or tile kind. */
function fixtureKey(query: SceneQuery): string {
  const variant = query.level === "Head" ? query.headStage : query.level === "Weights" ? query.tileKind : "-";
  return `${query.snapshot}|${query.level}|${variant}`;
}

/** The scene for `query`. Phase 1: `fetchJson<ScenePayload>("/viewer/scene?...")`. */
export async function fetchScene(query: SceneQuery): Promise<ScenePayload> {
  const { scenes, snapshots } = await loadFixture();
  const scene = scenes[fixtureKey({ ...query, snapshot: Math.max(0, Math.min(snapshots - 1, query.snapshot)) })];
  if (!scene) throw new Error(`The fixture has no scene for ${fixtureKey(query)}.`);
  // The fixture stores Build; the other modes are the frames the serializer made against it, applied as a live client would.
  const frame = scene.frames[query.mode];
  return frame ? applyFrame(scene.payload, frame) : scene.payload;
}

export type ProbeResult = { snapshot: number; note: string };

/**
 * "Run one real step": the server runs one training step on the version's checkpoint and the new snapshot becomes the
 * current one. Phase 1: POST the `probe_step` action and read the snapshot that comes back. The fixture waits as a real
 * step would and answers with its newest snapshot.
 */
export async function runProbeStep(): Promise<ProbeResult> {
  const { snapshots } = await loadFixture();
  await new Promise((resolve) => setTimeout(resolve, 1200));
  return { snapshot: snapshots - 1, note: "Ran one real step. Press Play to watch it pass through the blocks." };
}
