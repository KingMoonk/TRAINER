import type { HeadStage, Level, Mode, ScenePayload, TileKind } from "@/lib/scene";
import { api, ApiError, unwrap } from "@/lib/api";
import type { components } from "@/lib/api-schema";
import { awaitJob } from "@/lib/job-result";

/** Controls stay in the browser; scenes and snapshots come from the workspace. */
export type SceneQuery = {
  version?: number;
  level: Level;
  mode: Mode;
  headStage: HeadStage;
  tileKind: TileKind;
  snapshot: number;
  /** Heat-map tiles of the real matrices instead of bar matrices. */
  thermal?: boolean;
};

export async function fetchScene(query: SceneQuery, signal: AbortSignal): Promise<ScenePayload> {
  const scene = await unwrap(api.GET("/viewer/scene", { params: { query }, signal }));
  return scene as unknown as ScenePayload;
}

/** The snapshots a scene can show, oldest first; `SceneQuery.snapshot` is an index into this list. */
export type SnapshotEntry = components["schemas"]["SnapshotOut"];

export function fetchSnapshots(version: number | undefined, signal: AbortSignal): Promise<SnapshotEntry[]> {
  return unwrap(api.GET("/viewer/snapshots", { params: { query: { version } }, signal }));
}

export type ProbeResult = { snapshot: number; note: string };

export function viewerError(error: unknown): string {
  if (error instanceof ApiError) {
    try {
      const body = JSON.parse(error.body) as { detail?: unknown };
      if (typeof body.detail === "string") return body.detail;
    } catch { /* Fall back to the request error for a non-JSON response. */ }
  }
  return error instanceof Error ? error.message : String(error);
}

export async function runProbeStep(version: number | undefined, signal: AbortSignal): Promise<ProbeResult> {
  // No `text`: the server probes with its own sentence (the schema lists the default as required, so the body is cast).
  const body = { version } as components["schemas"]["ProbeRequest"];
  const job = await unwrap(api.POST("/viewer/probe", { body, signal }));
  return awaitJob<ProbeResult>(job, signal, { events: true });
}
