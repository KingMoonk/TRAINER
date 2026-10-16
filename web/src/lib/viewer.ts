import type { HeadStage, Level, Mode, ScenePayload, TileKind } from "@/lib/scene";
import { api, ApiError, unwrap } from "@/lib/api";
import { awaitJob } from "@/lib/job-result";

/** Controls stay in the browser; scenes and snapshots come from the workspace. */
export type SceneQuery = {
  version?: number;
  level: Level;
  mode: Mode;
  headStage: HeadStage;
  tileKind: TileKind;
  snapshot: number;
};

export async function fetchScene(query: SceneQuery, signal: AbortSignal): Promise<ScenePayload> {
  const scene = await unwrap(api.GET("/viewer/scene", { params: { query }, signal }));
  return scene as unknown as ScenePayload;
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
  const job = await unwrap(api.POST("/viewer/probe", { body: { version, text: "The cat sat on the mat because it was tired." }, signal }));
  return awaitJob<ProbeResult>(job, signal);
}
