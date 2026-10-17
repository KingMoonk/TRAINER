import { api, unwrap } from "@/lib/api";
import type { components } from "@/lib/api-schema";
import { awaitJob } from "@/lib/job-result";

/** The directions: two random filter-normalised ones, or the two main directions of the training path. */
export type LandscapeMode = "random" | "pca";

export type LandscapeRequest = components["schemas"]["LandscapeRequest"];

export type LandscapePoint = { x: number; y: number; loss: number };

/** POST /viewer/landscape's job result (the server's LandscapeResult; jobs are untyped in the OpenAPI schema). */
export type LandscapeResult = {
  mode: LandscapeMode;
  unit: "filter" | "trajectory";
  grid: number;
  span: number;
  seed: number;
  live: boolean;
  axis: number[];
  /** nats per piece; loss[row][column] is at (axis[column], axis[row]) */
  loss: number[][];
  ceiling: number;
  current: LandscapePoint & { step: number };
  minimum: LandscapePoint & { inside: boolean };
  trajectory: (LandscapePoint & { step: number })[];
  center: { step: number; name: string };
  batch: { rows: number; tokens: number };
  cost: { seconds: number; evaluations: number };
  note: string;
};

export const LIVE_GRID_MAX = 11;
export const GRIDS = [9, 15, 21, 31, 41] as const;

export async function fetchLandscape(body: Partial<LandscapeRequest>, signal: AbortSignal): Promise<LandscapeResult> {
  // The schema lists the defaults as required, so a partial body is cast (the server fills the rest in).
  const job = await unwrap(api.POST("/viewer/landscape", { body: body as LandscapeRequest, signal }));
  return awaitJob<LandscapeResult>(job, signal, { events: true });
}

/** The loss range the colours and heights span: the lowest cell up to the 95th percentile, so steep walls do not flatten the valley. */
export function lossRange(result: LandscapeResult): { low: number; high: number } {
  const low = result.minimum.loss;
  return { low, high: Math.max(result.ceiling, low + 1e-6) };
}

/** The grid flattened row by row (row 0 first), the order a texture and the surface's vertices use. */
export function flatten(result: LandscapeResult): Float32Array {
  const out = new Float32Array(result.grid * result.grid);
  result.loss.forEach((row, r) => row.forEach((value, c) => { out[r * result.grid + c] = value; }));
  return out;
}

/** Cells x parameters is what the server caps; the page shows the cells and the seconds, not the product. */
export function costLine(result: LandscapeResult): string {
  const seconds = result.cost.seconds < 10 ? result.cost.seconds.toFixed(1) : Math.round(result.cost.seconds).toString();
  return `${result.cost.evaluations.toLocaleString()} loss evaluations in ${seconds} s on ${result.batch.tokens} tokens of held-back text`;
}

/** What the unit of an axis means, for the axis label. */
export function axisUnit(result: LandscapeResult): string {
  return result.unit === "filter" ? "filter norms" : "path lengths";
}
