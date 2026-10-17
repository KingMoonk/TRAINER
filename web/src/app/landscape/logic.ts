import { LIVE_GRID_MAX } from "@/lib/landscape";

/** The grid the page asks for: live frames are small whatever the picker says. */
export function requestedGrid(grid: number, live: boolean): number {
  return live ? Math.min(grid, LIVE_GRID_MAX) : grid;
}

/** A training version's id for the title line, or "" when none is picked. */
export function versionLabel(version: number | undefined, name?: string | null): string {
  return version === undefined ? "" : `Version ${version}${name ? ` · ${name}` : ""}`;
}

/** True when an event from the server is a saved training snapshot. */
export function isSnapshotEvent(payload: unknown): boolean {
  return !!payload && typeof payload === "object" && "type" in payload && payload.type === "viz_snapshot";
}
