// The heat ramp: nine viridis stops from the theme (--ramp-1 .. --ramp-9, low to high), blended into a lookup table that
// three.js vertex colours, textures and 2D canvases all read, so one number is one colour everywhere.

export const RAMP_TOKENS = ["--ramp-1", "--ramp-2", "--ramp-3", "--ramp-4", "--ramp-5", "--ramp-6", "--ramp-7", "--ramp-8", "--ramp-9"] as const;
export const RAMP_STEPS = 256;

/** RAMP_STEPS rgb triples (0-255) blended between the stops, or null while a stop has not been resolved yet. */
export function buildRamp(stops: readonly string[]): Uint8Array | null {
  const rgb = stops.map((stop) => /^#([0-9a-f]{6})$/i.exec(stop)?.[1]).map((hex) => hex ? [0, 2, 4].map((i) => parseInt(hex.slice(i, i + 2), 16)) : null);
  if (rgb.length < 2 || rgb.some((stop) => stop === null)) return null;
  const table = new Uint8Array(RAMP_STEPS * 3);
  for (let i = 0; i < RAMP_STEPS; i++) {
    const at = (i / (RAMP_STEPS - 1)) * (rgb.length - 1);
    const low = Math.min(Math.floor(at), rgb.length - 2);
    const blend = at - low;
    for (let c = 0; c < 3; c++) table[i * 3 + c] = Math.round(rgb[low]![c] * (1 - blend) + rgb[low + 1]![c] * blend);
  }
  return table;
}

/** Where a value sits between `low` and `high`, 0 to 1 (0 when the range is empty or the value is not finite). */
export function unit(value: number, low: number, high: number): number {
  if (!Number.isFinite(value) || !(high > low)) return 0;
  return Math.min(1, Math.max(0, (value - low) / (high - low)));
}

/** The ramp's colour at t (0 to 1) as [r, g, b] in 0-255. */
export function rampAt(table: Uint8Array, t: number): [number, number, number] {
  const i = Math.round(Math.min(1, Math.max(0, t)) * (RAMP_STEPS - 1)) * 3;
  return [table[i], table[i + 1], table[i + 2]];
}

/**
 * A heat-map texture's pixels: `values` is rows x columns (row 0 first), shown from `low` (ramp start) to `high` (ramp end).
 * Returns RGBA bytes, row 0 at the top, ready for `new ImageData(...)` or a three.js DataTexture.
 */
export function heatPixels(values: ArrayLike<number>, table: Uint8Array, low: number, high: number): Uint8ClampedArray<ArrayBuffer> {
  const pixels = new Uint8ClampedArray(new ArrayBuffer(values.length * 4));
  for (let i = 0; i < values.length; i++) {
    const [r, g, b] = rampAt(table, unit(values[i], low, high));
    pixels.set([r, g, b, 255], i * 4);
  }
  return pixels;
}
