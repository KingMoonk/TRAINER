"use client";

import { useEffect, useRef } from "react";
import { heatPixels } from "@/lib/ramp";
import { flatten, lossRange, type LandscapeResult } from "@/lib/landscape";

/** Where a coordinate on an axis sits in the box, 0 to 100 (cells are drawn centred in their share of the box). */
function place(value: number, span: number, grid: number): number {
  return ((((value / span + 1) / 2) * (grid - 1) + 0.5) / grid) * 100;
}

/**
 * The 2D heat-map of a landscape: the fallback for reduced motion and narrow screens, and the picture for anyone who cannot
 * use the 3D one. One cell per grid point, smoothed by the browser, +y up, with the path, the current weights and the lowest cell.
 */
export function HeatMap({ result, ramp }: { result: LandscapeResult; ramp: Uint8Array | null }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const element = canvas.current;
    const context = element?.getContext("2d");
    if (!element || !context || !ramp) return;
    const { grid } = result;
    const { low, high } = lossRange(result);
    const flat = flatten(result);
    const upright = new Float32Array(flat.length);
    for (let row = 0; row < grid; row++) upright.set(flat.subarray(row * grid, (row + 1) * grid), (grid - 1 - row) * grid);
    element.width = element.height = grid;
    context.putImageData(new ImageData(heatPixels(upright, ramp, low, high), grid, grid), 0, 0);
  }, [result, ramp]);
  const at = (x: number, y: number) => ({ cx: place(x, result.span, result.grid), cy: 100 - place(y, result.span, result.grid) });
  const path = result.trajectory.map((point) => { const p = at(point.x, point.y); return `${p.cx},${p.cy}`; }).join(" ");
  const current = at(result.current.x, result.current.y);
  const minimum = at(result.minimum.x, result.minimum.y);
  return <div className="relative aspect-square w-full max-w-xl" data-testid="landscape-heatmap">
    <canvas ref={canvas} className="h-full w-full rounded-lg border border-border" role="img"
      aria-label={`Heat-map of the loss around the current weights. Lowest ${result.minimum.loss.toFixed(2)} nats, here ${result.current.loss.toFixed(2)} nats.`} />
    <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="pointer-events-none absolute inset-0 h-full w-full" aria-hidden="true">
      {result.trajectory.length > 1 && <polyline points={path} fill="none" strokeWidth="1.5" vectorEffect="non-scaling-stroke" style={{ stroke: "var(--landscape-path)" }} />}
      <circle cx={minimum.cx} cy={minimum.cy} r="2.2" fill="none" strokeWidth="2" vectorEffect="non-scaling-stroke" style={{ stroke: "var(--landscape-minimum)" }} />
      <circle cx={current.cx} cy={current.cy} r="1.8" style={{ fill: "var(--landscape-current)", stroke: "var(--landscape-path)" }} strokeWidth="1" vectorEffect="non-scaling-stroke" />
    </svg>
  </div>;
}
