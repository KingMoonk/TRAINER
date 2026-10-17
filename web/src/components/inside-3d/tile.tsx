"use client";

// One heat-map tile of the thermal look: a real matrix (src/lib/scene.ts SceneTile) as a plane facing the camera, coloured by the
// theme's ramp. The texture is made once and written in place, so a live run's new numbers ease in without a remount; the number
// of texture writes a second is capped, and with reduced motion each new frame is shown as it arrives.

import { Line } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import { Color, DataTexture, NearestFilter, RGBAFormat, SRGBColorSpace } from "three";
import { rampAt, unit } from "@/lib/ramp";
import type { SceneTile } from "@/lib/scene";

/** Texture writes per second at most; a tile is at most 32 x 32, so each is cheap. */
export const TILE_FPS = 20;
/** How fast new numbers ease in (per second); 1 / EASE is about the time to cover two thirds of the way. */
const EASE = 7;
const SETTLED = 1e-3; // of the range

export type TileColors = { mask: string; frame: string };

type Shown = { rows: number; cols: number; value: Float32Array; target: Float32Array; hidden: Float32Array | null; low: number; high: number; goalLow: number; goalHigh: number };

function sizeOf(tile: SceneTile): [number, number] {
  const shape = tile.values?.shape ?? [1, 1];
  return [Math.max(1, shape[0]), Math.max(1, shape[1] ?? 1)];
}

export function Tile({ tile, ramp, colors, still }: { tile: SceneTile; ramp: Uint8Array | null; colors: TileColors; still: boolean }) {
  const [rows, cols] = sizeOf(tile);
  const state = useRef<Shown | null>(null);
  const dirty = useRef(true);
  const clock = useRef(0);
  const texture = useMemo(() => {
    const data = new Uint8Array(rows * cols * 4);
    const made = new DataTexture(data, cols, rows, RGBAFormat);
    made.magFilter = made.minFilter = NearestFilter; // each cell stays a cell
    made.colorSpace = SRGBColorSpace;
    made.needsUpdate = true;
    return made;
  }, [rows, cols]);
  const current = useRef(texture);
  useEffect(() => {
    current.current = texture;
    dirty.current = true;
    return () => texture.dispose();
  }, [texture]);
  const maskRgb = useMemo(() => {
    const hex = colors.mask ? new Color(colors.mask).getHex() : 0;
    return [(hex >> 16) & 255, (hex >> 8) & 255, hex & 255];
  }, [colors.mask]);

  // New numbers from the server become the goal; the first ones (or a changed shape, or a still screen) are shown at once.
  useEffect(() => {
    const incoming = tile.values?.data;
    const target = Float32Array.from(incoming ?? []);
    const hidden = tile.mask ? Float32Array.from(tile.mask.data) : null;
    const same = state.current && state.current.rows === rows && state.current.cols === cols && target.length === state.current.value.length;
    const previous = state.current;
    state.current = {
      rows, cols, target, hidden,
      value: same && !still && previous ? previous.value : target.slice(),
      low: same && !still && previous ? previous.low : tile.range[0],
      high: same && !still && previous ? previous.high : tile.range[1],
      goalLow: tile.range[0], goalHigh: tile.range[1],
    };
    dirty.current = true;
  }, [tile, rows, cols, still]);
  useEffect(() => { dirty.current = true; }, [ramp, maskRgb]);

  useFrame((_, delta) => {
    const shown = state.current;
    if (!shown) return;
    clock.current += delta;
    if (clock.current < 1 / TILE_FPS) return;
    const step = clock.current;
    clock.current = 0;
    let moving = false;
    if (!still && shown.value.length === shown.target.length) {
      const k = 1 - Math.exp(-step * EASE);
      const span = Math.max(1e-12, Math.abs(shown.goalHigh - shown.goalLow));
      let worst = 0;
      for (let i = 0; i < shown.value.length; i++) {
        const move = (shown.target[i] - shown.value[i]) * k;
        shown.value[i] += move;
        worst = Math.max(worst, Math.abs(shown.target[i] - shown.value[i]));
      }
      shown.low += (shown.goalLow - shown.low) * k;
      shown.high += (shown.goalHigh - shown.high) * k;
      worst = Math.max(worst, Math.abs(shown.goalLow - shown.low), Math.abs(shown.goalHigh - shown.high));
      moving = worst > SETTLED * span;
      if (!moving) { shown.value.set(shown.target); shown.low = shown.goalLow; shown.high = shown.goalHigh; }
    } else {
      shown.value.set(shown.target.length === shown.value.length ? shown.target : shown.value);
      shown.low = shown.goalLow;
      shown.high = shown.goalHigh;
    }
    if (!moving && !dirty.current) return;
    dirty.current = false;
    const live = current.current;
    const out = live.image.data as unknown as Uint8Array;
    for (let r = 0; r < shown.rows; r++) {
      const at = (shown.rows - 1 - r) * shown.cols; // a texture's first row is its bottom
      for (let c = 0; c < shown.cols; c++) {
        const i = r * shown.cols + c;
        const [red, green, blue] = !ramp || !tile.values || shown.hidden?.[i] ? maskRgb : rampAt(ramp, unit(shown.value[i], shown.low, shown.high));
        out.set([red, green, blue, 255], (at + c) * 4);
      }
    }
    live.needsUpdate = true;
  });

  const [width, height] = tile.size;
  const outline = useMemo(
    () => [[-width / 2, -height / 2, 0.02], [width / 2, -height / 2, 0.02], [width / 2, height / 2, 0.02], [-width / 2, height / 2, 0.02], [-width / 2, -height / 2, 0.02]] as [number, number, number][],
    [width, height],
  );
  // The plane stands in the scene's x-z plane facing -y, where the camera is: turn three's x-y plane a quarter about x.
  return (
    <group position={tile.origin} rotation={[Math.PI / 2, 0, 0]} name={`tile:${tile.id}`}>
      <mesh>
        <planeGeometry args={[width, height]} />
        <meshBasicMaterial map={texture} toneMapped={false} />
      </mesh>
      {tile.selected && colors.frame && <Line points={outline} color={colors.frame} lineWidth={2} />}
    </group>
  );
}
