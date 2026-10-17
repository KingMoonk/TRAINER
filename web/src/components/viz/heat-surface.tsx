"use client";

// The 3D surface of a loss landscape: height and colour are the loss (the theme's heat ramp, lowest cell to the 95th
// percentile), a wireframe over it, the path of training, the current weights at the middle and the lowest cell. It is
// loaded client-only (WebGL does not exist while the static export prerenders). The canvas renders on demand: nothing
// moves, so nothing draws, until the data or the camera changes; a new frame eases into place unless motion is reduced.

import { Line, OrbitControls } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useEffect, useLayoutEffect, useMemo, useRef } from "react";
import { BufferAttribute, Color, DoubleSide, PlaneGeometry, SRGBColorSpace } from "three";
import { useThemeColor, useThemeColors } from "@/components/theme";
import { flatten, lossRange, type LandscapeResult } from "@/lib/landscape";
import { rampAt, unit } from "@/lib/ramp";

const HALF = 5; // the surface spans -5..5 on both ground axes
const HEIGHT = 4; // and 0..4 up
const EASE = 0.22; // the share of the way to the new height each frame
const MARKERS = ["--landscape-wire", "--landscape-path", "--landscape-current", "--landscape-minimum"] as const;

type Props = { result: LandscapeResult; ramp: Uint8Array; reduceMotion: boolean };

export default function HeatSurface({ result, ramp, reduceMotion }: Props) {
  const background = useThemeColor("--scene-background");
  const [wire, path, current, minimum] = useThemeColors(MARKERS);
  if (!background || !wire || !path || !current || !minimum) return null; // the tokens are read after the first client render
  return <Canvas frameloop="demand" dpr={[1, 2]} camera={{ fov: 40, near: 0.1, far: 200, position: [9, 7.5, 10] }} className="touch-none"
    aria-label="3D surface of the loss around the current weights. Drag to turn it, scroll to zoom.">
    <color attach="background" args={[background]} />
    <Surface result={result} ramp={ramp} wire={wire} reduceMotion={reduceMotion} />
    <Markers result={result} path={path} current={current} minimum={minimum} />
    <OrbitControls enableDamping={!reduceMotion} target={[0, 1.4, 0]} maxPolarAngle={Math.PI * 0.495} minDistance={6} maxDistance={40} />
  </Canvas>;
}

function Surface({ result, ramp, wire, reduceMotion }: Props & { wire: string }) {
  const { grid } = result;
  const invalidate = useThree((state) => state.invalidate);
  const geometry = useMemo(() => {
    const plane = new PlaneGeometry(1, 1, grid - 1, grid - 1);
    plane.setAttribute("color", new BufferAttribute(new Float32Array(grid * grid * 3), 3));
    return plane;
  }, [grid]);
  useEffect(() => () => geometry.dispose(), [geometry]);
  const aim = useRef(new Float32Array(0)); // the heights the surface is heading for
  const shown = useRef(new Float32Array(0)); // the heights it shows now

  useLayoutEffect(() => {
    const { low, high } = lossRange(result);
    const losses = flatten(result);
    const ground = geometry.getAttribute("position") as BufferAttribute;
    const colours = geometry.getAttribute("color") as BufferAttribute;
    const paint = new Color();
    const next = new Float32Array(losses.length);
    for (let i = 0; i < losses.length; i++) {
      const t = unit(losses[i], low, high);
      next[i] = t * HEIGHT;
      const [r, g, b] = rampAt(ramp, t);
      paint.setRGB(r / 255, g / 255, b / 255, SRGBColorSpace);
      colours.setXYZ(i, paint.r, paint.g, paint.b);
      ground.setX(i, (result.axis[i % grid] / result.span) * HALF);
      ground.setZ(i, (result.axis[Math.floor(i / grid)] / result.span) * HALF);
    }
    colours.needsUpdate = true;
    aim.current = next;
    if (reduceMotion || shown.current.length !== next.length) {
      shown.current = next.slice();
      for (let i = 0; i < next.length; i++) ground.setY(i, next[i]);
      ground.needsUpdate = true;
      geometry.computeBoundingSphere();
    }
    invalidate();
  }, [result, ramp, geometry, grid, reduceMotion, invalidate]);

  useFrame(() => {
    const from = shown.current;
    const to = aim.current;
    if (from.length !== to.length) return;
    let moving = false;
    const ground = geometry.getAttribute("position") as BufferAttribute;
    for (let i = 0; i < to.length; i++) {
      const gap = to[i] - from[i];
      if (Math.abs(gap) > 0.002) { from[i] += gap * EASE; moving = true; } else from[i] = to[i];
      ground.setY(i, from[i]);
    }
    if (moving) { ground.needsUpdate = true; invalidate(); }
  });

  return <group>
    <mesh geometry={geometry}>
      <meshBasicMaterial vertexColors side={DoubleSide} toneMapped={false} polygonOffset polygonOffsetFactor={1} polygonOffsetUnits={1} />
    </mesh>
    <mesh geometry={geometry}>
      <meshBasicMaterial wireframe color={wire} toneMapped={false} />
    </mesh>
  </group>;
}

function Markers({ result, path, current, minimum }: { result: LandscapeResult; path: string; current: string; minimum: string }) {
  const { low, high } = lossRange(result);
  const place = (point: { x: number; y: number; loss: number }, lift = 0.12): [number, number, number] =>
    [(point.x / result.span) * HALF, unit(point.loss, low, high) * HEIGHT + lift, (point.y / result.span) * HALF];
  const trail = result.trajectory.map((point) => place(point));
  return <group>
    {trail.length > 1 && <Line points={trail} color={path} lineWidth={2} />}
    {trail.map((position, i) => <mesh key={result.trajectory[i].step} position={position}>
      <sphereGeometry args={[0.07, 10, 8]} /><meshBasicMaterial color={path} toneMapped={false} />
    </mesh>)}
    <mesh position={place(result.minimum, 0.25)}>
      <octahedronGeometry args={[0.22]} /><meshBasicMaterial color={minimum} toneMapped={false} />
    </mesh>
    <mesh position={place(result.current, 0.25)}>
      <sphereGeometry args={[0.2, 16, 12]} /><meshBasicMaterial color={current} toneMapped={false} />
    </mesh>
  </group>;
}
