"use client";

// The react-three-fiber half of Inside 3D: draws a ScenePayload (src/lib/scene.ts). It is loaded client-only by
// inside-3d.tsx, because WebGL does not exist while the static export prerenders.

import { Line, OrbitControls } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useEffect, useLayoutEffect, useMemo, useRef, type ComponentRef, type MutableRefObject } from "react";
import { Color, InstancedMesh, Matrix4, Quaternion, Vector3, type Mesh, type MeshBasicMaterial } from "three";
import {
  pulsePose,
  type BoxEntry,
  type LineEntry,
  type PaletteEntry,
  type ScenePayload,
  type SceneGroup,
  type SceneLabel,
} from "@/lib/scene";

// The scene is z-up (x right, y into the screen); three.js is y-up. Rotating the whole scene by -90 degrees about x
// turns scene (x, y, z) into three (x, z, -y), so every position in the payload is used as it arrives.
const Z_UP: [number, number, number] = [-Math.PI / 2, 0, 0];
const HEADING = (25 * Math.PI) / 180;
const PITCH = (32 * Math.PI) / 180;

type Palette = ScenePayload["palette"];

export type SceneCanvasProps = {
  scene: ScenePayload;
  playing: boolean;
  reduceMotion: boolean;
  showNumbers: boolean;
  /** The pulse went forward and back through every block (`playback.pass_seconds` after Play): the page moves to the next snapshot. */
  onPassDone: () => void;
};

export default function SceneCanvas({ scene, playing, reduceMotion, showNumbers, onPassDone }: SceneCanvasProps) {
  const background = scene.palette.background.color;
  const pulse = useRef(new Vector3());
  const elements = useRef(new Map<string, HTMLElement>());
  const labels = useMemo(() => sceneLabels(scene, showNumbers), [scene, showNumbers]);
  return (
    <div className="relative h-full w-full overflow-hidden">
      <Canvas
        camera={{ fov: 40, near: 0.1, far: 500, position: [0, 12, 30] }}
        dpr={[1, 2]}
        className="touch-none"
        aria-label="3D view of the network"
      >
        <color attach="background" args={[background]} />
        <ambientLight intensity={1.1} />
        <directionalLight position={[8, 14, 10]} intensity={1.6} />
        <group rotation={Z_UP}>
          {Object.entries(scene.groups).map(([key, group]) => (
            <Group key={key} group={group} palette={scene.palette} />
          ))}
          <Replay scene={scene} playing={playing} reduceMotion={reduceMotion} onPassDone={onPassDone} position={pulse} />
        </group>
        <CameraRig scene={scene} playing={playing} pulse={pulse} />
        <LabelProjector labels={labels} elements={elements} />
      </Canvas>
      <div className="pointer-events-none absolute inset-0 overflow-hidden" aria-hidden>
        {labels.map((label) => (
          <Label
            key={label.key}
            label={label}
            palette={scene.palette[label.kind]}
            register={(element) => {
              if (element) elements.current.set(label.key, element);
              else elements.current.delete(label.key);
            }}
          />
        ))}
      </div>
    </div>
  );
}

function Group({ group, palette }: { group: SceneGroup; palette: Palette }) {
  return (
    <group position={group.origin}>
      {group.boxes.map((entry, i) => (
        <Boxes key={`${entry.kind}-${i}`} entry={entry} palette={palette[entry.kind]} />
      ))}
      {group.lines.map((entry, i) => (
        <Lines key={`${entry.kind}-${i}`} entry={entry} palette={palette[entry.kind]} />
      ))}
    </group>
  );
}

/** `entry.count` boxes of one kind as one draw call. */
function Boxes({ entry, palette }: { entry: BoxEntry; palette: PaletteEntry }) {
  const mesh = useRef<InstancedMesh>(null);
  const opacity = entry.alpha.data.length ? entry.alpha.data[0] : palette.alpha;
  useLayoutEffect(() => {
    const target = mesh.current;
    if (!target) return;
    const matrix = new Matrix4();
    const rotation = new Quaternion();
    const position = new Vector3();
    const scale = new Vector3();
    const color = new Color();
    for (let i = 0; i < entry.count; i++) {
      position.fromArray(entry.position.data, i * 3);
      scale.fromArray(entry.size.data, i * 3);
      target.setMatrixAt(i, matrix.compose(position, rotation, scale));
      if (entry.color) target.setColorAt(i, color.setRGB(entry.color.data[i * 4], entry.color.data[i * 4 + 1], entry.color.data[i * 4 + 2]));
    }
    target.instanceMatrix.needsUpdate = true;
    if (target.instanceColor) target.instanceColor.needsUpdate = true;
  }, [entry]);
  // `args` fixes the instance count, so a different count makes a new mesh.
  return (
    <instancedMesh ref={mesh} args={[undefined, undefined, entry.count]} key={entry.count} frustumCulled={false}>
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial color={entry.color ? "#ffffff" : palette.color} transparent={opacity < 1} opacity={opacity} />
    </instancedMesh>
  );
}

function Lines({ entry, palette }: { entry: LineEntry; palette: PaletteEntry }) {
  const per = entry.count ? entry.points.data.length / entry.count / 3 : 0;
  const lines = useMemo(
    () =>
      Array.from({ length: entry.count }, (_, i) => ({
        points: Array.from({ length: per }, (_, p) => {
          const at = (i * per + p) * 3;
          return [entry.points.data[at], entry.points.data[at + 1], entry.points.data[at + 2]] as [number, number, number];
        }),
        width: entry.width.data[i],
      })),
    [entry, per],
  );
  return (
    <>
      {lines.map((line, i) => (
        <Line
          key={i}
          points={line.points}
          color={palette.color}
          lineWidth={line.width}
          transparent={palette.alpha < 1}
          opacity={palette.alpha}
        />
      ))}
    </>
  );
}

type PlacedLabel = SceneLabel & { key: string; world: Vector3 };

/** Every label in the scene with its position in three.js world space (scene (x, y, z) is three (x, z, -y)). */
function sceneLabels(scene: ScenePayload, showNumbers: boolean): PlacedLabel[] {
  return Object.entries(scene.groups).flatMap(([key, group]) =>
    group.labels
      .map((label, i) => ({ label, i }))
      .filter(({ label }) => showNumbers || label.kind !== "token")
      .map(({ label, i }) => {
        const [x, y, z] = [0, 1, 2].map((axis) => group.origin[axis] + label.position[axis]);
        return { ...label, key: `${key}/${i}`, world: new Vector3(x, z, -y) };
      }),
  );
}

/**
 * Labels are plain DOM over the canvas, so they stay sharp and need no font download. (drei's `Html` unmounts a React
 * root while React is rendering, which React 19 reports as an error in development.) `LabelProjector` moves them every frame.
 */
function Label({ label, palette, register }: { label: PlacedLabel; palette: PaletteEntry; register: (element: HTMLElement | null) => void }) {
  const small = label.kind === "token";
  return (
    <div
      ref={register}
      className={`absolute left-0 top-0 whitespace-nowrap rounded-sm font-mono will-change-transform ${small ? "px-1 text-[10px]" : "px-2 py-0.5 text-xs"}`}
      style={{
        color: palette.color,
        background: small ? "transparent" : `color-mix(in srgb, ${palette.frame} ${(palette.frame_alpha ?? 1) * 100}%, transparent)`,
        visibility: "hidden",
      }}
    >
      {label.text}
    </div>
  );
}

function LabelProjector({ labels, elements }: { labels: PlacedLabel[]; elements: MutableRefObject<Map<string, HTMLElement>> }) {
  const projected = useMemo(() => new Vector3(), []);
  useFrame(({ camera, size }) => {
    for (const label of labels) {
      const element = elements.current.get(label.key);
      if (!element) continue;
      projected.copy(label.world).project(camera);
      const visible = projected.z > -1 && projected.z < 1;
      element.style.visibility = visible ? "visible" : "hidden";
      if (visible) {
        const x = ((projected.x + 1) / 2) * size.width;
        const y = ((1 - projected.y) / 2) * size.height;
        element.style.transform = `translate(${x}px, ${y}px) translate(-50%, -50%)`;
      }
    }
  });
  return null;
}

/**
 * The replay pulse: one cube that runs forward through every block and back, two blocks a second, white on the way
 * forward and orange on the way back. It also times Play: when it has gone forward and back, `onPassDone` moves to the
 * next snapshot (at every level, as the Panda3D viewer does). `position` is where it is, in scene coordinates.
 */
function Replay({
  scene,
  playing,
  reduceMotion,
  onPassDone,
  position,
}: Pick<SceneCanvasProps, "scene" | "playing" | "reduceMotion" | "onPassDone"> & { position: { current: Vector3 } }) {
  const { playback, view } = scene;
  const pulse = playback.pulse;
  const cube = useRef<Mesh>(null);
  const elapsed = useRef(0);
  const done = useRef(onPassDone);
  useEffect(() => {
    done.current = onPassDone;
  });
  useEffect(() => {
    elapsed.current = 0;
  }, [playing, playback.index]);

  useFrame((_, delta) => {
    const mesh = cube.current;
    if (!mesh) return;
    if (playing) elapsed.current += Math.min(0.1, delta);
    const pose = playing ? pulsePose(pulse.blocks, elapsed.current, reduceMotion) : { block: 0, depth: 0, forward: true };
    position.current.set(pulse.token_x, -pose.depth * pulse.spacing, pulse.z);
    mesh.position.copy(position.current);
    const size = playing ? pulse.sizes[pose.forward ? "forward" : "backward"]?.[pose.block] : null;
    if (size == null) mesh.scale.set(...pulse.idle_size);
    else mesh.scale.set(size, size, size);
    const color = pose.forward ? pulse.colors.forward : pulse.colors.backward;
    (mesh.material as MeshBasicMaterial).color.setRGB(color[0], color[1], color[2]);
    if (playing && elapsed.current >= playback.pass_seconds) {
      elapsed.current = 0;
      done.current();
    }
  });
  return (
    <mesh ref={cube} visible={view.level === "Model" || view.level === "Window"}>
      <boxGeometry args={[1, 1, 1]} />
      <meshBasicMaterial />
    </mesh>
  );
}

/**
 * Orbit camera. It frames each level when the level changes (`view.camera`, the Panda3D viewer's own framing), keeping the
 * user's angle; in "Ride a token" it follows the pulse while Play runs.
 */
function CameraRig({ scene, playing, pulse }: { scene: ScenePayload; playing: boolean; pulse: { current: Vector3 } }) {
  const controls = useRef<ComponentRef<typeof OrbitControls>>(null);
  const camera = useThree((state) => state.camera);
  const { level, camera: framing } = scene.view;
  const follow = scene.playback.pulse.follow && playing && (level === "Model" || level === "Window");
  const angle = useRef({ heading: HEADING, pitch: PITCH });
  const [tx, ty, tz] = framing.target;
  // a narrow canvas needs the camera further back to fit the same width; steps of a quarter keep a resize from reframing every pixel
  const aspect = useThree((state) => Math.max(0.25, Math.round((state.size.width / Math.max(1, state.size.height)) * 4) / 4));
  const distance = framing.distance * Math.max(1, 1.2 / aspect);
  const offset = useMemo(() => new Vector3(), []);
  const goal = useMemo(() => new Vector3(), []);

  // scene (x, y, z) is three (x, z, -y); the camera sits on the -y side so x runs left to right as it reads.
  useLayoutEffect(() => {
    const orbit = controls.current;
    if (!orbit) return;
    const { heading, pitch } = angle.current;
    orbit.target.set(tx, tz, -ty);
    camera.position.set(
      tx + distance * Math.sin(heading) * Math.cos(pitch),
      tz + distance * Math.sin(pitch),
      -ty + distance * Math.cos(heading) * Math.cos(pitch),
    );
    orbit.update();
  }, [level, tx, ty, tz, distance, camera]); // frame again when the level (or its framing) changes, not on every snapshot

  useFrame(() => {
    const orbit = controls.current;
    if (!orbit) return;
    if (follow) {
      goal.set(pulse.current.x, pulse.current.z, -pulse.current.y);
      offset.copy(goal).sub(orbit.target);
      orbit.target.add(offset);
      camera.position.add(offset);
    }
    // remember the user's angle for the next framing
    offset.copy(camera.position).sub(orbit.target);
    const length = offset.length() || 1;
    angle.current = { heading: Math.atan2(offset.x, offset.z), pitch: Math.asin(Math.max(-1, Math.min(1, offset.y / length))) };
  });

  return <OrbitControls ref={controls} makeDefault enableDamping dampingFactor={0.12} minDistance={2} maxDistance={160} />;
}
