// The Inside 3D scene as the server sends it: types for `studio_scene_json.py` (the scene-to-JSON serializer that the
// Phase 1 `viewer` router will return). Coordinates are the viewer's own: x right, y into the screen, z up (`up: "z"`).
// A box is centred on its position and `size` is its full extent. Colours are kind ids; `palette` maps each to a colour.

export type Vec3 = [number, number, number];

export const LEVELS = ["Model", "Window", "Head", "Weights"] as const;
export const MODES = ["Build", "Train", "Ride a token"] as const;
export const HEAD_STAGES = ["scores", "scale", "mask", "pattern", "values"] as const;
export const TILE_KINDS = ["weight", "gradient", "update"] as const;

export type Level = (typeof LEVELS)[number];
export type Mode = (typeof MODES)[number];
export type HeadStage = (typeof HEAD_STAGES)[number];
export type TileKind = (typeof TILE_KINDS)[number];

/** A row-major float array. The server can also send `{ shape, name }` (a slice of its binary buffer); this client asks for inline numbers. */
export type NumberArray = { shape: number[]; data: number[] };

export type BoxKind =
  | "activation"
  | "channel"
  | "token"
  | "gate"
  | "positive"
  | "negative"
  | "masked"
  | "picked"
  | "update";
export type LineKind = "mixer" | "lane" | "connector";
export type LabelKind = "label" | "label_selected" | "token" | "tile_title" | "tile_dims" | "tile_axis";
export type PaletteKey = BoxKind | LineKind | LabelKind | "background";

export type PaletteEntry = { color: string; alpha: number; frame?: string; frame_alpha?: number };

/** `count` boxes of one kind. Optional columns (`value`, `cell`, `color`) are present when every box of the entry has them. */
export type BoxEntry = {
  kind: BoxKind;
  count: number;
  position: NumberArray; // [count, 3]
  size: NumberArray; // [count, 3]
  alpha: NumberArray; // [count]
  value?: NumberArray; // [count]: a matrix value, or an update bar's brightness
  cell?: NumberArray; // [count, 2]: row, column in the visible tile
  color?: NumberArray; // [count, 4]: rgba in 0..1, for `update` boxes
};

/** `count` polylines of one kind, each `points` row a run of 3D points. */
export type LineEntry = {
  kind: LineKind;
  count: number;
  points: NumberArray; // [count, points per line, 3]
  width: NumberArray; // [count]
};

export type SceneLabel = { text: string; position: Vec3; kind: LabelKind; selected: boolean; size: number };

/**
 * A real matrix drawn as a heat-map plane facing the camera (the thermal look). `values` is rows x columns, at most 32 each way
 * (block-averaged when `shape`, the true size, is bigger); `range` is the ramp's low and high over the whole matrix. A `mask` of 1
 * marks a hidden cell (a later key). A tile keeps its `id` from frame to frame, so a client updates it in place. No `values`: no data yet.
 */
export type SceneTile = {
  id: string;
  label: string;
  origin: Vec3; // the centre, in the group
  size: [number, number]; // width, height
  selected: boolean;
  shape: number[];
  range: [number, number];
  values?: NumberArray;
  mask?: NumberArray;
};

type GroupBase = { origin: Vec3; boxes: BoxEntry[]; lines: LineEntry[]; labels: SceneLabel[]; tiles?: SceneTile[] };

/** One block at the Model and Window levels, keyed `gate:<index>`. */
export type GateGroup = GroupBase & { index: number; selected: boolean };

/** The matrix at the Head and Weights levels, keyed `matrix`. */
export type MatrixGroup = GroupBase & {
  level: "Head" | "Weights";
  kind: HeadStage | TileKind;
  weight_name: string | null;
  shape: number[] | null;
  origin_cell: [number, number];
  pitch: number;
  caption: string;
};

export type SceneGroup = GateGroup | MatrixGroup;

export function isMatrixGroup(group: SceneGroup): group is MatrixGroup {
  return "pitch" in group;
}

export type ControlRow = "level" | "mode" | "mixer" | "stage" | "tile" | "matrix";

export type SceneView = {
  level: Level;
  mode: Mode;
  head_stage: HeadStage;
  tile_kind: TileKind;
  /** The server drew heat-map tiles (the `thermal` control); absent from servers that predate it. */
  thermal?: boolean;
  reduce_motion: boolean;
  show_numbers: boolean;
  levels: Level[];
  modes: Mode[];
  head_stages: HeadStage[];
  tile_kinds: TileKind[];
  controls: {
    rows: ControlRow[];
    matrices: string[];
    snapshots: { show: boolean; probe: string; probe_enabled: boolean; note: string };
    toggles: ("motion" | "numbers")[];
  };
};

/** How a level is framed (the Panda3D viewer's own framing). Only the whole scene carries it: a frame keeps the user's orbit. */
export type SceneCamera = { target: Vec3; distance: number; level: Level };

export type Selection = {
  run: string | null;
  layer: number;
  token: number;
  head: number;
  weight: string | null;
  row: number;
  column: number;
  block_id: string;
  cell: [number, number];
};

export type TrainingStats = {
  step?: number;
  total?: number;
  loss?: number;
  lr?: number;
  tokens_per_second?: number;
};

export type SceneStats = {
  step: number | null;
  loss: number | null;
  info: string;
  training: TrainingStats;
  training_line: string;
  note: string;
};

export type ModuleEntry = { id: string; config: Record<string, unknown> };
export type Glyph = { kind: string; hooks: string[] };

export type DesignGate = {
  index: number;
  id: string;
  label: string;
  mixer: ModuleEntry;
  channel: ModuleEntry;
  glyphs: { mixer: Glyph; channel: Glyph };
};

export type SceneDesign = { name: string; d_model: number; n_layer: number; n_ctx: number; gates: DesignGate[] };

export type RGBA = [number, number, number, number];

/** What a renderer needs to animate the replay pulse (see `pulsePose`). */
export type Pulse = {
  animate: boolean;
  blocks: number;
  spacing: number;
  speed: number;
  z: number;
  follow: boolean; // "Ride a token": the camera follows the pulse
  token_x: number;
  colors: { forward: RGBA; backward: RGBA };
  idle_size: Vec3;
  sizes: Partial<Record<"forward" | "backward", (number | null)[]>>; // edge length per block
};

export type Playback = {
  show: boolean;
  play: "Play" | "Pause";
  position: string;
  previous: boolean;
  next: boolean;
  count: number;
  index: number;
  playing: boolean;
  pass_seconds: number;
  pulse: Pulse;
};

/** The section ids a frame is cursored by; `groups/<key>` is one entry of `groups`. */
export type Cursor = Record<string, string>;

type Sections = {
  view: SceneView;
  selection: Selection;
  stats: SceneStats;
  design: SceneDesign;
  playback: Playback;
};

/** The whole scene to draw from scratch (`scene_payload`). */
export type ScenePayload = Sections & {
  schema: number;
  full: true;
  up: "z";
  palette: Record<PaletteKey, PaletteEntry>;
  camera: SceneCamera;
  groups: Record<string, SceneGroup>;
  removed_groups: string[];
  cursor: Cursor;
};

/** Only what changed since the client's last cursor (`scene_frame`). Apply it with `applyFrame`. */
export type SceneFrame = Partial<Sections> & {
  schema: number;
  full: false;
  groups: Record<string, SceneGroup>;
  removed_groups: string[];
  cursor: Cursor;
};

/** Replace each section the frame carries, replace the groups it carries, drop `removed_groups`, keep the new cursor. */
export function applyFrame(scene: ScenePayload, frame: SceneFrame): ScenePayload {
  const { groups, removed_groups, cursor, schema, full: _full, ...sections } = frame;
  void _full;
  const kept = Object.fromEntries(Object.entries(scene.groups).filter(([key]) => !removed_groups.includes(key)));
  return { ...scene, ...sections, schema, groups: { ...kept, ...groups }, cursor, removed_groups: [] };
}

/**
 * (block, depth, forward) of the replay pulse `elapsed` seconds into a snapshot; depth is the y distance in block
 * spacings. The same as `pulse_pose` in studio_scene_json.py.
 */
export function pulsePose(blocks: number, elapsed: number, reduceMotion: boolean) {
  const span = 2 * blocks;
  const phase = (((elapsed * 2) % span) + span) % span;
  const forward = phase < blocks;
  const block = Math.min(blocks - 1, Math.floor(forward ? phase : span - phase - 1e-6));
  return { block, depth: reduceMotion ? block : forward ? phase : span - phase, forward };
}

/** Reads a `[count, ...]` array as the flat numbers of row `index` (a stride of `width` numbers per row). */
export function row(array: NumberArray, index: number, width: number): number[] {
  return array.data.slice(index * width, (index + 1) * width);
}
