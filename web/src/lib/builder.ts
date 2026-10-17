// The Builder's draft and its maths, mirroring studio_ia/pages/builder.py: the draft is the page's state (and a copy in
// localStorage so a reload keeps it); every sum comes from the stateless /design routes (studio_api/routers/design.py).
// Nothing here touches the DOM: the component (components/builder) renders what these functions return.

import { api, apiFetch, ApiError, unwrap } from "@/lib/api";
import type { components } from "@/lib/api-schema";
import { awaitJob } from "@/lib/job-result";
import { BUILDER, fill } from "@/lib/copy/builder";

export type Choices = components["schemas"]["Choices"];
export type Spec = Record<string, unknown>;
export type Registry = components["schemas"]["Registry"];
export type ModuleInfo = components["schemas"]["ModuleInfo"];
export type Tower = components["schemas"]["Tower"];
export type TowerBlock = components["schemas"]["Block"];
export type Sizes = components["schemas"]["Sizes"];
export type Preset = components["schemas"]["Preset"];
export type PresetCost = components["schemas"]["PresetCost"];
export type Line = components["schemas"]["Line"];
export type Expectations = components["schemas"]["Expectations"];
export type Changes = components["schemas"]["Changes"];
export type Option = components["schemas"]["Option"];
export type Question = components["schemas"]["Question"];
export type SplitStep = components["schemas"]["SplitStep"];
export type SplitPreview = components["schemas"]["SplitPreview"];
export type LearningCheck = components["schemas"]["LearningCheck"];
/** A guess: {error: "lower", step: "faster"}; a question left alone has no key. Skipped is the empty guess. */
export type Guess = Record<string, string>;
export type Version = components["schemas"]["Version"];
export type Slot = components["schemas"]["OptionsBody"]["slot"];
export type Preview = components["schemas"]["ComputeBody"]["preview"];
export type BlockSlot = "norm" | "mixer" | "channel";

export const PART_SLOTS: Slot[] = ["embedding", "position", "head"];
/** A block's sub-layers top-down, as the Tk page lists them; null is the add-back (residual) line. */
export const BLOCK_LINES: (BlockSlot | null)[] = [null, "channel", "norm", null, "mixer", "norm"];

/** A design loaded from elsewhere (a preset, an earlier version): the draft is written onto it, not onto the current version. */
export type Anchor = { spec: Spec; was: Choices };

/** The page's draft: what it edits, against which version. `base` is the version's number, null before the first save. */
export type Draft = { base: number | null; draft: Choices; anchor: Anchor | null };

// ------------------------------------------------------------------------------------------------ the /design calls

export const design = {
  registry: () => unwrap(api.GET("/design/registry")),
  presets: () => unwrap(api.GET("/design/presets")),
  read: (spec: Spec) => unwrap(api.POST("/design/read", { body: { spec } })),
  write: (base: Spec, choices: Choices, was: Choices | null) => unwrap(api.POST("/design/write", { body: { base, choices, was } })),
  choose: (choices: Choices, slot: Slot, short: string, block: number | null) =>
    unwrap(api.POST("/design/choose", { body: { choices, slot, short, block } })),
  resize: (choices: Choices, n: number) => unwrap(api.POST("/design/resize", { body: { choices, n } })),
  tower: (choices: Choices) => unwrap(api.POST("/design/tower", { body: { choices } })),
  partOptions: (base: Spec, choices: Choices, was: Choices | null, slot: Slot, block: number | null) =>
    unwrap(api.POST("/design/part-options", { body: { base, choices, was, slot, block } })),
  sizes: (spec: Spec, v: number) => unwrap(api.POST("/design/sizes", { body: { spec, v } })),
  presetCost: (spec: Spec) => unwrap(api.POST("/design/preset-cost", { body: { spec } })),
  computeLines: (spec: Spec, v: number, preview: Preview) =>
    unwrap(api.POST("/design/compute-lines", { body: { spec, v, preview } })),
  expectations: (cur: Choices, draft: Choices, base: Spec, draftSpec: Spec | null) =>
    unwrap(api.POST("/design/expectations", { body: { cur, draft, base, draft_spec: draftSpec } })),
  changes: (base: Spec, draftSpec: Spec, cur: Choices, draft: Choices) =>
    unwrap(api.POST("/design/changes", { body: { base, draft_spec: draftSpec, cur, draft } })),
  /** The dimensions worth a guess for the change from `oldSpec` to `newSpec` (design.prediction_questions). */
  questions: (oldSpec: Spec, newSpec: Spec) => unwrap(api.POST("/design/questions", { body: { old_spec: oldSpec, new_spec: newSpec } })),
  /** The draft as one version per change, with the chained pretrain jobs (design.split_plan, split_jobs); nothing is saved. */
  splitPlan: (base: Spec, was: Choices, cur: Choices, draft: Choices, parentVersion: number | null, datasets: string[]) =>
    unwrap(api.POST("/design/split-plan", { body: { base, was, cur, draft, parent_version: parentVersion, datasets } })),
  /** Version `newVersion`'s stored guess against what version `oldVersion` and it did (a job; follow it to its result). */
  guessCheck: async (oldVersion: number, newVersion: number, signal: AbortSignal) => {
    const job = await unwrap(api.POST("/design/guess-check", { body: { old_version: oldVersion, new_version: newVersion }, signal }));
    return (await awaitJob<LearningCheck | null>(job, signal, { events: true })) ?? null;
  },
};

/** The sets of text to train on (the ids a catch-up job reads), and the queue. */
export const queueJobs = {
  textSets: () => unwrap(api.GET("/pretraining/text_sets")),
  /** Add a split plan's job to the end of Up next. */
  enqueue: (job: SplitPreview["jobs"][number]) =>
    unwrap(api.POST("/queue/jobs", { body: { kind: job.kind, label: job.label, note: job.note, params: job.params ?? {}, eta_s: job.eta_s } })),
};

/** The cached count of your text (null while the words are still being counted), for compute-lines. */
export const pretraining = {
  preview: () => unwrap(api.GET("/pretraining/preview")),
};

export const versions = {
  list: () => unwrap(api.GET("/versions")),
  current: () => unwrap(api.GET("/versions/current")),
  /** `prediction` is the guess made before saving; none (or an empty one) when it was skipped. */
  save: (spec: Spec, change: string, prediction?: Guess) =>
    unwrap(api.POST("/versions", { body: { spec, change, ...(prediction && Object.keys(prediction).length ? { prediction } : {}) } })),
  /** Start from a pretrained open model (POST /post-training/imports, a form as the Post-training page sends it): its
   * design is saved as a new version now; its weights come when the queue finishes the download and conversion. */
  importPretrained: async (name: string): Promise<Version> => {
    const form = new FormData();
    form.set("name", name);
    const res = await apiFetch("/post-training/imports", { method: "POST", body: form });
    if (!res.ok) throw new ApiError(res.status, "/post-training/imports", await res.text());
    return (await res.json()) as Version;
  },
};

/** The sentence a failed call carries: the server's `detail` for a 422, else the error's own message. */
export function detailOf(error: unknown): string {
  if (error instanceof ApiError) {
    try {
      const detail = (JSON.parse(error.body) as { detail?: unknown }).detail;
      if (typeof detail === "string") return detail;
    } catch {
      // not JSON: the status line below says what happened
    }
    return error.message;
  }
  return error instanceof Error ? error.message : String(error);
}

// ------------------------------------------------------------------------------------------------ the draft

/** The spec a version's choices are read from: its own, or the default design (an empty spec validates to the default). */
export function specOf(version: Version | null | undefined): Spec {
  return version?.spec && Object.keys(version.spec).length ? version.spec : {};
}

/** (spec, its choices) the draft is written onto: the anchor, else the current version (BuilderPage.spec_base). */
export function specBase(draft: Draft, baseSpec: Spec, cur: Choices): { base: Spec; was: Choices | null } {
  return draft.anchor ? { base: draft.anchor.spec, was: draft.anchor.was } : { base: baseSpec, was: cur };
}

/** The draft that loads `spec` (a preset or an earlier version): its own choices, anchored so what the picture has no
 * control for (width, reading window, the training recipe) comes with it (builder.py preset_draft). */
export function anchoredDraft(base: number | null, spec: Spec, choices: Choices): Draft {
  return { base, draft: choices, anchor: { spec, was: choices } };
}

export function wordChanged(cur: Choices, draft: Choices): boolean {
  return cur.V !== draft.V;
}

export function sameChoices(a: Choices, b: Choices): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

/** The change text a save records: the chips, or what the Tk page says when there are none. */
export function changeText(chips: string[], base: number | null): string {
  return chips.join(", ") || (base === null ? "first design" : "no change");
}

/** The picks of a guess panel as a stored guess: the questions left unpicked are left out. */
export function guessOf(picks: Guess | undefined): Guess {
  return Object.fromEntries(Object.entries(picks ?? {}).filter(([, value]) => value));
}

/** A split plan's step is saved as `number`, which can differ from the one planned if a version was saved meanwhile: the job follows the number. */
export function jobFor(job: SplitPreview["jobs"][number], planned: number, number: number): SplitPreview["jobs"][number] {
  if (planned === number) return job;
  return { ...job, label: job.label.replace(`Version ${planned}`, `Version ${number}`), params: { ...job.params, version: number } };
}

export const versionName = (n: number) => `version ${n}`;

/** The version a save makes: one past the current one (BuilderPage.nv). */
export const nextVersionName = (base: number | null) => versionName((base ?? 0) + 1);

// ------------------------------------------------------------------------------------------------ keeping the draft across reloads

const STORE_KEY = "builder.draft";

/** The draft a reload finds, or null. Storage can be missing or blocked, and a stale shape is a fresh start. */
export function loadDraft(): Draft | null {
  try {
    const raw = window.localStorage.getItem(STORE_KEY);
    if (!raw) return null;
    const found = JSON.parse(raw) as Partial<Draft>;
    if (!found.draft || typeof found.draft !== "object" || !("L" in found.draft)) return null;
    return { base: typeof found.base === "number" ? found.base : null, draft: found.draft, anchor: found.anchor ?? null };
  } catch {
    return null;
  }
}

export function storeDraft(draft: Draft | null): void {
  try {
    if (draft) window.localStorage.setItem(STORE_KEY, JSON.stringify(draft));
    else window.localStorage.removeItem(STORE_KEY);
  } catch {
    // nothing kept: the draft lives on in the page's state until it is saved or discarded
  }
}

// ------------------------------------------------------------------------------------------------ the cost rows

/** "about the same" or "about 2.3× bigger" for y against x (builder.py say). */
export function say(x: number, y: number, more: string, less: string): string {
  if (!x) return BUILDER.about_same;
  const r = y / x;
  if (Math.abs(r - 1) < 0.05) return BUILDER.about_same;
  return `${fill(BUILDER.about_multiplier, { mult: (r > 1 ? r : 1 / r).toFixed(1) })} ${r > 1 ? more : less}`;
}

/** The two bars' widths as percentages of the longer one (builder.py bar). */
export function bar(x: number, y: number): [number, number] {
  const m = Math.max(x, y) || 1;
  return [Math.round((x / m) * 100), Math.round((y / m) * 100)];
}

export type CostRow = { name: string; key: "params" | "time" | "memory"; more: string; less: string };

export const COST_ROWS: CostRow[] = [
  { name: BUILDER.size_label, key: "params", more: BUILDER.cost_bigger, less: BUILDER.cost_smaller },
  { name: BUILDER.time_per_step, key: "time", more: BUILDER.cost_slower, less: BUILDER.cost_faster },
  { name: BUILDER.memory_while_training, key: "memory", more: BUILDER.cost_more, less: BUILDER.cost_less },
];

/** "Training memory: fits" | "doesn't fit" | the unknown text (builder.py _paint_costs). */
export function fitsText(fits: boolean | null | undefined): string {
  const verdict = fits == null ? BUILDER.training_memory_fits : fits ? "fits" : BUILDER.wont_fit;
  return `${BUILDER.training_memory} ${verdict}`;
}

// ------------------------------------------------------------------------------------------------ the tower

/** The registry's entry for a module, for its name and settings. */
export function moduleOf(registry: Registry | undefined, slot: string, short: string): ModuleInfo | undefined {
  return registry?.slots.find((s) => s.slot === slot)?.modules.find((m) => m.short === short);
}

export function moduleName(registry: Registry | undefined, slot: string, short: string): string {
  return moduleOf(registry, slot, short)?.name ?? short;
}

export function hasField(registry: Registry | undefined, slot: string, short: string, field: string): boolean {
  return moduleOf(registry, slot, short)?.fields.some((f) => (f as { name?: string }).name === field) ?? false;
}

export function partName(slot: string): string {
  return (BUILDER as Record<string, string>)[`part_${slot}`] ?? slot;
}

export type BlockView = {
  index: number;
  /** The parts it shows: the draft's, or the current version's for a block the draft removed. */
  parts: Record<string, string>;
  own: string[];
  added: boolean;
  removed: boolean;
  /** "new" | "removed" | "changed" | "own parts" | "" */
  tag: string;
  /** The current version's parts of this block, for the "was" labels; null for a new block. */
  before: Record<string, string> | null;
  choices: Choices;
};

/** Every block row the tower paints, newest on top (builder.py _paint_tower). */
export function blockViews(cur: Choices, draft: Choices, curTower: Tower | undefined, draftTower: Tower | undefined): BlockView[] {
  if (!curTower || !draftTower) return [];
  const knobs = draft.H !== cur.H || draft.W !== cur.W;
  const rows: BlockView[] = [];
  for (let i = 0; i < Math.max(cur.L, draft.L); i++) {
    const added = i >= cur.L;
    const removed = i >= draft.L;
    const shown = removed ? curTower.blocks[i] : draftTower.blocks[i];
    const before = added ? null : curTower.blocks[i].parts;
    const changed = !added && !removed && (knobs || JSON.stringify(before) !== JSON.stringify(shown.parts));
    const own = shown.own.length > 0;
    const tag = added ? BUILDER.block_new : removed ? BUILDER.block_removed : changed ? BUILDER.changed : own ? BUILDER.own_parts : "";
    rows.push({ index: i, parts: shown.parts, own: shown.own, added, removed, tag, before, choices: removed ? cur : draft });
  }
  return rows.reverse();
}

/** The picker's option label: the module's name, with the server's reason when the design cannot take it. */
export function optionLabel(option: Option): string {
  return option.why ? `${option.name} · ${BUILDER.wont_fit}: ${option.why}` : option.name;
}

/** The versions list with `version` added or replaced, in number order (the `versions` event's payload). */
export function upsertVersion(list: Version[] | undefined, version: Version): Version[] {
  const rest = (list ?? []).filter((v) => v.number !== version.number);
  return [...rest, version].sort((a, b) => a.number - b.number);
}

export function isVersion(payload: unknown): payload is Version {
  return typeof payload === "object" && payload !== null && typeof (payload as Version).number === "number";
}
