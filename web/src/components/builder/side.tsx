"use client";

// The Builder's side column and tray (builder.py _side, _paint_costs, _paint_expect, _tray): where to start from, the
// knobs, what the draft costs, what to expect, the versions, and the not-saved tray with its save and discard.

import { useQuery } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { Badge } from "@/components/ui/badge";
import Link from "next/link";
import { Button, buttonVariants } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { DiscreteControl } from "@/components/builder/discrete-control";
import {
  bar,
  COST_ROWS,
  design,
  fitsText,
  PARAM_PARTS,
  paramBars,
  paramNote,
  paramShares,
  nextVersionName,
  say,
  specOf,
  versionName,
  type Changes,
  type Choices,
  type Expectations,
  type Line,
  type Preset,
  type Registry,
  type Sizes,
  type Spec,
  type Tower,
  type Version,
} from "@/lib/builder";
import { BUILDER, fill, presetAbout } from "@/lib/copy/builder";
import { routeFor } from "@/lib/routes";
import { cn } from "@/lib/utils";

const AMBER = "text-warning";

export function Panel({ title, tip, children, testId }: { title: string; tip?: string; children: React.ReactNode; testId?: string }) {
  return (
    <Card size="sm" data-testid={testId}>
      <CardHeader>
        <CardTitle title={tip}>{title}</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-2">{children}</CardContent>
    </Card>
  );
}

// ------------------------------------------------------------------------------------------------ start from

type StartFromProps = {
  presets: Preset[];
  versions: Version[];
  onLoad: (spec: Spec) => void;
  onGpt2: () => void;
  note: string | null;
};

export function StartFrom({ presets, versions, onLoad, onGpt2, note }: StartFromProps) {
  const [key, setKey] = useState<string | null>(null);
  const preset = presets.find((p) => p.key === key) ?? null;
  const costQ = useQuery({
    queryKey: ["design", "preset-cost", preset?.spec],
    queryFn: () => design.presetCost(preset!.spec),
    enabled: !!preset,
    staleTime: Infinity,
    retry: false,
  });
  const cost = costQ.data;
  const warn = cost ? cost.fits === false || cost.slow : false;
  const saved = versions.filter((v) => v.spec && Object.keys(v.spec).length);
  return (
    <Panel title={BUILDER.start_from} tip={BUILDER.start_from_tip} testId="start-from">
      <Select value={key ?? ""} onValueChange={setKey}>
        <SelectTrigger size="sm" className="w-full" data-testid="preset-pick" aria-label={BUILDER.start_from}>
          <SelectValue placeholder={BUILDER.start_from_pick} />
        </SelectTrigger>
        <SelectContent>
          {presets.map((p) => (
            <SelectItem key={p.key} value={p.key}>
              {p.name}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      {preset && (
        <div className="flex flex-col gap-1 text-xs">
          <p className="text-muted-foreground">{presetAbout(preset.key)}</p>
          {cost ? (
            [...cost.lines, fill(BUILDER.preset_recipe, { recipe: cost.recipe })].map((line, i) => (
              <p key={i} className={cn(warn && AMBER)}>
                {line}
              </p>
            ))
          ) : costQ.isError ? (
            <p className={AMBER}>{String(costQ.error)}</p>
          ) : (
            <Skeleton className="h-4 w-3/4" />
          )}
          <div>
            <Button size="sm" variant="outline" onClick={() => onLoad(preset.spec)} data-testid="preset-load">
              {BUILDER.preset_load}
            </Button>
          </div>
        </div>
      )}
      {saved.length > 0 && (
        <Select value="" onValueChange={(n) => onLoad(specOf(saved.find((v) => String(v.number) === n)))}>
          <SelectTrigger size="sm" className="w-full" data-testid="version-pick" aria-label={BUILDER.start_from_version}>
            <SelectValue placeholder={BUILDER.start_from_version} />
          </SelectTrigger>
          <SelectContent>
            {saved.map((v) => (
              <SelectItem key={v.number} value={String(v.number)} title={BUILDER.start_from_version_tip}>
                {v.name} · {v.change}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      )}
      <div>
        <Button size="sm" variant="outline" title={BUILDER.import_gpt2_tip} onClick={onGpt2} data-testid="import-gpt2">
          {BUILDER.import_gpt2}
        </Button>
      </div>
      {note && <p className="text-xs text-muted-foreground">{note}</p>}
    </Panel>
  );
}

// ------------------------------------------------------------------------------------------------ the knobs

type ControlsProps = {
  registry: Registry;
  cur: Choices;
  draft: Choices;
  knobs: Tower["knobs"] | undefined;
  onKnob: (key: "H" | "W", value: number) => void;
  onModelSetting?: (key: ModelSettingKey, value: number) => void;
};

type ModelSettingKey = "d" | "T" | "U" | "token_limit_millions";

export function Controls({ registry, cur, draft, knobs, onKnob, onModelSetting = () => {} }: ControlsProps) {
  const vocabEditable = draft.V === 1;
  const vocabMin = 1024;
  const vocabMax = 65536;
  const mixer = registry.slots.find((slot) => slot.slot === "mixer")?.modules.find((module) => module.short === draft.M.mixer);
  const hasHeads = mixer?.fields.some((field) => field.name === "heads") ?? false;
  const numberInput = (label: string, key: ModelSettingKey, value: number, min: number, max: number, disabled = false) => (
    <ModelNumberField label={label} value={value} min={min} max={max} disabled={disabled} onCommit={(next) => onModelSetting(key, next)}
      problem={key !== "d" ? undefined : (next) =>
        hasHeads && next % draft.H !== 0 ? `Must be a multiple of the ${draft.H} look-back heads.` :
        draft.M.position === "rope" && next % 2 !== 0 ? "Must be even with rotary positions." : undefined} />
  );
  return (
    <Panel title={BUILDER.model_controls} testId="controls">
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-1">
        <div className="min-w-0">
          <DiscreteControl label={BUILDER.look_back_heads} help={BUILDER.heads_help} options={registry.heads} value={registry.heads.indexOf(draft.H)} saved={registry.heads.indexOf(cur.H)} onChange={(i) => onKnob("H", registry.heads[i])} disabled={!knobs?.H} testId="heads" />
          {knobs && !knobs.H && <p className="mt-2 text-xs text-muted-foreground">{BUILDER.no_heads}</p>}
        </div>
        <div className="min-w-0">
          <DiscreteControl label={BUILDER.thinking_room} help={BUILDER.width_help} options={registry.widths} value={draft.W} saved={cur.W} onChange={(i) => onKnob("W", i)} disabled={!knobs?.W} testId="width" />
          {knobs && !knobs.W && <p className="mt-2 text-xs text-muted-foreground">{BUILDER.no_room}</p>}
        </div>
      </div>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-1" data-testid="model-dimensions">
        {numberInput("Hidden size", "d", draft.d, 16, 1024)}
        {numberInput("Token context", "T", draft.T, 32, 2048)}
        {numberInput("Tokenizer size", "U", draft.U ?? 263, vocabEditable ? vocabMin : 263, vocabEditable ? vocabMax : 65536, !vocabEditable)}
        {numberInput("Training token limit (millions)", "token_limit_millions", draft.token_limit_millions ?? 0, 0, 1000000)}
      </div>
      <p className="text-xs text-muted-foreground">Tokenizer size is adjustable for learned tokenizers. The token limit is off at 0 and stops at an optimizer-step boundary.</p>
    </Panel>
  );
}

/** A number box you can click and type into: the text is held locally while typing and only a valid whole number in range is passed up, so a half-typed value is not snapped back. */
function ModelNumberField({ label, value, min, max, disabled, problem, onCommit }: {
  label: string; value: number; min: number; max: number; disabled: boolean;
  problem?: (next: number) => string | undefined; onCommit: (next: number) => void;
}) {
  const [typed, setTyped] = useState<string | null>(null);
  const text = typed ?? String(value);
  const next = text.trim() === "" ? NaN : Number(text);
  const error = text.trim() === "" ? undefined
    : !Number.isInteger(next) ? "Whole numbers only."
    : next < min || next > max ? `Between ${min.toLocaleString()} and ${max.toLocaleString()}.`
    : problem?.(next);
  return (
    <label className="flex flex-col gap-1 text-xs font-medium">
      {label}
      <Input type="number" inputMode="numeric" min={min} max={max} step={1} value={text} disabled={disabled} className="h-9" aria-invalid={error ? true : undefined}
        onBlur={() => setTyped(null)}
        onChange={(event) => {
          const raw = event.currentTarget.value;
          setTyped(raw);
          const parsed = raw.trim() === "" ? NaN : Number(raw);
          if (Number.isInteger(parsed) && parsed >= min && parsed <= max && !problem?.(parsed)) onCommit(parsed);
        }} />
      {error && <span className="text-xs font-normal text-destructive" role="alert">{error}</span>}
    </label>
  );
}

/** Native disclosure keeps optional tools mounted and preserves their local work. */
export function BuilderDisclosure({ title, hint, reveal, children }: { title: string; hint: string; reveal?: unknown; children: React.ReactNode }) {
  const ref = useRef<HTMLDetailsElement>(null);
  useEffect(() => {
    if (reveal && ref.current) ref.current.open = true;
  }, [reveal]);
  return (
    <details ref={ref} className="min-w-0 rounded-xl border bg-card/50">
      <summary className="cursor-pointer rounded-xl px-4 py-3 text-sm font-medium focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring">
        {title}
        <span className="mt-1 block text-xs font-normal text-muted-foreground">{hint}</span>
      </summary>
      <div className="flex min-w-0 flex-col gap-3 border-t p-3">{children}</div>
    </details>
  );
}

// ------------------------------------------------------------------------------------------------ what it costs

type CostsProps = {
  now: Sizes | undefined;
  next: Sizes | undefined;
  warm: Changes["warm"] | null;
  lines: Line[] | undefined;
};

/** "Where the size goes": the current design's parameters stacked faintly above the draft's, on one scale. */
function ParamBar({ now, next }: { now: Sizes["parts"]; next: Sizes["parts"] }) {
  const [a, b] = paramBars(now, next);
  const note = paramNote(next);
  const row = (widths: number[], faint: boolean, testId: string) => (
    <div className="flex h-1.5 w-full" aria-hidden="true" data-testid={testId}>
      {PARAM_PARTS.map((p, i) => (
        <span key={p.key} className={cn("h-full", p.fill, faint && "opacity-40")} style={{ width: `${widths[i]}%` }} />
      ))}
    </div>
  );
  return (
    <div className="flex flex-col gap-1 pt-1" data-testid="param-bar">
      <span className="text-xs font-medium" title={BUILDER.params_where_tip}>
        {BUILDER.params_where}
      </span>
      {row(a, true, "param-now")}
      {row(b, false, "param-next")}
      <div className="flex flex-wrap gap-x-3 text-xs text-muted-foreground">
        {paramShares(next).map((share) => {
          const part = PARAM_PARTS.find((p) => p.key === share.key)!;
          return (
            <span key={share.key} className="inline-flex items-center gap-1">
              <span className={cn("inline-block size-2 rounded-sm", part.fill)} aria-hidden="true" />
              {share.text}
            </span>
          );
        })}
      </div>
      {note ? <p className={cn("text-xs", AMBER)}>{note}</p> : null}
    </div>
  );
}

export function Costs({ now, next, warm, lines }: CostsProps) {
  return (
    <Panel title={BUILDER.what_costs} testId="costs">
      {now && next ? (
        COST_ROWS.map((row) => {
          const x = now[row.key];
          const y = next[row.key];
          const said = say(x, y, row.more, row.less);
          const same = said === BUILDER.about_same;
          const [wx, wy] = bar(x, y);
          return (
            <div key={row.key} className="flex flex-col gap-1" data-testid={`cost-${row.key}`}>
              <div className="flex items-baseline justify-between gap-2 text-xs">
                <span className="font-medium">{row.name}</span>
                <span className={cn(same ? "text-muted-foreground" : AMBER)}>{said}</span>
              </div>
              <div className="h-1.5 rounded-sm bg-muted-foreground/40" style={{ width: `${wx}%` }} aria-hidden="true" />
              <div className={cn("h-1.5 rounded-sm", same ? "bg-part-cost" : "bg-warning")} style={{ width: `${wy}%` }} aria-hidden="true" />
            </div>
          );
        })
      ) : (
        <Skeleton className="h-16 w-full" />
      )}
      {now && next ? <ParamBar now={now.parts} next={next.parts} /> : null}
      <p className="pt-1 text-sm font-medium">{BUILDER.carry_over}</p>
      {warm ? (
        <p className="text-xs">
          <Badge variant="outline" className="mr-1">
            {warm.chip}
          </Badge>
          {warm.text}
        </p>
      ) : (
        <p className="text-xs text-muted-foreground">{BUILDER.carry_over_none}</p>
      )}
      <p className="text-xs" title={BUILDER.training_memory_desc} data-testid="fits">
        {next ? fitsText(next.fits) : BUILDER.training_memory}
      </p>
      {lines ? (
        lines.map((line, i) => (
          <p key={i} className={cn("text-xs", line.bad ? AMBER : "text-muted-foreground")}>
            {line.text}
          </p>
        ))
      ) : (
        <Skeleton className="h-4 w-2/3" />
      )}
    </Panel>
  );
}

// ------------------------------------------------------------------------------------------------ what to expect

export function Expect({ expectations }: { expectations: Expectations | undefined }) {
  return (
    <Panel title={BUILDER.what_expect} testId="expect">
      {expectations ? (
        <>
          <p className="text-sm" title={expectations.details}>
            {expectations.headline}
          </p>
          {expectations.details && <p className="text-xs text-muted-foreground">{expectations.details}</p>}
          {expectations.lines.length > 0 && (
            <ul className="list-disc pl-4 text-xs">
              {expectations.lines.map((line, i) => (
                <li key={i}>{line}</li>
              ))}
            </ul>
          )}
        </>
      ) : (
        <Skeleton className="h-10 w-full" />
      )}
    </Panel>
  );
}

// ------------------------------------------------------------------------------------------------ the versions

export function VersionList({ versions, current }: { versions: Version[] | undefined; current: number | null }) {
  return (
    <Panel title={BUILDER.versions} testId="versions">
      {!versions ? (
        <Skeleton className="h-8 w-full" />
      ) : versions.length === 0 ? (
        <p className="text-xs text-muted-foreground">{BUILDER.versions_empty}</p>
      ) : (
        <ul className="flex flex-col gap-1 text-xs">
          {[...versions].reverse().map((v) => (
            <li key={v.number} data-testid={`version-${v.number}`} className="flex flex-wrap items-baseline gap-2">
              <span className={cn("font-medium", v.number === current && "text-highlight")}>{v.name}</span>
              <span className="text-muted-foreground">{v.change}</span>
              <span className="text-muted-foreground">· {v.status}</span>
            </li>
          ))}
        </ul>
      )}
    </Panel>
  );
}

// ------------------------------------------------------------------------------------------------ the tray

type TrayProps = {
  base: number | null;
  chips: string[];
  cannotWrite: string | null;
  saving: boolean;
  canSave: boolean;
  /** How many versions the draft can be saved as, one change each; 0 when it cannot be split. */
  splitCount: number;
  onSave: () => void;
  onSplit: () => void;
  onDiscard: () => void;
};

export function Tray({ base, chips, cannotWrite, saving, canSave, splitCount, onSave, onSplit, onDiscard }: TrayProps) {
  const nv = nextVersionName(base);
  const unchanged = base !== null && !cannotWrite && chips.length === 0;
  const saveButton = (
    <Button size="sm" onClick={onSave} disabled={!canSave || saving} id="builder-save" data-testid="save">
      {fill(BUILDER.save_as, { version: nv })}
    </Button>
  );
  const quickTest = (
    <Link href={`${routeFor("quicktest").href}/`} className={buttonVariants({ variant: "outline", size: "sm" })} data-testid="quick-test-first">
      {BUILDER.quick_test_first}
    </Link>
  );
  let kicker: string;
  let text: string;
  let actions: React.ReactNode = null;
  if (cannotWrite) {
    kicker = BUILDER.not_saved;
    text = `${BUILDER.cannot_write} ${cannotWrite}`;
    actions = (
      <Button size="sm" variant="outline" onClick={onDiscard}>
        {BUILDER.discard}
      </Button>
    );
  } else if (chips.length) {
    kicker = BUILDER.not_saved;
    text = fill(BUILDER.not_saved_text, { version: nv });
    actions = (
      <>
        {quickTest}
        {saveButton}
        {splitCount > 1 && (
          <Button size="sm" variant="outline" onClick={onSplit} disabled={!canSave || saving} id="builder-split" data-testid="split">
            {fill(BUILDER.split, { count: splitCount })}
          </Button>
        )}
        <Button size="sm" variant="outline" onClick={onDiscard}>
          {BUILDER.discard}
        </Button>
      </>
    );
  } else if (base === null) {
    kicker = BUILDER.first_save;
    text = fill(BUILDER.first_save_text, { version: nv });
    actions = (
      <>
        {saveButton}
        {quickTest}
      </>
    );
  } else {
    kicker = BUILDER.no_changes;
    text = fill(BUILDER.no_changes_text, { version: versionName(base) });
  }
  return (
    <div className={cn("flex flex-wrap items-center gap-3 rounded-xl border bg-card px-4", unchanged ? "py-2" : "py-3")} data-testid="tray" aria-live="polite">
      <div className={cn("flex min-w-0 flex-1 gap-1", unchanged ? "flex-wrap items-baseline gap-x-3" : "flex-col")}>
        <p className={cn("text-sm font-medium", (cannotWrite || chips.length > 0) && AMBER)}>{kicker}</p>
        <p className="text-xs text-muted-foreground">{text}</p>
        {chips.length > 1 && base !== null && !cannotWrite && (
          <p className="text-xs text-muted-foreground" data-testid="split-offer">
            {fill(BUILDER.one_change, { count: chips.length })}{" "}
            {splitCount > 1 ? fill(BUILDER.one_change_split, { count: splitCount }) : BUILDER.one_change_cannot}
          </p>
        )}
        {chips.length > 0 && !cannotWrite && (
          <div className="flex flex-wrap gap-1">
            {chips.map((chip) => (
              <Badge key={chip} variant="outline">
                {chip}
              </Badge>
            ))}
          </div>
        )}
      </div>
      <div className="flex flex-wrap gap-2">{actions}</div>
    </div>
  );
}

/** Compare runs and Quick test, which read the staged draft; the look-inside pages hang off the parts and the versions below. */
export function TryIt() {
  return (
    <section className="flex flex-wrap items-center gap-3" data-testid="try-it" aria-label={BUILDER.try_it}>
      <h2 className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{BUILDER.try_it}</h2>
      {(["compare", "quicktest"] as const).map((key) => (
        <Link key={key} href={`${routeFor(key).href}/`} className="text-sm underline underline-offset-4">
          {key === "compare" ? BUILDER.compare_runs : BUILDER.quick_test}
        </Link>
      ))}
    </section>
  );
}
