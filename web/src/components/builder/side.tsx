"use client";

// The Builder's side column and tray (builder.py _side, _paint_costs, _paint_expect, _tray): where to start from, the
// knobs, what the draft costs, what to expect, the versions, and the not-saved tray with its save and discard.

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Segmented } from "@/components/builder/tower";
import {
  bar,
  COST_ROWS,
  design,
  fitsText,
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
import { cn } from "@/lib/utils";

const AMBER = "text-warning";

function Panel({ title, tip, children, testId }: { title: string; tip?: string; children: React.ReactNode; testId?: string }) {
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
};

export function Controls({ registry, cur, draft, knobs, onKnob }: ControlsProps) {
  const headIndex = registry.heads.indexOf(draft.H);
  const curHeadIndex = registry.heads.indexOf(cur.H);
  return (
    <Panel title={BUILDER.look_back_heads} tip={BUILDER.look_back_heads_tip} testId="controls">
      <Segmented
        options={registry.heads}
        value={headIndex}
        saved={curHeadIndex}
        onChange={(i) => onKnob("H", registry.heads[i])}
        disabled={!knobs?.H}
        testId="heads"
      />
      {knobs && !knobs.H && <p className="text-xs text-muted-foreground">{BUILDER.no_heads}</p>}
      <p className="pt-1 text-sm font-medium" title={BUILDER.thinking_room_tip}>
        {BUILDER.thinking_room}
      </p>
      <Segmented options={registry.widths} value={draft.W} saved={cur.W} onChange={(i) => onKnob("W", i)} disabled={!knobs?.W} testId="width" />
      {knobs && !knobs.W && <p className="text-xs text-muted-foreground">{BUILDER.no_room}</p>}
    </Panel>
  );
}

// ------------------------------------------------------------------------------------------------ what it costs

type CostsProps = {
  now: Sizes | undefined;
  next: Sizes | undefined;
  warm: Changes["warm"] | null;
  lines: Line[] | undefined;
};

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
  onSave: () => void;
  onDiscard: () => void;
};

export function Tray({ base, chips, cannotWrite, saving, canSave, onSave, onDiscard }: TrayProps) {
  const nv = nextVersionName(base);
  const saveButton = (
    <Button size="sm" onClick={onSave} disabled={!canSave || saving} data-testid="save">
      {fill(BUILDER.save_as, { version: nv })}
    </Button>
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
        {saveButton}
        <Button size="sm" variant="outline" onClick={onDiscard}>
          {BUILDER.discard}
        </Button>
      </>
    );
  } else if (base === null) {
    kicker = BUILDER.first_save;
    text = fill(BUILDER.first_save_text, { version: nv });
    actions = saveButton;
  } else {
    kicker = BUILDER.no_changes;
    text = fill(BUILDER.no_changes_text, { version: versionName(base) });
  }
  return (
    <div className="flex flex-wrap items-center gap-3 rounded-md border px-3 py-2" data-testid="tray" aria-live="polite">
      <div className="flex min-w-0 flex-1 flex-col gap-1">
        <p className={cn("text-sm font-medium", (cannotWrite || chips.length > 0) && AMBER)}>{kicker}</p>
        <p className="text-xs text-muted-foreground">{text}</p>
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
      <div className="flex gap-2">{actions}</div>
    </div>
  );
}
