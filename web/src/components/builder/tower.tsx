"use client";

// The tower (builder.py _tower, _paint_tower): the model-wide parts top and bottom, the blocks between them newest on
// top, and the word list under it all. Every part is a picker whose menu comes from /design/part-options, so a module
// the design cannot take is greyed out with the server's reason.

import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import {
  BLOCK_LINES,
  blockViews,
  design,
  hasField,
  moduleName,
  optionLabel,
  partName,
  type BlockView,
  type Choices,
  type Explaining,
  type Part,
  type Registry,
  type Slot,
  type Spec,
  type Tower as TowerShape,
} from "@/lib/builder";
import { BUILDER, fill } from "@/lib/copy/builder";
import { cn } from "@/lib/utils";

const AMBER = "text-warning";
const VIOLET = "text-part-own";
const WIDTHS = [36, 72, 144];

type PickerProps = {
  registry: Registry;
  slot: Slot;
  block: number | null;
  value: string;
  /** The current version's module here; null for a block it does not have. */
  before: string | null;
  writeBase: Spec;
  draft: Choices;
  was: Choices | null;
  onPick: (slot: Slot, short: string, block: number | null) => void;
};

/** One part's menu. Its options load as soon as it shows, so the closed menu names the module and its disabled items
 * carry the server's reason (part_options' why_not). */
function Picker({ registry, slot, block, value, before, writeBase, draft, was, onPick }: PickerProps) {
  const { data: options } = useQuery({
    queryKey: ["design", "part-options", writeBase, draft, was, slot, block],
    queryFn: () => design.partOptions(writeBase, draft, was, slot, block),
    placeholderData: keepPreviousData,
    staleTime: Infinity,
    retry: false,
  });
  const items = options?.some((o) => o.short === value) ? options : [{ short: value, name: moduleName(registry, slot, value), blurb: "", why: "" }];
  const changed = before !== null && before !== value;
  const id = block === null ? `picker-${slot}` : `picker-${slot}-${block}`;
  return (
    <div className="flex min-w-0 flex-wrap items-center gap-2">
      <Select value={value} onValueChange={(short) => onPick(slot, short, block)}>
        <SelectTrigger size="sm" data-testid={id} aria-label={partName(slot)} className={cn("min-w-44", changed && AMBER)}>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {items.map((o) => (
            <SelectItem key={o.short} value={o.short} disabled={!!o.why} title={o.blurb}>
              {optionLabel(o)}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      {changed && (
        <span className="text-xs text-muted-foreground">
          {fill(BUILDER.was, { name: moduleName(registry, slot, before) })} · {BUILDER.changed}
        </span>
      )}
    </div>
  );
}

type Explain = { explaining: Explaining | null; onExplain: (part: Part, block: number | null) => void };

const isExplained = (explaining: Explaining | null, part: Part, block: number | null) => explaining?.part === part && explaining.block === block;

/** A part's name; clicking it opens the explain panel on that part. */
function PartName({ part, block, label, tip, className, explain }: { part: Part; block: number | null; label: string; tip?: string; className?: string; explain: Explain }) {
  const on = isExplained(explain.explaining, part, block);
  return (
    <button
      type="button"
      className={cn("rounded-sm border border-transparent px-1 text-left text-sm hover:underline", on && "selected-box", className)}
      title={tip}
      aria-pressed={on}
      aria-label={fill(BUILDER.explain_open, { part: label })}
      data-testid={`explain-${part}${block === null ? "" : `-${block}`}`}
      onClick={() => explain.onExplain(part, block)}
    >
      {label}
    </button>
  );
}

function Row({ label, tip, children, className, part, explain }: { label: string; tip?: string; children: React.ReactNode; className?: string; part?: Part; explain?: Explain }) {
  return (
    <div className={cn("flex flex-wrap items-center gap-3 rounded-md border px-3 py-2", className)}>
      {part && explain ? (
        <PartName part={part} block={null} label={label} tip={tip} className="w-28 shrink-0 font-medium" explain={explain} />
      ) : (
        <span className="w-28 shrink-0 text-sm font-medium" title={tip}>
          {label}
        </span>
      )}
      {children}
    </div>
  );
}

type TowerProps = {
  registry: Registry;
  cur: Choices;
  draft: Choices;
  curTower: TowerShape | undefined;
  draftTower: TowerShape | undefined;
  writeBase: Spec;
  was: Choices | null;
  onPick: (slot: Slot, short: string, block: number | null) => void;
  onResize: (n: number) => void;
  onWords: (v: number) => void;
} & Explain;

export function Tower({ registry, cur, draft, curTower, draftTower, writeBase, was, onPick, onResize, onWords, explaining, onExplain }: TowerProps) {
  const explain = { explaining, onExplain };
  const picker = (slot: Slot, block: number | null, value: string, before: string | null) => (
    <Picker registry={registry} slot={slot} block={block} value={value} before={before} writeBase={writeBase} draft={draft} was={was} onPick={onPick} />
  );
  const blocks = blockViews(cur, draft, curTower, draftTower);
  const normChanged = draft.M.norm !== cur.M.norm;
  return (
    <section className="flex flex-col gap-2" aria-label={BUILDER.h1}>
      <Row label={BUILDER.next_word} tip={BUILDER.next_word_desc} part="head" explain={explain}>
        {picker("head", null, draft.M.head, cur.M.head)}
      </Row>
      <Row label={BUILDER.final_norm} tip={BUILDER.final_norm_tip} part="final_norm" explain={explain}>
        <span className={cn("text-sm", normChanged && AMBER)}>{moduleName(registry, "norm", draft.M.norm)}</span>
        <span className="text-xs text-muted-foreground">{BUILDER.same_as_blocks}</span>
      </Row>
      <div className="flex flex-col gap-1" data-testid="blocks">
        {blocks.map((view) => (
          <Block key={view.index} view={view} registry={registry} draft={draft} picker={picker} explain={explain} />
        ))}
      </div>
      <Row label={BUILDER.blocks} tip={BUILDER.blocks_desc}>
        <Button variant="outline" size="xs" aria-label="fewer blocks" onClick={() => onResize(draft.L - 1)} disabled={draft.L <= 1}>
          −
        </Button>
        <span className={cn("text-sm tabular-nums", draft.L !== cur.L && AMBER)} data-testid="block-count">
          {draft.L}
        </span>
        <Button variant="outline" size="xs" aria-label="more blocks" onClick={() => onResize(draft.L + 1)} disabled={draft.L >= 12}>
          +
        </Button>
        {draft.L !== cur.L && <span className="text-xs text-muted-foreground">{fill(BUILDER.was, { name: String(cur.L) })}</span>}
      </Row>
      <Row label={partName("position")} tip={registry.slots.find((s) => s.slot === "position")?.tip} part="position" explain={explain}>
        {picker("position", null, draft.M.position, cur.M.position)}
      </Row>
      <Row label={partName("embedding")} tip={registry.slots.find((s) => s.slot === "embedding")?.tip} part="embedding" explain={explain}>
        {picker("embedding", null, draft.M.embedding, cur.M.embedding)}
      </Row>
      <Row label={BUILDER.your_text_title}>
        <Segmented options={registry.words} value={draft.V} saved={cur.V} onChange={onWords} testId="words" />
      </Row>
    </section>
  );
}

/** A row of choices with the saved one labelled "· saved" when the draft moved off it (builder.py _paint_seg). */
export function Segmented({
  options,
  value,
  saved,
  onChange,
  disabled,
  testId,
}: {
  options: readonly (string | number)[];
  value: number;
  saved: number;
  onChange: (index: number) => void;
  disabled?: boolean;
  testId?: string;
}) {
  return (
    <ToggleGroup
      type="single"
      variant="outline"
      size="sm"
      value={String(value)}
      onValueChange={(v) => v && onChange(Number(v))}
      disabled={disabled}
      data-testid={testId}
      className="flex-wrap"
    >
      {options.map((option, i) => {
        const wasSaved = saved === i && saved !== value;
        return (
          <ToggleGroupItem key={i} value={String(i)} className={cn(wasSaved && "border-part-own")}>
            {wasSaved ? `${option} · saved` : String(option)}
          </ToggleGroupItem>
        );
      })}
    </ToggleGroup>
  );
}

type BlockProps = {
  view: BlockView;
  registry: Registry;
  draft: Choices;
  picker: (slot: Slot, block: number | null, value: string, before: string | null) => React.ReactNode;
  explain: Explain;
};

function Block({ view, registry, draft, picker, explain }: BlockProps) {
  const [open, setOpen] = useState(false);
  const [scope, setScope] = useState<"all" | "one">("all");
  const { index: i, parts, own, added, removed, tag, before } = view;
  const heads = hasField(registry, "mixer", parts.mixer, "heads") ? view.choices.H : 0;
  const width = hasField(registry, "channel", parts.channel, "d_ff") ? WIDTHS[view.choices.W] : 0;
  const showDetail = open && !removed;
  const block = scope === "one" ? i : null;
  return (
    <div
      className={cn("rounded-md border", added && "bg-warning/10", own.length > 0 && !added && "bg-part-own/10")}
      data-testid={`block-${i}`}
    >
      <div className="flex flex-wrap items-center gap-3 px-3 py-2">
        <Button variant="ghost" size="xs" aria-label={showDetail ? "close block" : "open block"} aria-expanded={showDetail} onClick={() => setOpen(!open)} disabled={removed} title={BUILDER.block_parts_tip}>
          {showDetail ? "▾" : "▸"}
        </Button>
        <span className={cn("w-16 text-sm font-medium", removed && "text-muted-foreground line-through")}>Block {i + 1}</span>
        {heads > 0 && (
          <button
            type="button"
            className={cn("rounded-sm border border-transparent px-1 text-xs text-part-attention", isExplained(explain.explaining, "mixer", i) && "selected-box")}
            aria-label={fill(BUILDER.explain_open, { part: `${partName("mixer")} · block ${i + 1}` })}
            data-testid={`explain-heads-${i}`}
            onClick={() => explain.onExplain("mixer", i)}
          >
            {"●".repeat(heads)}
          </button>
        )}
        {width > 0 && (
          <button
            type="button"
            className={cn("rounded-sm border border-transparent px-1 py-1", isExplained(explain.explaining, "channel", i) && "selected-box")}
            aria-label={fill(BUILDER.explain_open, { part: `${partName("channel")} · block ${i + 1}` })}
            data-testid={`explain-room-${i}`}
            onClick={() => explain.onExplain("channel", i)}
          >
            <span className="block h-2 rounded-sm bg-part-mlp" style={{ width }} aria-hidden="true" />
          </button>
        )}
        <span className="text-xs text-muted-foreground">
          {moduleName(registry, "mixer", parts.mixer)} · {moduleName(registry, "channel", parts.channel)}
        </span>
        {tag && (
          <Badge variant="outline" className={cn(tag === BUILDER.own_parts ? VIOLET : AMBER)}>
            {tag}
          </Badge>
        )}
      </div>
      {showDetail && (
        <div className="flex flex-col gap-2 border-t px-3 py-2">
          <ToggleGroup type="single" variant="outline" size="sm" value={scope} onValueChange={(v) => v && setScope(v as "all" | "one")}>
            <ToggleGroupItem value="all">{BUILDER.scope_all}</ToggleGroupItem>
            <ToggleGroupItem value="one">{fill(BUILDER.scope_one, { n: i + 1 })}</ToggleGroupItem>
          </ToggleGroup>
          {BLOCK_LINES.map((slot, line) =>
            slot === null ? (
              <div key={line} className="text-xs text-muted-foreground">
                ⊕ {BUILDER.add_back}
              </div>
            ) : (
              <div key={line} className="flex flex-wrap items-center gap-3">
                <span className="w-28 shrink-0">
                  <PartName part={slot} block={block} label={partName(slot)} tip={registry.slots.find((s) => s.slot === slot)?.tip} explain={explain} />
                </span>
                {picker(slot, block, parts[slot], before?.[slot] ?? null)}
                {draft.O.some((o) => o.block === i && o.slot === slot) && <span className={cn("text-xs", VIOLET)}>{BUILDER.only_here}</span>}
              </div>
            ),
          )}
        </div>
      )}
    </div>
  );
}
