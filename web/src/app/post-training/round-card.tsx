"use client";

import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { cn } from "@/lib/utils";
import { T, fill, type RoundSpec } from "./copy";
import { count, spreadText, type Options, type Status } from "./logic";

export type Choices = {
  lora: boolean | null; // null = follow the version's own setting
  rank: number;
  teacher: "version" | "correction";
  teacherVersion: number | null; // null = the previous best
  leash: "Close" | "Medium" | "Far";
};

/** What the running round's `train` events have shown so far. */
export type Live = {
  group?: { prompt: string; rule: string; mean: number; skipped?: boolean; answers: { text: string; pass: boolean; score: number; push: number }[] };
  heat?: { prompt: string; words: string[]; surprise: number[] };
  moved?: { k: number; n: number };
  spread?: { taught_total: number; groups_total: number };
};

function Choice<V extends string>({
  label,
  value,
  items,
  onChange,
  disabled,
}: {
  label: string;
  value: V;
  items: { value: V; label: string }[];
  onChange: (value: V) => void;
  disabled?: boolean;
}) {
  return (
    <div className="flex flex-col gap-2">
      <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{label}</span>
      <ToggleGroup
        type="single"
        variant="outline"
        aria-label={label}
        value={value}
        disabled={disabled}
        onValueChange={(next) => next && onChange(next as V)}
      >
        {items.map((item) => (
          <ToggleGroupItem key={item.value} value={item.value}>
            {item.label}
          </ToggleGroupItem>
        ))}
      </ToggleGroup>
    </div>
  );
}

function Stat({ value, label }: { value: string; label: string }) {
  return (
    <div>
      <div className="font-mono text-2xl">{value}</div>
      <div className="text-xs text-muted-foreground">{label}</div>
    </div>
  );
}

function Round1({ counts, options, choices, set, example, loading }: RoundProps & { example: Status["example"] }) {
  const blocker = options?.lora_blocker ?? null;
  const on = choices.lora ?? options?.lora_default ?? false;
  const sample = (example ?? {}) as { prompt?: string; answer?: string };
  return (
    <div className="flex flex-col gap-5">
      <div className="grid gap-3 sm:grid-cols-2">
        <div className="rounded-lg bg-muted/50 p-3">
          <div className="text-xs text-muted-foreground">PROMPT · NOT SCORED</div>
          <p className="mt-1">{sample.prompt || "No conversations yet: add some in Your data."}</p>
        </div>
        <div className="rounded-lg bg-muted/50 p-3">
          <div className="text-xs text-muted-foreground">ANSWER · IT LEARNS TO WRITE THIS</div>
          <p className="mt-1">{sample.answer || "The answer from the same conversation appears here."}</p>
        </div>
      </div>
      <div className="flex flex-wrap gap-8">
        <Stat value={count(counts.conversations)} label="conversations (Dolly, Alpaca, yours)" />
        <Stat value={count(counts.corrections)} label="corrections you wrote" />
        <Stat value={count(counts.problems)} label="practice problems with answers" />
      </div>
      <div className="flex flex-col gap-2" aria-label="LoRA options">
        <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{T.loraLabel}</span>
        {loading ? (
          <Skeleton className="h-8 w-64" />
        ) : blocker ? (
          <p className="text-sm text-warning">
            {T.loraOff} · {blocker}
          </p>
        ) : (
          <>
            <div className="flex items-center gap-3">
              <Switch
                id="lora-switch"
                aria-label="Train LoRA adapters"
                checked={on}
                onCheckedChange={(next) => set({ lora: next })}
              />
              <label htmlFor="lora-switch" className="text-sm">
                {on ? T.loraOn : T.loraOff}
              </label>
            </div>
            <p className="max-w-prose text-sm text-muted-foreground">{on ? T.loraDescOn : T.loraDescOff}</p>
            {on && (
              <Choice
                label={T.loraRankLabel}
                value={String(choices.rank)}
                items={(options?.lora_ranks ?? []).map((r) => ({ value: String(r), label: String(r) }))}
                onChange={(rank) => set({ rank: Number(rank) })}
              />
            )}
          </>
        )}
      </div>
    </div>
  );
}

export type RoundProps = {
  counts: Record<string, number>;
  options: Options | undefined;
  choices: Choices;
  set: (patch: Partial<Choices>) => void;
  live: Live;
  gate?: { taught?: number; groups?: number };
  loading: boolean;
};

function Round2({ live, gate }: RoundProps) {
  const group = live.group;
  const taught = live.spread?.taught_total ?? gate?.taught ?? 0;
  const groups = live.spread?.groups_total ?? gate?.groups ?? 0;
  const line = spreadText(taught, groups);
  return (
    <div className="flex flex-col gap-3">
      {group ? (
        <>
          <p className="text-xs text-muted-foreground">
            PROMPT · RULE: {group.rule.replaceAll("_", " ").toUpperCase()}: {group.prompt}
          </p>
          <table className="w-full text-sm">
            <thead className="text-left text-xs text-muted-foreground">
              <tr>
                <th className="py-1 pr-4 font-normal">its answer</th>
                <th className="py-1 pr-4 font-normal">rule</th>
                <th className="py-1 pr-4 font-normal">score</th>
                <th className="py-1 font-normal">vs group average ({group.mean.toFixed(2)})</th>
              </tr>
            </thead>
            <tbody>
              {group.answers.map((a, i) => (
                <tr key={i} className={a.score >= 1 ? "text-success" : "text-destructive"}>
                  <td className="py-1 pr-4 font-mono">{a.text}</td>
                  <td className="py-1 pr-4">{a.score >= 1 ? "pass" : "fail"}</td>
                  <td className="py-1 pr-4">{a.score.toFixed(1)}</td>
                  <td className="py-1">
                    {Math.abs(a.push) < 1e-9 ? "no push" : a.push > 0 ? `↑ ${a.push.toFixed(2)} · pushed up` : `↓ ${a.push.toFixed(2)} · pushed down`}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      ) : (
        <p className="text-sm text-muted-foreground">The prompts and rule scores appear here while the round trains.</p>
      )}
      <p className={cn("text-sm", line ? (taught ? "text-foreground" : "text-warning") : "text-muted-foreground")}>{line || T.spreadHint}</p>
    </div>
  );
}

function Round3({ live, options, choices, set }: RoundProps) {
  const heat = live.heat;
  const top = heat ? Math.max(1, ...heat.surprise) : 1;
  const teachers = options?.teachers ?? [];
  const chosen = choices.teacherVersion ?? options?.default_teacher ?? null;
  return (
    <div className="flex flex-col gap-4">
      {heat ? (
        <div className="flex flex-col gap-2">
          <p className="text-xs text-muted-foreground">PROMPT: {heat.prompt}</p>
          <p className="text-xs uppercase text-muted-foreground">{T.answerShaded}</p>
          <p className="rounded-lg bg-muted/50 p-3 text-base leading-8">
            {heat.words.map((w, i) => (
              <span
                key={i}
                className="rounded px-0.5"
                style={{ backgroundColor: `color-mix(in oklab, var(--destructive) ${Math.round(Math.min(1, (heat.surprise[i] ?? 0) / top) * 70)}%, transparent)` }}
              >
                {w}{" "}
              </span>
            ))}
          </p>
          <p className="text-xs text-muted-foreground">
            {T.notSurprised} → <span className="text-destructive">{T.verySurprised}</span>
          </p>
        </div>
      ) : (
        <p className="text-sm text-muted-foreground">The teacher&apos;s answer, shaded by surprise, appears here while the round trains.</p>
      )}
      {live.moved && (
        <p className="text-sm">
          <span className="font-mono">{live.moved.k.toLocaleString("en-US")}</span> of{" "}
          <span className="font-mono">{live.moved.n.toLocaleString("en-US")}</span> words moved toward the teacher
        </p>
      )}
      <Choice
        label={T.teacherLabel}
        value={choices.teacher}
        items={[
          { value: "version", label: T.previousBest },
          { value: "correction", label: T.thisModelCorrection },
        ]}
        onChange={(teacher) => set({ teacher })}
      />
      <p className="max-w-prose text-sm text-muted-foreground">
        {choices.teacher === "version"
          ? chosen != null
            ? fill(T.previousDesc, { version: `Version ${chosen}` })
            : T.noPrevious
          : T.correctionDesc}
      </p>
      {choices.teacher === "version" && teachers.length > 1 && chosen != null && (
        <Choice
          label={T.whichVersion}
          value={String(chosen)}
          items={teachers.map((t) => ({ value: String(t.number), label: `Version ${t.number}` }))}
          onChange={(v) => set({ teacherVersion: Number(v) })}
        />
      )}
    </div>
  );
}

function Round4({ options, choices, set }: RoundProps) {
  const leash = options?.leash ?? [];
  const current = leash.find((l) => l.name === choices.leash);
  return (
    <div className="flex flex-col gap-4">
      <p className="max-w-prose text-sm text-muted-foreground">{T.forEachPair}</p>
      <Choice
        label={T.howFar}
        value={choices.leash}
        items={leash.map((l) => ({ value: l.name as Choices["leash"], label: l.name }))}
        onChange={(next) => set({ leash: next })}
      />
      {current && (
        <p className="text-sm">
          {current.note} (β = {current.beta})
        </p>
      )}
    </div>
  );
}

export function RoundCard({
  spec,
  props,
  example,
  blocker,
  outlook,
  status,
  queuing,
  onQueue,
}: {
  spec: RoundSpec;
  props: RoundProps;
  example: Status["example"];
  blocker: string | null;
  outlook: string | undefined;
  status: string;
  queuing: boolean;
  onQueue: () => void;
}) {
  const body = { 1: <Round1 {...props} example={example} />, 2: <Round2 {...props} />, 3: <Round3 {...props} />, 4: <Round4 {...props} /> }[spec.n];
  const help = [spec.method, ...(spec.n === 1 ? [T.fewThousand] : spec.n === 2 ? [T.rulesCheck] : spec.n === 3 ? [T.teacherNote] : [T.forEachPair, T.leashNote]), spec.foot];
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-xs font-medium uppercase tracking-wide text-highlight">{spec.kicker}</CardTitle>
        <CardDescription className="max-w-prose text-base">{spec.lead}</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-5">
        {body}
        <div className="flex flex-wrap items-center gap-3">
          <Button onClick={onQueue} disabled={!!blocker || queuing}>
            {spec.action}
          </Button>
          {blocker && <span className="text-sm text-warning">{blocker}</span>}
          {spec.link && (
            <Link href={spec.link.href} className="text-sm text-primary underline-offset-4 hover:underline">
              {spec.link.label} ›
            </Link>
          )}
        </div>
        {!blocker && status !== "running" && outlook && <p className="text-sm text-muted-foreground">{outlook}</p>}
        <details className="text-sm text-muted-foreground">
          <summary className="cursor-pointer select-none">How this round works</summary>
          <div className="mt-2 flex max-w-prose flex-col gap-2">
            {help.map((h) => (
              <p key={h}>{h}</p>
            ))}
          </div>
        </details>
      </CardContent>
    </Card>
  );
}
