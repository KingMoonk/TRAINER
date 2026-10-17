"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { api, unwrap } from "@/lib/api";
import { HEAD_LAB } from "@/app/head-lab/copy";
import { HeatMap, patternCells } from "@/app/head-lab/heat-map";
import {
  Stale,
  TESTS,
  THRESHOLD,
  fill,
  headKey,
  headPattern,
  headTests,
  parseKey,
  readVersion,
  runJob,
  sentence,
  topRelied,
  type Forward,
  type Pattern,
  type TestKey,
  type Version,
} from "@/app/head-lab/inspect";

type Mode = "mean" | "zero";
type Loaded = { version: Version; ids: number[]; tokens: string[]; patterns: Record<number, Pattern>; tests: Record<number, Record<TestKey, number[]>> };
type Changed = { on?: string | null; off?: string | null; bits?: { before: number; after: number } | null };

const GUESSES = [HEAD_LAB.lot, HEAD_LAB.little, HEAD_LAB.not_at_all];
const EXAMPLE_PROMPT = HEAD_LAB.example_prompt.replace(/ …$/, "");

async function load(number?: number): Promise<Loaded | string> {
  const version = await readVersion(undefined, number);
  if (!version?.best_checkpoint) return HEAD_LAB.no_checkpoint;
  const prepared = await runJob<{ ids: number[]; tokens: string[] }>(unwrap(api.POST("/inspect/prepared", { body: { version: version.number } })));
  const forward = await runJob<Forward>(unwrap(api.POST("/inspect/forward", { body: { version: version.number, ids: prepared.ids, hooks: true } })));
  const patterns: Record<number, Pattern> = {};
  for (const [name, value] of Object.entries(forward.hooks)) {
    const match = /^block(\d+)\.attn\.pattern$/.exec(name);
    if (match) patterns[Number(match[1])] = value as Pattern;
  }
  if (Object.keys(patterns).length === 0) return "This model has no attention heads to experiment on.";
  const tests = Object.fromEntries(Object.entries(patterns).map(([block, pattern]) => [Number(block), headTests(pattern, prepared.ids)]));
  return { version, ids: prepared.ids, tokens: prepared.tokens, patterns, tests };
}

export function HeadLab({ requestedVersion }: { requestedVersion?: number }) {
  const [loaded, setLoaded] = useState<Loaded | string | null>(null);
  const [sel, setSel] = useState<string | null>(null);
  const [off, setOff] = useState<ReadonlySet<string>>(new Set());
  const [mode, setMode] = useState<Mode>("mean");
  const [rises, setRises] = useState<Map<string, number> | null>(null);
  const [ranking, setRanking] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [tested, setTested] = useState<ReadonlySet<string>>(new Set());
  const [guess, setGuess] = useState<string | null>(null);
  const [changedFor, setChangedFor] = useState<{ key: string; value: Changed } | null>(null);
  const rankRun = useRef(0);

  useEffect(() => {
    let dead = false;
    load(requestedVersion).then(
      (result) => {
        if (dead) return;
        setLoaded(result);
        if (typeof result !== "string") {
          const block = Math.min(...Object.keys(result.patterns).map(Number));
          setSel(headKey(block, 0));
        }
      },
      (error) => !dead && setLoaded(sentence(error)),
    );
    return () => {
      dead = true;
    };
  }, [requestedVersion]);

  const version = typeof loaded === "object" && loaded ? loaded.version.number : null;

  // What changed: the answer with every head on, the answer with these off, and the held-back error, once the set or the mode changes.
  // The results are filed under the request they answer, so a result for an older set is never shown for the current one.
  const heads = useMemo(() => [...off].sort().map(parseKey), [off]);
  const changeKey = `${version}|${mode}|${heads.join(";")}`;
  useEffect(() => {
    if (version === null || heads.length === 0) return;
    let dead = false;
    const stale = () => dead;
    const attempt = async <T,>(job: () => Promise<T>) => {
      try {
        return await job();
      } catch (error) {
        if (error instanceof Stale) throw error;
        return null;
      }
    };
    const record = (part: Changed) => !dead && setChangedFor((previous) => ({ key: changeKey, value: { ...(previous?.key === changeKey ? previous.value : {}), ...part } }));
    (async () => {
      const reply = (list: [number, number][]) =>
        attempt(() => runJob<string>(unwrap(api.POST("/inspect/reply", { body: { version, prompt: EXAMPLE_PROMPT, off: list, mode } })), stale));
      record({ on: await reply([]) });
      record({ off: await reply(heads) });
      record({
        bits: await attempt(() => runJob<{ before: number; after: number }>(unwrap(api.POST("/inspect/heads-off", { body: { version, off: heads, mode } })), stale)),
      });
    })().catch(() => {});
    return () => {
      dead = true;
    };
  }, [version, heads, mode, changeKey]);
  const changed = changedFor?.key === changeKey ? changedFor.value : null;

  if (loaded === null || (typeof loaded === "object" && sel === null)) {
    return (
      <Shell>
        <p className="text-muted-foreground" role="status">Reading the model…</p>
      </Shell>
    );
  }
  if (typeof loaded === "string") {
    return (
      <Shell>
        <p className="max-w-prose text-muted-foreground">{loaded}</p>
      </Shell>
    );
  }

  const { patterns, tests, tokens } = loaded;
  const current = sel as string;
  const blocks = Object.keys(patterns).map(Number).sort((a, b) => a - b);
  const total = blocks.reduce((n, b) => n + patterns[b][0].length, 0);
  const [selBlock, selHead] = parseKey(current);
  const selOff = off.has(current);
  const checkpoint = loaded.version.best_checkpoint?.split(/[\\/]/).pop() ?? "";

  const flip = () =>
    setOff((previous) => {
      const next = new Set(previous);
      if (!next.delete(current)) next.add(current);
      return next;
    });
  const chooseMode = (value: string) => {
    if (value !== "mean" && value !== "zero") return;
    rankRun.current += 1;
    setRanking(false);
    setMode(value);
    setRises(null);
    setNotice(null);
  };
  const rank = () => {
    if (rises) {
      setRises(null);
      return;
    }
    const run = (rankRun.current += 1);
    setRanking(true);
    setNotice(null);
    runJob<{ block: number; head: number; rise: number }[]>(
      unwrap(api.POST("/inspect/rank-heads", { body: { version: loaded.version.number, mode } })),
      () => run !== rankRun.current,
    )
      .then((rows) => {
        if (run !== rankRun.current) return;
        setRises(new Map(rows.map((r) => [headKey(r.block, r.head), r.rise])));
        setRanking(false);
      })
      .catch((error) => {
        if (error instanceof Stale || run !== rankRun.current) return;
        setNotice(sentence(error));
        setRanking(false);
      });
  };

  const tileSub = (key: string, block: number, head: number): { text: string; off: boolean } => {
    if (off.has(key)) return { text: HEAD_LAB.head_off, off: true };
    const rise = rises?.get(key);
    if (rise !== undefined) return { text: fill(HEAD_LAB.rank_unit, { rise: rise.toFixed(2) }), off: false };
    return { text: headPattern(tests[block], head), off: false };
  };

  return (
    <Shell>
      <p className="max-w-prose text-sm text-muted-foreground" title={HEAD_LAB.checkpoint_tip}>
        {fill(HEAD_LAB.checkpoint, { version: loaded.version.number, checkpoint })}
      </p>

      <section aria-label="Heads" className="flex flex-col gap-3">
        <h2 className="text-xs font-medium tracking-wide text-muted-foreground">{total} HEADS · CLICK ONE</h2>
        <p className="max-w-prose text-sm text-muted-foreground">
          {HEAD_LAB.heads_note} {rises ? HEAD_LAB.grid_note_ranked : HEAD_LAB.grid_note_normal}
        </p>
        <div className="flex flex-wrap items-center gap-3">
          <Button onClick={rank} disabled={ranking}>
            {ranking ? "Ranking…" : rises ? HEAD_LAB.hide_importance : HEAD_LAB.rank_heads}
          </Button>
          {off.size > 0 && (
            <Button variant="outline" onClick={() => setOff(new Set())}>
              {HEAD_LAB.switch_all_on}
            </Button>
          )}
          <span className="ml-2 text-sm text-muted-foreground" title={HEAD_LAB.mean_exact} id="mode-label">
            {HEAD_LAB.mode_label}
          </span>
          <ToggleGroup type="single" variant="outline" value={mode} onValueChange={chooseMode} aria-labelledby="mode-label">
            <ToggleGroupItem value="mean">{HEAD_LAB.mode_mean}</ToggleGroupItem>
            <ToggleGroupItem value="zero">{HEAD_LAB.mode_zero}</ToggleGroupItem>
          </ToggleGroup>
        </div>
        {notice && (
          <p role="alert" className="text-sm text-warning">
            {notice}
          </p>
        )}
        <div className="flex flex-col gap-2">
          {blocks.map((block) => (
            <div key={block} className="flex flex-wrap items-center gap-2">
              <span className="w-16 text-sm text-muted-foreground">Block {block + 1}</span>
              {patterns[block][0].map((pattern, head) => {
                const key = headKey(block, head);
                const sub = tileSub(key, block, head);
                return (
                  <button
                    key={key}
                    type="button"
                    aria-pressed={key === current}
                    aria-label={`Block ${block + 1} head ${head + 1}`}
                    onClick={() => setSel(key)}
                    className={`flex flex-col gap-1 rounded-lg border p-2 text-left outline-none focus-visible:ring-3 focus-visible:ring-ring/50 ${
                      key === current ? "selected-box" : off.has(key) ? "border-warning" : "border-border"
                    }`}
                  >
                    <HeatMap cells={patternCells(pattern, off.has(key))} size={9} label={`Where block ${block + 1} head ${head + 1} looks`} />
                    <span className="text-xs">Head {head + 1}</span>
                    <span className={`text-[11px] ${sub.off ? "text-warning" : "text-muted-foreground"}`}>{sub.text}</span>
                  </button>
                );
              })}
            </div>
          ))}
        </div>
        {rises && rises.size > 0 && (
          <div>
            <h3 className="text-xs font-medium tracking-wide text-muted-foreground">{HEAD_LAB.most_relied}</h3>
            <ol className="font-mono text-sm">
              {topRelied(rises).map(([key, rise]) => {
                const [block, head] = parseKey(key);
                return (
                  <li key={key}>
                    Block {block + 1} · head {head + 1} {fill(HEAD_LAB.rank_unit, { rise: rise.toFixed(2) })}
                  </li>
                );
              })}
            </ol>
          </div>
        )}
      </section>

      <Card>
        <CardContent className="flex flex-col gap-3">
          <h2 className="text-lg font-semibold">
            Block {selBlock + 1} · head {selHead + 1}
          </h2>
          <p className={`text-sm ${selOff ? "text-warning" : "text-muted-foreground"}`}>{selOff ? HEAD_LAB.head_off : HEAD_LAB.head_on}</p>
          <div title={HEAD_LAB.prepared_text}>
            <HeatMap
              cells={patternCells(patterns[selBlock][0][selHead])}
              size={22}
              tokens={tokens}
              label={`Attention of block ${selBlock + 1} head ${selHead + 1} on the prepared text`}
            />
          </div>
          <p className="max-w-prose text-xs text-muted-foreground">{HEAD_LAB.prepared_text}</p>
          {!selOff && (
            <div className="flex flex-col gap-2">
              <p className="text-sm">{HEAD_LAB.before_switch}</p>
              <ToggleGroup type="single" variant="outline" value={guess ?? ""} onValueChange={(value) => setGuess(value || null)} aria-label="Your guess">
                {GUESSES.map((g) => (
                  <ToggleGroupItem key={g} value={g}>
                    {g}
                  </ToggleGroupItem>
                ))}
              </ToggleGroup>
            </div>
          )}
          <div className="flex flex-wrap gap-2">
            <Button variant={selOff ? "default" : "outline"} onClick={flip}>
              {selOff ? HEAD_LAB.switch_on : HEAD_LAB.switch_off}
            </Button>
            <Button variant="outline" onClick={() => setTested((previous) => new Set(previous).add(current))}>
              {HEAD_LAB.test_pattern}
            </Button>
          </div>
          <Link className="text-primary underline-offset-4 hover:underline" href={`/head-zoom/?version=${version}&block=${selBlock}&head=${selHead}`}>
            {HEAD_LAB.step_through} ›
          </Link>
          {tested.has(current) && (
            <div className="flex flex-col gap-1" role="group" aria-label="The three pattern tests">
              <h3 className="text-xs font-medium tracking-wide text-muted-foreground">{HEAD_LAB.three_tests}</h3>
              {TESTS.map((test) => {
                const value = tests[selBlock][test.key][selHead];
                return (
                  <div key={test.key} className="flex flex-wrap gap-4 text-sm">
                    <span className="w-64" title={HEAD_LAB[test.what]}>
                      {HEAD_LAB[test.name]}
                    </span>
                    <span className={`font-mono ${value >= THRESHOLD ? "text-success" : "text-muted-foreground"}`}>
                      {value.toFixed(2)} {value >= THRESHOLD ? "passes" : "no"}
                    </span>
                  </div>
                );
              })}
              <p className="max-w-prose text-xs text-muted-foreground">{HEAD_LAB.tests_caveat}</p>
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardContent className="flex flex-col gap-2">
          <h2 className="text-xs font-medium tracking-wide text-muted-foreground">
            {off.size
              ? fill(HEAD_LAB.what_changed, { count: off.size }).replace("HEAD OFF", off.size > 1 ? "HEADS OFF" : "HEAD OFF")
              : HEAD_LAB.what_changes}
          </h2>
          {off.size === 0 ? (
            <p className="max-w-prose text-sm text-muted-foreground">{HEAD_LAB.switch_head}</p>
          ) : (
            <>
              <p>{HEAD_LAB.example_prompt}</p>
              <Reply label={HEAD_LAB.all_heads_on} reply={changed?.on} />
              <Reply label={fill(HEAD_LAB.with_heads_off, { count: off.size })} reply={changed?.off} />
              <p className="font-mono text-sm" title={HEAD_LAB.big_rise}>
                {HEAD_LAB.error_change}:{" "}
                {changed?.bits === undefined ? "…" : changed.bits === null ? "—" : `${changed.bits.before.toFixed(2)} → ${changed.bits.after.toFixed(2)} bits/char`}
              </p>
              <p className="max-w-prose text-xs text-muted-foreground">{HEAD_LAB.big_rise}</p>
              <p className="text-sm text-notice">
                {HEAD_LAB.you_guessed}: {guess ? guess.toLowerCase() : "no guess"}
              </p>
            </>
          )}
        </CardContent>
      </Card>

      <details className="max-w-prose text-sm text-muted-foreground">
        <summary className="cursor-pointer text-foreground">{HEAD_LAB.how_switching}</summary>
        <div className="mt-2 flex flex-col gap-2">
          <p>
            {HEAD_LAB.switching_note} {HEAD_LAB.replaces_output}
          </p>
          <p>
            {HEAD_LAB.ranking_note} {HEAD_LAB.switches_each}
          </p>
          <p>
            {HEAD_LAB.tests_note} {HEAD_LAB.measure_attention}
          </p>
          <p>{HEAD_LAB.mean_exact}</p>
        </div>
      </details>
    </Shell>
  );
}

function Reply({ label, reply }: { label: string; reply: string | null | undefined }) {
  return (
    <div>
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="max-w-prose">{reply === undefined ? "…" : (reply ?? "—")}</p>
    </div>
  );
}

function Shell({ children }: { children: React.ReactNode }) {
  return (
    <main className="mx-auto flex w-full max-w-5xl flex-col gap-4 p-8">
      <p className="text-xs text-muted-foreground">Network designer › Look inside</p>
      <h1 className="text-2xl font-semibold tracking-tight">{HEAD_LAB.h1}</h1>
      <p className="max-w-prose text-muted-foreground">{HEAD_LAB.lead}</p>
      {children}
    </main>
  );
}
