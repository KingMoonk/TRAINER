"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { api, unwrap } from "@/lib/api";
import { HEAD_ZOOM } from "@/app/head-zoom/copy";
import {
  EXAMPLE_PROMPT,
  TOLERANCE,
  attentionBlocks,
  derive,
  fill,
  readVersion,
  runJob,
  sentence,
  showToken,
  type Derived,
  type Forward,
  type Pattern,
  type Version,
} from "@/app/head-lab/inspect";
import { PAGE_MAIN, PageHeader } from "@/components/page-header";

/** A longer prompt is cut from the left, keeping the query at the end. */
const MAX_TOKENS = 12;
const STEPS = ["raw", "scale", "mask", "softmax", "weighted"] as const;
type StepKey = (typeof STEPS)[number];
const STEP_NAME: Record<StepKey, string> = {
  raw: HEAD_ZOOM.raw_scores,
  scale: HEAD_ZOOM.scale_step,
  mask: HEAD_ZOOM.mask_step,
  softmax: HEAD_ZOOM.softmax_step,
  weighted: HEAD_ZOOM.weighted_step,
};

type HeadStats = { entropy: number[]; mean_distance: number[]; first_token_mass: number[]; previous_token_mass: number[] };
type Loaded = { version: Version; forward: Forward; blocks: Record<number, number> };

async function load(number?: number): Promise<Loaded | string> {
  const version = await readVersion(undefined, number);
  if (!version?.best_checkpoint) return "No trained checkpoint yet: train a version in Pretraining to look inside its heads.";
  let forward = await runJob<Forward>(unwrap(api.POST("/inspect/forward", { body: { version: version.number, text: EXAMPLE_PROMPT, hooks: true } })));
  if (forward.ids[0].length > MAX_TOKENS) {
    forward = await runJob<Forward>(unwrap(api.POST("/inspect/forward", { body: { version: version.number, ids: forward.ids[0].slice(-MAX_TOKENS), hooks: true } })));
  }
  const blocks = attentionBlocks(forward.hooks);
  if (Object.keys(blocks).length === 0) return "This model has no attention head to look inside.";
  return { version, forward, blocks };
}

const number = (value: string | null, fallback: number) => (value !== null && /^\d+$/.test(value) ? Number(value) : fallback);
const three = (value: number) => String(Number(value.toPrecision(3)));

export function HeadZoom({ requestedVersion }: { requestedVersion?: number }) {
  const params = useSearchParams();
  const [loaded, setLoaded] = useState<Loaded | string | null>(null);
  const [pick, setPick] = useState<[number, number]>([number(params.get("block"), 0), number(params.get("head"), 0)]);
  const [step, setStep] = useState(1);
  const [cell, setCell] = useState<[number, number] | null>(null);

  useEffect(() => {
    let dead = false;
    load(requestedVersion).then(
      (result) => !dead && setLoaded(result),
      (error) => !dead && setLoaded(sentence(error)),
    );
    return () => {
      dead = true;
    };
  }, [requestedVersion]);

  const data = typeof loaded === "object" && loaded ? loaded : null;
  // A block or head the model does not have lands on its first block and head, as the Tk page does.
  const blockNumbers = data ? Object.keys(data.blocks).map(Number) : [];
  const valid = data !== null && data.blocks[pick[0]] !== undefined && pick[1] < data.blocks[pick[0]];
  const block = valid ? pick[0] : (blockNumbers[0] ?? 0);
  const head = valid ? pick[1] : 0;

  const derived = useMemo<Derived | null>(() => (data ? derive(data.forward, block, head) : null), [data, block, head]);
  const stats = useHeadStats(data, block);

  const select = (b: number, h: number) => {
    setPick([b, h]);
    window.history.replaceState(null, "", `?block=${b}&head=${h}`);
  };

  if (loaded === null) {
    return (
      <Shell>
        <p className="text-muted-foreground" role="status">Reading the model…</p>
      </Shell>
    );
  }
  if (typeof loaded === "string" || data === null || derived === null) {
    return (
      <Shell>
        <p className="max-w-prose text-muted-foreground">{typeof loaded === "string" ? loaded : ""}</p>
      </Shell>
    );
  }

  const T = derived.tokens.length;
  const selected: [number, number] = cell && Math.max(...cell) < T ? cell : [Math.max(0, T - 2), T - 1];
  const [i, j] = selected;
  const last = T - 1;
  const key = STEPS[step - 1];
  const sqrtD = Math.sqrt(derived.dHead);
  const values = { d: derived.dHead, sqrt_d: three(sqrtD), token: showToken(derived.tokens[last]) };
  const ranked = [...derived.weights.keys()].sort((a, b) => derived.weights[b][last] - derived.weights[a][last]);
  const note = () => {
    if (step === 1) return HEAD_ZOOM.raw_note;
    if (step === 2) return HEAD_ZOOM.scale_note;
    if (step === 3) return HEAD_ZOOM.mask_note;
    const [a, b] = [ranked[0], ranked[ranked.length > 1 ? 1 : 0]];
    if (step === 4) {
      return fill(HEAD_ZOOM.softmax_note, {
        token: showToken(derived.tokens[last]),
        p1: derived.weights[a][last].toFixed(2),
        token1: showToken(derived.tokens[a]),
        p2: derived.weights[b][last].toFixed(2),
        token2: showToken(derived.tokens[b]),
      });
    }
    return fill(HEAD_ZOOM.weighted_note, values);
  };
  let why = fill(HEAD_ZOOM[`${key}_why`], values);
  if (step <= 3) why += ` ${note()}`;
  if (step === 2 && !derived.plainScale) why += " These are the model's real pre-mask scores, so a position bias or a μP scale is included.";
  const d_model = (data.forward.hooks.embed as number[][][])[0][0].length;
  const blockCounts = data.blocks[block];
  const selectedStats = typeof stats === "object" && stats ? stats : null;

  return (
    <Shell>
      <div className="flex flex-wrap items-center gap-4">
        <ToggleGroup
          type="single"
          variant="outline"
          value={String(block)}
          onValueChange={(value) => value && select(Number(value), 0)}
          aria-label="Block"
        >
          {blockNumbers.map((b) => (
            <ToggleGroupItem key={b} value={String(b)}>
              Block {b + 1}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
        <ToggleGroup
          type="single"
          variant="outline"
          value={String(head)}
          onValueChange={(value) => value && select(block, Number(value))}
          aria-label="Head"
        >
          {Array.from({ length: blockCounts }, (_, h) => (
            <ToggleGroupItem key={h} value={String(h)}>
              Head {h + 1}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
      </div>

      <p className="text-muted-foreground" title={derived.error <= TOLERANCE ? fill(HEAD_ZOOM.checked, { error: derived.error.toExponential(1), tol: TOLERANCE }) : undefined}>
        {fill(HEAD_ZOOM.head_label, { block: block + 1, head: head + 1, tokens: T })}
      </p>
      {derived.error > TOLERANCE && (
        <p role="alert" className="max-w-prose rounded-md border border-warning/50 bg-warning/10 p-3 text-sm text-warning">
          The numbers derived here differ from the model&apos;s own attention pattern by {derived.error.toPrecision(2)} (allowed {TOLERANCE}). A hook may be wrong, so do not
          trust this picture.
        </p>
      )}

      <div className="flex flex-wrap gap-2" role="group" aria-label="Steps">
        {STEPS.map((s, n) => (
          <Button key={s} variant={step === n + 1 ? "default" : "outline"} aria-pressed={step === n + 1} onClick={() => setStep(n + 1)}>
            {n + 1}. {STEP_NAME[s]}
          </Button>
        ))}
      </div>
      <p className="font-mono text-sm text-muted-foreground">{fill(HEAD_ZOOM[`${key}_unit`], values)}</p>

      <div className="flex flex-wrap items-start gap-4">
        <ScoreGrid derived={derived} step={step} selected={selected} onPick={setCell} />
        <Card className="min-w-72 flex-1">
          <CardContent className="flex flex-col gap-2">
            <p className="text-xs font-medium tracking-wide text-muted-foreground">{fill(HEAD_ZOOM.step_label, { n: step })}</p>
            <h2 className="text-lg font-semibold" title={why}>
              {STEP_NAME[key]}
            </h2>
            <p className="text-sm text-muted-foreground">{why}</p>
            <p className="font-mono text-sm">{fill(HEAD_ZOOM[`${key}_formula`], values)}</p>
            {step > 3 && <p className="text-xs text-muted-foreground">{note()}</p>}

            <p className="mt-2 text-xs font-medium tracking-wide text-muted-foreground">{HEAD_ZOOM.selected_cell}</p>
            <p className="text-sm">
              k{i + 1} “{showToken(derived.tokens[i])}” · q{j + 1} “{showToken(derived.tokens[j])}”
            </p>
            <dl className="grid grid-cols-[1fr_auto] gap-x-4 gap-y-1 text-sm">
              <dt className="text-muted-foreground">{HEAD_ZOOM.cell_label}</dt>
              <dd className="font-mono">{derived.raw[i][j].toFixed(2)}</dd>
              <dt className="text-muted-foreground">{fill(HEAD_ZOOM.scaled, values)}</dt>
              <dd className="font-mono">{derived.scaled[i][j].toFixed(2)}</dd>
              <dt className="text-muted-foreground">{HEAD_ZOOM.after_mask}</dt>
              <dd className="font-mono">{i > j ? "−∞" : derived.scaled[i][j].toFixed(2)}</dd>
              <dt className="text-muted-foreground">{HEAD_ZOOM.softmax_weight}</dt>
              <dd className="font-mono">{derived.weights[i][j].toFixed(2)}</dd>
            </dl>

            {step === 5 && (
              <>
                <p
                  className="mt-2 text-xs font-medium tracking-wide text-muted-foreground"
                  title={fill(HEAD_ZOOM.result_is, { dim: derived.dHead, token: showToken(derived.tokens[ranked[0]]), d: d_model })}
                >
                  {fill(HEAD_ZOOM.output_at, { token: showToken(derived.tokens[last]) })}
                </p>
                <p className="text-xs text-muted-foreground">{fill(HEAD_ZOOM.result_is, { dim: derived.dHead, token: showToken(derived.tokens[ranked[0]]), d: d_model })}</p>
                <ul className="flex flex-col gap-1" aria-label="Biggest contributors">
                  {ranked.slice(0, 5).map((r) => {
                    const w = derived.weights[r][last];
                    return (
                      <li key={r} className="flex items-center gap-2 text-sm">
                        <span className="w-20 truncate">{showToken(derived.tokens[r])}</span>
                        <span className="h-2.5 w-32 rounded-sm bg-muted">
                          <span className="block h-full rounded-sm bg-foreground/70" style={{ width: `${w * 100}%` }} />
                        </span>
                        <span className="text-xs text-muted-foreground">
                          {w.toFixed(2)} · {Math.round(w * 100)}%
                        </span>
                      </li>
                    );
                  })}
                </ul>
                <Link className="text-sm text-primary underline-offset-4 hover:underline" href={`/value-map/?version=${data.version.number}`}>
                  {HEAD_ZOOM.where_meet} ›
                </Link>
              </>
            )}
          </CardContent>
        </Card>
      </div>
      <p className="text-xs text-muted-foreground">{HEAD_ZOOM.rows_keys}</p>

      <Card>
        <CardContent className="flex flex-col gap-2">
          <h2 className="text-xs font-medium tracking-wide text-muted-foreground">THIS HEAD&apos;S ATTENTION, AVERAGED OVER THE PROMPT</h2>
          {typeof stats === "string" ? (
            <p className="text-sm text-muted-foreground">{stats}</p>
          ) : selectedStats === null ? (
            <p className="text-sm text-muted-foreground" role="status">Measuring…</p>
          ) : (
            <dl className="grid max-w-md grid-cols-[1fr_auto] gap-x-4 gap-y-1 text-sm" aria-label="Head statistics">
              <dt className="text-muted-foreground" title="How spread out the attention is: 0 means all on one piece.">Entropy</dt>
              <dd className="font-mono">{selectedStats.entropy[head].toFixed(2)} nats</dd>
              <dt className="text-muted-foreground" title="How far back the head looks on average, in pieces.">Mean distance</dt>
              <dd className="font-mono">{selectedStats.mean_distance[head].toFixed(2)} pieces</dd>
              <dt className="text-muted-foreground" title="Share of attention on the first piece. A high share often means the head is idle here.">First-piece share</dt>
              <dd className="font-mono">{selectedStats.first_token_mass[head].toFixed(2)}</dd>
              <dt className="text-muted-foreground" title="Share of attention on the piece just before.">Previous-piece share</dt>
              <dd className="font-mono">{selectedStats.previous_token_mass[head].toFixed(2)}</dd>
            </dl>
          )}
        </CardContent>
      </Card>

      <Link className="text-sm text-primary underline-offset-4 hover:underline" href={`/head-lab/${data ? `?version=${data.version.number}` : ""}`}>
        Back to the Head lab ›
      </Link>
    </Shell>
  );
}

/** One head's numbers for the selected block from POST /inspect/head-stats, or a sentence when they could not be read. */
function useHeadStats(data: Loaded | null, block: number): HeadStats | string | null {
  const [result, setResult] = useState<{ key: string; value: HeadStats | string } | null>(null);
  const key = data ? `${data.version.number}:${block}` : "";
  useEffect(() => {
    if (!data) return;
    let dead = false;
    const pattern = data.forward.hooks[`block${block}.attn.pattern`] as Pattern;
    unwrap(api.POST("/inspect/head-stats", { body: { pattern } })).then(
      (value) => !dead && setResult({ key, value: value as HeadStats }),
      (error) => !dead && setResult({ key, value: sentence(error) }),
    );
    return () => {
      dead = true;
    };
  }, [data, block, key]);
  return result && result.key === key ? result.value : null;
}

/** Amber for positive, blue for negative, stronger with size. */
function signed(value: number, scale: number): string {
  const share = Math.round((Math.min(1, Math.abs(value) / scale) * 0.85 + 0.08) * 100);
  return `color-mix(in oklch, ${value >= 0 ? "var(--heat-positive)" : "var(--heat-negative)"} ${share}%, var(--muted))`;
}

type Look = { text: string; background: string; dim?: boolean; bright?: boolean };

function look(d: Derived, step: number, i: number, j: number): Look {
  const T = d.tokens.length;
  const masked = i > j;
  const scale = Math.max(1e-9, ...d.scaled.flat().map(Math.abs));
  const rawScale = Math.max(1e-9, ...d.raw.flat().map(Math.abs));
  if (step === 1) return { text: d.raw[i][j].toFixed(1), background: signed(d.raw[i][j], rawScale) };
  if (step === 2) return { text: d.scaled[i][j].toFixed(2), background: signed(d.scaled[i][j], scale) };
  if (masked) return { text: step === 3 ? "−∞" : "0", background: "var(--cell-masked)", dim: true };
  if (step === 3) return { text: d.scaled[i][j].toFixed(2), background: signed(d.scaled[i][j], scale) };
  const w = d.weights[i][j];
  const faded = step === 5 && j !== T - 1;
  const share = Math.round((faded ? w * 0.25 : w * 0.92 + 0.04) * 100);
  return { text: w.toFixed(2), background: `color-mix(in oklch, var(--foreground) ${share}%, var(--muted))`, dim: faded, bright: w > 0.5 && !faded };
}

/** Rows are keys, columns are queries. Click a cell or use the arrow keys to move the selection. */
function ScoreGrid({ derived, step, selected, onPick }: { derived: Derived; step: number; selected: [number, number]; onPick: (cell: [number, number]) => void }) {
  const T = derived.tokens.length;
  const move = (event: React.KeyboardEvent) => {
    const delta: Record<string, [number, number]> = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] };
    const d = delta[event.key];
    if (!d) return;
    event.preventDefault();
    onPick([Math.min(T - 1, Math.max(0, selected[0] + d[0])), Math.min(T - 1, Math.max(0, selected[1] + d[1]))]);
  };
  return (
    <div
      role="grid"
      tabIndex={0}
      aria-label="Scores: rows are keys, columns are queries"
      onKeyDown={move}
      className="inline-grid gap-px rounded-md border bg-card p-2 outline-none focus-visible:ring-3 focus-visible:ring-ring/50"
      style={{ gridTemplateColumns: `96px repeat(${T}, 52px)` }}
    >
      <span />
      {derived.tokens.map((token, col) => (
        <span key={col} role="columnheader" className={`text-center text-xs ${step === 5 && col === T - 1 ? "text-warning" : "text-muted-foreground"}`}>
          {showToken(token).slice(0, 7)}
          <br />
          <span className="text-[10px]">q{col + 1}</span>
        </span>
      ))}
      {derived.tokens.map((token, row) => (
        <Row key={row} derived={derived} step={step} row={row} token={token} selected={selected} onPick={onPick} />
      ))}
    </div>
  );
}

function Row({ derived, step, row, token, selected, onPick }: { derived: Derived; step: number; row: number; token: string; selected: [number, number]; onPick: (cell: [number, number]) => void }) {
  return (
    <>
      <span role="rowheader" className="self-center pr-2 text-right text-xs text-muted-foreground">
        {showToken(token).slice(0, 8)} k{row + 1}
      </span>
      {derived.tokens.map((_, col) => {
        const view = look(derived, step, row, col);
        const on = selected[0] === row && selected[1] === col;
        return (
          <button
            key={col}
            type="button"
            tabIndex={-1}
            role="gridcell"
            aria-selected={on}
            aria-label={`k${row + 1} q${col + 1}: ${view.text}`}
            onClick={() => onPick([row, col])}
            className={`h-8 rounded-sm font-mono text-xs ${on ? "outline-2 outline-primary" : ""} ${view.dim ? "text-muted-foreground" : view.bright ? "text-background" : "text-foreground"}`}
            style={{ background: view.background }}
          >
            {view.text}
          </button>
        );
      })}
    </>
  );
}

function Shell({ children }: { children: React.ReactNode }) {
  return (
    <main className={PAGE_MAIN}>
      <PageHeader breadcrumb="Network designer › Look inside" title={HEAD_ZOOM.h1} lead={HEAD_ZOOM.lead} />
      {children}
    </main>
  );
}
