"use client";

import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { Suspense, useState } from "react";
import { useVersionParameter } from "@/lib/version-param";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { api, unwrap } from "@/lib/api";
import { routeFor } from "@/lib/routes";
import { cn } from "@/lib/utils";
import { reason, runJob } from "../trace/use-job";
import { VALUE_MAP, fill } from "./copy";

type ValueMap = { d_model: number; heads: number; blocks: Record<string, number>; singular: number[] | null };

const HEADS = [1, 2, 4, 8];
const STRIPES = [
  "fill-amber-500",
  "fill-blue-500",
  "fill-violet-500",
  "fill-pink-400",
  "fill-indigo-300",
  "fill-emerald-500",
  "fill-orange-400",
  "fill-purple-300",
];

/** Numerical rank: singular values above tol of the largest. */
function rankOf(singular: number[], tol = 1e-6): number {
  const top = Math.max(0, ...singular);
  return top > 0 ? singular.filter((s) => s > tol * top).length : 0;
}

export default function Page() {
  return <Suspense fallback={<Skeleton className="m-8 h-64" />}><VersionedMap /></Suspense>;
}

function VersionedMap() {
  const version = useVersionParameter();
  return <ValueMapPage key={version ?? "best"} version={version} />;
}

function ValueMapPage({ version }: { version?: number }) {
  const route = routeFor("value_map");
  const [heads, setHeads] = useState(4);
  const [block, setBlock] = useState(0);
  const [head, setHead] = useState(0);

  // version null is the best trained version.
  const map = useQuery({
    queryKey: ["inspect", "value-map", version, heads, block, head],
    queryFn: ({ signal }) => runJob<ValueMap>(() => unwrap(api.POST("/inspect/value-map", { body: { version, heads, block, head } })), signal),
    placeholderData: keepPreviousData,
    retry: false,
    staleTime: Infinity,
    refetchOnWindowFocus: false,
  });
  const info = map.data;
  const d = info?.d_model;
  const fit = (text: string) => (d === undefined ? text : text.replace("256", String(d)));

  return (
    <main className="mx-auto flex w-full max-w-5xl min-w-0 flex-col gap-4 p-4 md:p-8">
      <p className="text-xs text-muted-foreground">Network designer › Look inside</p>
      <h1 className="text-2xl font-semibold tracking-tight">{route.title}</h1>
    {version !== undefined && <p className="text-sm text-muted-foreground">Version {version}</p>}
      <p className="max-w-prose text-muted-foreground">{route.lead}</p>

      <div className="flex flex-wrap items-center gap-2">
        {d === undefined ? (
          map.isError ? (
            <span className="text-muted-foreground">{VALUE_MAP.heads_label}</span>
          ) : (
            <Skeleton aria-label="Loading d_model" className="h-5 w-32" />
          )
        ) : (
          <span className="text-muted-foreground">{fit(VALUE_MAP.d_model)}</span>
        )}
        <ToggleGroup
          type="single"
          variant="outline"
          spacing={0}
          aria-label={VALUE_MAP.heads_label}
          value={String(heads)}
          onValueChange={(value) => {
            if (!value) return;
            setHeads(Number(value));
            setBlock(0);
            setHead(0);
          }}
        >
          {HEADS.map((n) => (
            <ToggleGroupItem key={n} value={String(n)}>
              {n}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
      </div>

      {map.isError && (
        <p role="alert" className="text-muted-foreground">
          {reason(map.error, VALUE_MAP.no_numbers, VALUE_MAP.server_error)}
        </p>
      )}

      {map.isPending && !map.isError && (
        <div role="status" aria-label="Loading the value map" className="flex flex-col gap-3">
          <Skeleton className="h-7 w-64 max-w-full" />
          <Skeleton className="h-72 w-full" />
        </div>
      )}

      {info && d !== undefined && d % heads !== 0 && (
        <p role="alert" className="text-amber-500">
          {fill(VALUE_MAP.not_equal, { d, heads })}
        </p>
      )}

      {info && d !== undefined && d % heads === 0 && (
        <div aria-busy={map.isFetching} className={cn("flex min-w-0 flex-col gap-4", map.isFetching && "opacity-60")}>
          <Result
            info={info}
            heads={heads}
            block={block}
            head={head}
            fit={fit}
            pick={(b, h) => {
              setBlock(b);
              setHead(h);
            }}
          />
        </div>
      )}
    </main>
  );
}

function Result({
  info,
  heads,
  block,
  head,
  fit,
  pick,
}: {
  info: ValueMap;
  heads: number;
  block: number;
  head: number;
  fit: (text: string) => string;
  pick: (block: number, head: number) => void;
}) {
  const d = info.d_model;
  const hd = d / heads;
  const blocks = Object.keys(info.blocks)
    .map(Number)
    .sort((a, b) => a - b);
  const singular = info.singular;
  const matches = heads === info.heads;

  return (
    <>
      <h2 className="text-lg font-bold break-words">
        {fill(fit(VALUE_MAP.head_dim), { heads })} {hd}
      </h2>
      <Card>
        <CardContent className="flex min-w-0 flex-col gap-4">
          <h3 className="text-xs font-medium tracking-wider text-muted-foreground">{VALUE_MAP.one_head}</h3>
          <div className="flex max-w-prose flex-col gap-2 text-sm text-muted-foreground">
            <p>
              {fit(VALUE_MAP.together)} {hd}. {fill(VALUE_MAP.each_head, { hd })}
            </p>
            <p>{fill(VALUE_MAP.weighting, { hd })}</p>
            <p>
              {fill(VALUE_MAP.params, {
                per_head: (2 * d * hd).toLocaleString("en-US"),
                heads,
                total: (2 * d * d).toLocaleString("en-US"),
                d,
              })}
            </p>
          </div>
          <Funnel d={d} hd={hd} heads={heads} fit={fit} />

          {singular && matches && (
            <>
              {(blocks.length > 1 || heads > 1) && (
                <div className="flex flex-wrap gap-3">
                  <ToggleGroup
                    type="single"
                    variant="outline"
                    spacing={0}
                    aria-label="Block"
                    className="flex-wrap"
                    value={String(block)}
                    onValueChange={(value) => value && pick(Number(value), 0)}
                  >
                    {blocks.map((b) => (
                      <ToggleGroupItem key={b} value={String(b)}>
                        Block {b + 1}
                      </ToggleGroupItem>
                    ))}
                  </ToggleGroup>
                  <ToggleGroup
                    type="single"
                    variant="outline"
                    spacing={0}
                    aria-label="Head"
                    className="flex-wrap"
                    value={String(head)}
                    onValueChange={(value) => value && pick(block, Number(value))}
                  >
                    {Array.from({ length: heads }, (_, h) => (
                      <ToggleGroupItem key={h} value={String(h)}>
                        Head {h + 1}
                      </ToggleGroupItem>
                    ))}
                  </ToggleGroup>
                </div>
              )}
              <p className="max-w-prose text-sm">{fill(VALUE_MAP.measured, { block: block + 1, head: head + 1, rank: rankOf(singular), hd })}</p>
              <Bars singular={singular} hd={hd} />
            </>
          )}
          {!matches && <p className="text-sm text-muted-foreground">{fill(VALUE_MAP.measured_other, { heads: info.heads })}</p>}
          {matches && !singular && <p className="text-sm text-muted-foreground">{VALUE_MAP.measured_none}</p>}
        </CardContent>
      </Card>
    </>
  );
}

/** x (d) -> W_V -> v (hd) -> W_O -> delta x (d). Heights are to scale, with an 8px floor so the bottleneck shows. Scrolls sideways on a phone. */
function Funnel({ d, hd, heads, fit }: { d: number; hd: number; heads: number; fit: (text: string) => string }) {
  const H = 260;
  const mid = H / 2 - 8;
  const full = 200;
  const narrow = Math.max(8, (full * hd) / d);
  const [x0, x1, x2, x3, x4] = [20, 150, 330, 510, 640];
  const top = mid - full / 2;
  const bottom = mid + full / 2;
  const mono = "font-mono fill-muted-foreground text-[12px]";
  const ends: [number, string][] = [
    [x0, fit(VALUE_MAP.x_label)],
    [x4, fit(VALUE_MAP.delta_x)],
  ];
  return (
    <div className="max-w-full overflow-x-auto" tabIndex={0} aria-label="Funnel diagram">
      <svg viewBox={`0 0 720 ${H}`} role="img" aria-label={`${d} to ${hd} to ${d}`} className="h-auto w-full min-w-[560px]">
        {ends.map(([x, label]) => (
          <g key={label}>
            <rect x={x} y={top} width={24} height={full} className="fill-muted stroke-border" />
            <text x={x + 12} y={bottom + 14} textAnchor="middle" className={mono}>
              {label}
            </text>
          </g>
        ))}
        <text x={(x1 + x2) / 2} y={14} textAnchor="middle" className="fill-foreground text-[13px] font-bold">
          {`${VALUE_MAP.w_v}  ${d} → ${hd}`}
        </text>
        <text x={(x2 + x3) / 2} y={14} textAnchor="middle" className="fill-foreground text-[13px] font-bold">
          {`${VALUE_MAP.w_o}  ${hd} → ${d}`}
        </text>
        <polygon
          points={`${x0 + 24},${top} ${x2},${mid - narrow / 2} ${x2},${mid + narrow / 2} ${x0 + 24},${bottom}`}
          className="fill-amber-500/20 stroke-amber-500"
        />
        <polygon
          points={`${x2},${mid - narrow / 2} ${x4},${top} ${x4},${bottom} ${x2},${mid + narrow / 2}`}
          className="fill-blue-500/20 stroke-blue-500"
        />
        <rect x={x2 - 6} y={mid - narrow / 2} width={12} height={narrow} className="fill-violet-500" />
        <text x={x2} y={bottom + 14} textAnchor="middle" className="fill-violet-500 font-mono text-[12px]">
          {fill(VALUE_MAP.v_label, { hd })}
        </text>
        {Array.from({ length: Math.min(heads, STRIPES.length) }, (_, i) => (
          <rect key={i} x={x4 + 24} y={top + (full * i) / heads} width={10} height={full / heads - 1} className={STRIPES[i]} />
        ))}
      </svg>
    </div>
  );
}

/** The singular values, largest first: the first head_dim bars are the ones that can be non-zero. */
function Bars({ singular, hd }: { singular: number[]; hd: number }) {
  const n = singular.length;
  const top = Math.max(...singular, 1e-12);
  return (
    <svg viewBox="0 0 420 60" preserveAspectRatio="none" role="img" aria-label={`${n} singular values`} className="h-16 w-full max-w-[420px]">
      {singular.map((s, i) => {
        const h = (56 * s) / top;
        return <rect key={i} x={(i * 420) / n + 1} y={58 - h} width={Math.max(0.5, 420 / n - 2)} height={h} className={i < hd ? "fill-violet-500" : "fill-border"} />;
      })}
    </svg>
  );
}
