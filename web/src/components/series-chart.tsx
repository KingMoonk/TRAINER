"use client";

import { CartesianGrid, Line, LineChart, ReferenceArea, XAxis, YAxis } from "recharts";
import {
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart";

export type Point = { x: number; y: number };
export type Series = {
  id: string; label: string; points: Point[];
  axis?: "left" | "right";
  color?: string;
  opacity?: number;
  strokeWidth?: number;
  dash?: string;
  interpolation?: "linear" | "monotone";
};
export type Band = { id: string; label: string; from: number; to: number; color?: string };

const COLORS = ["var(--chart-1)", "var(--chart-2)", "var(--chart-3)", "var(--chart-4)", "var(--chart-5)"];

/**
 * The one line chart: loss, BPC and run comparison all draw through it, and no page draws its own. Each series is a
 * line over a shared x axis (step or tokens); pass several to compare runs. Series ids must be unique.
 */
export function SeriesChart({
  series,
  xLabel,
  yLabel,
  rightLabel,
  bands = [],
  className,
}: {
  series: Series[];
  xLabel: string;
  yLabel: string;
  rightLabel?: string;
  bands?: Band[];
  className?: string;
}) {
  const config: ChartConfig = Object.fromEntries(
    series.map((s, i) => [s.id, { label: s.label, color: s.color ?? COLORS[i % COLORS.length] }]),
  );
  // One row per x with a column per series, which is the shape recharts wants.
  const rows = new Map<number, Record<string, number>>();
  for (const s of series) for (const p of s.points) rows.set(p.x, { ...rows.get(p.x), x: p.x, [s.id]: p.y });
  const data = [...rows.values()].sort((a, b) => a.x - b.x);
  const firstX = Math.min(data[0]?.x ?? 0, ...bands.map((band) => band.from));
  return (
    <ChartContainer config={config} className={className ?? "h-64 w-full"}>
      <LineChart data={data} margin={{ left: 8, right: 8 }}>
        <CartesianGrid vertical={false} />
        <XAxis dataKey="x" type="number" domain={[firstX, "dataMax"]} tickLine={false} label={{ value: xLabel, position: "insideBottom", offset: -4 }} height={40} />
        <YAxis yAxisId="left" tickLine={false} axisLine={false} width={48} label={{ value: yLabel, angle: -90, position: "insideLeft" }} />
        {series.some((s) => s.axis === "right") && <YAxis yAxisId="right" orientation="right" domain={[0, "auto"]}
          tickLine={false} axisLine={false} width={68} tickFormatter={(value: number) => value === 0 ? "0" : value.toExponential(0)}
          label={{ value: rightLabel, angle: 90, position: "insideRight" }} />}
        {bands.map((band, index) => <ReferenceArea key={band.id} yAxisId="left" x1={band.from} x2={band.to} ifOverflow="hidden"
          className="schedule-band" data-phase={band.label} fill={band.color ?? COLORS[index % COLORS.length]} fillOpacity={0.08} strokeOpacity={0}
          label={{ value: band.label, position: "insideTop", fontSize: 10, fill: "var(--muted-foreground)" }} />)}
        <ChartTooltip content={<ChartTooltipContent formatter={(value, name) => <>
          <span className="text-muted-foreground">{config[String(name)]?.label ?? String(name)}</span>
          <span className="ml-auto font-mono font-medium tabular-nums">{typeof value === "number"
            ? series.find((s) => s.id === name)?.axis === "right" ? value.toExponential(2) : value.toLocaleString("en-US", { maximumFractionDigits: 4 })
            : String(value)}</span>
        </>} />} />
        {series.length > 1 && <ChartLegend content={<ChartLegendContent className="flex-wrap gap-x-4 gap-y-1" />} />}
        {series.map((s) => (
          <Line key={s.id} dataKey={s.id} yAxisId={s.axis ?? "left"} type={s.interpolation ?? "monotone"}
            className={`series-${s.id}`} stroke={`var(--color-${s.id})`} strokeOpacity={s.opacity ?? 1}
            strokeWidth={s.strokeWidth ?? 2} strokeDasharray={s.dash} dot={false} connectNulls isAnimationActive={false} />
        ))}
      </LineChart>
    </ChartContainer>
  );
}
