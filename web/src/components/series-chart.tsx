"use client";

import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts";
import {
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart";

export type Point = { x: number; y: number };
export type Series = { id: string; label: string; points: Point[] };

const COLORS = ["var(--chart-1)", "var(--chart-2)", "var(--chart-3)", "var(--chart-4)", "var(--chart-5)"];

/**
 * The one line chart: loss, BPC and run comparison all draw through it, and no page draws its own. Each series is a
 * line over a shared x axis (step or tokens); pass several to compare runs. Series ids must be unique.
 */
export function SeriesChart({
  series,
  xLabel,
  yLabel,
  className,
}: {
  series: Series[];
  xLabel: string;
  yLabel: string;
  className?: string;
}) {
  const config: ChartConfig = Object.fromEntries(
    series.map((s, i) => [s.id, { label: s.label, color: COLORS[i % COLORS.length] }]),
  );
  // One row per x with a column per series, which is the shape recharts wants.
  const rows = new Map<number, Record<string, number>>();
  for (const s of series) for (const p of s.points) rows.set(p.x, { ...rows.get(p.x), x: p.x, [s.id]: p.y });
  const data = [...rows.values()].sort((a, b) => a.x - b.x);
  return (
    <ChartContainer config={config} className={className ?? "h-64 w-full"}>
      <LineChart data={data} margin={{ left: 8, right: 8 }}>
        <CartesianGrid vertical={false} />
        <XAxis dataKey="x" type="number" domain={["dataMin", "dataMax"]} tickLine={false} label={{ value: xLabel, position: "insideBottom", offset: -4 }} height={40} />
        <YAxis tickLine={false} axisLine={false} width={48} label={{ value: yLabel, angle: -90, position: "insideLeft" }} />
        <ChartTooltip content={<ChartTooltipContent />} />
        {series.length > 1 && <ChartLegend content={<ChartLegendContent />} />}
        {series.map((s) => (
          <Line key={s.id} dataKey={s.id} type="monotone" stroke={`var(--color-${s.id})`} strokeWidth={2} dot={false} connectNulls isAnimationActive={false} />
        ))}
      </LineChart>
    </ChartContainer>
  );
}
