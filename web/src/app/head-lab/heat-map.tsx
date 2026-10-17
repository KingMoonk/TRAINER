import { cn } from "@/lib/utils";
import { showToken } from "@/app/head-lab/inspect";

/**
 * A grid of attention shares: rows are queries, columns are keys. A null cell is the future, which a piece cannot see. Values are
 * 0 to 1; brighter is more attention. `tokens` adds a label to each row and column.
 */
export function HeatMap({
  cells,
  size,
  tokens,
  label,
  className,
}: {
  cells: (number | null)[][];
  size: number;
  tokens?: string[];
  label: string;
  className?: string;
}) {
  const T = cells.length;
  const labelled = tokens !== undefined;
  const labelWidth = labelled ? 48 : 0;
  return (
    <div
      role="img"
      aria-label={label}
      className={cn("inline-grid gap-px", className)}
      style={{ gridTemplateColumns: `${labelled ? `${labelWidth}px ` : ""}repeat(${T}, ${size}px)` }}
    >
      {labelled && <span />}
      {labelled && tokens.map((token, k) => <span key={k} className="truncate text-center text-[10px] text-muted-foreground">{showToken(token).slice(0, 4)}</span>)}
      {cells.map((row, q) => (
        <HeatRow key={q} row={row} size={size} label={labelled ? showToken(tokens[q]).slice(0, 6) : undefined} />
      ))}
    </div>
  );
}

function HeatRow({ row, size, label }: { row: (number | null)[]; size: number; label?: string }) {
  return (
    <>
      {label !== undefined && <span className="truncate pr-1 text-right text-[10px] leading-none text-muted-foreground" style={{ lineHeight: `${size}px` }}>{label}</span>}
      {row.map((value, k) => (
        <span
          key={k}
          className="block rounded-[1px]"
          style={{
            height: size,
            backgroundColor:
              value === null
                ? "var(--cell-masked)"
                : `color-mix(in oklch, var(--primary) ${Math.round(Math.min(1, Math.max(0, value)) * 100)}%, var(--muted))`,
          }}
        />
      ))}
    </>
  );
}

/** The heat of one attention share: it saturates at two thirds so soft patterns still show (the Tk tiles scale by 1.5). */
export function heat(value: number): number {
  return Math.min(1, value * 1.5);
}

/** The lower triangle of a head's [query][key] pattern as heat-map cells; `off` shows a switched-off head as dark. */
export function patternCells(pattern: number[][], off = false): (number | null)[][] {
  return pattern.map((row, q) => row.map((value, k) => (k > q ? null : off ? 0 : heat(value))));
}
