import Link from "next/link";
import { T, fill } from "./copy";
import { completedText, titleText, type RunRow } from "./logic";

export function CompletedList({ rows }: { rows: RunRow[] }) {
  return (
    <ul className="flex flex-col gap-2" data-testid="completed">
      {rows.map((row) => {
        const { version, name } = titleText(row);
        const where = row.distributed ? row.workers.join(", ") : fill(T.local, { node: row.workers[0] ?? "" });
        return (
          <li key={row.id}>
            <Link
              href={`/runs/view/?id=${encodeURIComponent(row.id)}`}
              data-testid={`completed-${row.id}`}
              className="flex flex-wrap items-baseline justify-between gap-2 rounded-xl bg-card px-4 py-3 text-sm ring-1 ring-foreground/10 outline-none transition-colors hover:bg-muted/40 focus-visible:ring-3 focus-visible:ring-ring/50"
            >
              <span>
                <b>{[version, name].filter(Boolean).join(" · ")}</b> <span className="text-muted-foreground">· {row.kindText} · {where}</span>
              </span>
              <span className="text-muted-foreground">{completedText(row)}</span>
            </Link>
          </li>
        );
      })}
    </ul>
  );
}
