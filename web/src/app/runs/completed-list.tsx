import Link from "next/link";
import { Button } from "@/components/ui/button";
import { T, fill } from "./copy";
import { completedText, titleText, type RunRow } from "./logic";

/** Finished runs, newest first. A failed one still in the queue's failed list has a Restart beside it, which queues it again first. */
export function CompletedList({ rows, onRestart, restarting = false }: { rows: RunRow[]; onRestart?: (id: string) => void; restarting?: boolean }) {
  return (
    <ul className="flex flex-col gap-2" data-testid="completed">
      {rows.map((row) => {
        const { version, name } = titleText(row);
        const title = [version, name].filter(Boolean).join(" · ");
        const where = row.distributed ? row.workers.join(", ") : fill(T.local, { node: row.workers[0] ?? "" });
        return (
          <li key={row.id} className="flex items-center gap-2">
            <Link
              href={`/runs/view/?id=${encodeURIComponent(row.id)}`}
              data-testid={`completed-${row.id}`}
              className="flex min-w-0 flex-1 flex-wrap items-baseline justify-between gap-2 rounded-xl bg-card px-4 py-3 text-sm ring-1 ring-foreground/10 outline-none transition-colors hover:bg-muted/40 focus-visible:ring-3 focus-visible:ring-ring/50"
            >
              <span>
                <b>{title}</b> <span className="text-muted-foreground">· {row.kindText} · {where}</span>
              </span>
              <span className="text-muted-foreground">{completedText(row)}</span>
            </Link>
            {row.restartable && onRestart && (
              <Button size="sm" aria-label={`${T.restart}: ${title}`} disabled={restarting} onClick={() => onRestart(row.id)}>
                {T.restart}
              </Button>
            )}
          </li>
        );
      })}
    </ul>
  );
}
