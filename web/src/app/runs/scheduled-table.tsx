import { Button } from "@/components/ui/button";
import { useJobOutlook, type QueueActions } from "@/app/queue/use-queue";
import { T, fill } from "./copy";
import { Hint } from "./hint";
import { titleText, type RunRow } from "./logic";

/** When a waiting job starts: the outlook sentence, else why it waits, else the plan's hours. */
function Schedule({ row }: { row: RunRow }) {
  const outlook = useJobOutlook(row.id).data?.outlook;
  if (row.held) return <>{T.onHold}</>;
  if (outlook) return <>{outlook}</>;
  if (row.note) return <>{row.note}</>;
  return <Hint tip={T.allowedHoursTip}>{T.allowedHours.toLowerCase()}</Hint>;
}

function Machine({ name }: { name: string }) {
  return name === T.anyIdleNode ? <Hint tip={T.anyIdleNodeTip}>{name}</Hint> : <>{name}</>;
}

export function ScheduledTable({ rows, actions }: { rows: RunRow[]; actions: QueueActions }) {
  return (
    <div className="overflow-x-auto rounded-xl ring-1 ring-foreground/10">
      <table className="w-full min-w-xl text-left text-sm" data-testid="scheduled">
        <thead className="text-xs text-muted-foreground">
          <tr className="border-b border-border">
            <th scope="col" className="px-3 py-2 font-medium">{T.colRun}</th>
            <th scope="col" className="px-3 py-2 font-medium"><Hint tip={T.colWorkersTip}>{T.colWorkers}</Hint></th>
            <th scope="col" className="px-3 py-2 font-medium"><Hint tip={T.colCoordinatorTip}>{T.colCoordinator}</Hint></th>
            <th scope="col" className="px-3 py-2 font-medium">{T.colSchedule}</th>
            <th scope="col" className="px-3 py-2 text-right font-medium">{T.colActions}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const { version, name } = titleText(row);
            const title = [version, name].filter(Boolean).join(" · ");
            const inQueue = row.queueIndex >= 0;
            return (
              <tr key={row.id} data-testid={`scheduled-${row.id}`} className="border-b border-border last:border-b-0">
                <td className="px-3 py-2.5">
                  <div className="font-semibold">{title}</div>
                  <div className="text-xs text-muted-foreground">{row.kindText}</div>
                </td>
                <td className="px-3 py-2.5">
                  {row.workers.map((w, i) => (
                    <span key={w}>
                      {i ? ", " : ""}
                      <Machine name={w} />
                    </span>
                  ))}
                </td>
                <td className="px-3 py-2.5">
                  <Machine name={row.coordinator} />
                </td>
                <td className="px-3 py-2.5 text-muted-foreground">
                  <Schedule row={row} />
                </td>
                <td className="px-3 py-2.5">
                  {inQueue && (
                    <div className="flex justify-end gap-1">
                      <Button
                        variant="outline"
                        size="sm"
                        aria-label={`${T.moveUp}: ${title}`}
                        disabled={row.queueIndex === 0}
                        onClick={() => actions.move.mutate({ id: row.id, index: row.queueIndex - 1 })}
                      >
                        {"▲"}
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        aria-label={fill("{action}: {title}", { action: row.held ? T.release : T.hold, title })}
                        onClick={() => actions.hold.mutate({ id: row.id, held: !row.held })}
                      >
                        {row.held ? T.release : T.hold}
                      </Button>
                    </div>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
