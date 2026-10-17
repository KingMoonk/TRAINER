"use client";

import { Badge } from "@/components/ui/badge";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { TRAINING as T } from "@/lib/copy/training";
import type { Run } from "@/app/cluster/training/logic";
import { T as V, fill } from "./copy";
import { eventAge, eventsNewestFirst, type RunEvent } from "./live";
import { clock } from "./logic";

const BAD = ["dropped", "stale", "late"];

/** The fields of one event, as the coordinator recorded them. The raw request behind it is for the API console (#341), not here. */
function Details({ run, event }: { run: Run; event: RunEvent }) {
  const rows: [string, string][] = [
    [V.eventKind, T.eventKinds[event.kind]],
    ...(event.node ? ([[V.eventWho, event.node]] as [string, string][]) : []),
    [V.eventRound, String(event.round + 1)],
    [V.eventWhen, `${clock(event.t)} · ${fill(V.eventAgo, { n: Math.max(0, Math.round(run.now_s - event.t)) })}`],
    [V.eventSeq, `#${event.seq}`],
  ];
  return (
    <div className="flex flex-col gap-2">
      <p className="text-sm">{event.text}</p>
      <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
        {rows.map(([name, value]) => (
          <div key={name} className="contents">
            <dt className="text-muted-foreground">{name}</dt>
            <dd className="tabular-nums">{value}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

/** What happened in the run, newest first: machines joining, leaving, dropped or late, changes refused, each outer step and each checkpoint. A row opens its details. */
export function EventLog({ run }: { run: Run }) {
  const events = eventsNewestFirst(run);
  return (
    <section aria-label={T.events} className="flex flex-col gap-2" data-testid="event-log">
      <h3 className="text-sm font-medium">{T.events}</h3>
      {events.length === 0 ? (
        <p className="text-sm text-muted-foreground">{T.noEvents}</p>
      ) : (
        <ol className="flex max-h-72 flex-col gap-0.5 overflow-y-auto text-sm">
          {events.map((event) => (
            <li key={event.seq} data-testid={`event-${event.seq}`} data-kind={event.kind}>
              <Popover>
                <PopoverTrigger asChild>
                  <button type="button" className="flex w-full flex-wrap items-center gap-2 rounded-md px-2 py-1.5 text-left hover:bg-muted focus-visible:ring-3 focus-visible:ring-ring/50 focus-visible:outline-none">
                    <Badge variant={BAD.includes(event.kind) ? "destructive" : "secondary"}>{T.eventKinds[event.kind]}</Badge>
                    <span>{event.text}</span>
                    <span className="ml-auto text-xs tabular-nums text-muted-foreground">{eventAge(run, event)}</span>
                    <span aria-hidden="true" className="text-xs text-muted-foreground">▸</span>
                  </button>
                </PopoverTrigger>
                <PopoverContent align="end" aria-label={V.eventDetails}>
                  <Details run={run} event={event} />
                </PopoverContent>
              </Popover>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
