"use client";

import { Badge } from "@/components/ui/badge";
import { TRAINING as T } from "@/lib/copy/training";
import { eventAge, eventsNewestFirst } from "./live";
import type { Run } from "./logic";

/** What happened in the run, newest first: machines joining, leaving, dropped or late, changes refused, each outer step and each checkpoint. */
export function EventLog({ run }: { run: Run }) {
  const events = eventsNewestFirst(run);
  return (
    <section aria-label={T.events} className="flex flex-col gap-2" data-testid="event-log">
      <h3 className="text-sm font-medium">{T.events}</h3>
      {events.length === 0 ? (
        <p className="text-sm text-muted-foreground">{T.noEvents}</p>
      ) : (
        <ol className="flex max-h-64 flex-col gap-1 overflow-y-auto text-sm">
          {events.map((event) => (
            <li key={event.seq} className="flex flex-wrap items-center gap-2" data-testid={`event-${event.seq}`} data-kind={event.kind}>
              <Badge variant={event.kind === "dropped" || event.kind === "stale" || event.kind === "late" ? "destructive" : "secondary"}>{T.eventKinds[event.kind]}</Badge>
              <span>{event.text}</span>
              <span className="ml-auto text-xs tabular-nums text-muted-foreground">{eventAge(run, event)}</span>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
