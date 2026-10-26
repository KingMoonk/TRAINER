"use client";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { T } from "./copy";
import { scalingView, type Plan } from "./logic";

/** What this computer can train and how the run was scaled to it: each change with its reason, and "use my settings anyway" where the
 * run only fits tightly or is only slow. Every figure comes from the server's measured profile; nothing is shown before it is read. */
export function ScalingCard({ plan, keepMine, onKeepMine }: { plan: Plan | undefined; keepMine: boolean; onKeepMine: (keep: boolean) => void }) {
  const view = scalingView(plan);
  if (!view) return null;
  return (
    <Card>
      <CardHeader>
        <CardTitle>{T.scalingTitle}</CardTitle>
        <CardDescription>{T.scalingDesc}</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <p className="text-sm text-muted-foreground" data-testid="hardware-line">
          {view.hardware}
        </p>
        <p className="text-sm font-medium" data-testid="scaling-lead" role={view.refused ? "alert" : undefined}>
          {view.lead}
        </p>
        {view.refused && (
          <p className="text-sm text-warning" data-testid="scaling-refused">
            {view.refused}
          </p>
        )}
        {view.notes.length > 0 && (
          <ul className="flex list-disc flex-col gap-1 pl-5 text-sm" data-testid="scaling-notes">
            {view.notes.map((note, i) => (
              <li key={i} data-kind={note.kind}>
                {note.text}
              </li>
            ))}
          </ul>
        )}
        {(view.canKeep || view.kept) && (
          <div className="flex flex-col gap-1">
            <div>
              <Button variant="outline" size="sm" data-testid="keep-mine" aria-pressed={keepMine} onClick={() => onKeepMine(!keepMine)}>
                {keepMine ? T.scalingScale : T.scalingKeep}
              </Button>
            </div>
            <p className="text-xs text-muted-foreground">{keepMine ? T.scalingKept : T.scalingKeepHint}</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
