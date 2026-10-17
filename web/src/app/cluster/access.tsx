"use client";

import { useState, type FormEvent } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { problem } from "@/lib/cluster";
import { CLUSTER as T } from "@/lib/copy/cluster";
import { changed, cleanList } from "./logic";
import { useAccess, useSaveAccess } from "./use-cluster-page";

/** Who may call this computer: the allowed Tailscale logins and tags (GET and PUT /cluster/access). Counts after the app restarts. */
export function AccessCard() {
  const read = useAccess();
  const [note, setNote] = useState("");
  const save = useSaveAccess(setNote);
  const [draft, setDraft] = useState<{ users: string[]; tags: string[] } | null>(null);
  if (read.isError) return <p role="alert" className="text-sm text-destructive">{T.accessLoadFailed.replace("{why}", problem(read.error))}</p>;
  const saved = { users: read.data?.allow_users ?? [], tags: read.data?.allow_tags ?? [] };
  const now = draft ?? saved;
  const dirty = draft !== null && (changed(cleanList(draft.users), saved.users) || changed(cleanList(draft.tags), saved.tags));
  const edit = (next: { users: string[]; tags: string[] }) => {
    setDraft(next);
    setNote("");
    save.reset();
  };
  return (
    <Card data-testid="access">
      <CardHeader>
        <CardTitle>{T.accessTitle}</CardTitle>
        <CardDescription>{T.accessLead}</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        {read.isPending ? (
          <Skeleton className="h-24 w-full" />
        ) : (
          <>
            <p className="text-sm text-muted-foreground" data-testid="access-login">
              {read.data.login ? T.yourLogin.replace("{login}", read.data.login) : T.noLogin}
            </p>
            <Entries id="users" title={T.usersTitle} desc={T.usersDesc} empty={T.noUsers} label={T.usersAddLabel} placeholder={T.usersPlaceholder} items={now.users} disabled={save.isPending} onChange={(users) => edit({ ...now, users })} />
            <Entries id="tags" title={T.tagsTitle} desc={T.tagsDesc} empty={T.noTags} label={T.tagsAddLabel} placeholder={T.tagsPlaceholder} items={now.tags} disabled={save.isPending} onChange={(tags) => edit({ ...now, tags })} />
            <p className="text-xs text-muted-foreground">{T.accessRestart}</p>
            {note && <p role="alert" className="text-sm text-destructive" data-testid="access-problem">{note}</p>}
            <div className="flex flex-wrap gap-2">
              <Button
                disabled={!dirty || save.isPending}
                onClick={() => draft && save.mutate({ allow_users: cleanList(draft.users), allow_tags: cleanList(draft.tags) }, { onSuccess: () => setDraft(null) })}
              >
                {save.isPending ? T.saving : T.accessSave}
              </Button>
              {draft !== null && <Button variant="ghost" disabled={save.isPending} onClick={() => { setDraft(null); setNote(""); }}>{T.undo}</Button>}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}

function Entries({ id, title, desc, empty, label, placeholder, items, disabled, onChange }: {
  id: string; title: string; desc: string; empty: string; label: string; placeholder: string; items: string[]; disabled: boolean; onChange: (items: string[]) => void;
}) {
  const [typed, setTyped] = useState("");
  const add = (event: FormEvent) => {
    event.preventDefault();
    const entry = typed.trim();
    if (!entry || items.includes(entry)) return;
    onChange([...items, entry]);
    setTyped("");
  };
  return (
    <div className="flex flex-col gap-2" data-testid={`access-${id}`}>
      <div>
        <div className="font-medium">{title}</div>
        <div className="text-xs text-muted-foreground">{desc}</div>
      </div>
      {items.length === 0 ? (
        <p className="text-sm text-muted-foreground">{empty}</p>
      ) : (
        <ul className="flex flex-col gap-1.5" aria-label={title}>
          {items.map((item) => (
            <li key={item} className="flex items-center gap-2 rounded-lg border border-border bg-surface-faint px-3 py-1.5 text-sm">
              <span className="min-w-0 flex-1 truncate">{item}</span>
              <Button variant="ghost" size="sm" aria-label={`${T.removeEntry} ${item}`} disabled={disabled} onClick={() => onChange(items.filter((other) => other !== item))}>{T.removeEntry}</Button>
            </li>
          ))}
        </ul>
      )}
      <form onSubmit={add} className="flex flex-wrap items-center gap-2">
        <input aria-label={label} placeholder={placeholder} value={typed} disabled={disabled} onChange={(e) => setTyped(e.target.value)}
          className="h-8 min-w-0 flex-1 rounded-md border border-input bg-background px-2 text-sm text-foreground" />
        <Button type="submit" variant="outline" disabled={!typed.trim() || disabled}>{T.add}</Button>
      </form>
    </div>
  );
}
