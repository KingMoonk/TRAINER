"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Copy } from "lucide-react";
import { useState, type FormEvent } from "react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { api, unwrap } from "@/lib/api";
import { problem } from "@/lib/cluster";
import { MACHINES as T } from "@/lib/copy/machines";
import { cn } from "@/lib/utils";
import { changed, cleanList, deviceText, refusedUs, type ClusterNode } from "./machines-logic";

const NODES = ["cluster", "nodes"] as const;

/** The machines this one pools compute with (GET /cluster/nodes), and the list of them to edit and save (PUT /cluster/nodes). */
export function MachinesSection() {
  const client = useQueryClient();
  const read = useQuery({ queryKey: NODES, queryFn: ({ signal }) => unwrap(api.GET("/cluster/nodes", { signal })), retry: false });
  const rows = read.data ?? [];
  const me = rows.find((row) => row.local);
  const saved = rows.filter((row) => !row.local).map((row) => row.address);
  const [draft, setDraft] = useState<string[] | null>(null); // null: nothing edited, the list is what the server has
  const [typed, setTyped] = useState("");
  const [note, setNote] = useState("");
  const list = draft ?? saved;
  const dirty = draft !== null && changed(cleanList(draft), saved);

  const save = useMutation({
    mutationFn: (addresses: string[]) => unwrap(api.PUT("/cluster/nodes", { body: { addresses } })),
    onSuccess: async () => {
      setNote("");
      toast.success(T.saved);
      await client.invalidateQueries({ queryKey: NODES });
      void client.invalidateQueries({ queryKey: ["cluster", "queue"] });
      setDraft(null);
    },
    onError: (error) => setNote(problem(error)),
  });

  const edit = (next: string[]) => {
    setDraft(next);
    setNote("");
    save.reset();
  };
  const add = (event: FormEvent) => {
    event.preventDefault();
    const address = typed.trim();
    if (!address) return;
    if (list.includes(address)) return setNote(T.duplicate);
    edit([...list, address]);
    setTyped("");
  };

  return (
    <section aria-labelledby="machines" className="flex flex-col gap-4" data-testid="machines">
      <div className="flex flex-col gap-1">
        <h2 id="machines" className="text-lg font-semibold tracking-tight">{T.section}</h2>
        <p className="max-w-prose text-sm text-muted-foreground">{T.sectionLead}</p>
      </div>
      {read.isError ? (
        <p role="alert" className="text-sm text-destructive">{T.loadFailed.replace("{why}", problem(read.error))}</p>
      ) : (
        <>
          <Card>
            <CardHeader>
              <CardTitle>{T.nameTitle}</CardTitle>
              <CardDescription>{T.nameDesc}</CardDescription>
            </CardHeader>
            <CardContent>{read.isPending ? <Skeleton className="h-8 w-72" /> : <TailnetName name={me?.tailnet_name ?? ""} />}</CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>{T.listTitle}</CardTitle>
              <CardDescription>{T.listDesc}</CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-4">
              {read.isPending ? (
                <div className="flex flex-col gap-2" aria-hidden="true">
                  <Skeleton className="h-14 w-full" />
                  <Skeleton className="h-14 w-full" />
                </div>
              ) : (
                <ul className="flex flex-col gap-2" aria-label={T.listTitle}>
                  {me && <MachineRow node={me} />}
                  {list.map((address) => (
                    <MachineRow
                      key={address}
                      address={address}
                      node={rows.find((row) => !row.local && row.address === address)}
                      onRemove={() => edit(list.filter((other) => other !== address))}
                      disabled={save.isPending}
                    />
                  ))}
                </ul>
              )}
              {rows.some(refusedUs) && (
                <div role="note" className="rounded-lg border border-warning/40 bg-warning-soft p-3 text-sm" data-testid="machines-refused">
                  <div className="font-medium">{T.refusedTitle}</div>
                  <p className="text-muted-foreground">{T.refused}</p>
                </div>
              )}
              <form onSubmit={add} className="flex flex-wrap items-center gap-2">
                <input
                  aria-label={T.addLabel}
                  placeholder={T.addPlaceholder}
                  value={typed}
                  disabled={read.isPending || save.isPending}
                  onChange={(event) => setTyped(event.target.value)}
                  className="h-8 min-w-0 flex-1 rounded-md border border-input bg-background px-2 text-sm text-foreground"
                />
                <Button type="submit" variant="outline" disabled={!typed.trim() || read.isPending || save.isPending}>{T.add}</Button>
              </form>
              {note && <p role="alert" className="text-sm text-destructive" data-testid="machines-problem">{note}</p>}
              <div className="flex flex-wrap gap-2">
                <Button disabled={!dirty || save.isPending} onClick={() => save.mutate(cleanList(list))}>{save.isPending ? T.saving : T.save}</Button>
                {draft !== null && (
                  <Button variant="ghost" disabled={save.isPending} onClick={() => {
                    setDraft(null);
                    setNote("");
                  }}>{T.undo}</Button>
                )}
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </section>
  );
}

/** This machine's MagicDNS name, to copy and give to someone, or why it is not known. */
function TailnetName({ name }: { name: string }) {
  if (!name) return <p className="text-sm text-muted-foreground" data-testid="tailnet-name-missing">{T.noName}</p>;
  const copy = () =>
    navigator.clipboard.writeText(name).then(
      () => toast.success(T.copied),
      () => toast.error(T.copyFailed),
    );
  return (
    <div className="flex flex-wrap items-center gap-2">
      <code className="rounded-md border border-border bg-muted px-2 py-1 font-mono text-sm select-all" data-testid="tailnet-name">{name}</code>
      <Button variant="outline" size="sm" onClick={copy}>
        <Copy aria-hidden="true" />
        {T.copy}
      </Button>
    </div>
  );
}

/** One machine: its name and what it computes on, whether it answers, and why not. A listed one can be removed. */
function MachineRow({ node, address, onRemove, disabled }: { node?: ClusterNode; address?: string; onRemove?: () => void; disabled?: boolean }) {
  const key = node?.local ? "local" : address ?? node?.address ?? "";
  const title = node?.local ? `${node.hostname} · ${T.thisMachine}` : node?.hostname ? `${node.hostname} (${key})` : key;
  const device = node ? deviceText(node) : "";
  return (
    <li
      data-testid={`machine-${key}`}
      data-online={node ? String(node.online) : undefined}
      className={cn("flex flex-wrap items-center gap-3 rounded-lg border border-border bg-surface-faint p-3", node && !node.online && "opacity-80")}
    >
      <div className="min-w-0 flex-1">
        <div className="truncate font-medium">{title}</div>
        <div className="text-xs text-muted-foreground">
          {node ? [device, node.version && `version ${node.version}`].filter(Boolean).join(" · ") : T.unsaved}
        </div>
        {node && !node.online && node.error && <div className="text-sm text-muted-foreground" data-testid={`machine-error-${key}`}>{node.error}</div>}
      </div>
      {node && <Badge variant={node.online ? "secondary" : "outline"}>{node.online ? T.online : T.offline}</Badge>}
      {node?.online && node.busy && <Badge variant="outline">{T.busy}</Badge>}
      {onRemove && (
        <Button variant="ghost" size="sm" aria-label={`${T.remove} ${key}`} disabled={disabled} onClick={onRemove}>{T.remove}</Button>
      )}
    </li>
  );
}
