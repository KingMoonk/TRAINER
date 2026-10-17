"use client";

import { Copy } from "lucide-react";
import { useState, type FormEvent } from "react";
import { toast } from "sonner";
import { PAGE_MAIN, PageHeader } from "@/components/page-header";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { problem, useCluster } from "@/lib/cluster";
import { CLUSTER as T } from "@/lib/copy/cluster";
import { useQueue } from "@/app/queue/use-queue";
import { AccessCard } from "./access";
import { changed, cleanList, jobRows } from "./logic";
import { MachineCard } from "./machine-card";
import { useAccess, useNodes, useSaveNodes, useStopJob } from "./use-cluster-page";

export default function ClusterPage() {
  const read = useNodes();
  const merged = useCluster().data;
  const queue = useQueue().data;
  const access = useAccess().data;
  const stop = useStopJob();
  const nodes = read.data ?? [];
  const me = nodes.find((node) => node.local);
  const saved = nodes.filter((node) => !node.local).map((node) => node.address);
  const [draft, setDraft] = useState<string[] | null>(null); // null: nothing edited, the list is what the server has
  const [typed, setTyped] = useState("");
  const [note, setNote] = useState("");
  const save = useSaveNodes(() => setDraft(null), setNote);
  const list = draft ?? saved;
  const dirty = draft !== null && changed(cleanList(draft), saved);
  const rows = jobRows(merged, queue);
  const seen = Object.fromEntries((merged?.nodes ?? []).map((node) => [node.address, node.seen_at]));
  const names = Object.fromEntries(nodes.map((node) => [node.address, node.hostname]));

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
  const card = (address: string, onRemove?: () => void) => (
    <MachineCard
      key={address}
      address={address}
      node={nodes.find((node) => node.address === address)}
      rows={rows}
      seenAt={seen[address] ?? 0}
      login={access?.login ?? ""}
      names={names}
      busy={save.isPending || stop.isPending}
      onRemove={onRemove}
      onStop={(row) => stop.mutate(row)}
    />
  );

  return (
    <main className={PAGE_MAIN}>
      <PageHeader title={T.title} lead={T.lead} />
      {read.isError ? (
        <p role="alert" className="text-sm text-destructive">{T.couldNotLoad.replace("{why}", problem(read.error))}</p>
      ) : (
        <>
          <Card>
            <CardHeader>
              <CardTitle>{T.nameTitle}</CardTitle>
              <CardDescription>{T.nameDesc}</CardDescription>
            </CardHeader>
            <CardContent>{read.isPending ? <Skeleton className="h-8 w-72" /> : <TailnetName name={me?.tailnet_name ?? ""} />}</CardContent>
          </Card>
          <section aria-labelledby="machines" className="flex flex-col gap-3" data-testid="machines">
            <div className="flex flex-col gap-1">
              <h2 id="machines" className="text-lg font-semibold tracking-tight">{T.machines}</h2>
              <p className="max-w-prose text-sm text-muted-foreground">{T.machinesLead}</p>
            </div>
            {read.isPending ? (
              <div className="flex flex-col gap-2" aria-hidden="true">
                <Skeleton className="h-32 w-full" />
              </div>
            ) : (
              <div className="flex flex-col gap-3">
                {me && card(me.address)}
                {list.map((address) => card(address, () => edit(list.filter((other) => other !== address))))}
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
                <Button variant="ghost" disabled={save.isPending} onClick={() => { setDraft(null); setNote(""); }}>{T.undo}</Button>
              )}
            </div>
          </section>
          <AccessCard />
        </>
      )}
    </main>
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
