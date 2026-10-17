"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useState } from "react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { ApiError, api, apiFetch, unwrap } from "@/lib/api";
import { ROW_KINDS } from "./copy";
import { errorText, rowText, type VersionRow } from "./logic";
import { useJob } from "./use-events";

/** The versions, keyed by number so a row updates in place when its checkpoint arrives or a new one is saved. */
export function VersionsCard({ versions, startsFrom, loading }: { versions: VersionRow[] | undefined; startsFrom: number | undefined; loading: boolean }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Versions</CardTitle>
        <CardDescription>Each round starts from the best trained one.</CardDescription>
      </CardHeader>
      <CardContent>
        {loading ? (
          <div className="flex flex-col gap-2">
            <Skeleton className="h-6 w-full" />
            <Skeleton className="h-6 w-2/3" />
          </div>
        ) : !versions?.length ? (
          <p className="text-sm text-muted-foreground">No versions yet. Build one in the Network designer, or start from a pretrained model below.</p>
        ) : (
          <ul className="flex flex-col gap-2" aria-label="Versions">
            {[...versions].reverse().map((v) => (
              <li key={v.number} data-testid={`version-${v.number}`} className="flex flex-wrap items-center gap-2 text-sm">
                <span className="font-medium">Version {v.number}</span>
                <span className="text-muted-foreground">{v.change || v.name}</span>
                {v.best_checkpoint ? <Badge variant="secondary">trained</Badge> : <Badge variant="outline">no checkpoint yet</Badge>}
                {v.number === startsFrom && <Badge>starts here</Badge>}
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}

/** Start from a pretrained open model: a new version, with the same LoRA choice as round 1. */
export function ImportCard({ ranks, onImported }: { ranks: number[]; onImported: (number: number) => void }) {
  const queryClient = useQueryClient();
  const models = useQuery({ queryKey: ["post-training", "import-models"],
    queryFn: ({ signal }) => unwrap(api.GET("/post-training/import-models", { signal })), retry: false });
  const [chosen, setChosen] = useState<string>();
  const selected = chosen === undefined ? models.data?.[0] : models.data?.find((model) => model.id === chosen);
  const [lora, setLora] = useState(false);
  const [rank, setRank] = useState(8);
  const [file, setFile] = useState<File | null>(null);
  const importModel = useMutation({
    mutationFn: async () => {
      if (!selected) throw new Error("Choose an available model first.");
      const form = new FormData();
      form.set("name", selected.id);
      form.set("lora", String(lora));
      form.set("lora_rank", String(rank));
      if (file) form.set("weights", file);
      const res = await apiFetch("/post-training/imports", { method: "POST", body: form });
      if (!res.ok) throw new ApiError(res.status, "/post-training/imports", await res.text());
      return (await res.json()) as VersionRow;
    },
    onSuccess: (version) => {
      toast.success(`Saved version ${version.number}. It trains when the conversion job ends.`);
      queryClient.invalidateQueries({ queryKey: ["versions"] });
      queryClient.invalidateQueries({ queryKey: ["queue"] });
      onImported(version.number);
    },
    onError: (error) => toast.error(errorText(error)),
  });
  return (
    <Card>
      <CardHeader>
        <CardTitle>Start from a pretrained model</CardTitle>
        <CardDescription>
          Choose a model this server supports. It becomes a new version, and the conversion is queued.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        {models.isPending ? <Skeleton className="h-9 w-full" aria-label="Loading importable models" />
          : models.isError ? <div role="alert" className="space-y-2 text-sm"><p>Could not load importable models: {errorText(models.error)}</p>
            <Button variant="outline" onClick={() => void models.refetch()}>Retry model list</Button></div>
          : !models.data?.length ? <p className="text-sm text-muted-foreground">This server has no importable models.</p>
          : <div className="space-y-2">
            <Select value={selected?.id ?? ""} onValueChange={setChosen} disabled={importModel.isPending}>
              <SelectTrigger aria-label="Pretrained model" className="w-full"><SelectValue placeholder="Choose a model" /></SelectTrigger>
              <SelectContent>{models.data.map((model) => <SelectItem key={model.id} value={model.id}>{model.name}</SelectItem>)}</SelectContent>
            </Select>
            {selected && <p className="text-xs text-muted-foreground">About {Math.ceil(selected.size / 1_000_000).toLocaleString("en-US")} MB to download once.</p>}
          </div>}
        <div className="flex items-center gap-3">
          <Switch id="import-lora" aria-label="Fine-tune LoRA adapters" checked={lora} onCheckedChange={setLora} />
          <label htmlFor="import-lora" className="text-sm">
            Fine-tune LoRA adapters instead of every weight
          </label>
        </div>
        {lora && (
          <ToggleGroup type="single" variant="outline" aria-label="Import adapter size (rank)" value={String(rank)} onValueChange={(v) => v && setRank(Number(v))}>
            {ranks.map((r) => (
              <ToggleGroupItem key={r} value={String(r)}>
                {r}
              </ToggleGroupItem>
            ))}
          </ToggleGroup>
        )}
        <div className="flex flex-col gap-1">
          <label htmlFor="import-weights" className="text-sm">
            model.safetensors (optional)
          </label>
          <input
            id="import-weights"
            type="file"
            accept=".safetensors"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            className="text-sm file:mr-3 file:rounded-md file:border file:bg-muted file:px-3 file:py-1"
          />
          <p className="text-xs text-muted-foreground">Leave it out and the server downloads it. Choose it for a server that cannot.</p>
        </div>
        <div>
          <Button onClick={() => importModel.mutate()} disabled={importModel.isPending || !selected || models.isError}>
            {importModel.isPending ? "Importing…" : selected ? `Import ${selected.name}` : "Import model"}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

/** The rows behind a round (or the held-back ones): a job, so a skeleton shows until its result event arrives. */
export function RowsCard() {
  const [kind, setKind] = useState<(typeof ROW_KINDS)[number]["value"]>("conversations");
  const [asked, setAsked] = useState<(typeof ROW_KINDS)[number]["value"] | null>(null);
  const request = useCallback(() => unwrap(api.POST("/post-training/rows", { body: { kind } })), [kind]);
  const { job, start, pending } = useJob(request);
  const ask = useMutation({
    mutationFn: start,
    onMutate: () => setAsked(kind),
    onError: (error) => toast.error(errorText(error)),
  });
  const rows = job?.status === "done" && Array.isArray(job.result) ? (job.result as unknown[]) : null;
  const waiting = ask.isPending || pending;
  return (
    <Card>
      <CardHeader>
        <CardTitle>Look at the data</CardTitle>
        <CardDescription>The rows a round uses, or the held-back ones it is checked on.</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <div className="flex flex-wrap items-center gap-3">
          <ToggleGroup type="single" variant="outline" aria-label="Which rows" value={kind} onValueChange={(v) => v && setKind(v as typeof kind)} className="flex-wrap">
            {ROW_KINDS.map((k) => (
              <ToggleGroupItem key={k.value} value={k.value}>
                {k.label}
              </ToggleGroupItem>
            ))}
          </ToggleGroup>
          <Button variant="outline" onClick={() => ask.mutate()} disabled={waiting}>
            Show rows
          </Button>
        </div>
        {waiting && (
          <div className="flex flex-col gap-2" aria-label="Reading rows">
            <Skeleton className="h-5 w-full" />
            <Skeleton className="h-5 w-5/6" />
            <Skeleton className="h-5 w-2/3" />
          </div>
        )}
        {!waiting && job?.status === "failed" && <p className="text-sm text-destructive">{job.error}</p>}
        {!waiting && rows && (
          <div className="flex flex-col gap-2">
            <p className="text-sm text-muted-foreground">
              {rows.length.toLocaleString("en-US")} {ROW_KINDS.find((k) => k.value === asked)?.label.toLowerCase()}
              {rows.length > 20 ? ", the first 20 shown" : ""}
            </p>
            {rows.slice(0, 20).map((row, i) => (
              <pre key={i} className="whitespace-pre-wrap rounded-lg bg-muted/50 p-2 text-xs">
                {rowText(row)}
              </pre>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
