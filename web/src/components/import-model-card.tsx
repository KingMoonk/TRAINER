"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { ApiError, api, apiFetch, unwrap } from "@/lib/api";
import { detailOf, type Version } from "@/lib/builder";

/** Start from a pretrained open model: a new version, with the same LoRA choice as round 1. */
export function ImportModelCard({ onImported }: { onImported?: (number: number) => void }) {
  const queryClient = useQueryClient();
  const ranks = useQuery({ queryKey: ["post-training", "options", "import"],
    queryFn: ({ signal }) => unwrap(api.GET("/post-training/options", { signal })), retry: false }).data?.lora_ranks ?? [];
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
      return (await res.json()) as Version;
    },
    onSuccess: (version) => {
      toast.success(`Saved version ${version.number}. It trains when the conversion job ends.`);
      queryClient.invalidateQueries({ queryKey: ["versions"] });
      queryClient.invalidateQueries({ queryKey: ["queue"] });
      onImported?.(version.number);
    },
    onError: (error) => toast.error(detailOf(error)),
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
          : models.isError ? <div role="alert" className="space-y-2 text-sm"><p>Could not load importable models: {detailOf(models.error)}</p>
            <Button variant="outline" onClick={() => void models.refetch()}>Retry model list</Button></div>
          : !models.data?.length ? <p className="text-sm text-muted-foreground">This server has no importable models.</p>
          : <div className="space-y-2">
            <Select value={selected?.id ?? ""} onValueChange={setChosen} disabled={importModel.isPending}>
              <SelectTrigger aria-label="Pretrained model" className="w-full"><SelectValue placeholder="Choose a model" /></SelectTrigger>
              <SelectContent>{models.data.map((model) => <SelectItem key={model.id} value={model.id}>{model.name}</SelectItem>)}</SelectContent>
            </Select>
            {selected && <p className="text-xs text-muted-foreground">About {Math.ceil(selected.size / 1_000_000).toLocaleString("en-US")} MB to download once. Each version started from it adds about 0.5 GB, and a full fine-tune saves checkpoints of about 1.5 GB each.</p>}
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
