"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { ApiError, api, apiFetch, unwrap } from "@/lib/api";
import { IMPORT, fill, type Kind } from "./copy";
import { INPUT } from "./forms";
import { runJob, type JobLike } from "./jobs";
import { errorText } from "./logic";

const plural = (n: number) => (n === 1 ? "file" : "files");
const sizeText = (bytes: number | null | undefined) =>
  bytes == null ? "" : bytes < 1024 ? `${bytes} B` : bytes < 1024 ** 2 ? `${(bytes / 1024).toFixed(1)} KB` : `${(bytes / 1024 ** 2).toFixed(1)} MB`;

/** POST /data/{kind}/import: a multipart upload, answered with the import job. */
async function upload(kind: Kind, files: File[]): Promise<JobLike> {
  const form = new FormData();
  for (const file of files) form.append("files", file);
  const res = await apiFetch(`/data/${kind}/import`, { method: "POST", body: form });
  if (!res.ok) throw new ApiError(res.status, `/data/${kind}/import`, await res.text());
  return (await res.json()) as JobLike;
}

/** The server-side picker: folders and files under the workspace and the home folder, for a corpus already on the box. */
function ServerPicker({ picked, setPicked }: { picked: string[]; setPicked: (paths: string[]) => void }) {
  const [path, setPath] = useState<string | null>(null);
  const listing = useQuery({
    queryKey: ["data", "browse", path],
    queryFn: () => unwrap(api.GET("/data/browse", { params: { query: { path } } })),
  });
  const toggle = (file: string) => setPicked(picked.includes(file) ? picked.filter((p) => p !== file) : [...picked, file]);
  const data = listing.data;
  return (
    <div className="flex flex-col gap-2" data-testid="server-picker">
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <span className="text-muted-foreground">{data?.path ?? IMPORT.places}</span>
        {data?.path && (
          <Button size="xs" variant="outline" onClick={() => setPath(data.parent ?? null)}>
            {IMPORT.up}
          </Button>
        )}
      </div>
      {listing.isLoading ? (
        <div className="flex flex-col gap-1.5" aria-label="Reading the folder">
          <Skeleton className="h-6 w-full" />
          <Skeleton className="h-6 w-2/3" />
        </div>
      ) : listing.isError ? (
        <p role="alert" className="text-sm text-destructive">
          {errorText(listing.error)}
        </p>
      ) : (
        <ul className="max-h-64 overflow-auto rounded-md border" aria-label="Folder contents">
          {data?.entries.length === 0 && <li className="p-2 text-sm text-muted-foreground">{IMPORT.noFiles}</li>}
          {data?.entries.map((entry) => (
            <li key={entry.path} className="flex items-center gap-2 border-b px-2 py-1 text-sm last:border-b-0">
              {entry.is_dir ? (
                <button type="button" className="flex-1 text-left text-violet-400 hover:underline" onClick={() => setPath(entry.path)}>
                  {entry.name}/
                </button>
              ) : (
                <label className="flex flex-1 cursor-pointer items-center gap-2">
                  <input type="checkbox" checked={picked.includes(entry.path)} onChange={() => toggle(entry.path)} />
                  <span>{entry.name}</span>
                  <span className="text-xs text-muted-foreground">{sizeText(entry.size)}</span>
                </label>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/** Import files into the open kind: a file from this computer (uploaded) or from a folder on the server (picked). */
export function ImportCard({ kind, onClose }: { kind: Kind; onClose: () => void }) {
  const queryClient = useQueryClient();
  const [files, setFiles] = useState<File[]>([]);
  const [picked, setPicked] = useState<string[]>([]);
  const [report, setReport] = useState<string[] | null>(null);
  const run = useMutation({
    mutationFn: (start: () => Promise<JobLike>) => runJob<string[]>(start),
    onSuccess: (lines) => {
      setReport(lines);
      setFiles([]);
      setPicked([]);
      toast.success(fill("Imported {n} {files}.", { n: lines.length, files: plural(lines.length) }));
      queryClient.invalidateQueries({ queryKey: ["data"] });
    },
    onError: (error) => toast.error(errorText(error)),
  });
  return (
    <Card>
      <CardHeader>
        <CardTitle>{IMPORT.title}</CardTitle>
        <CardDescription>{IMPORT.desc}</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-5">
        <div className="flex flex-col gap-2">
          <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{IMPORT.fromHere}</div>
          <input
            type="file"
            multiple
            aria-label={IMPORT.choose}
            className={INPUT}
            onChange={(e) => setFiles(Array.from(e.target.files ?? []))}
          />
          <div>
            <Button disabled={!files.length || run.isPending} onClick={() => run.mutate(() => upload(kind, files))}>
              {fill(IMPORT.upload, { n: files.length, files: plural(files.length) })}
            </Button>
          </div>
        </div>
        <div className="flex flex-col gap-2">
          <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{IMPORT.fromServer}</div>
          <ServerPicker picked={picked} setPicked={setPicked} />
          <div>
            <Button
              disabled={!picked.length || run.isPending}
              onClick={() => run.mutate(() => unwrap(api.POST("/data/{kind}/import_paths", { params: { path: { kind } }, body: { paths: picked } })) as Promise<JobLike>)}
            >
              {fill(IMPORT.pick, { n: picked.length, files: plural(picked.length) })}
            </Button>
          </div>
        </div>
        {run.isPending && <Skeleton className="h-5 w-64" aria-label={IMPORT.importing} />}
        {report && (
          <div data-testid="import-report">
            <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{IMPORT.report}</div>
            <ul className="mt-1 text-sm">
              {report.map((line, i) => (
                <li key={i}>{line}</li>
              ))}
            </ul>
          </div>
        )}
        <div>
          <Button variant="outline" onClick={onClose}>
            {IMPORT.close}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
