"use client";

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { toast } from "sonner";
import { PAGE_MAIN, PageHeader } from "@/components/page-header";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { downloadName, errorText, fill } from "@/app/storage/logic";
import { api, apiFetch, ApiError, unwrap } from "@/lib/api";
import { routeFor } from "@/lib/routes";
import { EXPORT } from "./copy";

type ExportKind = "hf" | "gguf";

export default function Page() {
  const route = routeFor("export");
  return (
    <main className={PAGE_MAIN}>
      <PageHeader title={route.title} lead={route.lead} />
      <ExportCard />
    </main>
  );
}

function ExportCard() {
  const rows = useQuery({ queryKey: ["storage", "exports"], queryFn: () => unwrap(api.GET("/storage/exports")) });
  const [busy, setBusy] = useState<string | null>(null);

  async function download(number: number, name: string, kind: ExportKind) {
    setBusy(`${number}:${kind}`);
    try {
      const res = await apiFetch(`/versions/${number}/export?kind=${kind}`);
      if (!res.ok) throw new ApiError(res.status, `/versions/${number}/export`, await res.text());
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = downloadName(res.headers.get("content-disposition"), `${name}-${kind}`);
      document.body.append(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
      toast.success(fill(EXPORT.done, { name }));
    } catch (error) {
      toast.error(errorText(error));
    } finally {
      setBusy(null);
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-xs font-medium tracking-wider text-muted-foreground">{EXPORT.kicker}</CardTitle>
        <p className="max-w-prose text-sm text-muted-foreground">{EXPORT.desc}</p>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        {rows.isPending ? (
          <div className="flex flex-col gap-2" aria-label="Loading versions">
            <Skeleton className="h-9 w-full max-w-md" />
            <Skeleton className="h-9 w-full max-w-md" />
          </div>
        ) : rows.isError ? (
          <p className="text-sm text-destructive">{EXPORT.loadFailed}</p>
        ) : rows.data.length === 0 ? (
          <p className="text-sm text-muted-foreground">{EXPORT.none}</p>
        ) : (
          <ul className="flex flex-col gap-3">
            {rows.data.map((v) => (
              <li key={v.number} className="flex flex-col gap-1">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="min-w-0 flex-1 truncate text-sm">{v.name}</span>
                  <Button variant="outline" disabled={busy !== null} onClick={() => download(v.number, v.name, "hf")}>
                    {busy === `${v.number}:hf` ? EXPORT.working : EXPORT.hf}
                  </Button>
                  <Button
                    variant="outline"
                    disabled={busy !== null || !!v.gguf_problem}
                    aria-describedby={v.gguf_problem ? `gguf-why-${v.number}` : undefined}
                    onClick={() => download(v.number, v.name, "gguf")}
                  >
                    {busy === `${v.number}:gguf` ? EXPORT.working : EXPORT.gguf}
                  </Button>
                </div>
                {v.gguf_problem && (
                  <p id={`gguf-why-${v.number}`} className="max-w-prose text-xs text-muted-foreground">
                    {v.gguf_problem}
                  </p>
                )}
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}
