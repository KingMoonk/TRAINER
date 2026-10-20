"use client";

import Link from "next/link";
import { Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { PAGE_MAIN } from "@/components/page-header";
import { Skeleton } from "@/components/ui/skeleton";
import { problem } from "@/lib/cluster";
import { T, fill } from "./copy";
import { DilocoView, MemberView, SoloView } from "./run-view";
import { useRunView } from "./use-run-view";

function Message({ title, lead }: { title: string; lead?: string }) {
  return (
    <div className="flex flex-col gap-2" role="status">
      <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
      {lead && <p className="max-w-prose text-muted-foreground">{lead}</p>}
      <Link href="/runs/" className="w-fit text-sm text-highlight underline-offset-4 hover:underline">← {T.back}</Link>
    </div>
  );
}

function RunPage() {
  const id = useSearchParams().get("id") ?? "";
  const view = useRunView(id);
  switch (view.kind) {
    case "loading":
      return <Skeleton className="h-64 w-full" aria-label={T.loading} />;
    case "error":
      return <p role="alert" className="text-sm text-destructive">{fill(T.couldNotLoad, { why: problem(view.error) })}</p>;
    case "missing":
      return id === "" ? <Message title={T.noId} /> : <Message title={T.notFound} lead={T.notFoundLead} />;
    case "diloco":
      return <DilocoView run={view.run} />;
    case "member":
      return <MemberView view={view} />;
    default:
      return <SoloView view={view} />;
  }
}

/** One page for any run, whatever the number of machines: `/runs/view/?id=<queue job id or distributed run id>`. */
export default function RunViewPage() {
  return (
    <main className={PAGE_MAIN}>
      <Suspense fallback={<Skeleton className="h-64 w-full" aria-label={T.loading} />}>
        <RunPage />
      </Suspense>
    </main>
  );
}
