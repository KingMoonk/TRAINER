"use client";

import { Suspense } from "react";
import { Skeleton } from "@/components/ui/skeleton";
import { useVersionParameter } from "@/lib/version-param";
import { HeadZoom } from "@/app/head-zoom/head-zoom";

// HeadZoom reads ?block=&head= (a link from the Head lab opens one head), so it sits in a Suspense boundary for the static export.
export default function Page() {
  return (
    <Suspense fallback={<Skeleton className="m-8 h-64" />}>
      <VersionedHeadZoom />
    </Suspense>
  );
}

function VersionedHeadZoom() {
  const version = useVersionParameter();
  return <HeadZoom key={version ?? "best"} requestedVersion={version} />;
}
