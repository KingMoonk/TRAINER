"use client";

import { Suspense } from "react";
import { HeadLab } from "@/app/head-lab/head-lab";
import { Skeleton } from "@/components/ui/skeleton";
import { useVersionParameter } from "@/lib/version-param";

export default function Page() {
  return <Suspense fallback={<Skeleton className="m-8 h-64" />}><VersionedHeadLab /></Suspense>;
}

function VersionedHeadLab() {
  const version = useVersionParameter();
  return <HeadLab key={version ?? "best"} requestedVersion={version} />;
}
