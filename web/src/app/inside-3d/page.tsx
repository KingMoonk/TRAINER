"use client";

import { Suspense } from "react";
import { Inside3d } from "@/components/inside-3d/inside-3d";
import { Skeleton } from "@/components/ui/skeleton";
import { useVersionParameter } from "@/lib/version-param";

export default function Page() {
  return <Suspense fallback={<Skeleton className="m-8 h-64" />}><VersionedViewer /></Suspense>;
}

function VersionedViewer() {
  const version = useVersionParameter();
  return <Inside3d key={version ?? "current"} version={version} />;
}
