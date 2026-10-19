"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { CLUSTER_TABS } from "./tabs";

// /cluster/ opens the area's first tab. A static export has no server-side redirect, so it is done here.
export default function ClusterHome() {
  const router = useRouter();
  const first = CLUSTER_TABS[0];
  useEffect(() => {
    router.replace(first.href);
  }, [router, first.href]);
  return (
    <main className="p-4 text-sm text-muted-foreground">
      <Link href={first.href} className="underline underline-offset-2">{first.title}</Link>
    </main>
  );
}
