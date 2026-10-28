"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { routeFor } from "@/lib/routes";

// /cluster/ opens the Machines page, the area's home. A static export has no server-side redirect, so it is done here.
export default function ClusterHome() {
  const router = useRouter();
  const machines = routeFor("machines");
  useEffect(() => {
    router.replace(machines.href);
  }, [router, machines.href]);
  return (
    <main className="p-4 text-sm text-muted-foreground">
      <Link href={machines.href} className="underline underline-offset-2">{machines.title}</Link>
    </main>
  );
}
