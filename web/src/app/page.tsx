"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { HOME } from "@/lib/routes";

// The app opens on the Builder, as the desktop app does. A static export has no server-side redirect, so it is done here.
export default function Home() {
  const router = useRouter();
  useEffect(() => {
    router.replace(HOME);
  }, [router]);
  return null;
}
