import type { ReactNode } from "react";
import { ClusterTabs } from "./tabs";

export default function ClusterLayout({ children }: { children: ReactNode }) {
  return (
    <div className="flex flex-col">
      <div className="mx-auto w-full max-w-5xl px-4 pt-4 md:px-8 md:pt-6">
        <ClusterTabs />
      </div>
      {children}
    </div>
  );
}
