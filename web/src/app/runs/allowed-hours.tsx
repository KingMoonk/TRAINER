"use client";

import type { ReactNode } from "react";
import { PlanCard } from "@/app/queue/plan-card";
import { useQueue, useQueueActions } from "@/app/queue/use-queue";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { T } from "./copy";

/** The queue's plan card in a Sheet: when this computer may train. `children` is the trigger (one element). */
export function AllowedHours({ children }: { children: ReactNode }) {
  const queue = useQueue();
  const actions = useQueueActions();
  return (
    <Sheet>
      <SheetTrigger asChild>{children}</SheetTrigger>
      <SheetContent>
        <SheetHeader>
          <SheetTitle>{T.allowedHoursTitle}</SheetTitle>
          <SheetDescription>{T.allowedHoursLead}</SheetDescription>
        </SheetHeader>
        <PlanCard queue={queue.data} actions={actions} />
      </SheetContent>
    </Sheet>
  );
}
