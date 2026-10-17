import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

/** What a card or list says when there is nothing to show yet: one muted sentence, optionally followed by a next step. */
export function EmptyState({ children, className }: { children: ReactNode; className?: string }) {
  return <p className={cn("text-sm text-muted-foreground", className)}>{children}</p>;
}
