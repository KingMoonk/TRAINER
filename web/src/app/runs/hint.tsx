import type { ReactNode } from "react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

/** A word with a shadcn Tooltip: for what is read once or twice and not needed to act. Keyboard-focusable. */
export function Hint({ tip, children, className, plain }: { tip: ReactNode; children: ReactNode; className?: string; plain?: boolean }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span tabIndex={0} className={cn("rounded-sm outline-none focus-visible:ring-3 focus-visible:ring-ring/50", !plain && "cursor-help underline decoration-dotted underline-offset-2", className)}>
          {children}
        </span>
      </TooltipTrigger>
      <TooltipContent>{tip}</TooltipContent>
    </Tooltip>
  );
}
