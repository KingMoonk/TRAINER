import { cn } from "@/lib/utils";

/**
 * The class list of a choosable box (a radio, checkbox, tab or toggle button drawn as a card). A chosen box takes the
 * neutral `selected-box` look, never an accent border; an unchosen one is a plain bordered box with a hover wash.
 */
export function optionBox(on: boolean, className?: string) {
  return cn(
    "rounded-lg border-2 text-left outline-none transition-colors focus-visible:ring-3 focus-visible:ring-ring/50",
    on ? "selected-box" : "border-border hover:bg-muted/30",
    className,
  );
}
