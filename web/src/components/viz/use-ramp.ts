"use client";

import { useMemo } from "react";
import { useThemeColors } from "@/components/theme";
import { buildRamp, RAMP_TOKENS } from "@/lib/ramp";

/** The theme's heat ramp as a lookup table (lib/ramp.ts), rebuilt when the mode or accent changes; null before the first client render. */
export function useRamp(): Uint8Array | null {
  const stops = useThemeColors(RAMP_TOKENS);
  return useMemo(() => buildRamp(stops), [stops]);
}
