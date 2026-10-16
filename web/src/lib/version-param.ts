"use client";

import { useSearchParams } from "next/navigation";

export function useVersionParameter(): number | undefined {
  const value = useSearchParams().get("version");
  if (!value || !/^[1-9]\d*$/.test(value)) return undefined;
  const number = Number(value);
  return Number.isSafeInteger(number) ? number : undefined;
}
