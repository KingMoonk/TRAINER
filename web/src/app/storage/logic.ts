import { ApiError } from "@/lib/api";

/** The "older than" choices in days, in the Tk page's order; "all" follows them and means null days. */
export const AGES = [1, 7, 30, 90] as const;
export const ALL = "all";
export const DEFAULT_CHOICE = "30";

export function choiceDays(id: string): number | null {
  return id === ALL ? null : Number(id);
}

/** "Also delete my data" only counts with "All history". */
export function dataApplies(id: string, data: boolean): boolean {
  return data && id === ALL;
}

export function sizeText(bytes: number): string {
  return bytes >= 1e9 ? `${(bytes / 1e9).toFixed(1)} GB` : `${(bytes / 1e6).toFixed(1)} MB`;
}

export function fill(template: string, values: Record<string, string | number>): string {
  return template.replace(/\{(\w+)\}/g, (_, key: string) => String(values[key] ?? ""));
}

/** A sentence for a failed call: the server's own `detail` when it sent one. */
export function errorText(error: unknown): string {
  if (error instanceof ApiError) {
    try {
      const detail = (JSON.parse(error.body) as { detail?: unknown }).detail;
      if (typeof detail === "string") return detail;
    } catch {
      // not JSON: fall through
    }
    return error.body || error.message;
  }
  return error instanceof Error ? error.message : String(error);
}

/** The file name a download response names in Content-Disposition, or the fallback. */
export function downloadName(header: string | null, fallback: string): string {
  const match = header?.match(/filename\*=UTF-8''([^;]+)|filename="?([^";]+)"?/i);
  const raw = match?.[1] ?? match?.[2];
  if (!raw) return fallback;
  try {
    return decodeURIComponent(raw);
  } catch {
    return raw;
  }
}
