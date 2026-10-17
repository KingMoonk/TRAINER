// The pieces of the API client that the stream readers share (kept apart from api.ts so the two do not import each other).

import { authHeaders } from "@/lib/token";

export const BASE = (process.env.NEXT_PUBLIC_API_URL ?? "").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly path: string,
    readonly body: string,
  ) {
    super(`${status} from ${path}`);
    this.name = "ApiError";
  }
}

export function apiUrl(path: string): string {
  return `${BASE}${path.startsWith("/") ? path : `/${path}`}`;
}

/** `fetch` of an API path with the Funnel token header (token.ts) when one is stored. For calls whose reply is not JSON (uploads, downloads). */
export function apiFetch(path: string, init?: RequestInit): Promise<Response> {
  return fetch(apiUrl(path), { ...init, headers: { ...authHeaders(), ...init?.headers } });
}
