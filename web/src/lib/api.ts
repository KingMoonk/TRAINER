// The API client. Phase 1 (docs/rewrite.md) adds the real endpoints; until then this is the transport only.
// NEXT_PUBLIC_API_URL is the server's origin (e.g. http://127.0.0.1:8000). Empty means same origin, which is what the
// static export served by the API uses.

const BASE = (process.env.NEXT_PUBLIC_API_URL ?? "").replace(/\/$/, "");

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

/** GET (or `init.method`) `path` and parse the JSON reply as `T`. Throws ApiError on a non-2xx status. */
export async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(apiUrl(path), {
    ...init,
    headers: { Accept: "application/json", ...(init?.body ? { "Content-Type": "application/json" } : {}), ...init?.headers },
  });
  if (!res.ok) throw new ApiError(res.status, path, await res.text());
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

/** Every `Services.emit(kind, payload)` on the server, as the API's `GET /events` SSE stream will send it. */
export type ApiEvent = { kind: string; payload: unknown };

/** Open the event stream and call `onEvent` for each event. Returns the function that closes it. */
export function subscribe(
  onEvent: (event: ApiEvent) => void,
  onError?: (error: Event) => void,
  onOpen?: () => void,
): () => void {
  const source = new EventSource(apiUrl("/events"));
  if (onOpen) source.onopen = onOpen;
  source.onmessage = (message: MessageEvent<string>) => {
    try {
      onEvent(JSON.parse(message.data) as ApiEvent);
    } catch {
      // a malformed frame is dropped; the next one is independent
    }
  };
  if (onError) source.onerror = onError;
  return () => source.close();
}
