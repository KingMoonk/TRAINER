// A reader for a server-sent-events stream that answers a POST. EventSource (used for GET /events in api.ts) can only GET,
// and POST /chat/reply streams on its own response, so this reads the body with fetch. Abort the signal to stop the stream.

import { ApiError, apiUrl } from "@/lib/api";

/** One message of the stream. The API frames each as `data: {"kind", "payload"}`; a frame without that shape is skipped. */
export type SseEvent = { kind: string; payload: unknown };

/**
 * Turns chunks of an SSE body into events. A frame ends at a blank line and a chunk can end mid-frame, so the tail is
 * kept until the next `push`. Comment lines (`: ping`) and malformed frames are dropped.
 */
export class SseParser {
  private buffer = "";

  push(chunk: string): SseEvent[] {
    this.buffer += chunk;
    const frames = this.buffer.split(/\r\n\r\n|\n\n|\r\r/);
    this.buffer = frames.pop() ?? "";
    const events: SseEvent[] = [];
    for (const frame of frames) {
      const data = frame
        .split(/\r\n|\n|\r/)
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).replace(/^ /, ""))
        .join("\n");
      if (!data) continue;
      try {
        const parsed: unknown = JSON.parse(data);
        if (parsed && typeof parsed === "object" && typeof (parsed as SseEvent).kind === "string") {
          events.push({ kind: (parsed as SseEvent).kind, payload: (parsed as SseEvent).payload });
        }
      } catch {
        // a malformed frame is dropped; the next one is independent
      }
    }
    return events;
  }
}

/**
 * POST `body` as JSON to `path` and call `onEvent` for each event on the response, until it ends or `signal` aborts.
 * Resolves when the stream ends; an abort resolves too (the caller knows it asked). A non-2xx status throws ApiError.
 */
export async function postStream(
  path: string,
  body: unknown,
  onEvent: (event: SseEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  try {
    const res = await fetch(apiUrl(path), {
      method: "POST",
      headers: { Accept: "text/event-stream", "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    });
    if (!res.ok) throw new ApiError(res.status, path, await res.text());
    if (!res.body) return;
    const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
    const parser = new SseParser();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) return;
      for (const event of parser.push(value)) onEvent(event);
    }
  } catch (error) {
    if (signal?.aborted) return;
    throw error;
  }
}
