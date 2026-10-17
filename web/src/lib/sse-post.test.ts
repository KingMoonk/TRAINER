import { describe, expect, test, vi } from "vitest";
import { ApiError } from "./api-core";
import { getStream, postStream, SseParser, type SseEvent } from "./sse-post";
import { TOKEN_KEY } from "./token";
import { mockApi } from "@/test/network";

const frame = (kind: string, payload: object) => `data: ${JSON.stringify({ kind, payload })}\n\n`;

describe("SseParser", () => {
  test("joins frames split across chunks, and skips comments and malformed frames", () => {
    const parser = new SseParser();
    const whole = frame("text", { text: "Hel" }) + ": ping\n\n" + "data: not json\n\n" + frame("done", { text: "Hello" });
    const events = [...parser.push(whole.slice(0, 20)), ...parser.push(whole.slice(20, 70)), ...parser.push(whole.slice(70))];
    expect(events).toEqual([
      { kind: "text", payload: { text: "Hel" } },
      { kind: "done", payload: { text: "Hello" } },
    ]);
    expect(new SseParser().push(frame("text", { text: "a" }).replace(/\n/g, "\r\n"))).toEqual([{ kind: "text", payload: { text: "a" } }]);
  });

  test("an unfinished frame waits for its blank line", () => {
    const parser = new SseParser();
    expect(parser.push('data: {"kind":"a","payload":1}\n')).toEqual([]);
    expect(parser.push("\n")).toEqual([{ kind: "a", payload: 1 }]);
  });

  test("a CRLF blank line split between chunks still ends the frame", () => {
    const parser = new SseParser();
    expect(parser.push('data: {"kind":"a","payload":1}\r\n\r')).toEqual([]);
    expect(parser.push("\n")).toEqual([{ kind: "a", payload: 1 }]);
  });

  test("accepts bare CR line endings", () => {
    expect(new SseParser().push('data: {"kind":"a","payload":2}\r\r')).toEqual([{ kind: "a", payload: 2 }]);
  });

  test("several frames in one chunk come out in order", () => {
    const events = new SseParser().push(frame("a", { n: 1 }) + frame("b", { n: 2 }) + frame("c", { n: 3 }));
    expect(events.map((e) => e.kind)).toEqual(["a", "b", "c"]);
  });

  test("a chunk without a finished frame yields nothing, then the whole frame later", () => {
    const parser = new SseParser();
    const text = frame("a", { n: 1 });
    const out: SseEvent[] = [];
    for (const ch of text) out.push(...parser.push(ch)); // one character at a time
    expect(out).toEqual([{ kind: "a", payload: { n: 1 } }]);
  });

  test("data: without the optional space is read, and only one leading space is stripped", () => {
    expect(new SseParser().push('data:{"kind":"a","payload":1}\n\n')).toEqual([{ kind: "a", payload: 1 }]);
    expect(new SseParser().push('data:  {"kind":"a","payload":"x"}\n\n')).toEqual([{ kind: "a", payload: "x" }]);
  });

  test("a payload split over several data lines is joined with newlines", () => {
    const events = new SseParser().push('data: {"kind":"a",\ndata: "payload":{"n":1}}\n\n');
    expect(events).toEqual([{ kind: "a", payload: { n: 1 } }]);
  });

  test("other SSE fields (event, id, retry) are ignored", () => {
    const events = new SseParser().push('event: message\nid: 7\nretry: 100\ndata: {"kind":"a","payload":1}\n\n');
    expect(events).toEqual([{ kind: "a", payload: 1 }]);
  });

  test("a frame with no data line is skipped", () => {
    expect(new SseParser().push("event: ping\nid: 1\n\n")).toEqual([]);
  });

  test.each([
    ["no kind", '{"payload":1}'],
    ["a numeric kind", '{"kind":3,"payload":1}'],
    ["a null body", "null"],
    ["a number", "42"],
    ["a string", '"hello"'],
    ["an array", "[1,2]"],
    ["truncated JSON", '{"kind":"a","payload":'],
  ])("skips a frame that is %s, and reads the next", (_, data) => {
    const events = new SseParser().push(`data: ${data}\n\n` + frame("ok", { n: 1 }));
    expect(events).toEqual([{ kind: "ok", payload: { n: 1 } }]);
  });

  test("a frame without a payload yields an event with an undefined one", () => {
    expect(new SseParser().push('data: {"kind":"a"}\n\n')).toEqual([{ kind: "a", payload: undefined }]);
  });

  test("a malformed frame does not poison the buffer for later pushes", () => {
    const parser = new SseParser();
    expect(parser.push("data: {oops\n\n")).toEqual([]);
    expect(parser.push(frame("a", {}))).toEqual([{ kind: "a", payload: {} }]);
  });
});

/** A streamed body that sends each chunk on its own read, then ends (or stays open with `hold`, or fails with `fail`). */
function sseResponse(chunks: string[], opts: { hold?: AbortSignal; fail?: Error; status?: number } = {}): Response {
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      if (opts.fail) controller.error(opts.fail);
      else if (!opts.hold) controller.close();
      else opts.hold.addEventListener("abort", () => controller.error(new DOMException("aborted", "AbortError")), { once: true });
    },
  });
  return new Response(body, { status: opts.status ?? 200, headers: { "Content-Type": "text/event-stream" } });
}

describe("postStream", () => {
  test("posts the body as JSON with the SSE Accept header and delivers each event", async () => {
    let seen: { headers: Headers; body: unknown } | undefined;
    mockApi({
      "POST /chat/reply": (call, request) => {
        seen = { headers: request.headers, body: call.body };
        return sseResponse([frame("text", { text: "Hel" }), frame("done", { text: "Hello" })]);
      },
    });
    const events: SseEvent[] = [];
    await postStream("/chat/reply", { version: 2, messages: [{ role: "user", content: "hi" }] }, (e) => events.push(e));
    expect(events).toEqual([
      { kind: "text", payload: { text: "Hel" } },
      { kind: "done", payload: { text: "Hello" } },
    ]);
    expect(seen?.body).toEqual({ version: 2, messages: [{ role: "user", content: "hi" }] });
    expect(seen?.headers.get("Accept")).toBe("text/event-stream");
    expect(seen?.headers.get("Content-Type")).toBe("application/json");
    expect(seen?.headers.has("X-Groundwork-Token")).toBe(false);
  });

  test("sends the stored token", async () => {
    localStorage.setItem(TOKEN_KEY, "secret");
    let token: string | null = null;
    mockApi({
      "POST /chat/reply": (_, request) => {
        token = request.headers.get("X-Groundwork-Token");
        return sseResponse([]);
      },
    });
    await postStream("/chat/reply", {}, () => {});
    expect(token).toBe("secret");
  });

  test("a frame split across network chunks is delivered once whole", async () => {
    const whole = frame("text", { text: "split" });
    mockApi({ "POST /chat/reply": () => sseResponse([whole.slice(0, 7), whole.slice(7, 25), whole.slice(25)]) });
    const events: SseEvent[] = [];
    await postStream("/chat/reply", {}, (e) => events.push(e));
    expect(events).toEqual([{ kind: "text", payload: { text: "split" } }]);
  });

  test("a multi-byte character split between chunks is decoded whole", async () => {
    const bytes = new TextEncoder().encode(frame("text", { text: "héllo ✓" }));
    const cut = bytes.indexOf(0xc3) + 1; // inside the two-byte é
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(bytes.slice(0, cut));
        controller.enqueue(bytes.slice(cut));
        controller.close();
      },
    });
    mockApi({ "POST /chat/reply": () => new Response(body, { status: 200 }) });
    const events: SseEvent[] = [];
    await postStream("/chat/reply", {}, (e) => events.push(e));
    expect(events).toEqual([{ kind: "text", payload: { text: "héllo ✓" } }]);
  });

  test("a non-2xx status throws an ApiError carrying the status, path and body", async () => {
    mockApi({ "POST /chat/reply": { status: 422, body: { detail: "No trained version" } } });
    const onEvent = vi.fn();
    const error = await postStream("/chat/reply", {}, onEvent).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({ status: 422, path: "/chat/reply", body: '{"detail":"No trained version"}' });
    expect(onEvent).not.toHaveBeenCalled();
  });

  test("a response with no body resolves without events", async () => {
    mockApi({ "POST /chat/reply": { status: 204 } });
    const onEvent = vi.fn();
    await expect(postStream("/chat/reply", {}, onEvent)).resolves.toBeUndefined();
    expect(onEvent).not.toHaveBeenCalled();
  });

  test("a failure while reading the body rejects with that error", async () => {
    mockApi({ "POST /chat/reply": () => sseResponse([frame("a", {})], { fail: new TypeError("connection reset") }) });
    const events: SseEvent[] = [];
    await expect(postStream("/chat/reply", {}, (e) => events.push(e))).rejects.toThrow("connection reset");
    expect(events.length).toBeLessThanOrEqual(1);
  });

  test("a network failure rejects", async () => {
    mockApi({
      "POST /chat/reply": () => {
        throw new TypeError("Failed to fetch");
      },
    });
    await expect(postStream("/chat/reply", {}, () => {})).rejects.toThrow("Failed to fetch");
  });

  test("aborting mid-stream resolves quietly and stops delivering", async () => {
    const controller = new AbortController();
    mockApi({ "POST /chat/reply": (_, request) => sseResponse([frame("text", { text: "a" })], { hold: request.signal }) });
    const events: SseEvent[] = [];
    const done = postStream("/chat/reply", {}, (e) => { events.push(e); controller.abort(); }, controller.signal);
    await expect(done).resolves.toBeUndefined();
    expect(events).toEqual([{ kind: "text", payload: { text: "a" } }]);
  });

  test("aborting before the response arrives resolves rather than throwing", async () => {
    const controller = new AbortController();
    mockApi({
      "POST /chat/reply": (_, request) =>
        new Promise((_resolve, reject) => {
          const abort = () => reject(new DOMException("aborted", "AbortError"));
          if (request.signal.aborted) abort(); // the abort can land before the handler runs
          else request.signal.addEventListener("abort", abort);
        }),
    });
    const done = postStream("/chat/reply", {}, () => {}, controller.signal);
    controller.abort();
    await expect(done).resolves.toBeUndefined();
  });

  test("an error after an abort is swallowed, but without an abort it is thrown", async () => {
    const controller = new AbortController();
    mockApi({
      "POST /chat/reply": () => {
        controller.abort();
        throw new Error("boom");
      },
    });
    await expect(postStream("/chat/reply", {}, () => {}, controller.signal)).resolves.toBeUndefined();
    await expect(postStream("/chat/reply", {}, () => {}, new AbortController().signal)).rejects.toThrow("boom");
  });
});

describe("getStream", () => {
  test("GETs the path and calls onOpen once the response is accepted, before any event", async () => {
    const order: string[] = [];
    let method = "";
    mockApi({
      "GET /events": (call) => {
        method = call.method;
        return sseResponse([frame("status", { n: 1 })]);
      },
    });
    await getStream("/events", (e) => order.push(e.kind), undefined, () => order.push("open"));
    expect(method).toBe("GET");
    expect(order).toEqual(["open", "status"]);
  });

  test("does not call onOpen when the server refuses", async () => {
    mockApi({ "GET /events": { status: 401, body: { detail: "bad token" } } });
    const onOpen = vi.fn();
    await expect(getStream("/events", () => {}, undefined, onOpen)).rejects.toMatchObject({ status: 401 });
    expect(onOpen).not.toHaveBeenCalled();
  });

  test("onOpen is optional", async () => {
    mockApi({ "GET /events": () => sseResponse([frame("a", {})]) });
    const events: SseEvent[] = [];
    await getStream("/events", (e) => events.push(e));
    expect(events).toHaveLength(1);
  });
});
