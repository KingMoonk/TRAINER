import { describe, expect, test, vi } from "vitest";
import { ApiError } from "./api-core";
import { emit, mockApi } from "@/test/network";
import { fetchScene, fetchSnapshots, runProbeStep, viewerError, type SceneQuery } from "./viewer";

const query: SceneQuery = { version: 2, level: "Head", mode: "Train", headStage: "pattern", tileKind: "gradient", snapshot: 4 };
const signal = () => new AbortController().signal;
const job = (over: object = {}) => ({ id: "probe-1", kind: "probe", status: "done", created_at: 0, progress: "", error: "", ...over });

describe("fetchScene", () => {
  test("asks for the scene with every control as a query parameter", async () => {
    const calls = mockApi({ "GET /viewer/scene": { body: { schema: 1, full: true } } });
    await expect(fetchScene(query, signal())).resolves.toEqual({ schema: 1, full: true });
    const params = new URLSearchParams(calls[0].search);
    expect(Object.fromEntries(params)).toEqual({ version: "2", level: "Head", mode: "Train", headStage: "pattern", tileKind: "gradient", snapshot: "4" });
  });

  test("leaves the version out when none is chosen", async () => {
    const calls = mockApi({ "GET /viewer/scene": { body: {} } });
    await fetchScene({ ...query, version: undefined }, signal());
    expect(new URLSearchParams(calls[0].search).has("version")).toBe(false);
  });

  test("a mode with spaces is encoded and arrives intact", async () => {
    const calls = mockApi({ "GET /viewer/scene": { body: {} } });
    await fetchScene({ ...query, mode: "Ride a token" }, signal());
    expect(new URLSearchParams(calls[0].search).get("mode")).toBe("Ride a token");
  });

  test("a refusal becomes an ApiError with the server's body", async () => {
    mockApi({ "GET /viewer/scene": { status: 422, body: { detail: "No such version" } } });
    const error = await fetchScene(query, signal()).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    // (the path is read from the response URL, which a mocked Response does not have)
    expect(error).toMatchObject({ status: 422, body: '{"detail":"No such version"}' });
  });
});

describe("fetchSnapshots", () => {
  test("returns the list for a version", async () => {
    const calls = mockApi({ "GET /viewer/snapshots": { body: [{ step: 0 }, { step: 100 }] } });
    await expect(fetchSnapshots(3, signal())).resolves.toEqual([{ step: 0 }, { step: 100 }]);
    expect(calls[0].search).toBe("?version=3");
  });

  test("asks for the current version when none is given", async () => {
    const calls = mockApi({ "GET /viewer/snapshots": { body: [] } });
    await fetchSnapshots(undefined, signal());
    expect(calls[0].search).toBe("");
  });
});

describe("viewerError", () => {
  test("prefers the server's detail sentence", () => {
    expect(viewerError(new ApiError(422, "/viewer/scene", JSON.stringify({ detail: "No snapshots yet" })))).toBe("No snapshots yet");
  });

  test.each([
    ["a body that is not JSON", "<html>bad gateway</html>"],
    ["JSON without a detail", JSON.stringify({ error: "x" })],
    ["a detail that is not a string", JSON.stringify({ detail: [{ msg: "bad" }] })],
  ])("falls back to the request error for %s", (_, body) => {
    expect(viewerError(new ApiError(502, "/viewer/scene", body))).toBe("502 from /viewer/scene");
  });

  test("uses a plain Error's message, and stringifies anything else", () => {
    expect(viewerError(new TypeError("Failed to fetch"))).toBe("Failed to fetch");
    expect(viewerError("just text")).toBe("just text");
    expect(viewerError(null)).toBe("null");
  });
});

describe("runProbeStep", () => {
  test("posts the version and follows the job to its result", async () => {
    const calls = mockApi({ "POST /viewer/probe": { body: job({ status: "done", result: { snapshot: 5, note: "ok" } }) } });
    await expect(runProbeStep(7, signal())).resolves.toEqual({ snapshot: 5, note: "ok" });
    expect(calls[0].body).toEqual({ version: 7 });
  });

  test("sends no version, so the server picks, when there is none", async () => {
    const calls = mockApi({ "POST /viewer/probe": { body: job({ status: "done", result: { snapshot: 0, note: "" } }) } });
    await runProbeStep(undefined, signal());
    expect(calls[0].body).toEqual({});
  });

  test("a probe still running finishes on its /events message", async () => {
    const calls = mockApi({
      "POST /viewer/probe": { body: job({ status: "running" }) },
      "GET /jobs/probe-1": { body: job({ status: "running" }) },
    });
    const waiting = runProbeStep(1, signal());
    await vi.waitFor(() => expect(calls.some((c) => c.path === "/jobs/probe-1")).toBe(true));
    emit("job", job({ status: "done", result: { snapshot: 2, note: "from the stream" } }));
    await expect(waiting).resolves.toEqual({ snapshot: 2, note: "from the stream" });
  });

  test("a failed probe job rejects with the job's error", async () => {
    mockApi({ "POST /viewer/probe": { body: job({ status: "failed", error: "No checkpoint" }) } });
    await expect(runProbeStep(1, signal())).rejects.toThrow("No checkpoint");
  });

  test("a refused submission rejects before any job is followed", async () => {
    const calls = mockApi({ "POST /viewer/probe": { status: 409, body: { detail: "Busy" } } });
    await expect(runProbeStep(1, signal())).rejects.toMatchObject({ status: 409 });
    expect(calls).toHaveLength(1);
  });
});
