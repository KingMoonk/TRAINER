import { afterEach, describe, expect, test, vi } from "vitest";
import { mockApi } from "@/test/network";
import { TOKEN_KEY } from "./token";

// BASE is read from the environment when the module loads, so each case loads a fresh copy.
async function load(base?: string) {
  vi.resetModules();
  if (base === undefined) vi.stubEnv("NEXT_PUBLIC_API_URL", undefined);
  else vi.stubEnv("NEXT_PUBLIC_API_URL", base);
  return import("./api-core");
}

afterEach(() => vi.unstubAllEnvs());

describe("BASE and apiUrl", () => {
  test("BASE drops one trailing slash", async () => {
    expect((await load("http://node:8787/")).BASE).toBe("http://node:8787");
    expect((await load("http://node:8787")).BASE).toBe("http://node:8787");
  });

  test("an unset or empty origin means same origin", async () => {
    expect((await load()).BASE).toBe("");
    const core = await load("");
    expect(core.apiUrl("/events")).toBe("/events");
  });

  test("apiUrl joins the origin and path, adding the leading slash a path may lack", async () => {
    const { apiUrl } = await load("http://node:8787/");
    expect(apiUrl("/jobs/1")).toBe("http://node:8787/jobs/1");
    expect(apiUrl("jobs/1")).toBe("http://node:8787/jobs/1");
  });

  test("the dev stack reaches the local API directly but keeps the tailnet proxy", async () => {
    const { resolvedBase } = await load("/api");
    expect(resolvedBase("/api", "127.0.0.1")).toBe("http://127.0.0.1:8787");
    expect(resolvedBase("/api", "localhost")).toBe("http://127.0.0.1:8787");
    expect(resolvedBase("/api", "gpu-box.tailnet.ts.net")).toBe("/api");
    expect(resolvedBase("http://node:8787/", "localhost")).toBe("http://node:8787");
  });
});

describe("ApiError", () => {
  test("carries the status, path and body, and says status and path in its message", async () => {
    const { ApiError } = await load("http://node:8787");
    const error = new ApiError(409, "/versions/published", "no store");
    expect(error).toBeInstanceOf(Error);
    expect(error.name).toBe("ApiError");
    expect(error.message).toBe("409 from /versions/published");
    expect([error.status, error.path, error.body]).toEqual([409, "/versions/published", "no store"]);
  });
});

describe("apiFetch", () => {
  test("requests the API url and passes the init through", async () => {
    const { apiFetch } = await load("http://localhost:8787");
    const calls = mockApi({ "POST /uploads": { body: { ok: true } } });
    const res = await apiFetch("/uploads", { method: "POST", body: "payload" });
    expect(res.ok).toBe(true);
    expect(calls).toEqual([{ method: "POST", path: "/uploads", search: "", body: "payload" }]);
  });

  test("adds the token header when one is stored, and none otherwise", async () => {
    const { apiFetch } = await load("http://localhost:8787");
    const seen: (string | null)[] = [];
    mockApi({ "GET /x": (_, request) => (seen.push(request.headers.get("X-Groundwork-Token")), { body: {} }) });
    await apiFetch("/x");
    localStorage.setItem(TOKEN_KEY, "secret");
    await apiFetch("/x");
    expect(seen).toEqual([null, "secret"]);
  });

  test("the caller's own headers are kept and win over the token header", async () => {
    const { apiFetch } = await load("http://localhost:8787");
    localStorage.setItem(TOKEN_KEY, "secret");
    const seen: Record<string, string | null> = {};
    mockApi({
      "GET /x": (_, request) => {
        seen.token = request.headers.get("X-Groundwork-Token");
        seen.accept = request.headers.get("Accept");
        return { body: {} };
      },
    });
    await apiFetch("/x", { headers: { Accept: "text/csv", "X-Groundwork-Token": "override" } });
    expect(seen).toEqual({ token: "override", accept: "text/csv" });
  });
});
