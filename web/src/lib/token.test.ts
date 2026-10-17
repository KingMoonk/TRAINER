import { afterEach, describe, expect, test, vi } from "vitest";
import { apiToken, authHeaders, TOKEN_HEADER, TOKEN_KEY } from "./token";

// Run `body` with `localStorage` replaced on globalThis, then put the original back (the setup file clears it after each test).
function withStorage(replacement: PropertyDescriptor, body: () => void) {
  const original = Object.getOwnPropertyDescriptor(globalThis, "localStorage");
  Object.defineProperty(globalThis, "localStorage", { ...replacement, configurable: true });
  try {
    body();
  } finally {
    if (original) Object.defineProperty(globalThis, "localStorage", original);
  }
}

afterEach(() => localStorage.removeItem(TOKEN_KEY));

describe("apiToken", () => {
  test("is empty when nothing is stored", () => expect(apiToken()).toBe(""));

  test("returns the stored token", () => {
    localStorage.setItem(TOKEN_KEY, "abc123");
    expect(apiToken()).toBe("abc123");
  });

  test("trims whitespace a pasted token may carry", () => {
    localStorage.setItem(TOKEN_KEY, "  abc123\n");
    expect(apiToken()).toBe("abc123");
  });

  test("a blank token counts as none", () => {
    localStorage.setItem(TOKEN_KEY, "   ");
    expect(apiToken()).toBe("");
  });

  test("is empty when reading storage throws", () => {
    localStorage.setItem(TOKEN_KEY, "abc123");
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new DOMException("blocked", "SecurityError");
    });
    expect(apiToken()).toBe("");
  });

  test("is empty when touching localStorage itself throws (site data blocked)", () => {
    localStorage.setItem(TOKEN_KEY, "abc123");
    withStorage(
      {
        get: () => {
          throw new DOMException("blocked", "SecurityError");
        },
      },
      () => expect(apiToken()).toBe(""),
    );
  });

  test("is empty when there is no localStorage at all", () => {
    localStorage.setItem(TOKEN_KEY, "abc123");
    withStorage({ value: undefined }, () => expect(apiToken()).toBe(""));
  });
});

describe("authHeaders", () => {
  test("is empty without a token, so spreading it adds nothing", () => expect(authHeaders()).toEqual({}));

  test("carries the token under the Funnel header", () => {
    localStorage.setItem(TOKEN_KEY, "abc123");
    expect(TOKEN_HEADER).toBe("X-Groundwork-Token");
    expect(authHeaders()).toEqual({ "X-Groundwork-Token": "abc123" });
  });

  test("is empty when storage is blocked", () => {
    localStorage.setItem(TOKEN_KEY, "abc123");
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new DOMException("blocked", "SecurityError");
    });
    expect(authHeaders()).toEqual({});
  });
});
