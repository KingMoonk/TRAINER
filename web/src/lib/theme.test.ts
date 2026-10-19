import { afterEach, describe, expect, test, vi } from "vitest";
import { ACCENT_KEY, ACCENT_SCRIPT, ACCENTS, MODES } from "./theme";

describe("theme constants", () => {
  test("the modes are the ones next-themes understands", () => expect([...MODES]).toEqual(["light", "dark", "system"]));

  test("neutral is the first accent and the rest are distinct non-empty ids", () => {
    expect(ACCENTS[0]).toBe("neutral");
    expect(new Set(ACCENTS).size).toBe(ACCENTS.length);
    expect(ACCENTS.every((id) => /^[a-z]+$/.test(id))).toBe(true);
  });
});

// The script runs in <head> as a string before the app loads, so it is tested as one: run it against jsdom's document.
describe("ACCENT_SCRIPT", () => {
  const run = () => new Function(ACCENT_SCRIPT)();
  afterEach(() => document.documentElement.removeAttribute("data-theme"));

  test.each(ACCENTS.filter((id) => id !== "neutral"))("puts a saved %s accent on <html>", (accent) => {
    localStorage.setItem(ACCENT_KEY, accent);
    run();
    expect(document.documentElement.getAttribute("data-theme")).toBe(accent);
  });

  test("neutral is no attribute at all", () => {
    localStorage.setItem(ACCENT_KEY, "neutral");
    run();
    expect(document.documentElement.hasAttribute("data-theme")).toBe(false);
  });

  test("an unknown or missing value leaves <html> alone", () => {
    run();
    expect(document.documentElement.hasAttribute("data-theme")).toBe(false);
    localStorage.setItem(ACCENT_KEY, "magenta");
    run();
    expect(document.documentElement.hasAttribute("data-theme")).toBe(false);
  });

  test("blocked storage does not throw before first paint", () => {
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new DOMException("blocked", "SecurityError");
    });
    expect(run).not.toThrow();
    expect(document.documentElement.hasAttribute("data-theme")).toBe(false);
  });
});
