import { describe, expect, it } from "vitest";
import { ApiError } from "@/lib/api";
import { AGES, ALL, DEFAULT_CHOICE, choiceDays, dataApplies, downloadName, errorText, fill, sizeText } from "./logic";

describe("sizes, server sentences and file names are read the way the Tk page shows them", () => {
  it("covers the examples the page was first checked against", () => {
    expect(sizeText(1.5e9)).toBe("1.5 GB");
    expect(sizeText(2.5e6)).toBe("2.5 MB");
    expect(errorText(new ApiError(409, "/storage/clean", JSON.stringify({ detail: "Stop the running job first." })))).toBe("Stop the running job first.");
    expect(downloadName('attachment; filename="my-model-hf.zip"', "x")).toBe("my-model-hf.zip");
    expect(downloadName(null, "fallback")).toBe("fallback");
  });
});

describe("choices", () => {
  it("lists the ages in days with 'all' apart from them, defaulting to 30 days", () => {
    expect(AGES).toEqual([1, 7, 30, 90]);
    expect(ALL).toBe("all");
    expect(AGES).toContain(Number(DEFAULT_CHOICE));
  });
  it("maps a choice to days, 'all' to null", () => {
    expect(choiceDays("7")).toBe(7);
    expect(choiceDays("all")).toBeNull();
  });
  it("counts 'also delete my data' only with all history", () => {
    expect(dataApplies("all", true)).toBe(true);
    expect(dataApplies("all", false)).toBe(false);
    expect(dataApplies("30", true)).toBe(false);
  });
});

describe("sizeText", () => {
  it.each([
    [0, "0.0 MB"],
    [49_999, "0.0 MB"],
    [50_000, "0.1 MB"],
    [999_999_999, "1000.0 MB"],
    [1e9, "1.0 GB"],
    [120e9, "120.0 GB"],
  ])("%d bytes is %s", (bytes, text) => expect(sizeText(bytes)).toBe(text));
});

describe("fill", () => {
  it("replaces each placeholder, repeats included", () => expect(fill("{a} and {a} and {b}", { a: 1, b: "x" })).toBe("1 and 1 and x"));
  it("blanks a placeholder with no value, and leaves other text alone", () => {
    expect(fill("Hello {name}!", {})).toBe("Hello !");
    expect(fill("no braces", { a: 1 })).toBe("no braces");
  });
  it("keeps a value of 0", () => expect(fill("{n} items", { n: 0 })).toBe("0 items"));
});

describe("errorText", () => {
  it("prefers the server's detail sentence", () => expect(errorText(new ApiError(409, "/s", '{"detail":"Busy."}'))).toBe("Busy."));
  it("shows the body when it is not JSON, or has no string detail", () => {
    expect(errorText(new ApiError(502, "/s", "Bad gateway"))).toBe("Bad gateway");
    expect(errorText(new ApiError(422, "/s", '{"detail":[1]}'))).toBe('{"detail":[1]}');
  });
  it("shows the status line when the body is empty", () => expect(errorText(new ApiError(500, "/s", ""))).toBe("500 from /s"));
  it("handles other errors and values", () => {
    expect(errorText(new Error("offline"))).toBe("offline");
    expect(errorText("boom")).toBe("boom");
  });
});

describe("downloadName", () => {
  it("reads a quoted or unquoted filename", () => {
    expect(downloadName('attachment; filename="a b.zip"', "x")).toBe("a b.zip");
    expect(downloadName("attachment; filename=model.zip", "x")).toBe("model.zip");
    expect(downloadName("attachment; filename=model.zip; size=3", "x")).toBe("model.zip");
  });
  it("decodes the UTF-8 form, and takes whichever form comes first", () => {
    expect(downloadName("attachment; filename*=UTF-8''my%20model%C3%A9.zip", "x")).toBe("my modelé.zip");
    expect(downloadName("attachment; filename=\"plain.zip\"; filename*=UTF-8''better%20one.zip", "x")).toBe("plain.zip");
  });
  it("keeps the raw name when it does not decode", () => expect(downloadName("attachment; filename=100%.zip", "x")).toBe("100%.zip"));
  it("uses the fallback for no header, an empty header, or no filename", () => {
    expect(downloadName(null, "fb")).toBe("fb");
    expect(downloadName("", "fb")).toBe("fb");
    expect(downloadName("attachment", "fb")).toBe("fb");
  });
});
