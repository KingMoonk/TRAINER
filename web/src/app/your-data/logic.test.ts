import { describe, expect, it } from "vitest";
import { COLLECTION, KEYED, RULES, RULE_LABELS, T, fill } from "./copy";
import {
  STATUS_CLASS,
  allowedRules,
  backLink,
  checkFields,
  collectionShown,
  dupLabel,
  errorText,
  kindOf,
  parseRule,
  overviewCounts,
  setShare,
  shownLine,
  sourceAction,
  sourceStatus,
  visibleRows,
  type DataRow,
  type Summary,
} from "./logic";
import { ApiError } from "@/lib/api";

const rows = (...statuses: string[]) => statuses.map((status, i) => ({ id: i, status })) as unknown as DataRow[];

const summary = (...sets: [string, number][]) => ({ sets: sets.map(([name, count]) => ({ name, count })) }) as unknown as Summary;

describe("backLink", () => {
  it("sends text back to Pretraining and the other kinds to Post-training", () => {
    expect(backLink("text")).toEqual({ label: T.backPretraining, href: "/pretraining" });
    expect(backLink("conversations")).toEqual({ label: T.backPost, href: "/post-training" });
    expect(backLink("problems")).toEqual({ label: T.backPost, href: "/post-training" });
  });
});

describe("visibleRows", () => {
  it("keeps every row, or only the near-duplicates", () => {
    const all = rows("In use", "Near-duplicate", "Good", "Near-duplicate");
    expect(visibleRows(all, false)).toBe(all);
    expect(visibleRows(all, true).map((r) => r.id)).toEqual([1, 3]);
    expect(visibleRows([], true)).toEqual([]);
  });
});

describe("dupLabel", () => {
  it("counts the near-duplicates found, marking the filter when on", () => {
    expect(dupLabel(1234, false)).toBe(fill(T.nearDuplicates, { n: "1,234" }));
    expect(dupLabel(0, true)).toBe(fill(T.nearDuplicates, { n: "0" }) + T.showing);
  });
});

describe("shownLine", () => {
  it("is empty when nothing more waits behind Load more", () => expect(shownLine(50, 50, "", false)).toBe(""));
  it("says how many of all are shown", () => expect(shownLine(1000, 12345, "", true)).toBe(fill(T.shown, { shown: "1,000", count: "12,345" })));
  it("says matches when searching, and treats blank space as no search", () => {
    expect(shownLine(1000, 12345, "cat", true)).toBe(fill(T.shownMatches, { shown: "1,000" }));
    expect(shownLine(1000, 12345, "   ", true)).toBe(fill(T.shown, { shown: "1,000", count: "12,345" }));
  });
});

describe("collectionShown", () => {
  it("shows the recommended collection as a card while the kind has no sets, tucks it away once there are some", () => {
    expect(collectionShown(summary())).toBe("card");
    expect(collectionShown(summary(["a", 1]))).toBe("tucked");
    expect(collectionShown(undefined)).toBeNull();
  });
});

describe("STATUS_CLASS", () => {
  it("has a style for each status a row can have", () => {
    for (const status of ["In use", "Held back", "Near-duplicate", "Good", "Review", "From Chat", "Ready"]) expect(STATUS_CLASS[status]).toBeTruthy();
  });
});

describe("sourceAction", () => {
  it("offers to download, then to add, and nothing while it works or after", () => {
    expect(sourceAction({ state: "not downloaded" })).toEqual({ label: COLLECTION.download, action: "download" });
    expect(sourceAction({ state: "downloaded" })).toEqual({ label: COLLECTION.import, action: "import" });
    expect(sourceAction({ state: "downloading" })).toBeNull();
    expect(sourceAction({ state: "imported" })).toBeNull();
  });
});

describe("sourceStatus", () => {
  it("puts a failure first, with its trailing full stops collapsed to one", () => {
    expect(sourceStatus({ state: "downloading", error: "No space..." }, "50%")).toBe(fill(COLLECTION.failed, { error: "No space." }));
    expect(sourceStatus({ state: "not downloaded", error: "Offline" })).toBe(fill(COLLECTION.failed, { error: "Offline." }));
  });
  it("shows live progress while downloading, else the plain state", () => {
    expect(sourceStatus({ state: "downloading", error: "" }, "12 of 40 MB")).toBe("12 of 40 MB");
    expect(sourceStatus({ state: "downloading", error: "" })).toBe(COLLECTION.stateDownloading);
    expect(sourceStatus({ state: "not downloaded", error: "" }, "ignored")).toBe(COLLECTION.stateNotDownloaded);
    expect(sourceStatus({ state: "downloaded", error: "" })).toBe(COLLECTION.stateDownloaded);
    expect(sourceStatus({ state: "imported", error: "" })).toBe(COLLECTION.stateImported);
  });
  it("shows an unknown state as it is", () => expect(sourceStatus({ state: "paused", error: "" })).toBe("paused"));
});

describe("parseRule", () => {
  const allowed = ["copy", "small_addition", "max_length"];
  it("accepts the key, the key with spaces, or the label, in any case", () => {
    expect(parseRule("copy", allowed)).toBe("copy");
    expect(parseRule("  COPY ", allowed)).toBe("copy");
    expect(parseRule("small_addition", allowed)).toBe("small_addition");
    expect(parseRule("Small Addition", allowed)).toBe("small_addition");
    expect(parseRule(RULE_LABELS.max_length, allowed)).toBe("max_length");
  });
  it("rejects what is not allowed, or blank", () => {
    expect(parseRule("reverse", allowed)).toBeNull();
    expect(parseRule("", allowed)).toBeNull();
    expect(parseRule("copy", [])).toBeNull();
  });
});

describe("allowedRules", () => {
  it("keeps keyed rules out unless the problem already has one", () => {
    expect(allowedRules(undefined)).toBe(RULES);
    expect(allowedRules("copy")).toBe(RULES);
    expect(allowedRules("")).toBe(RULES);
    expect(allowedRules("final_number")).toEqual([...RULES, ...KEYED]);
    expect(allowedRules("choice")).toEqual([...RULES, ...KEYED]);
  });
});

describe("checkFields", () => {
  it("sends only the fields a kind takes", () => {
    expect(checkFields("text", { text: "hello", junk: "x" })).toEqual({ fields: { text: "hello" } });
    expect(checkFields("conversations", { user: "hi", assistant: "hey", junk: "x" })).toEqual({ fields: { user: "hi", assistant: "hey" } });
  });
  it("keeps the text as typed (padding included)", () => expect(checkFields("text", { text: "  hi " })).toEqual({ fields: { text: "  hi " } }));
  it("asks for every field, rejecting blank and missing ones", () => {
    const missing = { error: T.fillAll };
    expect(checkFields("text", { text: "   " })).toEqual(missing);
    expect(checkFields("text", {})).toEqual(missing);
    expect(checkFields("conversations", { user: "hi", assistant: "" })).toEqual(missing);
    expect(checkFields("problems", { prompt: "2+2" })).toEqual(missing);
    expect(checkFields("problems", { rule: "copy" })).toEqual(missing);
  });
  it("turns a problem's rule label into its key", () => {
    expect(checkFields("problems", { prompt: "copy this", rule: "Small Addition" })).toEqual({ fields: { prompt: "copy this", rule: "small_addition" } });
  });
  it("lists the rules when one is unknown", () => {
    expect(checkFields("problems", { prompt: "p", rule: "nope" })).toEqual({ error: fill(T.unknownRule, { rules: RULES.join(", ") }) });
  });
  it("accepts a keyed rule only for a problem that already has it", () => {
    expect(checkFields("problems", { prompt: "p", rule: "final_number" })).toHaveProperty("error");
    expect(checkFields("problems", { prompt: "p", rule: "final_number" }, "final_number")).toEqual({ fields: { prompt: "p", rule: "final_number" } });
    expect(checkFields("problems", { prompt: "p", rule: "nope" }, "choice")).toEqual({ error: fill(T.unknownRule, { rules: [...RULES, ...KEYED].join(", ") }) });
  });
});

describe("kindOf", () => {
  it("passes a known kind through, else conversations", () => {
    expect(kindOf("text")).toBe("text");
    expect(kindOf("problems")).toBe("problems");
    expect(kindOf("conversations")).toBe("conversations");
    expect(kindOf("TEXT")).toBe("conversations");
    expect(kindOf("")).toBe("conversations");
    expect(kindOf(null)).toBe("conversations");
    expect(kindOf(undefined)).toBe("conversations");
  });
});

describe("setShare", () => {
  it("is the whole percent of the total", () => {
    expect(setShare(1, 3)).toBe(33);
    expect(setShare(3, 3)).toBe(100);
  });
  it("is 0 for an empty kind and never above 100", () => {
    expect(setShare(0, 0)).toBe(0);
    expect(setShare(5, 3)).toBe(100);
  });
});

describe("overviewCounts", () => {
  it("lists total, held back and near-duplicates with thousands separators", () => {
    const s = { count: 12345, held_back: 20, near_duplicates: 0 } as unknown as Summary;
    expect(overviewCounts(s)).toEqual([
      { key: "total", label: T.statTotal, value: "12,345" },
      { key: "held", label: T.statHeldBack, value: "20" },
      { key: "dups", label: T.statDuplicates, value: "0" },
    ]);
  });
  it("is empty before the summary loads", () => expect(overviewCounts(undefined)).toEqual([]));
});

describe("errorText", () => {
  it("prefers the server's detail sentence", () => expect(errorText(new ApiError(409, "/d", '{"detail":"Busy."}'))).toBe("Busy."));
  it("joins the messages of a validation error list, skipping blanks", () => {
    const body = JSON.stringify({ detail: [{ msg: "field required" }, {}, { msg: "too long" }] });
    expect(errorText(new ApiError(422, "/d", body))).toBe("field required; too long");
  });
  it("shows an empty list of messages as nothing", () => expect(errorText(new ApiError(422, "/d", '{"detail":[]}'))).toBe(""));
  it("shows a body with no usable detail, or not JSON, as it is", () => {
    expect(errorText(new ApiError(500, "/d", '{"other":1}'))).toBe('{"other":1}');
    expect(errorText(new ApiError(502, "/d", "Bad gateway"))).toBe("Bad gateway");
    expect(errorText(new ApiError(422, "/d", '{"detail":5}'))).toBe('{"detail":5}');
  });
  it("falls back to the error message or the string form", () => {
    expect(errorText(new ApiError(500, "/d", ""))).toBe("500 from /d");
    expect(errorText(new Error("offline"))).toBe("offline");
    expect(errorText("boom")).toBe("boom");
    expect(errorText(null)).toBe("null");
  });
});
