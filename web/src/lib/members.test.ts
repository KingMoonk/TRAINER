import { describe, expect, it } from "vitest";
import { etaText, isBlocked, memberChanges, pasteAddress, stateLabel, type MemberState, type Seen } from "./members";

const seen = (state: MemberState, over: Partial<Seen> = {}): Seen => ({ key: "run:r1:gpu-box", node: "gpu-box", run: "Version 7", state, here: false, ...over });
const before = (state: MemberState, key = "run:r1:gpu-box", here = false) => new Map<string, Seen>([[key, seen(state, { key, here })]]);

describe("member vocabulary", () => {
  it("has one word for each state, the same on every page", () => {
    expect(stateLabel("invited")).toBe("Invited");
    expect(stateLabel("joining")).toBe("Joining");
    expect(stateLabel("training")).toBe("Training");
    expect(stateLabel("behind")).toBe("Behind");
    expect(stateLabel("left")).toBe("Left");
    expect(stateLabel("waiting")).toBe("Waiting");
    expect(stateLabel("unreachable")).toBe("Unreachable");
  });

  it("calls only waiting and unreachable blocked", () => {
    expect(["waiting", "unreachable"].every((s) => isBlocked(s as MemberState))).toBe(true);
    expect(["invited", "joining", "training", "behind", "left"].some((s) => isBlocked(s as MemberState))).toBe(false);
  });

  it("says how long a blocking job has left, or nothing when it cannot tell", () => {
    expect(etaText(720)).toBe(", about 12 min left");
    expect(etaText(5)).toBe(", about 1 min left");
    expect(etaText(null)).toBe("");
    expect(etaText(undefined)).toBe("");
  });
});

describe("pasteAddress", () => {
  it("puts a bare host behind https:// and drops a port", () => {
    expect(pasteAddress("beast")).toBe("https://beast");
    expect(pasteAddress("beast.tail1234.ts.net:8000")).toBe("https://beast.tail1234.ts.net");
    expect(pasteAddress(" beast ")).toBe("https://beast");
  });

  it("keeps a URL as it was given, without a trailing slash", () => {
    expect(pasteAddress("https://beast.tail1234.ts.net/")).toBe("https://beast.tail1234.ts.net");
    expect(pasteAddress("http://beast:8000")).toBe("http://beast:8000");
  });
});

describe("memberChanges", () => {
  it("only records on the first read, so opening a page toasts nothing", () => {
    const { next, changes } = memberChanges(null, [seen("training")]);
    expect(changes).toEqual([]);
    expect(next.get("run:r1:gpu-box")?.state).toBe("training");
  });

  it("says a machine joined when it was not yet training", () => {
    for (const was of ["invited", "waiting", "joining"] as const) {
      expect(memberChanges(before(was), [seen("training")]).changes).toEqual([
        { id: "run:r1:gpu-box:training", kind: "success", text: "gpu-box joined 'Version 7'." },
      ]);
    }
  });

  it("speaks of this machine on the machine that was added", () => {
    const [change] = memberChanges(before("joining", "invite:j1", true), [seen("training", { key: "invite:j1", here: true })]).changes;
    expect(change.text).toBe("This machine joined 'Version 7'.");
  });

  it("says a machine fell behind, dropped out or left", () => {
    expect(memberChanges(before("training"), [seen("behind")]).changes[0]).toMatchObject({ kind: "warning", text: "gpu-box fell behind in 'Version 7'." });
    expect(memberChanges(before("behind"), [seen("dropped")]).changes[0]).toMatchObject({ kind: "warning", text: "gpu-box dropped out of 'Version 7'." });
    expect(memberChanges(before("training"), [seen("left")]).changes[0]).toMatchObject({ kind: "info", text: "gpu-box left 'Version 7'." });
    expect(memberChanges(before("training"), [seen("stopped")]).changes[0]).toMatchObject({ text: "gpu-box left 'Version 7'." });
  });

  it("says nothing when the state did not change, or a machine that never joined is stopped", () => {
    expect(memberChanges(before("training"), [seen("training")]).changes).toEqual([]);
    expect(memberChanges(before("unreachable"), [seen("stopped")]).changes).toEqual([]);
  });

  it("says a machine left when the invite for it is gone", () => {
    const { changes } = memberChanges(before("training", "invite:j1", true), []);
    expect(changes).toEqual([{ id: "invite:j1:gone", kind: "info", text: "This machine left 'Version 7'." }]);
  });
});
