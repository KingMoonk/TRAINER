import { describe, expect, test, vi } from "vitest";
import { ApiError } from "./api-core";
import { emit, mockApi } from "@/test/network";
import { BUILDER } from "./copy/builder";
import {
  paramBars,
  paramNote,
  paramShares,
  anchoredDraft,
  answerRows,
  bar,
  BLOCK_LINES,
  blockViews,
  changeText,
  type Choices,
  COST_ROWS,
  design,
  detailOf,
  type Draft,
  fitsText,
  guessOf,
  hasField,
  isVersion,
  jobFor,
  loadDraft,
  moduleName,
  moduleOf,
  nextVersionName,
  optionLabel,
  partName,
  piecesFor,
  PART_SLOTS,
  pretraining,
  queueJobs,
  type Registry,
  sameChoices,
  say,
  specBase,
  specOf,
  storeDraft,
  type SplitPreview,
  type Tower,
  upsertVersion,
  type Version,
  versionName,
  versions,
  wordChanged,
} from "./builder";

const choices = (over: Record<string, unknown> = {}) => ({ L: 2, H: 4, W: 64, V: 100, ...over }) as unknown as Choices;
const ver = (number: number, over: Record<string, unknown> = {}) => ({ number, name: `version ${number}`, spec: {}, ...over }) as unknown as Version;
const signal = () => new AbortController().signal;

describe("detailOf", () => {
  test("is the server's detail sentence for an ApiError", () => {
    expect(detailOf(new ApiError(422, "/design/write", JSON.stringify({ detail: "Heads must divide the width." })))).toBe("Heads must divide the width.");
  });

  test.each([
    ["a body that is not JSON", "plain text"],
    ["JSON without a detail", JSON.stringify({ error: 1 })],
    ["a detail that is a list", JSON.stringify({ detail: [{ msg: "x" }] })],
    ["a JSON null", "null"],
  ])("falls back to the ApiError's own message for %s", (_, body) => {
    expect(detailOf(new ApiError(500, "/design/write", body))).toBe("500 from /design/write");
  });

  test("uses a plain Error's message, and stringifies anything else", () => {
    expect(detailOf(new Error("boom"))).toBe("boom");
    expect(detailOf("text")).toBe("text");
    expect(detailOf(undefined)).toBe("undefined");
  });
});

describe("specOf", () => {
  test("is the version's own spec", () => {
    expect(specOf(ver(1, { spec: { n_layer: 2 } }))).toEqual({ n_layer: 2 });
  });

  test("is the empty spec (the default design) for nothing, or an empty spec", () => {
    expect(specOf(undefined)).toEqual({});
    expect(specOf(null)).toEqual({});
    expect(specOf(ver(1, { spec: {} }))).toEqual({});
    expect(specOf(ver(1, { spec: undefined }))).toEqual({});
  });
});

describe("specBase and anchoredDraft", () => {
  const current = choices();
  const baseSpec = { n_layer: 2 };

  test("a draft with no anchor is written onto the current version", () => {
    const draft: Draft = { base: 1, draft: choices(), anchor: null };
    expect(specBase(draft, baseSpec, current)).toEqual({ base: baseSpec, was: current });
  });

  test("an anchored draft is written onto its anchor", () => {
    const loaded = choices({ L: 12 });
    const draft = anchoredDraft(3, { n_layer: 12 }, loaded);
    expect(specBase(draft, baseSpec, current)).toEqual({ base: { n_layer: 12 }, was: loaded });
  });

  test("anchoredDraft keeps the version it is against and starts with the loaded choices", () => {
    const loaded = choices({ L: 12 });
    expect(anchoredDraft(null, { a: 1 }, loaded)).toEqual({ base: null, draft: loaded, anchor: { spec: { a: 1 }, was: loaded } });
  });
});

describe("comparing choices", () => {
  test("wordChanged looks only at the word list", () => {
    expect(wordChanged(choices(), choices({ V: 200 }))).toBe(true);
    expect(wordChanged(choices(), choices({ L: 9 }))).toBe(false);
  });

  test("sameChoices compares by content", () => {
    expect(sameChoices(choices(), choices())).toBe(true);
    expect(sameChoices(choices(), choices({ H: 8 }))).toBe(false);
  });
});

describe("changeText", () => {
  test("joins the chips", () => {
    expect(changeText(["2 more blocks", "wider"], 1)).toBe("2 more blocks, wider");
  });

  test("says what the Tk page says when there are none", () => {
    expect(changeText([], null)).toBe("first design");
    expect(changeText([], 3)).toBe("no change");
  });
});

describe("guessOf", () => {
  test("keeps the picked questions and drops the unpicked", () => {
    expect(guessOf({ error: "lower", step: "", memory: "same" })).toEqual({ error: "lower", memory: "same" });
  });

  test("skipping is the empty guess", () => {
    expect(guessOf(undefined)).toEqual({});
    expect(guessOf({})).toEqual({});
    expect(guessOf({ error: "" })).toEqual({});
  });
});

describe("jobFor", () => {
  const job = { kind: "pretrain", label: "Version 3 pretraining", note: "", eta_s: 10, params: { version: 3, datasets: ["a"] } } as unknown as SplitPreview["jobs"][number];

  test("is the planned job when the version was saved under the planned number", () => {
    expect(jobFor(job, 3, 3)).toBe(job);
  });

  test("follows the number a version was saved as", () => {
    const moved = jobFor(job, 3, 5);
    expect(moved.label).toBe("Version 5 pretraining");
    expect(moved.params).toEqual({ version: 5, datasets: ["a"] });
    expect(job.label).toBe("Version 3 pretraining");
  });

  test("a label that does not name the planned version is left alone", () => {
    expect(jobFor({ ...job, label: "Pretraining" }, 3, 5).label).toBe("Pretraining");
  });
});

describe("version names", () => {
  test("versionName and nextVersionName", () => {
    expect(versionName(4)).toBe("version 4");
    expect(nextVersionName(null)).toBe("version 1");
    expect(nextVersionName(4)).toBe("version 5");
  });
});

describe("the draft kept across reloads", () => {
  const draft: Draft = { base: 2, draft: choices(), anchor: { spec: { a: 1 }, was: choices({ L: 3 }) } };

  test("what is stored is what loads", () => {
    storeDraft(draft);
    expect(loadDraft()).toEqual(draft);
  });

  test("nothing stored loads as null", () => {
    expect(loadDraft()).toBeNull();
  });

  test("storing null forgets the draft", () => {
    storeDraft(draft);
    storeDraft(null);
    expect(loadDraft()).toBeNull();
    expect(localStorage.getItem("builder.draft")).toBeNull();
  });

  test("a draft with no anchor loads with a null one", () => {
    localStorage.setItem("builder.draft", JSON.stringify({ base: 1, draft: choices() }));
    expect(loadDraft()).toEqual({ base: 1, draft: choices(), anchor: null });
  });

  test("a base that is not a number loads as null", () => {
    localStorage.setItem("builder.draft", JSON.stringify({ base: "2", draft: choices() }));
    expect(loadDraft()?.base).toBeNull();
  });

  test.each([
    ["JSON that does not parse", "{oops"],
    ["a JSON string", '"hello"'],
    ["no draft", JSON.stringify({ base: 1 })],
    ["a draft that is not an object", JSON.stringify({ base: 1, draft: 7 })],
    ["a draft from an older shape (no L)", JSON.stringify({ base: 1, draft: { layers: 2 } })],
    ["a null draft", JSON.stringify({ base: 1, draft: null })],
    ["null", "null"],
  ])("a stale or damaged entry (%s) is a fresh start", (_, raw) => {
    localStorage.setItem("builder.draft", raw);
    expect(loadDraft()).toBeNull();
  });

  test("blocked storage loads nothing and storing does not throw", () => {
    const blocked = () => {
      throw new DOMException("blocked", "SecurityError");
    };
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(blocked);
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(blocked);
    vi.spyOn(Storage.prototype, "removeItem").mockImplementation(blocked);
    expect(loadDraft()).toBeNull();
    expect(() => storeDraft(draft)).not.toThrow();
    expect(() => storeDraft(null)).not.toThrow();
  });

  test("a full disk does not throw either", () => {
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new DOMException("full", "QuotaExceededError");
    });
    expect(() => storeDraft(draft)).not.toThrow();
  });
});

describe("say", () => {
  const more = "bigger";
  const less = "smaller";

  test("about the same when nothing was there, or the ratio is within 5%", () => {
    expect(say(0, 10, more, less)).toBe(BUILDER.about_same);
    expect(say(100, 100, more, less)).toBe(BUILDER.about_same);
    expect(say(100, 104, more, less)).toBe(BUILDER.about_same);
    expect(say(100, 96, more, less)).toBe(BUILDER.about_same);
  });

  test("a rise is a multiplier and the `more` word", () => {
    expect(say(100, 230, more, less)).toBe("about 2.3× bigger");
    expect(say(100, 110, more, less)).toBe("about 1.1× bigger");
  });

  test("a fall is the inverse multiplier and the `less` word", () => {
    expect(say(200, 100, more, less)).toBe("about 2.0× smaller");
    expect(say(100, 50, more, less)).toBe("about 2.0× smaller");
  });

  test("the edge of the same band: 5% off is already a change", () => {
    expect(say(100, 106, more, less)).toBe("about 1.1× bigger");
    expect(say(100, 94, more, less)).toBe("about 1.1× smaller");
  });
});

describe("bar", () => {
  test("scales the longer bar to 100", () => {
    expect(bar(50, 100)).toEqual([50, 100]);
    expect(bar(100, 25)).toEqual([100, 25]);
    expect(bar(1, 3)).toEqual([33, 100]);
  });

  test("equal bars are both full", () => {
    expect(bar(7, 7)).toEqual([100, 100]);
  });

  test("two empty bars are both empty rather than dividing by zero", () => {
    expect(bar(0, 0)).toEqual([0, 0]);
    expect(bar(0, 5)).toEqual([0, 100]);
  });
});

describe("COST_ROWS and fitsText", () => {
  test("the cost rows pair each measure with its words", () => {
    expect(COST_ROWS.map((r) => [r.key, r.more, r.less])).toEqual([
      ["params", BUILDER.cost_bigger, BUILDER.cost_smaller],
      ["time", BUILDER.cost_slower, BUILDER.cost_faster],
      ["memory", BUILDER.cost_more, BUILDER.cost_less],
    ]);
  });

  test("fits, does not fit, and unknown", () => {
    expect(fitsText(true)).toBe(`${BUILDER.training_memory} fits`);
    expect(fitsText(false)).toBe(`${BUILDER.training_memory} ${BUILDER.wont_fit}`);
    expect(fitsText(null)).toBe(`${BUILDER.training_memory} ${BUILDER.training_memory_fits}`);
    expect(fitsText(undefined)).toBe(fitsText(null));
  });
});

describe("block lines and part slots", () => {
  test("a block lists its sub-layers top-down with the add-back lines as null", () => {
    expect(BLOCK_LINES).toEqual([null, "channel", "norm", null, "mixer", "norm"]);
    expect(PART_SLOTS).toEqual(["embedding", "position", "head"]);
  });
});

describe("the registry lookups", () => {
  const registry = {
    slots: [
      { slot: "mixer", modules: [{ short: "attn", name: "Attention", fields: [{ name: "heads" }, { name: "window" }] }, { short: "conv", name: "Convolution", fields: [] }] },
      { slot: "channel", modules: [{ short: "mlp", name: "MLP", fields: [{ name: "ratio" }] }] },
    ],
  } as unknown as Registry;

  test("moduleOf finds a module by slot and short name", () => {
    expect(moduleOf(registry, "mixer", "conv")?.name).toBe("Convolution");
    expect(moduleOf(registry, "mixer", "mlp")).toBeUndefined();
    expect(moduleOf(registry, "norm", "attn")).toBeUndefined();
    expect(moduleOf(undefined, "mixer", "attn")).toBeUndefined();
  });

  test("moduleName is the name, or the short name when the registry lacks it", () => {
    expect(moduleName(registry, "mixer", "attn")).toBe("Attention");
    expect(moduleName(registry, "mixer", "ghost")).toBe("ghost");
    expect(moduleName(undefined, "mixer", "attn")).toBe("attn");
  });

  test("hasField asks the module's settings", () => {
    expect(hasField(registry, "mixer", "attn", "heads")).toBe(true);
    expect(hasField(registry, "mixer", "attn", "ratio")).toBe(false);
    expect(hasField(registry, "mixer", "conv", "heads")).toBe(false);
    expect(hasField(registry, "mixer", "ghost", "heads")).toBe(false);
    expect(hasField(undefined, "mixer", "attn", "heads")).toBe(false);
  });

  test("partName is the copy's name for a slot, or the slot itself", () => {
    expect(partName("embedding")).toBe(BUILDER.part_embedding);
    expect(partName("mixer")).toBe(BUILDER.part_mixer);
    expect(partName("unheard_of")).toBe("unheard_of");
  });
});

describe("blockViews", () => {
  const tower = (...blocks: { parts: Record<string, string>; own?: string[] }[]) =>
    ({ blocks: blocks.map((b) => ({ parts: b.parts, own: b.own ?? [] })) }) as unknown as Tower;
  const plain = { norm: "rms", mixer: "attn", channel: "mlp" };

  test("is empty until both towers are known", () => {
    expect(blockViews(choices(), choices(), undefined, tower())).toEqual([]);
    expect(blockViews(choices(), choices(), tower(), undefined)).toEqual([]);
  });

  test("lists blocks newest on top, untagged when nothing changed", () => {
    const t = tower({ parts: plain }, { parts: plain });
    const views = blockViews(choices({ L: 2 }), choices({ L: 2 }), t, t);
    expect(views.map((v) => v.index)).toEqual([1, 0]);
    expect(views.every((v) => v.tag === "" && !v.added && !v.removed)).toBe(true);
    expect(views[0].before).toEqual(plain);
  });

  test("blocks beyond the current version are new, with no `before`", () => {
    const views = blockViews(choices({ L: 1 }), choices({ L: 3 }), tower({ parts: plain }), tower({ parts: plain }, { parts: plain }, { parts: plain }));
    expect(views.map((v) => [v.index, v.tag, v.added, v.before])).toEqual([
      [2, BUILDER.block_new, true, null],
      [1, BUILDER.block_new, true, null],
      [0, "", false, plain],
    ]);
  });

  test("blocks the draft dropped show the current version's parts and choices", () => {
    const dropped = { norm: "layer", mixer: "conv", channel: "mlp" };
    const cur = choices({ L: 2 });
    const views = blockViews(cur, choices({ L: 1 }), tower({ parts: plain }, { parts: dropped }), tower({ parts: plain }));
    expect(views[0]).toMatchObject({ index: 1, removed: true, tag: BUILDER.block_removed, parts: dropped, choices: cur });
  });

  test("a block whose parts differ is changed", () => {
    const swapped = { ...plain, mixer: "conv" };
    const views = blockViews(choices({ L: 1 }), choices({ L: 1 }), tower({ parts: plain }), tower({ parts: swapped }));
    expect(views[0]).toMatchObject({ tag: BUILDER.changed, parts: swapped, before: plain });
  });

  test("changing the heads or the width marks every kept block changed", () => {
    const t = tower({ parts: plain }, { parts: plain });
    expect(blockViews(choices(), choices({ H: 8 }), t, t).map((v) => v.tag)).toEqual([BUILDER.changed, BUILDER.changed]);
    expect(blockViews(choices(), choices({ W: 128 }), t, t).map((v) => v.tag)).toEqual([BUILDER.changed, BUILDER.changed]);
  });

  test("a block with parts of its own, otherwise unchanged, is tagged as such", () => {
    const t = tower({ parts: plain, own: ["mixer"] });
    const [view] = blockViews(choices({ L: 1 }), choices({ L: 1 }), t, t);
    expect(view).toMatchObject({ tag: BUILDER.own_parts, own: ["mixer"] });
  });

  test("a change outranks `own parts`", () => {
    const [view] = blockViews(choices({ L: 1 }), choices({ L: 1, H: 8 }), tower({ parts: plain, own: ["mixer"] }), tower({ parts: plain, own: ["mixer"] }));
    expect(view.tag).toBe(BUILDER.changed);
  });

  test("a kept block carries the draft's choices", () => {
    const draft = choices({ L: 1, H: 8 });
    const [view] = blockViews(choices({ L: 1 }), draft, tower({ parts: plain }), tower({ parts: plain }));
    expect(view.choices).toBe(draft);
  });
});

describe("optionLabel", () => {
  test("is the name when the design can take the option", () => {
    expect(optionLabel({ short: "a", name: "Attention", why: "" } as never)).toBe("Attention");
    expect(optionLabel({ short: "a", name: "Attention" } as never)).toBe("Attention");
  });

  test("adds the server's reason when it cannot", () => {
    expect(optionLabel({ short: "a", name: "Attention", why: "needs an even width" } as never)).toBe(`Attention · ${BUILDER.wont_fit}: needs an even width`);
  });
});

describe("upsertVersion and isVersion", () => {
  test("adds a version in number order", () => {
    expect(upsertVersion([ver(1), ver(3)], ver(2)).map((v) => v.number)).toEqual([1, 2, 3]);
  });

  test("replaces the version with the same number", () => {
    const next = ver(2, { name: "renamed" });
    const list = upsertVersion([ver(1), ver(2)], next);
    expect(list).toHaveLength(2);
    expect(list[1]).toBe(next);
  });

  test("starts a list from nothing, and does not change the old one", () => {
    expect(upsertVersion(undefined, ver(1))).toEqual([ver(1)]);
    const old = [ver(1)];
    upsertVersion(old, ver(2));
    expect(old).toHaveLength(1);
  });

  test("isVersion needs an object with a numeric number", () => {
    expect(isVersion(ver(1))).toBe(true);
    expect(isVersion({ number: "1" })).toBe(false);
    expect(isVersion({})).toBe(false);
    expect(isVersion(null)).toBe(false);
    expect(isVersion(3)).toBe(false);
    expect(isVersion(undefined)).toBe(false);
  });
});

// The /design calls: each one is a POST of the right body to the right route. A mismatch would 422 against the real server.
describe("design (the stateless /design calls)", () => {
  const cur = choices();
  const spec = { n_layer: 2 };

  test.each([
    ["registry", () => design.registry(), "GET", "/design/registry", undefined],
    ["presets", () => design.presets(), "GET", "/design/presets", undefined],
    ["read", () => design.read(spec), "POST", "/design/read", { spec }],
    ["write", () => design.write(spec, cur, null), "POST", "/design/write", { base: spec, choices: cur, was: null }],
    ["choose", () => design.choose(cur, "mixer", "attn", 1), "POST", "/design/choose", { choices: cur, slot: "mixer", short: "attn", block: 1 }],
    ["resize", () => design.resize(cur, 4), "POST", "/design/resize", { choices: cur, n: 4 }],
    ["tower", () => design.tower(cur), "POST", "/design/tower", { choices: cur }],
    ["partOptions", () => design.partOptions(spec, cur, cur, "head", null), "POST", "/design/part-options", { base: spec, choices: cur, was: cur, slot: "head", block: null }],
    ["sizes", () => design.sizes(spec, 100), "POST", "/design/sizes", { spec, v: 100 }],
    ["presetCost", () => design.presetCost(spec), "POST", "/design/preset-cost", { spec }],
    ["computeLines", () => design.computeLines(spec, 100, "cached" as never), "POST", "/design/compute-lines", { spec, v: 100, preview: "cached" }],
    ["expectations", () => design.expectations(cur, cur, spec, null), "POST", "/design/expectations", { cur, draft: cur, base: spec, draft_spec: null }],
    ["changes", () => design.changes(spec, { n_layer: 3 }, cur, cur), "POST", "/design/changes", { base: spec, draft_spec: { n_layer: 3 }, cur, draft: cur }],
    ["questions", () => design.questions(spec, { n_layer: 3 }), "POST", "/design/questions", { old_spec: spec, new_spec: { n_layer: 3 } }],
    [
      "splitPlan",
      () => design.splitPlan(spec, cur, cur, cur, 4, ["text-a"]),
      "POST",
      "/design/split-plan",
      { base: spec, was: cur, cur, draft: cur, parent_version: 4, datasets: ["text-a"] },
    ],
  ])("%s", async (_, call, method, path, body) => {
    const calls = mockApi({ [`${method} ${path}`]: { body: { ok: path } } });
    await expect(call()).resolves.toEqual({ ok: path });
    expect(calls).toHaveLength(1);
    expect(calls[0]).toMatchObject({ method, path });
    expect(calls[0].body).toEqual(body);
  });

  test("a refusal becomes an ApiError whose detail the page can show", async () => {
    mockApi({ "POST /design/write": { status: 422, body: { detail: "Heads must divide the width." } } });
    const error = await design.write(spec, cur, null).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(detailOf(error)).toBe("Heads must divide the width.");
  });

  describe("guessCheck", () => {
    const check = { verdict: "right" };
    const job = (over: object = {}) => ({ id: "gc-1", kind: "guess", status: "done", created_at: 0, progress: "", error: "", ...over });

    test("posts the two versions and returns the job's result", async () => {
      const calls = mockApi({ "POST /design/guess-check": { body: job({ result: check }) } });
      await expect(design.guessCheck(1, 2, signal())).resolves.toEqual(check);
      expect(calls[0].body).toEqual({ old_version: 1, new_version: 2 });
    });

    test("a job with no result is null", async () => {
      mockApi({ "POST /design/guess-check": { body: job({ result: null }) } });
      await expect(design.guessCheck(1, 2, signal())).resolves.toBeNull();
      mockApi({ "POST /design/guess-check": { body: job() } });
      await expect(design.guessCheck(1, 2, signal())).resolves.toBeNull();
    });

    test("follows a running job to the result its /events message brings", async () => {
      const calls = mockApi({
        "POST /design/guess-check": { body: job({ status: "running" }) },
        "GET /jobs/gc-1": { body: job({ status: "running" }) },
      });
      const waiting = design.guessCheck(1, 2, signal());
      await vi.waitFor(() => expect(calls.some((c) => c.path === "/jobs/gc-1")).toBe(true));
      emit("job", job({ result: check }));
      await expect(waiting).resolves.toEqual(check);
    });
  });
});

describe("queueJobs, pretraining and versions", () => {
  test("textSets and the pretraining preview are GETs", async () => {
    const calls = mockApi({ "GET /pretraining/text_sets": { body: ["a"] }, "GET /pretraining/preview": { body: { tokens: 5 } } });
    await expect(queueJobs.textSets()).resolves.toEqual(["a"]);
    await expect(pretraining.preview()).resolves.toEqual({ tokens: 5 });
    expect(calls.map((c) => `${c.method} ${c.path}`)).toEqual(["GET /pretraining/text_sets", "GET /pretraining/preview"]);
  });

  test("enqueue sends only the job's own fields, with empty params when it has none", async () => {
    const calls = mockApi({ "POST /queue/jobs": { body: {} } });
    await queueJobs.enqueue({ kind: "pretrain", label: "Version 2", note: "n", params: { version: 2 }, eta_s: 30, extra: "dropped" } as never);
    await queueJobs.enqueue({ kind: "pretrain", label: "Version 3", note: "", eta_s: null } as never);
    expect(calls[0].body).toEqual({ kind: "pretrain", label: "Version 2", note: "n", params: { version: 2 }, eta_s: 30 });
    expect(calls[1].body).toEqual({ kind: "pretrain", label: "Version 3", note: "", params: {}, eta_s: null });
  });

  test("list and current are GETs", async () => {
    const calls = mockApi({ "GET /versions": { body: [] }, "GET /versions/current": { body: { number: 1 } } });
    await versions.list();
    await expect(versions.current()).resolves.toEqual({ number: 1 });
    expect(calls.map((c) => c.path)).toEqual(["/versions", "/versions/current"]);
  });

  describe("save", () => {
    test("posts the spec and the change text", async () => {
      const calls = mockApi({ "POST /versions": { body: { number: 2 } } });
      await versions.save({ n_layer: 2 }, "wider");
      expect(calls[0].body).toEqual({ spec: { n_layer: 2 }, change: "wider" });
    });

    test("carries the guess when one was made", async () => {
      const calls = mockApi({ "POST /versions": { body: {} } });
      await versions.save({}, "wider", { error: "lower" });
      expect(calls[0].body).toEqual({ spec: {}, change: "wider", prediction: { error: "lower" } });
    });

    test("leaves the guess out when it was skipped or empty", async () => {
      const calls = mockApi({ "POST /versions": { body: {} } });
      await versions.save({}, "x", {});
      await versions.save({}, "x", undefined);
      expect(calls.map((c) => Object.keys(c.body as object))).toEqual([["spec", "change"], ["spec", "change"]]);
    });
  });

  describe("importPretrained", () => {
    test("sends the model name as a form and returns the new version", async () => {
      let form: FormData | undefined;
      mockApi({
        "POST /post-training/imports": async (_, request) => {
          form = await request.formData();
          return { body: { number: 3, name: "version 3" } };
        },
      });
      await expect(versions.importPretrained("gpt2")).resolves.toEqual({ number: 3, name: "version 3" });
      expect(form?.get("name")).toBe("gpt2");
    });

    test("a refusal is an ApiError with the response text", async () => {
      mockApi({ "POST /post-training/imports": { status: 409, body: { detail: "Already downloading." } } });
      const error = await versions.importPretrained("gpt2").catch((e: unknown) => e);
      expect(error).toBeInstanceOf(ApiError);
      expect(error).toMatchObject({ status: 409, path: "/post-training/imports", body: '{"detail":"Already downloading."}' });
      expect(detailOf(error)).toBe("Already downloading.");
    });
  });
});

describe("where the size goes", () => {
  const now = { table: 40, position: 10, blocks: 40, head: 10 };
  const next = { table: 40, position: 10, blocks: 140, head: 10 };

  test("both bars are drawn to the larger design's scale", () => {
    const [a, b] = paramBars(now, next);
    expect(a).toEqual([20, 5, 20, 5]);
    expect(b.reduce((x, y) => x + y, 0)).toBeCloseTo(100);
  });

  test("a missing part counts as zero and an empty design does not divide by zero", () => {
    expect(paramBars({}, {})).toEqual([[0, 0, 0, 0], [0, 0, 0, 0]]);
  });

  test("shares name each part in bar order", () => {
    expect(paramShares(now).map((s) => s.text)).toEqual(["word table 40%", "position 10%", "blocks 40%", "word scores 10%"]);
  });

  test("the note shows only when the word table is half the model or more", () => {
    expect(paramNote(now)).toBe("");
    expect(paramNote({ table: 60, position: 0, blocks: 30, head: 10 })).toBe(BUILDER.params_table_heavy);
    expect(paramNote({ table: 50, position: 0, blocks: 50, head: 0 })).toBe(BUILDER.params_table_heavy);
  });
});

describe("piecesFor", () => {
  const chip = (label: string, known = true) => ({ label, known });
  const lists = {
    sentence: "Hi there",
    lists: [
      { id: "letters", vocab_size: 256, count: 8, chips: [chip("H"), chip("i")] },
      { id: "small", vocab_size: 500, count: 1234, chips: [chip("Hi"), chip("there")] },
      null,
      { id: "words", vocab_size: 900, count: 2, chips: [chip("Hi"), chip("there", false)] },
    ],
  } as unknown as Parameters<typeof piecesFor>[1];

  test("while the lists are being read it says so and shows dashes", () => {
    expect(piecesFor(1, undefined)).toEqual({ chips: [], note: BUILDER.pieces_reading, line: "– · – · – · – pieces" });
  });

  test("chips and the count line for a list, with thousands separators", () => {
    const view = piecesFor(1, lists);
    expect(view.chips).toEqual([{ label: "Hi", state: "pieces" }, { label: "there", state: "pieces" }]);
    expect(view.line).toBe("8 · 1,234 · – · 2 pieces");
    expect(view.note).toBeNull();
  });

  test("words the whole-word list lacks are counted as lost", () => {
    expect(piecesFor(3, lists).line).toBe("8 · 1,234 · – · 2 pieces, 1 word lost to “unknown”");
    expect(piecesFor(3, lists).chips[1]).toEqual({ label: "there", state: "lost" });
  });

  test("a list that is not available says why", () => {
    expect(piecesFor(2, lists).note).toBe(BUILDER.pieces_gpt2_missing);
    const own = { ...lists!, lists: [lists!.lists[0], null, null, null] } as typeof lists;
    expect(piecesFor(1, own).note).toBe(BUILDER.pieces_needs_text);
  });
});

describe("answerRows", () => {
  test("nothing trained: no rows, so no prompt or button", () => {
    expect(answerRows(2, new Set(), true)).toEqual([]);
  });

  test("the untrained next version of an unsaved draft says to save and train it", () => {
    expect(answerRows(2, new Set([2]), true)).toEqual([
      { number: 2, why: "" },
      { number: 3, why: BUILDER.compare_later },
    ]);
  });

  test("a saved version that has not trained says so", () => {
    expect(answerRows(2, new Set([2]), false)[1]).toEqual({ number: 3, why: BUILDER.not_trained });
    expect(answerRows(2, new Set([3]), true)[0]).toEqual({ number: 2, why: BUILDER.not_trained });
  });

  test("version 0 never gets a row", () => {
    expect(answerRows(0, new Set([1]), true)).toEqual([{ number: 1, why: "" }]);
  });
});
