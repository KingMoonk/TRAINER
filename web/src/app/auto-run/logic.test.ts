import { describe, expect, test } from "vitest";
import { PHASES, boundTexts, buildPlan, budgetText, checkText, controls, cyclesText, growTexts, parseCycles, defaultVersion, errorText, keptRound, loopRounds, pickRun, progressLine, resultChecks, roundText, settingTexts, timelinePhases, valueText, when } from "./logic";
import { plan, run, summary } from "./test-data";

describe("the cycle limit", () => {
  test("is read from the plan, defaults to one and is written back into the controller", () => {
    expect(cyclesText(plan())).toBe("1");
    const built = buildPlan(plan(), {}, "", boundTexts(plan()), "3");
    expect(built).toMatchObject({ plan: { controller: { max_cycles: 3 } } });
  });

  test("must be a whole number from 1 to 5", () => {
    for (const bad of ["0", "6", "1.5", "", "x"]) {
      expect(parseCycles(bad)).toBeNull();
      expect(buildPlan(plan(), {}, "", boundTexts(plan()), bad)).toEqual({ error: "The most cycles must be a whole number from 1 to 5." });
    }
    expect(parseCycles("5")).toBe(5);
  });
});

describe("growing a design", () => {
  test("the limits are read from the plan and written back; consent stays as it was", () => {
    const base = plan({ controller: { ...plan().controller!, grow: { enabled: true, max_width: 96, max_layers: 3 } } });
    expect(growTexts(base)).toEqual({ width: "96", layers: "3" });
    expect(buildPlan(base, {}, "", boundTexts(base), "1", { width: "128", layers: "4" })).toMatchObject({ plan: { controller: { grow: { enabled: true, max_width: 128, max_layers: 4 } } } });
    expect(buildPlan(plan(), {}, "", boundTexts(plan()), "1", growTexts(plan()))).toMatchObject({ plan: { controller: { grow: { enabled: false, max_width: 256, max_layers: 6 } } } });
  });

  test("a limit that is not a whole number from 1 is refused", () => {
    for (const bad of [{ width: "0", layers: "2" }, { width: "64", layers: "2.5" }, { width: "", layers: "2" }]) {
      expect(buildPlan(plan(), {}, "", boundTexts(plan()), "1", bad)).toEqual({ error: "The widest design and the most layers must be whole numbers from 1." });
    }
  });
});

describe("buildPlan", () => {
  test("keeps the plan as it is when nothing was typed, and takes the budget from its text", () => {
    const built = buildPlan(plan(), {}, "6");
    expect(built).toEqual({ plan: plan({ budget: { hours: 6 } }) });
  });

  test("an empty budget is no limit", () => {
    expect(buildPlan(plan(), {}, " ")).toMatchObject({ plan: { budget: { hours: null } } });
  });

  test("puts typed settings in, and drops one that was emptied", () => {
    const base = plan();
    base.phases.find((p) => p.kind === "sft")!.settings = { steps: 50 };
    const built = buildPlan(base, { "sft.steps": "", "pretrain.hours": "1.5", "rlvr.group": "4" }, "");
    expect("plan" in built && built.plan.phases.find((p) => p.kind === "sft")!.settings).toEqual({});
    expect("plan" in built && built.plan.phases.find((p) => p.kind === "pretrain")!.settings).toEqual({ hours: 1.5 });
    expect("plan" in built && built.plan.phases.find((p) => p.kind === "rlvr")!.settings).toEqual({ group: 4 });
  });

  test("refuses a setting outside what the server takes, naming it", () => {
    expect(buildPlan(plan(), { "sft.steps": "0" }, "")).toEqual({ error: "Steps for Round 1 · Learn to answer must be a whole number from 1 to 5000, or empty." });
    expect(buildPlan(plan(), { "pretrain.hours": "abc" }, "")).toMatchObject({ error: expect.stringContaining("Hours to train") });
  });

  test("refuses a budget of nothing or less", () => {
    for (const bad of ["0", "-2", "x"]) expect(buildPlan(plan(), {}, bad)).toMatchObject({ error: expect.stringContaining("time budget") });
  });

  test("needs a name and at least one phase", () => {
    expect(buildPlan(plan({ name: "  " }), {}, "")).toMatchObject({ error: "Give the run a name." });
    const none = plan();
    none.phases.forEach((p) => (p.enabled = false));
    expect(buildPlan(none, {}, "")).toEqual({ error: "Turn on at least one phase." });
  });

  test("only settings the server reads are offered", () => {
    expect(PHASES.pretrain.settings.map((s) => s.key)).toEqual(["hours", "passes"]);
    expect(PHASES.rlvr.settings.map((s) => s.key)).toEqual(["steps", "prompts_per_step", "group", "temperature", "max_tokens", "kl"]);
    expect(PHASES.data.settings).toEqual([]);
  });
});

describe("the editor's starting text", () => {
  test("shows set numbers and an empty field for the rest", () => {
    const base = plan({ budget: { hours: null } });
    base.phases[1].settings = { hours: 2, other: "x" };
    expect(settingTexts(base)).toEqual({ "pretrain.hours": "2" });
    expect(budgetText(base)).toBe("");
    expect(budgetText(plan())).toBe("12");
  });
});

describe("controls", () => {
  test("a running run can be paused or stopped", () => {
    expect(controls(run("running", "pretrain", { pretrain: { status: "running" } }))).toMatchObject({ pause: true, stop: true, resume: false, decide: false, edit: false });
  });

  test("a pausing run can only be stopped", () => {
    expect(controls(run("pausing", "pretrain", { pretrain: { status: "running" } }))).toMatchObject({ pause: false, stop: true, resume: false, decide: false });
  });

  test("a run paused as planned continues; one paused on a failure asks what to do", () => {
    expect(controls(run("paused", "pretrain", { pretrain: { status: "passed" } }))).toMatchObject({ resume: true, decide: false, edit: true });
    expect(controls(run("paused", "sft", { sft: { status: "failed" } }))).toMatchObject({ resume: false, decide: true, edit: true });
    expect(controls(run("waiting_user", "data", { data: { status: "blocked" } }))).toMatchObject({ resume: false, decide: true, edit: false });
  });

  test("a draft is edited, a finished run has no controls", () => {
    expect(controls(run("draft", null))).toMatchObject({ edit: true, pause: false, stop: false });
    expect(controls(run("done", null))).toMatchObject({ edit: false, pause: false, stop: false, resume: false, decide: false });
  });
});

describe("keptRound", () => {
  const gate = { round: 1, ok: false, checks: [], checkpoint: "c/round1.npz" };

  test("a failed round that left a checkpoint is kept", () => {
    const view = run("paused", "sft", { sft: { status: "failed", checkpoint: "c/round1.npz", result: gate } });
    expect(keptRound(view)).toMatchObject({ round: 1, gate: { checkpoint: "c/round1.npz" } });
    expect(controls(view).kept?.round).toBe(1);
  });

  test("not kept without a checkpoint, for pretraining, or while it still runs", () => {
    expect(keptRound(run("paused", "sft", { sft: { status: "failed", result: { ...gate, checkpoint: undefined } } }))).toBeNull();
    expect(keptRound(run("paused", "pretrain", { pretrain: { status: "failed", checkpoint: "c.npz", result: gate } }))).toBeNull();
    expect(keptRound(run("running", "sft", { sft: { status: "failed", checkpoint: "c.npz", result: gate } }))).toBeNull();
  });
});

describe("the timeline", () => {
  test("lists the phases the plan includes, with their state", () => {
    const view = run("running", "pretrain", { pretrain: { status: "running" } });
    expect(timelinePhases(view).map((p) => [p.plan.kind, p.state.status])).toEqual([
      ["data", "pending"], ["pretrain", "running"], ["sft", "pending"], ["rlvr", "pending"], ["distill", "pending"], ["dpo", "pending"],
    ]);
  });

  test("reads a round's gate checks", () => {
    expect(resultChecks({ checks: [{ name: "Held-back loss", detail: "3.1 to 2.9", ok: true }] })).toHaveLength(1);
    expect(resultChecks(null)).toEqual([]);
    expect(resultChecks({ checks: "none" })).toEqual([]);
    expect([true, false, null].map(checkText)).toEqual(["passed", "failed", "not measured"]);
  });
});

describe("the RLHF loop's rounds", () => {
  test("reads the rounds a phase result lists, and nothing else", () => {
    expect(loopRounds(null)).toEqual([]);
    expect(loopRounds({ rounds: "none" })).toEqual([]);
    expect(loopRounds({ rounds: [{ round: 1, status: "kept", reason: "", judge_win_rate: null, card_better: [] }] })).toHaveLength(1);
  });

  test("says how a round ended, then the judge and the report card", () => {
    expect(roundText({ round: 3, status: "rolled_back", reason: "", judge_win_rate: 0.9, card_better: [] })).toBe("Round 3: rolled back · the judge preferred it in 90% of comparisons");
    expect(roundText({ round: 1, status: "kept", reason: "", judge_win_rate: null, card_better: ["multiple-choice accuracy", "chat win rate"] })).toBe(
      "Round 1: kept · report card better: multiple-choice accuracy, chat win rate",
    );
  });

  test("the phase takes the loop's own settings, and the editor keeps them", () => {
    expect(PHASES.rlhf.settings.map((s) => s.key)).toEqual(["max_rounds", "card_patience", "patience"]);
    const base = plan();
    base.phases.find((p) => p.kind === "rlhf")!.enabled = true;
    const built = buildPlan(base, { "rlhf.card_patience": "2" }, "");
    expect("plan" in built && built.plan.phases.find((p) => p.kind === "rlhf")!.settings).toEqual({ card_patience: 2 });
    expect(buildPlan(base, { "rlhf.card_patience": "0" }, "")).toMatchObject({ error: expect.stringContaining("Rounds without a better report card") });
  });
});

describe("progressLine", () => {
  const job = { id: "j", kind: "pretrain", label: "", status: "running", phase: "steady" };
  test("the step and the time left", () => {
    expect(progressLine({ ...job, step: 1200, total: 5000, eta_s: 600 })).toBe("step 1,200 of 5,000 · about 10 min left");
    expect(progressLine({ ...job, step: 3, total: 10 })).toBe("step 3 of 10");
  });
  test("or why it has not started", () => {
    expect(progressLine({ ...job, status: "waiting" })).toBe("Waiting in the queue");
    expect(progressLine({ ...job, status: "held" })).toBe("Held in the queue");
    expect(progressLine(job)).toBe("Preparing");
  });
});

describe("pickRun", () => {
  const runs = [summary(run("done", null)), { ...summary(run("paused", "sft")), id: "b" }, { ...summary(run("running", "dpo")), id: "c" }];
  test("opens the one asked for, else the newest that is not over", () => {
    expect(pickRun(runs, "c")).toBe("c");
    expect(pickRun(runs, null)).toBe("b");
    expect(pickRun([runs[0]], null)).toBeNull();
    expect(pickRun(undefined, null)).toBeNull();
  });
});

describe("small helpers", () => {
  test("defaultVersion prefers the newest trained one", () => {
    expect(defaultVersion([{ number: 1, best_checkpoint: "a" }, { number: 2, best_checkpoint: "b" }, { number: 3 }])).toBe(2);
    expect(defaultVersion([{ number: 1 }, { number: 2 }])).toBe(2);
    expect(defaultVersion([])).toBeNull();
  });

  test("errorText reads the sentence of a refusal", () => {
    expect(errorText({ body: JSON.stringify({ detail: "This run is not waiting for you." }) })).toBe("This run is not waiting for you.");
    expect(errorText(new Error("down"))).toBe("down");
  });

  test("when trims a time to the second", () => {
    expect(when("2026-10-27T10:20:30.123456+00:00")).toBe("2026-10-27 10:20:30");
    expect(when("2026-10-27T10:20:30Z")).toBe("2026-10-27 10:20:30");
  });
});

describe("the parameter controller's limits", () => {
  test("are shown as text and built back into the plan", () => {
    const texts = boundTexts(plan())!;
    expect(texts.steps).toEqual({ min: "10", max: "20000" });
    texts.steps.max = "8000";
    const built = buildPlan(plan(), {}, "12", texts);
    expect("plan" in built && built.plan.controller!.bounds.steps).toEqual({ min: 10, max: 8000 });
  });

  test("a plan without a controller has no limits to show", () => {
    expect(boundTexts(plan({ controller: null }))).toBeNull();
  });

  test("refuses limits that are not numbers, are reversed, or have no highest above 0", () => {
    for (const [key, min, max] of [["steps", "a", "5"], ["beta", "0.9", "0.1"], ["learning_rate", "0", "0"], ["kl", "", "1"]] as const) {
      const texts = boundTexts(plan())!;
      texts[key] = { min, max };
      expect(buildPlan(plan(), {}, "12", texts)).toMatchObject({ error: expect.stringContaining("The limits for") });
    }
  });

  test("names a value the way the page says it", () => {
    expect([valueText(0.00005), valueText(0.0001), valueText(3000), valueText(0.15), valueText(null)]).toEqual(["5e-5", "1e-4", "3,000", "0.15", ""]);
  });
});
