import { describe, expect, test } from "vitest";
import { initialRules, planText, rulesParams, rulesProblem, withPivot, type RulesForm } from "./rules";

const rules = (over: Partial<RulesForm> = {}): RulesForm => ({ ...initialRules(), ...over });

describe("the stall rules", () => {
  test("are off while the window is blank, and send nothing", () => {
    expect(rulesProblem(rules(), "", "10")).toBe("");
    expect(rulesParams(rules())).toBeUndefined();
    expect(planText(rules(), "", "10")).toBe("");
  });

  test("a window alone sends the plateau and no pivots or stages", () => {
    expect(rulesParams(rules({ plateauChecks: "3", plateauGain: "2" }))).toEqual({ plateau_checks: 3, plateau_min_improvement: 0.02 });
  });

  test("pivots go in the order they were ticked, and stages comes with next_dataset only", () => {
    let pivots = withPivot([], "next_dataset", true);
    pivots = withPivot(pivots, "halve_lr", true);
    expect(pivots).toEqual(["next_dataset", "halve_lr"]);
    expect(withPivot(pivots, "next_dataset", true)).toEqual(pivots);
    expect(rulesParams(rules({ plateauChecks: "2", pivots, replay: "30" }))).toEqual({ plateau_checks: 2, plateau_min_improvement: 0.01, pivots: ["next_dataset", "halve_lr"], stages: { replay: 0.3 } });
    expect(rulesParams(rules({ plateauChecks: "2", pivots: ["halve_lr"] }))).not.toHaveProperty("stages");
    expect(withPivot(pivots, "next_dataset", false)).toEqual(["halve_lr"]);
  });

  test("say why they cannot start, as the API would refuse", () => {
    expect(rulesProblem(rules({ pivots: ["halve_lr"] }), "", "10")).toMatch(/Set when the run stalls/);
    expect(rulesProblem(rules({ plateauChecks: "1" }), "", "10")).toMatch(/2 or more/);
    expect(rulesProblem(rules({ plateauChecks: "2.5" }), "", "10")).toMatch(/2 or more/);
    expect(rulesProblem(rules({ plateauChecks: "10" }), "", "10")).toMatch(/fewer than the number of rounds/);
    expect(rulesProblem(rules({ plateauChecks: "3", plateauGain: "0" }), "", "10")).toMatch(/above 0 and below 100/);
    expect(rulesProblem(rules({ plateauChecks: "3", plateauGain: "100" }), "", "10")).toMatch(/above 0 and below 100/);
    expect(rulesProblem(rules({ plateauChecks: "3", pivots: ["next_dataset"], replay: "100" }), "", "10")).toMatch(/from 0 to under 100/);
    expect(rulesProblem(rules({ plateauChecks: "3", pivots: ["drop_lessons"] }), "", "10")).toMatch(/needs a teacher that polishes/);
    expect(rulesProblem(rules({ plateauChecks: "3", pivots: ["drop_lessons"] }), "2", "10")).toBe("");
    expect(rulesProblem(rules({ plateauChecks: "3", pivots: ["next_dataset"], replay: "0" }), "", "10")).toBe("");
  });
});

describe("the plan text", () => {
  test("says the run ends by itself when only a window is set", () => {
    expect(planText(rules({ plateauChecks: "3" }), "", "10")).toBe("The run moves on by itself when the validation loss improves by less than 1% over 3 rounds. Then it ends, keeping its best weights.");
  });

  test("lists the changes in order, with the replay share, and says the run ends when none is left", () => {
    const text = planText(rules({ plateauChecks: "4", plateauGain: "2.5", pivots: ["halve_lr", "next_dataset"], replay: "25" }), "", "10");
    expect(text).toContain("improves by less than 2.5% over 4 rounds");
    expect(text).toContain("goes back to its best weights and first halves the learning rate, then moves on to the next unseen text, replaying 25% of the text it already saw.");
    expect(text).toContain("When nothing is left to try it ends");
  });

  test("a single change has no first or then", () => {
    expect(planText(rules({ plateauChecks: "3", pivots: ["halve_lr"] }), "", "10")).toContain("best weights and halves the learning rate.");
  });

  test("is empty while the rules cannot start", () => {
    expect(planText(rules({ plateauChecks: "3", pivots: ["drop_lessons"] }), "", "10")).toBe("");
  });
});
