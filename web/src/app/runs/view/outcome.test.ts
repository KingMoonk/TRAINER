import { describe, expect, test } from "vitest";
import { round, run } from "@/app/cluster/training/test-data";
import { outcomeSeries } from "./outcome";

describe("run outcome versus work", () => {
  test("cumulates accepted tokens, including late work, on both aligned plots", () => {
    const view = outcomeSeries(run({ best_round: 1, history: [round(0, { accepted_tokens: 100, members: 2, contributed: 1 }),
      round(1, { accepted_tokens: 130, members: 3, contributed: 2, late_contributed: 1, val_loss: 3.1 })] }));
    expect(view.axis.label).toBe("Accepted target tokens");
    expect(view.loss[0].points).toEqual([{ x: 100, y: 4.5 }, { x: 230, y: 3.1 }]);
    expect(view.participation.map((series) => [series.id, series.points])).toEqual([
      ["invited", [{ x: 100, y: 2 }, { x: 230, y: 3 }]],
      ["contributed", [{ x: 100, y: 1 }, { x: 230, y: 2 }]],
      ["late", [{ x: 100, y: 0 }, { x: 230, y: 1 }]],
    ]);
    expect(view.marks).toContainEqual({ id: "best", x: 230, label: "Best", color: "var(--chart-2)" });
  });

  test("labels old or empty-token history as planned steps rather than inventing accepted work", () => {
    const view = outcomeSeries(run({ history: [round(0, { accepted_tokens: null }), round(1, { accepted_tokens: 0 })] }));
    expect(view.axis.label).toBe("Planned global steps");
    expect(view.loss[0].points.map((point) => point.x)).toEqual([100, 200]);
  });

  test("omits unknown historical late-work counts instead of plotting false zeroes", () => {
    const view = outcomeSeries(run({ history: [round(0, { late_contributed: null }), round(1, { late_contributed: 1 })] }));
    expect(view.participation.find((series) => series.id === "late")?.points).toEqual([{ x: 2000, y: 1 }]);
  });
});
