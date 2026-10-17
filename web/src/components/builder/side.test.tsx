import { render, screen, within } from "@testing-library/react";
import { describe, expect, test } from "vitest";
import { BUILDER } from "@/lib/copy/builder";
import type { Sizes } from "@/lib/builder";
import { Costs } from "./side";

const sizes = (parts: Sizes["parts"]): Sizes => ({ params: 100, table: 40, time: 1, memory: 1600, activations: 1, parts, fits: true });

describe("Costs: where the size goes", () => {
  test("shows both bars, a legend of the draft's shares and the word-table note when it dominates", () => {
    const now = sizes({ table: 40, position: 10, blocks: 40, head: 10 });
    const next = sizes({ table: 120, position: 10, blocks: 40, head: 10 });
    render(<Costs now={now} next={next} warm={null} lines={[]} />);
    const bar = screen.getByTestId("param-bar");
    expect(within(bar).getByText(BUILDER.params_where)).toBeInTheDocument();
    expect(within(bar).getByText("word table 67%")).toBeInTheDocument();
    expect(within(bar).getByText(BUILDER.params_table_heavy)).toBeInTheDocument();
    expect(screen.getByTestId("param-now").children).toHaveLength(4);
    expect((screen.getByTestId("param-next").children[0] as HTMLElement).style.width).toBe("66.66666666666666%");
  });

  test("has no note when the blocks carry the size, and no bar before the sizes arrive", () => {
    const parts = { table: 10, position: 10, blocks: 70, head: 10 };
    const { rerender } = render(<Costs now={sizes(parts)} next={sizes(parts)} warm={null} lines={[]} />);
    expect(screen.queryByText(BUILDER.params_table_heavy)).toBeNull();
    rerender(<Costs now={undefined} next={undefined} warm={null} lines={[]} />);
    expect(screen.queryByTestId("param-bar")).toBeNull();
  });
});
