import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, test, vi } from "vitest";
import { BUILDER } from "@/lib/copy/builder";
import userEvent from "@testing-library/user-event";
import type { Choices, Registry, Sizes } from "@/lib/builder";
import { BuilderDisclosure, Controls, Costs } from "./side";

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


const registry: Registry = { heads: [1, 3, 8], widths: ["small", "medium", "large"], words: [], slots: [], block_slots: [], part_slots: [] };
const choices = { H: 3, W: 1, V: 1, d: 64, T: 256, U: 1024, token_limit_millions: 0, M: { mixer: "attention", position: "learned" } } as Choices;

describe("model controls", () => {
  test("edits hidden size, context, tokenizer size and the optional token cap", () => {
    const onModelSetting = vi.fn();
    render(<Controls registry={registry} cur={choices} draft={choices} knobs={{ H: true, W: true }} onKnob={vi.fn()} onModelSetting={onModelSetting} />);
    fireEvent.change(screen.getByLabelText("Hidden size"), { target: { value: "128" } });
    fireEvent.change(screen.getByLabelText("Token context"), { target: { value: "512" } });
    fireEvent.change(screen.getByLabelText("Tokenizer size"), { target: { value: "4096" } });
    fireEvent.change(screen.getByLabelText("Training token limit (millions)"), { target: { value: "12" } });
    expect(onModelSetting.mock.calls).toEqual([["d", 128], ["T", 512], ["U", 4096], ["token_limit_millions", 12]]);
  });

  test("maps nonconsecutive head choices to actual counts and width choices to indices", () => {
    const onKnob = vi.fn();
    render(<Controls registry={registry} cur={{ ...choices, H: 1, W: 0 }} draft={choices} knobs={{ H: true, W: true }} onKnob={onKnob} />);
    const heads = screen.getByRole("slider", { name: BUILDER.look_back_heads });
    expect(heads).toHaveAttribute("aria-valuetext", "3");
    expect(screen.getByTestId("width-value")).toHaveTextContent("medium");
    expect(screen.getByText("Saved value: 1")).toBeInTheDocument();
    fireEvent.change(heads, { target: { value: "2" } });
    expect(onKnob).toHaveBeenLastCalledWith("H", 8);
    fireEvent.change(screen.getByRole("slider", { name: BUILDER.thinking_room }), { target: { value: "2" } });
    expect(onKnob).toHaveBeenLastCalledWith("W", 2);
    fireEvent.click(within(screen.getByTestId("heads")).getByRole("button", { name: "1" }));
    expect(onKnob).toHaveBeenLastCalledWith("H", 1);
  });

  test("supports keyboard step buttons and disables steps at registry boundaries", async () => {
    const user = userEvent.setup();
    const onKnob = vi.fn();
    const { rerender } = render(<Controls registry={registry} cur={choices} draft={{ ...choices, H: 1 }} knobs={{ H: true, W: true }} onKnob={onKnob} />);
    expect(screen.getByRole("button", { name: "Decrease Attention heads" })).toBeDisabled();
    screen.getByRole("button", { name: "Increase Attention heads" }).focus();
    await user.keyboard("{Enter}");
    expect(onKnob).toHaveBeenLastCalledWith("H", 3);
    rerender(<Controls registry={registry} cur={choices} draft={{ ...choices, H: 8, W: 2 }} knobs={{ H: true, W: true }} onKnob={onKnob} />);
    expect(screen.getByRole("button", { name: "Increase Attention heads" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Increase MLP width" })).toBeDisabled();
  });

  test("keeps unavailable knobs disabled across sliders, steps and choices", () => {
    const onKnob = vi.fn();
    const { rerender } = render(<Controls registry={registry} cur={choices} draft={choices} knobs={undefined} onKnob={onKnob} />);
    for (const slider of screen.getAllByRole("slider")) expect(slider).toBeDisabled();
    rerender(<Controls registry={registry} cur={choices} draft={choices} knobs={{ H: false, W: false }} onKnob={onKnob} />);
    for (const button of screen.getAllByRole("button")) {
      expect(button).toBeDisabled();
      fireEvent.click(button);
    }
    expect(onKnob).not.toHaveBeenCalled();
    expect(screen.getByText(BUILDER.no_heads)).toBeInTheDocument();
    expect(screen.getByText(BUILDER.no_room)).toBeInTheDocument();
  });

  test("does not substitute an invalid stored choice with another registry value", () => {
    const onKnob = vi.fn();
    render(<Controls registry={registry} cur={choices} draft={{ ...choices, H: 9, W: 8 }} knobs={{ H: true, W: true }} onKnob={onKnob} />);
    for (const slider of screen.getAllByRole("slider")) expect(slider).toBeDisabled();
    for (const button of screen.getAllByRole("button")) expect(button).toBeDisabled();
    expect(onKnob).not.toHaveBeenCalled();
  });
});

describe("optional Builder sections", () => {
  test("opens for a selected part and retains entered work across closing and reopening", () => {
    const content = <input aria-label="Notes" defaultValue="" />;
    const { rerender } = render(<BuilderDisclosure title="Explanation" hint="Explore a part">{content}</BuilderDisclosure>);
    const details = screen.getByText("Explanation").closest("details")!;
    expect(details.open).toBe(false);
    rerender(<BuilderDisclosure title="Explanation" hint="Explore a part" reveal={{ part: "mixer" }}>{content}</BuilderDisclosure>);
    expect(details.open).toBe(true);
    fireEvent.change(screen.getByLabelText("Notes"), { target: { value: "Keep my work" } });
    details.open = false;
    rerender(<BuilderDisclosure title="Explanation" hint="Explore a part" reveal={{ part: "channel" }}>{content}</BuilderDisclosure>);
    expect(details.open).toBe(true);
    expect(screen.getByLabelText("Notes")).toHaveValue("Keep my work");
  });
});
