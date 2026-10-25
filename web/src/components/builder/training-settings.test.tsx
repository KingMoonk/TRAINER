import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, test, vi } from "vitest";
import type { Choices, Registry } from "@/lib/builder";
import { TrainingSettings } from "./training-settings";

const registry = {
  precisions: [
    { short: "fp32", name: "float32", blurb: "Full precision." },
    { short: "bf16", name: "bfloat16 mixed", blurb: "Half the memory." },
  ],
} as Registry;
const saved = { batch_size: 4, accumulation: 1, learning_rate: 0.001, weight_decay: 0.01, warmup: 20, dropout: 0.1, eval_every: 25, seed: 42, precision: "fp32" } as Choices;

describe("training settings", () => {
  test("show every saved value", () => {
    render(<TrainingSettings registry={registry} cur={saved} draft={saved} onChange={vi.fn()} />);
    const value = (label: string) => (screen.getByLabelText(label) as HTMLInputElement).value;
    expect(value("Batch size")).toBe("4");
    expect(value("Gradient accumulation")).toBe("1");
    expect(value("Learning rate")).toBe("0.001");
    expect(value("Weight decay")).toBe("0.01");
    expect(value("Warm-up steps")).toBe("20");
    expect(value("Dropout")).toBe("0.1");
    expect(value("Evaluate every (steps)")).toBe("25");
    expect(value("Seed")).toBe("42");
    expect(screen.getByRole("combobox", { name: "Precision" })).toHaveTextContent("float32");
  });

  test("edit the draft with a valid value", () => {
    const onChange = vi.fn();
    render(<TrainingSettings registry={registry} cur={saved} draft={saved} onChange={onChange} />);
    fireEvent.change(screen.getByLabelText("Batch size"), { target: { value: "32" } });
    fireEvent.change(screen.getByLabelText("Learning rate"), { target: { value: "0.0005" } });
    fireEvent.change(screen.getByLabelText("Dropout"), { target: { value: "0.2" } });
    expect(onChange.mock.calls).toEqual([[{ batch_size: 32 }], [{ learning_rate: 0.0005 }], [{ dropout: 0.2 }]]);
    expect(screen.queryByRole("alert")).toBeNull();
  });

  test("refuse a bad value with a sentence and leave the draft alone", () => {
    const onChange = vi.fn();
    render(<TrainingSettings registry={registry} cur={saved} draft={saved} onChange={onChange} />);
    fireEvent.change(screen.getByLabelText("Batch size"), { target: { value: "500" } });
    expect(screen.getByRole("alert")).toHaveTextContent("Batch size must be between 1 and 128.");
    expect(screen.getByLabelText("Batch size")).toHaveAttribute("aria-invalid", "true");
    fireEvent.change(screen.getByLabelText("Dropout"), { target: { value: "abc" } });
    expect(screen.getAllByRole("alert").map((a) => a.textContent)).toContain("Dropout must be a number.");
    expect(onChange).not.toHaveBeenCalled();
    fireEvent.change(screen.getByLabelText("Batch size"), { target: { value: "8" } });
    expect(onChange).toHaveBeenCalledWith({ batch_size: 8 });
  });

  test("name the saved value when the draft differs", () => {
    render(<TrainingSettings registry={registry} cur={saved} draft={{ ...saved, seed: 7, precision: "bf16" }} onChange={vi.fn()} />);
    expect(screen.getByText("Saved value: 42")).toBeInTheDocument();
    expect(screen.getByText("Saved value: float32")).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "Precision" })).toHaveTextContent("bfloat16 mixed");
  });

  test("follow a draft that changes from outside, such as a discard", () => {
    const { rerender } = render(<TrainingSettings registry={registry} cur={saved} draft={{ ...saved, batch_size: 8 }} onChange={vi.fn()} />);
    expect(screen.getByLabelText("Batch size")).toHaveValue("8");
    rerender(<TrainingSettings registry={registry} cur={saved} draft={saved} onChange={vi.fn()} />);
    expect(screen.getByLabelText("Batch size")).toHaveValue("4");
  });

  test("disable warm-up when the schedule has none", () => {
    render(<TrainingSettings registry={registry} cur={{ ...saved, warmup: null }} draft={{ ...saved, warmup: null }} onChange={vi.fn()} />);
    expect(screen.getByLabelText("Warm-up steps")).toBeDisabled();
    expect(screen.getByText("This design's schedule has no warm-up.")).toBeInTheDocument();
  });
});
