import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test } from "vitest";
import { Disclosure } from "./disclosure";

const details = () => screen.getByTestId("adv") as HTMLDetailsElement;

describe("Disclosure", () => {
  test("is closed by default, shows its title and hint, and keeps its children mounted", async () => {
    render(<Disclosure title="Advanced" hint="Rarely needed" testId="adv"><input aria-label="Inside" defaultValue="" /></Disclosure>);
    expect(details().open).toBe(false);
    expect(screen.getByText("Advanced")).toBeInTheDocument();
    expect(screen.getByText("Rarely needed")).toBeInTheDocument();
    await userEvent.click(screen.getByText("Advanced"));
    expect(details().open).toBe(true);
    await userEvent.type(screen.getByLabelText("Inside"), "kept");
    await userEvent.click(screen.getByText("Advanced"));
    expect(details().open).toBe(false);
    expect(screen.getByLabelText("Inside", { selector: "input" })).toHaveValue("kept");
  });

  test("defaultOpen starts it open", () => {
    render(<Disclosure title="Advanced" hint="h" defaultOpen testId="adv"><p>Body</p></Disclosure>);
    expect(details().open).toBe(true);
  });

  test("reveal opens it when it turns truthy, and the hint is optional", () => {
    const { rerender } = render(<Disclosure title="Advanced" reveal={false} testId="adv"><p>Body</p></Disclosure>);
    expect(details().open).toBe(false);
    rerender(<Disclosure title="Advanced" reveal testId="adv"><p>Body</p></Disclosure>);
    expect(details().open).toBe(true);
  });
});
