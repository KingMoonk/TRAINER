import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test } from "vitest";
import { mockInspect } from "@/app/head-lab/mock-inspect";
import { renderPage } from "@/test/render";
import Page from "./page";

// Inside a head against a mocked inspection API (the component tier of tests/heads.spec.ts; the version-link tests for both
// pages are in app/head-lab/page.test.tsx).

describe("Inside a head", () => {
  test("a link lands on its head, the steps change the grid, and the head's stats show", async () => {
    const { bodies } = mockInspect();
    const user = userEvent.setup();
    renderPage(<Page />, { route: "/head-zoom?block=1&head=2" });
    expect(await screen.findByRole("heading", { level: 1, name: "Inside a head" })).toBeVisible();
    expect(await screen.findByText(/Block 2 · head 3 · 5 tokens/)).toBeVisible();
    expect(screen.queryByText(/differ from the model/)).toBeNull(); // the derived softmax matches the model's pattern
    expect(screen.getByRole("gridcell", { name: /^k1 q1: -?\d\.\d$/ })).toBeVisible();

    await user.click(screen.getByRole("button", { name: /3\. Causal mask/ }));
    expect(screen.getByRole("gridcell", { name: "k5 q1: −∞" })).toBeVisible();
    await user.click(screen.getByRole("button", { name: /5\. Weighted sum of values/ }));
    expect(within(screen.getByRole("list", { name: "Biggest contributors" })).getAllByRole("listitem")).toHaveLength(5);
    expect(screen.getByRole("link", { name: /Where V and W_O meet/ })).toBeVisible();

    await user.click(screen.getByRole("gridcell", { name: /^k2 q4/ }));
    expect(screen.getByText("k2 “t7” · q4 “t7”")).toBeVisible();

    expect(screen.getAllByRole("term").some((term) => term.textContent === "Entropy")).toBe(true);
    expect(await screen.findByText("1.33 nats")).toBeVisible();
    expect((bodies("POST", "/inspect/head-stats")[0] as { pattern: number[][][][] }).pattern[0]).toHaveLength(3);

    await user.click(screen.getByRole("radio", { name: "Head 1" }));
    expect(await screen.findByText("1.11 nats")).toBeVisible();
    await waitFor(() => expect(window.location.search).toMatch(/block=1&head=0/));
  });

  test("a block or head the model does not have falls back to the first", async () => {
    mockInspect();
    renderPage(<Page />, { route: "/head-zoom?block=9&head=9" });
    expect(await screen.findByText(/Block 1 · head 1 · 5 tokens/)).toBeVisible();
  });
});
