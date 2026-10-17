import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test } from "vitest";
import HeadZoomPage from "@/app/head-zoom/page";
import { renderPage } from "@/test/render";
import { mockInspect, ids } from "./mock-inspect";
import Page from "./page";

// Head lab against a mocked inspection API (the component tier of tests/heads.spec.ts; Inside a head is in app/head-zoom/page.test.tsx).

describe.each([
  ["head-lab", Page],
  ["head-zoom", HeadZoomPage],
])("%s", (path, Component) => {
  test("a version link inspects that version without asking for the best", async () => {
    const { bodies, calls } = mockInspect();
    renderPage(<Component />, { route: `/${path}?version=7` });
    await waitFor(() => expect(bodies("POST", "/inspect/forward").length).toBeGreaterThan(0));
    expect(bodies("POST", "/inspect/forward")[0]).toMatchObject({ version: 7 });
    expect(calls.some((call) => call.path === "/versions/best")).toBe(false);
  });

  test("a missing requested version does not inspect a different one", async () => {
    const { calls } = mockInspect();
    renderPage(<Component />, { route: `/${path}?version=999` });
    expect(await screen.findByText("Version 999 is not available.")).toBeVisible();
    expect(calls.some((call) => call.path === "/inspect/forward")).toBe(false);
    expect(calls.some((call) => call.path === "/versions/best")).toBe(false);
  });
});

describe("Head lab", () => {
  test("shows the heads, switches one off, and ranks them", async () => {
    const { bodies } = mockInspect();
    const user = userEvent.setup();
    renderPage(<Page />, { route: "/head-lab" });
    expect(await screen.findByRole("heading", { level: 1, name: "Head lab" })).toBeVisible();
    expect(screen.getByText("Experiments on version 3 · checkpoint best.npz.")).toBeVisible();
    expect(screen.getByText("6 HEADS · CLICK ONE")).toBeVisible();
    expect(bodies("POST", "/inspect/forward")).toEqual([{ version: 3, ids, hooks: true }]);

    await user.click(screen.getByRole("button", { name: "Block 2 head 3" }));
    expect(screen.getByRole("heading", { name: "Block 2 · head 3" })).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Show the three tests" }));
    expect(screen.getByRole("group", { name: "The three pattern tests" })).toHaveTextContent("Previous word");
    // next.config's trailing slash is not applied here, so the link has no slash after the page name
    expect(screen.getByRole("link", { name: /Step through one head/ })).toHaveAttribute("href", "/head-zoom?version=3&block=1&head=2");

    await user.click(screen.getByRole("button", { name: "Switch it off" }));
    expect((await screen.findAllByText("switched off"))[0]).toBeVisible();
    expect(await screen.findByText(/2\.00 → 2\.50 bits\/char/)).toBeVisible();
    expect(await screen.findByText("on-reply")).toBeVisible();
    expect(await screen.findByText("off-reply")).toBeVisible();
    expect(bodies("POST", "/inspect/heads-off")).toEqual([{ version: 3, off: [[1, 2]], mode: "mean" }]);

    await user.click(screen.getByRole("radio", { name: "Zero (harsher)" }));
    await waitFor(() => expect(bodies("POST", "/inspect/heads-off")).toHaveLength(2));
    expect(bodies("POST", "/inspect/heads-off")[1]).toEqual({ version: 3, off: [[1, 2]], mode: "zero" });

    await user.click(screen.getByRole("button", { name: "Rank heads by importance" }));
    expect(await screen.findByRole("button", { name: "Hide importance" })).toBeVisible();
    expect(screen.getByText("Block 1 · head 1 error +0.10 bits/char")).toBeVisible();
    expect(bodies("POST", "/inspect/rank-heads")).toEqual([{ version: 3, mode: "zero" }]);

    await user.click(screen.getByRole("button", { name: "Switch all back on" }));
    expect(screen.getByText(/Switch a head off to see how the answer/)).toBeVisible();
  });

  test("without a trained checkpoint it says so", async () => {
    mockInspect({ trained: false });
    renderPage(<Page />, { route: "/head-lab" });
    expect(await screen.findByText(/No trained checkpoint yet/)).toBeVisible();
  });
});
