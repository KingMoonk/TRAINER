import { screen, within } from "@testing-library/react";
import { describe, expect, test } from "vitest";
import { renderPage } from "@/test/render";
import "@/test/navigation";
import { SectionTabs } from "./section-tabs";

describe("SectionTabs", () => {
  test("lists the Runs pages in order and leaves run_view out", () => {
    renderPage(<SectionTabs />, { route: "/runs" });
    const nav = screen.getByRole("navigation", { name: "Runs pages" });
    const links = within(nav).getAllByRole("link");
    expect(links.map((a) => a.textContent)).toEqual([
      "All runs",
      "New run",
      "Queue",
      "Training run",
      "Cluster training",
      "Post-training",
    ]);
    expect(within(nav).getByRole("link", { name: "All runs" })).toHaveAttribute("aria-current", "page");
    expect(within(nav).queryByRole("link", { name: "Run" })).toBeNull();
  });

  test("marks the current tab with aria-current='page'", () => {
    renderPage(<SectionTabs />, { route: "/runs/new" });
    const nav = screen.getByRole("navigation", { name: "Runs pages" });
    expect(within(nav).getByRole("link", { name: "New run" })).toHaveAttribute("aria-current", "page");
    expect(within(nav).getByRole("link", { name: "All runs" })).not.toHaveAttribute("aria-current");
  });

  test("on a hidden route with a parent (run_view), marks the parent tab", () => {
    renderPage(<SectionTabs />, { route: "/runs/view" });
    const nav = screen.getByRole("navigation", { name: "Runs pages" });
    expect(within(nav).getByRole("link", { name: "All runs" })).toHaveAttribute("aria-current", "page");
    expect(within(nav).queryByRole("link", { name: "Run" })).toBeNull();
  });

  test("renders the Models tabs on /builder with Build your model marked", () => {
    renderPage(<SectionTabs />, { route: "/builder" });
    const nav = screen.getByRole("navigation", { name: "Models pages" });
    const links = within(nav).getAllByRole("link");
    expect(links.map((a) => a.textContent)).toEqual([
      "Build your model",
      "Quick test",
      "Compare runs",
    ]);
    expect(within(nav).getByRole("link", { name: "Build your model" })).toHaveAttribute("aria-current", "page");
  });

  test("renders the Data tabs on /your-data", () => {
    renderPage(<SectionTabs />, { route: "/your-data" });
    const nav = screen.getByRole("navigation", { name: "Data pages" });
    const links = within(nav).getAllByRole("link");
    expect(links.map((a) => a.textContent)).toEqual([
      "Your data",
      "Pretraining",
      "Collect data",
      "Storage",
    ]);
    expect(within(nav).getByRole("link", { name: "Your data" })).toHaveAttribute("aria-current", "page");
  });

  test("renders the Eval tabs on /chat", () => {
    renderPage(<SectionTabs />, { route: "/chat" });
    const nav = screen.getByRole("navigation", { name: "Eval pages" });
    const links = within(nav).getAllByRole("link");
    expect(links.map((a) => a.textContent)).toEqual(["Chat", "Benchmark"]);
    expect(within(nav).getByRole("link", { name: "Chat" })).toHaveAttribute("aria-current", "page");
  });

  test("renders nothing for a single-page area (Machines)", () => {
    renderPage(<SectionTabs />, { route: "/cluster/machines" });
    expect(screen.queryByTestId("section-tabs")).toBeNull();
  });

  test("renders nothing for routes outside AREAS (/settings)", () => {
    renderPage(<SectionTabs />, { route: "/settings" });
    expect(screen.queryByTestId("section-tabs")).toBeNull();
  });
});
