import { screen, within } from "@testing-library/react";
import { describe, expect, test } from "vitest";
import { renderPage } from "@/test/render";
import "@/test/navigation";
import { SectionTabs } from "./section-tabs";

describe("SectionTabs", () => {
  test("lists the Train pages in order and leaves the hidden ones out", () => {
    renderPage(<SectionTabs />, { route: "/queue" });
    const nav = screen.getByRole("navigation", { name: "Train pages" });
    const links = within(nav).getAllByRole("link");
    expect(links.map((a) => a.textContent)).toEqual(["Auto Run", "Pretraining", "Post-training", "Queue"]);
    expect(within(nav).getByRole("link", { name: "Queue" })).toHaveAttribute("aria-current", "page");
    for (const hidden of ["New run", "Runs", "Training run", "Run"]) expect(within(nav).queryByRole("link", { name: hidden })).toBeNull();
  });

  test("marks the current tab with aria-current='page'", () => {
    renderPage(<SectionTabs />, { route: "/post-training" });
    const nav = screen.getByRole("navigation", { name: "Train pages" });
    expect(within(nav).getByRole("link", { name: "Post-training" })).toHaveAttribute("aria-current", "page");
    expect(within(nav).getByRole("link", { name: "Queue" })).not.toHaveAttribute("aria-current");
  });

  test.each(["/runs/new", "/runs", "/training-run", "/runs/view"])("on the hidden page %s, marks the parent Queue tab", (route) => {
    renderPage(<SectionTabs />, { route });
    const nav = screen.getByRole("navigation", { name: "Train pages" });
    expect(within(nav).getByRole("link", { name: "Queue" })).toHaveAttribute("aria-current", "page");
    expect(within(nav).getAllByRole("link")).toHaveLength(4);
  });

  test("lists the Machines tabs on /cluster/machines", () => {
    renderPage(<SectionTabs />, { route: "/cluster/machines" });
    const nav = screen.getByRole("navigation", { name: "Machines pages" });
    expect(within(nav).getAllByRole("link").map((a) => a.textContent)).toEqual(["Machines", "Cluster training"]);
    expect(within(nav).getByRole("link", { name: "Machines" })).toHaveAttribute("aria-current", "page");
  });

  test("lists the Chat tabs on /chat", () => {
    renderPage(<SectionTabs />, { route: "/chat" });
    const nav = screen.getByRole("navigation", { name: "Chat pages" });
    expect(within(nav).getAllByRole("link").map((a) => a.textContent)).toEqual(["Chat", "Benchmark", "Export"]);
    expect(within(nav).getByRole("link", { name: "Chat" })).toHaveAttribute("aria-current", "page");
  });

  test.each(["/builder", "/quick-test", "/compare", "/your-data", "/collect", "/storage"])("renders no tab bar on %s: its area has one visible page", (route) => {
    renderPage(<SectionTabs />, { route });
    expect(screen.queryByTestId("section-tabs")).toBeNull();
  });

  test("renders nothing for routes outside AREAS (/settings)", () => {
    renderPage(<SectionTabs />, { route: "/settings" });
    expect(screen.queryByTestId("section-tabs")).toBeNull();
  });
});
