import { screen, within } from "@testing-library/react";
import { describe, expect, test } from "vitest";
import { renderPage } from "@/test/render";
import { Sidebar } from "./sidebar";

describe("Sidebar", () => {
  test("lists the five areas in order, with the open one marked", () => {
    renderPage(<Sidebar />, { route: "/runs/new" });
    const studio = screen.getByRole("navigation", { name: "Studio" });
    const areas = within(studio).getAllByRole("link").filter((a) => a.hasAttribute("href") && a.getAttribute("href") !== "/settings");
    expect(areas.map((a) => a.textContent)).toEqual(["Models", "Data", "Runs", "Eval", "Machines"]);
    expect(within(studio).getByRole("link", { name: "Runs" })).toHaveAttribute("aria-current", "page");
  });

  test("lists the Runs pages in the page list, and leaves run_view out", () => {
    renderPage(<Sidebar />, { route: "/runs/view" });
    const group = screen.getByRole("group", { name: "Pages in this area" });
    expect(within(group).getAllByRole("link").map((a) => a.textContent)).toEqual([
      "Runs", "New run", "Training run", "Post-training", "Queue", "Passive training", "Training",
    ]);
  });

  test("pins a Settings gear at the foot of the nav", () => {
    renderPage(<Sidebar />, { route: "/builder" });
    const gears = screen.getAllByRole("link", { name: "Settings" });
    expect(gears.length).toBeGreaterThan(0);
    for (const gear of gears) expect(gear).toHaveAttribute("href", "/settings");
  });
});
