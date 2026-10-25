import { screen, within } from "@testing-library/react";
import { describe, expect, test } from "vitest";
import { TrafficProvider } from "@/components/traffic/traffic-provider";
import { renderPage } from "@/test/render";
import "@/test/navigation";
import { Sidebar } from "./sidebar";

describe("Sidebar", () => {
  test("lists the five areas in order, with the open one marked", () => {
    renderPage(<TrafficProvider><Sidebar /></TrafficProvider>, { route: "/runs/new" });
    const studio = screen.getByRole("navigation", { name: "Studio" });
    const areas = within(studio).getAllByRole("link");
    expect(areas.map((a) => a.textContent)).toEqual(["Models", "Data", "Runs", "Eval", "Machines"]);
    expect(within(studio).getByRole("link", { name: "Runs" })).toHaveAttribute("aria-current", "page");
  });

  test("pins Settings and traffic console at the foot of the nav", () => {
    renderPage(<TrafficProvider><Sidebar /></TrafficProvider>, { route: "/builder" });
    const gears = screen.getAllByRole("link", { name: "Settings" });
    expect(gears.length).toBeGreaterThan(0);
    for (const gear of gears) expect(gear).toHaveAttribute("href", "/settings");
    const consoles = screen.getAllByRole("button", { name: /traffic/i });
    expect(consoles.length).toBeGreaterThan(0);
  });

  test("contains only the five area links in the studio navigation", () => {
    renderPage(<TrafficProvider><Sidebar /></TrafficProvider>, { route: "/builder" });
    const studio = screen.getByRole("navigation", { name: "Studio" });
    const links = within(studio).getAllByRole("link");
    expect(links).toHaveLength(5);
    expect(screen.queryByRole("group", { name: "Pages in this area" })).toBeNull();
  });
});
