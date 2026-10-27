import { screen, within } from "@testing-library/react";
import { describe, expect, test } from "vitest";
import { TrafficProvider } from "@/components/traffic/traffic-provider";
import { renderPage } from "@/test/render";
import "@/test/navigation";
import { Sidebar } from "./sidebar";

describe("Sidebar", () => {
  test("lists the five areas in order, with the open one marked", () => {
    renderPage(<TrafficProvider><Sidebar /></TrafficProvider>, { route: "/runs" });
    const studio = screen.getByRole("navigation", { name: "Studio" });
    const areas = within(studio).getAllByRole("link");
    expect(areas.map((a) => a.textContent)).toEqual(["Models", "Data", "Train", "Chat", "Machines"]);
    expect(within(studio).getByRole("link", { name: "Train" })).toHaveAttribute("aria-current", "page");
  });

  test.each([
    ["/quick-test", "Models"],
    ["/compare", "Models"],
    ["/collect", "Data"],
    ["/storage", "Data"],
    ["/runs/new", "Train"],
    ["/queue", "Train"],
    ["/training-run", "Train"],
    ["/runs/view", "Train"],
    ["/cluster/training", "Machines"],
  ])("a hidden or moved page (%s) keeps its area %s marked", (route, area) => {
    renderPage(<TrafficProvider><Sidebar /></TrafficProvider>, { route });
    const studio = screen.getByRole("navigation", { name: "Studio" });
    expect(within(studio).getByRole("link", { name: area })).toHaveAttribute("aria-current", "page");
    expect(studio.querySelectorAll("a[aria-current=page]")).toHaveLength(1);
  });

  test("hides the traffic console button until Developer tools is on", () => {
    renderPage(<TrafficProvider><Sidebar /></TrafficProvider>, { route: "/builder" });
    expect(screen.queryByRole("button", { name: /traffic/i })).toBeNull();
    expect(screen.getAllByRole("link", { name: "Settings" }).length).toBeGreaterThan(0);
  });

  test("shows the traffic console button once Developer tools is on", () => {
    localStorage.setItem("settings.developerTools", "on");
    renderPage(<TrafficProvider><Sidebar /></TrafficProvider>, { route: "/builder" });
    expect(screen.getAllByRole("button", { name: /traffic/i }).length).toBeGreaterThan(0);
  });

  test("pins Settings and traffic console at the foot of the nav", () => {
    localStorage.setItem("settings.developerTools", "on");
    renderPage(<TrafficProvider><Sidebar /></TrafficProvider>, { route: "/builder" });
    const gears = screen.getAllByRole("link", { name: "Settings" });
    expect(gears.length).toBeGreaterThan(0);
    for (const gear of gears) expect(gear).toHaveAttribute("href", "/settings");
    const consoles = screen.getAllByRole("button", { name: /traffic/i });
    expect(consoles.length).toBeGreaterThan(0);
  });

  test("contains only the five area links in the studio navigation, each to its area's home", () => {
    renderPage(<TrafficProvider><Sidebar /></TrafficProvider>, { route: "/builder" });
    const studio = screen.getByRole("navigation", { name: "Studio" });
    const links = within(studio).getAllByRole("link");
    expect(links).toHaveLength(5);
    expect(links.map((a) => a.getAttribute("href"))).toEqual(["/builder", "/your-data", "/pretraining", "/chat", "/cluster/machines"]);
    expect(screen.queryByRole("group", { name: "Pages in this area" })).toBeNull();
  });
});
