import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, test } from "vitest";
import { TrafficProvider } from "@/components/traffic/traffic-provider";
import { renderPage } from "@/test/render";
import { nav } from "@/test/navigation";
import { ROUTES } from "@/lib/routes";
import { ShellEvents } from "@/lib/events";
import { Sidebar } from "./sidebar";

describe("Sidebar", () => {
  test("lists the five areas in order, with the open one marked", () => {
    renderPage(<TrafficProvider><Sidebar /></TrafficProvider>, { route: "/runs/new" });
    const studio = screen.getByRole("navigation", { name: "Studio" });
    const areas = within(studio).getAllByRole("link").filter((a) => a.hasAttribute("href") && a.getAttribute("href") !== "/settings");
    expect(areas.map((a) => a.textContent)).toEqual(["Models", "Data", "Runs", "Eval", "Machines"]);
    expect(within(studio).getByRole("link", { name: "Runs" })).toHaveAttribute("aria-current", "page");
  });

  test("lists the Runs pages in the page list, and leaves run_view out", () => {
    renderPage(<TrafficProvider><Sidebar /></TrafficProvider>, { route: "/runs/view" });
    const group = screen.getByRole("group", { name: "Pages in this area" });
    expect(within(group).getAllByRole("link").map((a) => a.textContent)).toEqual([
      "Runs", "New run", "Training run", "Post-training", "Queue", "Passive training", "Training",
    ]);
  });

  test("pins a Settings gear at the foot of the nav", () => {
    renderPage(<TrafficProvider><Sidebar /></TrafficProvider>, { route: "/builder" });
    const gears = screen.getAllByRole("link", { name: "Settings" });
    expect(gears.length).toBeGreaterThan(0);
    for (const gear of gears) expect(gear).toHaveAttribute("href", "/settings");
  });

  test("keeps model tasks visible and lets the keyboard reveal every diagnostic page", async () => {
    const user = userEvent.setup();
    renderPage(<TrafficProvider><Sidebar /></TrafficProvider>, { route: "/builder" });
    const group = screen.getByRole("group", { name: "Pages in this area" });
    expect(within(group).getAllByRole("link").map((link) => link.textContent)).toEqual([
      "Build your model", "Quick test", "Compare runs",
    ]);
    const disclosure = within(group).getByRole("button", { name: "Look inside" });
    expect(disclosure).toHaveAttribute("aria-expanded", "false");
    disclosure.focus();
    await user.keyboard("{Enter}");
    expect(disclosure).toHaveAttribute("aria-expanded", "true");
    const panel = document.getElementById(disclosure.getAttribute("aria-controls")!);
    expect(panel).toBeVisible();
    for (const route of ROUTES.filter((route) => route.lookInside)) {
      expect(within(panel!).getByRole("link", { name: route.title })).toHaveAttribute("href", route.href);
    }
    await user.keyboard(" ");
    expect(disclosure).toHaveAttribute("aria-expanded", "false");
    expect(panel).not.toBeVisible();
    expect(disclosure).toHaveFocus();
  });

  test.each(ROUTES.filter((route) => route.lookInside))("reveals the active diagnostic page at $href", (route) => {
    renderPage(<TrafficProvider><Sidebar /></TrafficProvider>, { route: `${route.href}/` });
    const group = screen.getByRole("group", { name: "Pages in this area" });
    expect(within(group).getByRole("button", { name: "Look inside" })).toHaveAttribute("aria-expanded", "true");
    expect(within(group).getByRole("link", { name: route.title })).toHaveAttribute("aria-current", "location");
    expect(screen.getByRole("link", { name: "Models", exact: true })).toHaveAttribute("aria-current", "page");
  });

  test("reveals a diagnostic route after navigation even when the group was collapsed", async () => {
    const user = userEvent.setup();
    const view = renderPage(<TrafficProvider><Sidebar /></TrafficProvider>, { route: "/trace" });
    await user.click(screen.getByRole("button", { name: "Look inside" }));
    expect(screen.getByRole("button", { name: "Look inside" })).toHaveAttribute("aria-expanded", "false");
    nav.path = "/head-lab";
    view.rerender(<QueryClientProvider client={view.client}><ShellEvents><TrafficProvider><Sidebar /></TrafficProvider></ShellEvents></QueryClientProvider>);
    expect(screen.getByRole("button", { name: "Look inside" })).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByRole("link", { name: "Head lab" })).toHaveAttribute("aria-current", "location");
  });
});
