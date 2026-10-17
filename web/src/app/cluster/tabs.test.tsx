import { screen, waitFor } from "@testing-library/react";
import { describe, expect, test } from "vitest";
import { nav } from "@/test/navigation";
import { renderPage } from "@/test/render";
import { AREAS, ROUTES, areaHome } from "@/lib/routes";
import ClusterHome from "./page";
import { CLUSTER_TABS, ClusterTabs } from "./tabs";

describe("Cluster tab bar", () => {
  test("lists the tabs in order and marks the open one", () => {
    renderPage(<ClusterTabs />, { route: "/cluster/machines" });
    const links = screen.getAllByRole("link");
    expect(links.map((a) => a.textContent)).toEqual(CLUSTER_TABS.map((t) => t.title));
    expect(screen.getByRole("link", { name: "Machines" })).toHaveAttribute("aria-current", "location");
    expect(screen.getByRole("link", { name: "Machines" })).toHaveAttribute("href", "/cluster/machines");
  });

  test("every tab is a route of the Cluster area, and the first one is the area's home", () => {
    const area = ROUTES.filter((r) => r.area === "cluster").map((r) => r.href);
    for (const tab of CLUSTER_TABS) expect(area).toContain(tab.href);
    expect(AREAS.some((a) => a.id === "cluster")).toBe(true);
    expect(areaHome("cluster").href).toBe(CLUSTER_TABS[0].href);
  });

  test("/cluster/ opens the first tab", async () => {
    renderPage(<ClusterHome />, { route: "/cluster" });
    await waitFor(() => expect(nav.replace).toHaveBeenCalledWith(CLUSTER_TABS[0].href));
  });
});
