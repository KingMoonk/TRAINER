import { screen, waitFor } from "@testing-library/react";
import { describe, expect, test } from "vitest";
import { nav } from "@/test/navigation";
import { renderPage } from "@/test/render";
import { ROUTES, routeFor } from "@/lib/routes";
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

  test("every tab is a listed route, and the first one is the default", () => {
    const hrefs = ROUTES.map((r) => r.href);
    for (const tab of CLUSTER_TABS) expect(hrefs).toContain(tab.href);
    expect(CLUSTER_TABS[0].href).toBe(routeFor("cluster_training").href);
  });

  test("/cluster/ opens the first tab", async () => {
    renderPage(<ClusterHome />, { route: "/cluster" });
    await waitFor(() => expect(nav.replace).toHaveBeenCalledWith(CLUSTER_TABS[0].href));
  });
});
