import { describe, expect, test } from "vitest";
import { AREAS, areaHome, HOME, ROUTES, routeFor } from "./routes";

describe("ROUTES and AREAS", () => {
  test("route names, hrefs and area ids are each unique", () => {
    expect(new Set(ROUTES.map((r) => r.route)).size).toBe(ROUTES.length);
    expect(new Set(ROUTES.map((r) => r.href)).size).toBe(ROUTES.length);
    expect(new Set(AREAS.map((a) => a.id)).size).toBe(AREAS.length);
  });

  test("every route belongs to a listed area and every area has a route", () => {
    const areas = new Set(AREAS.map((a) => a.id));
    for (const r of ROUTES) expect(areas, r.route).toContain(r.area);
    for (const a of AREAS) expect(ROUTES.some((r) => r.area === a.id), a.id).toBe(true);
  });

  test("hrefs are absolute paths without a trailing slash, at most one level deep", () => {
    for (const r of ROUTES) expect(r.href, r.route).toMatch(/^\/[a-z0-9-]+(\/[a-z0-9-]+)?$/);
  });

  test("every route has a title and a lead", () => {
    for (const r of ROUTES) {
      expect(r.title.trim(), r.route).not.toBe("");
      expect(r.lead.trim(), r.route).not.toBe("");
    }
  });

  test("HOME is the builder route's href", () => expect(HOME).toBe(routeFor("builder").href));
});

describe("areaHome", () => {
  test("is the first route of each area", () => {
    for (const area of AREAS) expect(areaHome(area.id)).toBe(ROUTES.find((r) => r.area === area.id));
  });

  test("names the expected homes", () => {
    expect(areaHome("models").route).toBe("builder");
    expect(areaHome("data").route).toBe("your_data");
    expect(areaHome("train").route).toBe("auto_run");
    expect(areaHome("eval").route).toBe("chat");
    expect(areaHome("machines").route).toBe("machines");
  });
});

describe("the five sidebar areas", () => {
  const pagesIn = (area: string) => ROUTES.filter((r) => r.area === area).map((r) => r.route);
  const tabsIn = (area: string) => ROUTES.filter((r) => r.area === area && !r.hidden).map((r) => r.route);

  test("are Models, Data, Train, Chat and Machines, in that order", () => {
    expect(AREAS.map((a) => a.title)).toEqual(["Models", "Data", "Train", "Chat", "Machines"]);
    expect(AREAS.map((a) => a.id)).toEqual(["models", "data", "train", "eval", "machines"]);
  });

  test("group the routes as the design does", () => {
    expect(pagesIn("models")).toEqual(["builder", "quicktest", "compare"]);
    expect(pagesIn("data")).toEqual(["your_data", "collect", "storage"]);
    expect(pagesIn("train")).toEqual(["auto_run", "pretraining", "post_training", "runs", "new_run", "queue", "training_run", "run_view"]);
    expect(pagesIn("eval")).toEqual(["chat", "benchmark", "export"]);
    expect(pagesIn("machines")).toEqual(["machines", "cluster_training"]);
  });

  test("the section tabs are the visible pages, eleven in all", () => {
    expect(tabsIn("models")).toEqual(["builder"]);
    expect(tabsIn("data")).toEqual(["your_data"]);
    expect(tabsIn("train")).toEqual(["auto_run", "pretraining", "post_training", "runs"]);
    expect(tabsIn("eval")).toEqual(["chat", "benchmark", "export"]);
    expect(tabsIn("machines")).toEqual(["machines", "cluster_training"]);
    expect(ROUTES.filter((r) => !r.hidden)).toHaveLength(11);
  });

  test("the hidden pages are the ones reached from elsewhere, each with a visible parent in its own area", () => {
    expect(ROUTES.filter((r) => r.hidden).map((r) => r.route)).toEqual(["quicktest", "compare", "collect", "storage", "new_run", "queue", "training_run", "run_view"]);
    for (const r of ROUTES.filter((x) => x.hidden)) {
      const parent = routeFor(r.parent!);
      expect(parent, r.route).toBeDefined();
      expect(parent.hidden, r.route).toBeFalsy();
      expect(parent.area, r.route).toBe(r.area);
    }
  });

  test("the run pages keep their titles and hrefs", () => {
    expect(routeFor("runs")).toMatchObject({ href: "/runs", area: "train", title: "Runs", tabTitle: "All runs" });
    expect(routeFor("cluster_training")).toMatchObject({ href: "/cluster/training", area: "machines", title: "Training", tabTitle: "Cluster training" });
    expect(routeFor("new_run")).toMatchObject({ href: "/runs/new", area: "train", hidden: true, parent: "runs" });
    expect(routeFor("run_view")).toMatchObject({ href: "/runs/view", area: "train", hidden: true, parent: "runs" });
  });

  test("keep the route names and hrefs", () => {
    expect(routeFor("cluster_training").href).toBe("/cluster/training");
    expect(routeFor("machines").href).toBe("/cluster/machines");
    expect(routeFor("pretraining").href).toBe("/pretraining");
  });
});

describe("routeFor", () => {
  test("finds a route by its Tk name", () => {
    expect(routeFor("compare")).toMatchObject({ href: "/compare", area: "models", title: "Compare runs" });
  });

  test("returns every listed route", () => {
    for (const r of ROUTES) expect(routeFor(r.route)).toBe(r);
  });
});
