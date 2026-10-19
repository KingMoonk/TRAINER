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
    expect(areaHome("runs").route).toBe("runs");
    expect(areaHome("eval").route).toBe("chat");
    expect(areaHome("machines").route).toBe("machines");
  });
});

describe("the five sidebar areas", () => {
  test("are Models, Data, Runs, Eval and Machines, in that order", () => {
    expect(AREAS.map((a) => a.title)).toEqual(["Models", "Data", "Runs", "Eval", "Machines"]);
  });

  test("group the routes as the design does", () => {
    const pagesIn = (area: string) => ROUTES.filter((r) => r.area === area).map((r) => r.route);
    expect(pagesIn("models")).toEqual(["builder", "quicktest", "compare", "trace", "head_zoom", "value_map", "loss", "landscape", "head_lab", "inside3d"]);
    expect(pagesIn("data").sort()).toEqual(["pretraining", "storage", "your_data"]);
    expect(pagesIn("runs")).toEqual(["runs", "new_run", "run_view", "training_run", "post_training", "queue", "passive", "cluster_training"]);
    expect(pagesIn("eval")).toEqual(["chat", "benchmark"]);
    expect(pagesIn("machines")).toEqual(["machines"]);
  });

  test("the new run pages are routes, and run_view is hidden from the nav", () => {
    expect(routeFor("runs")).toMatchObject({ href: "/runs", area: "runs" });
    expect(routeFor("new_run")).toMatchObject({ href: "/runs/new", area: "runs" });
    expect(routeFor("run_view")).toMatchObject({ href: "/runs/view", area: "runs", hidden: true });
    expect(ROUTES.filter((r) => r.hidden).map((r) => r.route)).toEqual(["run_view"]);
  });

  test("keep the route names and hrefs the Tk shell uses", () => {
    expect(routeFor("cluster_training").href).toBe("/cluster/training");
    expect(routeFor("machines").href).toBe("/cluster/machines");
    expect(routeFor("pretraining").href).toBe("/pretraining");
  });
});

describe("routeFor", () => {
  test("finds a route by its Tk name", () => {
    expect(routeFor("head_zoom")).toMatchObject({ href: "/head-zoom", area: "models", title: "Inside a head" });
  });

  test("returns every listed route", () => {
    for (const r of ROUTES) expect(routeFor(r.route)).toBe(r);
  });

  test("looks-inside pages are all in the models area", () => {
    expect(ROUTES.filter((r) => r.lookInside).every((r) => r.area === "models")).toBe(true);
  });
});
