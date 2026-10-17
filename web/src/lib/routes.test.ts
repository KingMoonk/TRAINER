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

  test("hrefs are absolute paths without a trailing slash", () => {
    for (const r of ROUTES) expect(r.href, r.route).toMatch(/^\/[a-z0-9-]+$/);
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
    expect(areaHome("designer").route).toBe("builder");
    expect(areaHome("pretraining").route).toBe("pretraining");
    expect(areaHome("post").route).toBe("post_training");
    expect(areaHome("chat").route).toBe("chat");
    expect(areaHome("queue").route).toBe("queue");
    expect(areaHome("resources").route).toBe("your_data");
  });
});

describe("routeFor", () => {
  test("finds a route by its Tk name", () => {
    expect(routeFor("head_zoom")).toMatchObject({ href: "/head-zoom", area: "designer", title: "Inside a head" });
  });

  test("returns every listed route", () => {
    for (const r of ROUTES) expect(routeFor(r.route)).toBe(r);
  });

  test("looks-inside pages are all in the designer area", () => {
    expect(ROUTES.filter((r) => r.lookInside).every((r) => r.area === "designer")).toBe(true);
  });
});
