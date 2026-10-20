import { render } from "@testing-library/react";
import { describe, expect, test } from "vitest";
import { SeriesChart, type Series } from "./series-chart";

// Every series gets its own colour, whatever the id: a machine called "Michaels-Mac.local" is a dotted id.

const points = (offset: number) => [1, 2, 3].map((x) => ({ x, y: x + offset }));
const two: Series[] = [
  { id: "BEAST", label: "BEAST", points: points(0) },
  { id: "Michaels-Mac.local", label: "Michaels-Mac.local", points: points(1) },
];

describe("SeriesChart colours", () => {
  test("two members, one with a dotted name, each get a colour of their own", () => {
    const { container } = render(<SeriesChart series={two} xLabel="Round" yLabel="loss" />);
    const css = [...container.querySelectorAll("style")].map((style) => style.textContent).join("\n");
    const colours = [...css.matchAll(/--color-(s\d+):\s*([^;]+);/g)].map((m) => [m[1], m[2].trim()]);
    expect([...new Set(colours.map(([key]) => key))]).toEqual(["s0", "s1"]);
    expect(new Set(colours.slice(0, 2).map(([, colour]) => colour)).size).toBe(2);
    expect(css).not.toContain("--color-Michaels-Mac.local");
  });
});
