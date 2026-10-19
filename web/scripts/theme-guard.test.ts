import { execFileSync } from "node:child_process";
import { resolve } from "node:path";
import { expect, test } from "vitest";
import { checkSource } from "./theme-guard.mjs";

const bad = (source: string, file = "app/x.tsx") => checkSource(file, source);

test("token-based code passes", () => {
  const good = `
    // text-amber-500 in a comment, and issue #148
    /* bg-white/5 in a block comment */
    export const a = "bg-card border-surface-border text-warning-soft bg-info/10 fill-cat-1 hover:bg-surface-overlay";
    export const b = <a href="#fee">x</a>;
    export const c = { background: "color-mix(in oklch, var(--foreground) 40%, var(--muted))", color: "hsl(var(--x))" };
    export const d = <svg fill="none" stroke="currentColor" />;
    const url = "http://example.com/*"; const ok = "bg-card";
  `;
  expect(bad(good)).toEqual([]);
});

test("raw palette classes fail, including per-side borders, inset rings and shadows", () => {
  for (const cls of ["text-amber-500", "bg-white/5", "hover:text-red-400", "border-t-white/10", "border-x-sky-500", "border-s-pink-400",
    "inset-ring-white/10", "drop-shadow-black", "text-shadow-blue-500", "ring-offset-white"]) {
    expect(bad(`const a = "${cls}";`).length, cls).toBe(1);
  }
});

test("named colours in arbitrary values, props and color-mix fail", () => {
  for (const s of [`"bg-[white]"`, `"text-[red]"`, `<rect fill="red" />`, `<path stroke='white' />`, `{ color: "black" }`,
    `style={{ backgroundColor: "navy" }}`, `"color-mix(in oklch, var(--a) 70%, black)"`]) {
    expect(bad(`const a = ${s};`).length >= 1, s).toBe(true);
  }
});

test("hex and colour functions fail, but not var() arguments or anchors", () => {
  for (const s of [`"#0e1014"`, `"#fff"`, `"bg-[#123456]"`, `"color: #aabbcc"`, `"rgb(0 0 0 / 50%)"`, `"oklch(0.5 0 0)"`, `"hsl(210 40% 90%)"`]) {
    expect(bad(`const a = ${s};`).length >= 1, s).toBe(true);
  }
  expect(bad(`const a = "hsl(var(--x))", b = "oklch(var(--y) / 50%)", c = <a href="#add" id="#bad" />;`)).toEqual([]);
  expect(bad(`const label = "See issue #148 and #1234";`)).toEqual([]);
});

test("a /* inside a string does not hide the rest of the file", () => {
  expect(bad(`const glob = "src/**/*.ts"; const a = "text-amber-500";`).length).toBe(1);
  expect(bad(`const a = "/*";\nconst b = "bg-white";`).length).toBe(1);
});

test("chart.tsx is exempt only for recharts' own selector colours", () => {
  const selectors = `"[&_.recharts-dot[stroke='#fff']]:stroke-transparent [&_.recharts-polar-grid_[stroke='#ccc']]:stroke-border"`;
  expect(bad(`const a = ${selectors};`, "components/ui/chart.tsx")).toEqual([]);
  expect(bad(`const a = ${selectors};`, "app/other.tsx").length).toBe(1);
  expect(bad(`const a = "text-amber-500";`, "components/ui/chart.tsx").length).toBe(1);
});

test("check-theme.mjs walks src/ to completion (exits 0 on the real tree)", () => {
  // Runs the script for real: on Windows a bad src path made it throw before checking anything.
  const script = resolve("scripts/check-theme.mjs"); // pnpm test runs from web/
  expect(execFileSync(process.execPath, [script], { encoding: "utf8" })).toContain("theme guard: no raw palette classes");
});
