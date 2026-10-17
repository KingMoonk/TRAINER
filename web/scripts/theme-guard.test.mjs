import assert from "node:assert/strict";
import { test } from "node:test";
import { checkSource } from "./theme-guard.mjs";

const bad = (source, file = "app/x.tsx") => checkSource(file, source);

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
  assert.deepEqual(bad(good), []);
});

test("raw palette classes fail, including per-side borders, inset rings and shadows", () => {
  for (const cls of ["text-amber-500", "bg-white/5", "hover:text-red-400", "border-t-white/10", "border-x-sky-500", "border-s-pink-400",
    "inset-ring-white/10", "drop-shadow-black", "text-shadow-blue-500", "ring-offset-white"]) {
    assert.equal(bad(`const a = "${cls}";`).length, 1, cls);
  }
});

test("named colours in arbitrary values, props and color-mix fail", () => {
  for (const s of [`"bg-[white]"`, `"text-[red]"`, `<rect fill="red" />`, `<path stroke='white' />`, `{ color: "black" }`,
    `style={{ backgroundColor: "navy" }}`, `"color-mix(in oklch, var(--a) 70%, black)"`]) {
    assert.equal(bad(`const a = ${s};`).length >= 1, true, s);
  }
});

test("hex and colour functions fail, but not var() arguments or anchors", () => {
  for (const s of [`"#0e1014"`, `"#fff"`, `"bg-[#123456]"`, `"color: #aabbcc"`, `"rgb(0 0 0 / 50%)"`, `"oklch(0.5 0 0)"`, `"hsl(210 40% 90%)"`]) {
    assert.equal(bad(`const a = ${s};`).length >= 1, true, s);
  }
  assert.deepEqual(bad(`const a = "hsl(var(--x))", b = "oklch(var(--y) / 50%)", c = <a href="#add" id="#bad" />;`), []);
  assert.deepEqual(bad(`const label = "See issue #148 and #1234";`), []);
});

test("a /* inside a string does not hide the rest of the file", () => {
  assert.equal(bad(`const glob = "src/**/*.ts"; const a = "text-amber-500";`).length, 1);
  assert.equal(bad(`const a = "/*";\nconst b = "bg-white";`).length, 1);
});

test("chart.tsx is exempt only for recharts' own selector colours", () => {
  const selectors = `"[&_.recharts-dot[stroke='#fff']]:stroke-transparent [&_.recharts-polar-grid_[stroke='#ccc']]:stroke-border"`;
  assert.deepEqual(bad(`const a = ${selectors};`, "components/ui/chart.tsx"), []);
  assert.equal(bad(`const a = ${selectors};`, "app/other.tsx").length, 1);
  assert.equal(bad(`const a = "text-amber-500";`, "components/ui/chart.tsx").length, 1);
});
