// The theme guard (run by `pnpm lint`): every colour lives in src/app/globals.css as a token, so nothing else in src/
// may name a Tailwind palette colour (text-amber-500, bg-white/5), a named colour (fill="red") or write a colour literal
// (#fff, rgb(...), oklch(...)). Use a semantic utility (bg-card, text-warning, bg-surface-overlay) or var(--token).
// See the top of globals.css. The rules are in theme-guard.mjs and tested by theme-guard.test.mjs.
import { readdirSync, readFileSync } from "node:fs";
import { join, relative, sep } from "node:path";
import { checkSource } from "./theme-guard.mjs";

const SRC = new URL("../src/", import.meta.url).pathname;
const THEME = "app/globals.css";
// Skipped whole, with the reason. (The one narrow exemption, for chart.tsx, is in theme-guard.mjs.)
const SKIP = new Map([["lib/api-schema.ts", "generated from openapi.json"]]);

function* walk(dir) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) yield* walk(path);
    else if (/\.(tsx?|jsx?|mjs|css)$/.test(entry.name)) yield path;
  }
}

const problems = [];
for (const path of walk(SRC)) {
  const file = relative(SRC, path).split(sep).join("/");
  if (file === THEME || SKIP.has(file)) continue;
  problems.push(...checkSource(file, readFileSync(path, "utf8")));
}

if (problems.length) {
  console.error(`${problems.join("\n")}\n\nColours belong in src/app/globals.css as tokens (see the comment at its top). Use a semantic utility or var(--token).`);
  process.exit(1);
}
console.log("theme guard: no raw palette classes or colour literals outside globals.css");
