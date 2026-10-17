// The theme guard (run by `pnpm lint`): every colour lives in src/app/globals.css as a token, so nothing else in src/
// may name a Tailwind palette colour (text-amber-500, bg-white/5) or write a colour literal (#fff, rgb(...), oklch(...)).
// Use a semantic utility (bg-card, text-warning, bg-surface-overlay) or var(--token). See the top of globals.css.
import { readdirSync, readFileSync } from "node:fs";
import { join, relative, sep } from "node:path";

const SRC = new URL("../src/", import.meta.url).pathname;
const THEME = "app/globals.css";
// Files exempt from the guard, each with the reason. Keep this short.
const ALLOW = new Map([
  ["lib/api-schema.ts", "generated from openapi.json"],
  ["components/ui/chart.tsx", "shadcn selectors that match the literal colours recharts writes into its own markup"],
]);

const PALETTE = "white|black|slate|gray|zinc|neutral|stone|red|orange|amber|yellow|lime|green|emerald|teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose";
const UTILITY = "bg|text|border|ring|ring-offset|fill|stroke|from|via|to|divide|outline|shadow|decoration|accent|caret|placeholder";
const RULES = [
  [new RegExp(`(?<![\\w-])(?:${UTILITY})-(?:${PALETTE})(?![a-z])`), "raw Tailwind palette class"],
  [/#[0-9a-fA-F]{3,8}\b/, "hex colour literal"],
  [/\b(?:rgba?|hsla?|oklch|oklab|lab|lch|hwb|color)\(/, "colour function literal"],
];

function* walk(dir) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) yield* walk(path);
    else if (/\.(tsx?|jsx?|mjs|css)$/.test(entry.name)) yield path;
  }
}

/** The line without comments, so a note like "issue #148" or "(text-amber-500)" in a comment does not trip the guard. */
function code(line, state) {
  let out = "";
  for (let i = 0; i < line.length; i++) {
    if (state.block) {
      const end = line.indexOf("*/", i);
      if (end < 0) return out;
      state.block = false;
      i = end + 1;
    } else if (line.startsWith("/*", i)) {
      state.block = true;
      i++;
    } else if (line.startsWith("//", i) && line[i - 1] !== ":") {
      return out;
    } else out += line[i];
  }
  return out;
}

const problems = [];
for (const path of walk(SRC)) {
  const file = relative(SRC, path).split(sep).join("/");
  if (file === THEME || ALLOW.has(file)) continue;
  const state = { block: false };
  readFileSync(path, "utf8").split("\n").forEach((line, index) => {
    const text = code(line, state);
    for (const [rule, what] of RULES) {
      const hit = text.match(rule);
      if (hit) problems.push(`src/${file}:${index + 1}: ${what} \`${hit[0].trim()}\``);
    }
  });
}

if (problems.length) {
  console.error(`${problems.join("\n")}\n\nColours belong in src/app/globals.css as tokens (see the comment at its top). Use a semantic utility or var(--token).`);
  process.exit(1);
}
console.log("theme guard: no raw palette classes or colour literals outside globals.css");
