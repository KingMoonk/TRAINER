// The rules of the theme guard (see check-theme.mjs for what it is for). Kept apart from the file walk so the unit test
// can feed it samples.

const PALETTE = "white|black|slate|gray|zinc|neutral|stone|red|orange|amber|yellow|lime|green|emerald|teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose";
// Colour names that can appear in arbitrary values and props, where there is no -500 shade: a superset of the palette.
// ("transparent", "currentColor" and "inherit" are not colours of the theme and are allowed.)
const NAMED = `${PALETTE}|grey|magenta|navy|maroon|silver|gold|brown|aqua|olive|tan|coral|salmon|crimson|ivory|khaki|lavender|beige`;
const UTILITY = "bg|text|border(?:-[trblxyse])?|ring|ring-offset|inset-ring|fill|stroke|from|via|to|divide|outline|shadow|inset-shadow|drop-shadow|text-shadow|decoration|accent|caret|placeholder";
const HEX = "#(?:[0-9a-fA-F]{8}|[0-9a-fA-F]{6}|[0-9a-fA-F]{4}|[0-9a-fA-F]{3})(?![0-9a-zA-Z_-])";
const HEX_LONG = "#(?:[0-9a-fA-F]{8}|[0-9a-fA-F]{6})(?![0-9a-zA-Z_-])";
// Where a "#abc" is an anchor or id, not a colour.
const ANCHOR = /(?:href|id|htmlFor|to|name|aria-[a-z]+)\s*[=:]\s*\{?$/;

export const RULES = [
  { what: "raw Tailwind palette class", find: (t) => t.match(new RegExp(`(?<![\\w-])(?:${UTILITY})-(?:${PALETTE})(?![a-z])`)) },
  { what: "named colour in an arbitrary value", find: (t) => t.match(new RegExp(`(?<![\\w-])(?:${UTILITY})-\\[(?:${NAMED})\\]`)) },
  { what: "named colour in a colour prop or style", find: (t) => t.match(new RegExp(`\\b(?:fill|stroke|color|backgroundColor|background-color|borderColor|stopColor|floodColor)\\s*[=:]\\s*\\{?\\s*["'\`](?:${NAMED})["'\`]`)) },
  { what: "named colour in color-mix()", find: (t) => t.match(new RegExp(`color-mix\\(.*\\b(?:${NAMED})\\b`)) },
  { what: "colour function literal", find: (t) => t.match(/\b(?:rgba?|hsla?|oklch|oklab|lab|lch|hwb|color)\((?!\s*var\()/) },
  {
    what: "hex colour literal",
    find: (t) => {
      // A hex colour is a whole string ("#0e1014"), a CSS value (color: #fff), or an argument/arbitrary value; "#148" in
      // prose is an issue number, so outside a whole string only 6- and 8-digit hex counts.
      for (const m of t.matchAll(new RegExp(`(["'\`])(${HEX})\\1|[:,(\\[]\\s*(${HEX_LONG})|-\\[(${HEX})`, "g"))) {
        const hex = m[2] ?? m[3] ?? m[4];
        if (m[1] && ANCHOR.test(t.slice(0, m.index + 0))) continue;
        return [hex];
      }
      return null;
    },
  },
];

/** `text` with comments removed; quote state is kept so `/*` or `//` inside a string does not hide the rest. */
export function stripComments(text) {
  const lines = text.split("\n");
  let block = false;
  let tick = false; // inside a multi-line template literal
  return lines.map((line) => {
    let out = "";
    let quote = tick ? "`" : null;
    for (let i = 0; i < line.length; i++) {
      const c = line[i];
      if (block) {
        const end = line.indexOf("*/", i);
        if (end < 0) return out;
        block = false;
        i = end + 1;
      } else if (quote) {
        out += c;
        if (c === "\\") out += line[++i] ?? "";
        else if (c === quote) quote = null;
      } else if (c === '"' || c === "'" || c === "`") {
        quote = c;
        out += c;
      } else if (c === "/" && line[i + 1] === "*") {
        block = true;
        i++;
      } else if (c === "/" && line[i + 1] === "/") {
        break;
      } else out += c;
    }
    tick = quote === "`";
    return out;
  });
}

// A narrow exemption, not a whole-file one: shadcn's chart.tsx selects the literal colours recharts writes into its own
// markup, e.g. [&_.recharts-dot[stroke='#fff']]. Only that selector pattern is removed, and only from that file.
const EXEMPT = [{ file: "components/ui/chart.tsx", pattern: /\[(?:stroke|fill)='#(?:ccc|fff)'\]/g }];

/** Problems in one file's source, as "src/<file>:<line>: <what> `<match>`". */
export function checkSource(file, source) {
  const problems = [];
  stripComments(source).forEach((line, index) => {
    let text = line;
    for (const rule of EXEMPT) if (rule.file === file) text = text.replace(rule.pattern, "");
    for (const { what, find } of RULES) {
      const hit = find(text);
      if (hit) problems.push(`src/${file}:${index + 1}: ${what} \`${hit[0].trim()}\``);
    }
  });
  return problems;
}
