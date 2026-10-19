// A static file server for web/out, for the Playwright smoke tests. It resolves a directory to its index.html, as the API
// server will, and answers anything else (including /events) with 404, so the shell sees the API as unreachable.
import { createServer } from "node:http";
import { existsSync, readFileSync, statSync } from "node:fs";
import { extname, join, normalize, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(fileURLToPath(new URL("../out", import.meta.url)));
const port = Number(process.env.PORT ?? 4173);
const types = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
  ".woff2": "font/woff2",
  ".txt": "text/plain; charset=utf-8",
};

if (!existsSync(join(root, "index.html"))) {
  console.error(`${root} has no export. Run \`pnpm build\` in web/ (or \`python tasks.py web:build\`) first.`);
  process.exit(1);
}

createServer((req, res) => {
  const path = decodeURIComponent(new URL(req.url ?? "/", "http://localhost").pathname);
  let file = normalize(join(root, path));
  if (!file.startsWith(root)) file = "";
  if (file && existsSync(file) && statSync(file).isDirectory()) file = join(file, "index.html");
  if (file && !existsSync(file)) {
    // The client prefetches <route>/__next.<segment>.__PAGE__.txt; the export writes <route>/__next.<segment>/__PAGE__.txt
    // (studio_api/static.py does the same mapping).
    const name = file.slice(file.lastIndexOf(sep) + 1);
    const parts = name.startsWith("__next.") && name.endsWith(".txt") ? name.slice(7, -4).split(".") : [];
    if (parts.length >= 2 && parts.every(Boolean)) {
      file = join(file.slice(0, file.lastIndexOf(sep)), `__next.${parts[0]}`, ...parts.slice(1, -1), `${parts.at(-1)}.txt`);
    }
  }
  if (!file || !existsSync(file)) {
    res.writeHead(404).end("not found");
    return;
  }
  res.writeHead(200, { "Content-Type": types[extname(file)] ?? "application/octet-stream" }).end(readFileSync(file));
}).listen(port, "127.0.0.1", () => console.log(`serving ${root} on http://127.0.0.1:${port}`));
