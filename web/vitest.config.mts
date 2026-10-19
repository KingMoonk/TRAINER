import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

// Tiers 1 and 2 of the web tests (logic.ts, lib/*, components and the recorded API fixtures): jsdom, no browser, no server.
// Playwright (playwright.config.ts, tests/) is tier 3 and only runs in CI.
export default defineConfig({
  resolve: { alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) } },
  test: {
    environment: "jsdom",
    // Node's Request (what openapi-fetch builds) rejects a relative URL; the browser would resolve it against the page.
    env: { NEXT_PUBLIC_API_URL: "http://localhost:8787" },
    include: ["src/**/*.test.{ts,tsx}", "scripts/**/*.test.{ts,mjs}"],
    setupFiles: ["./src/test/setup.ts"],
    restoreMocks: true,
  },
});
