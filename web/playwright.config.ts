import { defineConfig, devices } from "@playwright/test";

// Smoke tests against the static export in out/ (run `pnpm build` first). No API is running: tests/serve-out.mjs 404s /events,
// and a test that needs events fulfils them itself.
const PORT = 4173;

export default defineConfig({
  testDir: "tests",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : "list",
  use: { baseURL: `http://127.0.0.1:${PORT}`, trace: "on-first-retry" },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: "node tests/serve-out.mjs",
    env: { PORT: String(PORT) },
    url: `http://127.0.0.1:${PORT}/builder/`,
    reuseExistingServer: !process.env.CI,
  },
});
