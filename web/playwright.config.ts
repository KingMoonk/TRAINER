import { defineConfig, devices } from "@playwright/test";

// Smoke tests against the static export in out/ (run `pnpm build` first). No API is running: tests/serve-out.mjs 404s /events,
// and a test that needs events fulfils them itself.
//
// The "live" project runs tests/live.spec.ts and tests/live.<page>.spec.ts against `python -m studio_api --fake`, which serves out/ and a real /events. It
// needs the Python requirements (torch included, because FakeServices imports the engine), so it only runs when API_LIVE=1;
// API_PYTHON names the interpreter (default `python`; `.venv/bin/python` from a checkout).
const PORT = 4173;
const LIVE_PORT = 4174;
const live = process.env.API_LIVE === "1";

export default defineConfig({
  testDir: "tests",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : "list",
  use: { trace: "on-first-retry" },
  projects: [
    {
      name: "chromium",
      testIgnore: ["live.spec.ts", "live.*.spec.ts"],
      use: { ...devices["Desktop Chrome"], baseURL: `http://127.0.0.1:${PORT}` },
    },
    ...(live
      ? [
          {
            name: "live",
            testMatch: ["live.spec.ts", "live.*.spec.ts"],
            use: { ...devices["Desktop Chrome"], baseURL: `http://127.0.0.1:${LIVE_PORT}` },
          },
        ]
      : []),
  ],
  webServer: [
    {
      command: "node tests/serve-out.mjs",
      env: { PORT: String(PORT) },
      url: `http://127.0.0.1:${PORT}/builder/`,
      reuseExistingServer: !process.env.CI,
    },
    ...(live
      ? [
          {
            command: `${process.env.API_PYTHON ?? "python"} -m studio_api --fake --port ${LIVE_PORT}`,
            cwd: "..",
            url: `http://127.0.0.1:${LIVE_PORT}/builder/`,
            reuseExistingServer: !process.env.CI,
            timeout: 60_000,
          },
        ]
      : []),
  ],
});
