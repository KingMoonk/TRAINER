import { expect, test, type Page } from "@playwright/test";

test("generation links follow the selected data kind and passive training is reachable", async ({ page }) => {
  await mockApi(page);
  await page.goto("/your-data/");
  await expect(page.getByRole("link", { name: /Passive training: see or stop it/ })).toHaveAttribute("href", "/passive/");
  for (const [tab, link, kind] of [
    [/^Text it reads/, "Generate prose", "text"],
    [/^Conversations it practises/, "Make extra conversations", "conversations"],
    [/^Problems/, "Generate problems a rule can check", "problems"],
  ] as const) {
    await page.getByRole("tab", { name: tab }).click();
    await expect(page.getByRole("link", { name: link, exact: true })).toHaveAttribute("href", `/make-data/?kind=${kind}`);
  }
});

// The Your data page against a mocked API (as chat.spec.ts does): three summaries read as jobs, the rows a page at a time, the
// recommended collection with live download progress, an uploaded file imported as a job, a folder on the server picked.

const frame = (event: object) => `data: ${JSON.stringify(event)}\n\n`;
const json = (body: unknown, status = 200) => ({ status, contentType: "application/json", body: JSON.stringify(body) });

const job = (id: string, kind: string, result: unknown) => ({
  id,
  kind,
  status: "done",
  progress: "",
  fraction: null,
  result,
  error: "",
  created_at: 0,
  started_at: 0,
  finished_at: 1,
});

const row = (id: string, a: string, b: string, status = "In use") => ({ id, a, b, status, dataset: "Your files" });
const summary = (kind: string, rows: ReturnType<typeof row>[], sets: { id: string; name: string; count: number }[]) => ({
  kind,
  count: rows.length,
  characters: 100,
  used_by: kind === "text" ? "used by Pretraining" : "used by Post-training, round 1",
  count_line: `${rows.length} ${kind === "text" ? "documents" : "conversations"} · ${sets.length} sets`,
  sets,
  rules: {},
  near_duplicates: 0,
  held_back: 0,
  scanned: rows.length,
  rows,
});

const source = (id: string, name: string, state: string, extra: object = {}) => ({
  id,
  name,
  size: "64 MB",
  records: "77,941 stories",
  state,
  completed: 0,
  total: 0,
  error: "",
  ...extra,
});

type Mock = { uploads: string[]; paths: string[][]; downloads: unknown[]; imported: boolean; gate: () => void };

async function mockApi(page: Page): Promise<Mock> {
  let release = () => {};
  const gate = new Promise<void>((resolve) => (release = resolve));
  const state: Mock = { uploads: [], paths: [], downloads: [], imported: false, gate: release };
  let streams = 0;
  const conversations = () =>
    state.imported
      ? [row("c1", "Name a prime.", "7 is a prime number."), row("c2", "Say hello.", "Hello there.", "Review"), row("c3", "Imported prompt", "Imported answer")]
      : [row("c1", "Name a prime.", "7 is a prime number."), row("c2", "Say hello.", "Hello there.", "Review")];

  await page.route("**/data/refresh_downloads", (route) => route.fulfill(json(job("r1", "refresh_downloads", []), 202)));
  await page.route("**/data/*/summary", async (route) => {
    const kind = new URL(route.request().url()).pathname.split("/")[2];
    const rows = kind === "conversations" ? conversations() : [];
    const sets =
      kind === "conversations" ? [{ id: "d1", name: "Your conversations", count: rows.length }, ...(state.imported ? [{ id: "d2", name: "Imported set", count: 1 }] : [])] : [];
    await route.fulfill(json(job(`s-${kind}-${state.imported}`, "data", summary(kind, rows, sets)), 202));
  });
  await page.route("**/data/*/rows**", (route) => {
    const kind = new URL(route.request().url()).pathname.split("/")[2];
    return route.fulfill(json({ rows: kind === "conversations" ? conversations() : [], next_cursor: null }));
  });
  await page.route("**/data/collections/*", (route) =>
    route.fulfill(json([source("tinystories", "TinyStories", "downloading", { total: 100, completed: 10 }), source("wikitext2", "WikiText-2", "not downloaded")])),
  );
  await page.route("**/data/collections/*/download", async (route) => {
    state.downloads.push(route.request().postDataJSON());
    await route.fulfill(json({ started: true }));
  });
  await page.route("**/data/conversations/import", async (route) => {
    state.uploads.push(route.request().postDataBuffer()?.toString("latin1") ?? "");
    state.imported = true;
    await route.fulfill(json(job("i1", "import_data", ["notes.txt: 1 added"]), 202));
  });
  await page.route("**/data/conversations/import_paths", async (route) => {
    state.paths.push((route.request().postDataJSON() as { paths: string[] }).paths);
    await route.fulfill(json(job("i2", "import_data", ["corpus.txt: 4 added"]), 202));
  });
  await page.route("**/data/browse**", (route) => {
    const path = new URL(route.request().url()).searchParams.get("path");
    if (path === null) return route.fulfill(json({ path: null, parent: null, entries: [{ name: "/home/me", path: "/home/me", is_dir: true, size: null }] }));
    return route.fulfill(json({ path, parent: null, entries: [{ name: "corpus.txt", path: `${path}/corpus.txt`, is_dir: false, size: 2048 }] }));
  });
  await page.route("**/events", async (route) => {
    streams += 1;
    if (streams > 2) return route.abort(); // the shell opens the first stream, the page the second; neither is reopened
    await gate;
    return route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      body: frame({ kind: "download", payload: { id: "tinystories", text: "Downloading · 42 MB of 64 MB", completed: 42, total: 64 } }),
    });
  });
  return state;
}

test("the three kinds show their counts and the first kind's rows", async ({ page }) => {
  await mockApi(page);
  await page.goto("/your-data/");
  await expect(page.getByRole("heading", { level: 1, name: "Your data" })).toBeVisible();
  await expect(page.getByRole("tab", { name: /Conversations it practises/ })).toContainText("2 conversations · 1 sets");
  await expect(page.getByTestId("row-c1")).toContainText("Name a prime.");
  await expect(page.getByTestId("sets-line")).toHaveText("Your conversations (2)");
  await page.getByTestId("row-c2").click();
  await expect(page.getByTestId("detail")).toContainText("Hello there.");
  await expect(page.getByRole("button", { name: "Mark Good" })).toBeVisible();
  await expect(page.getByRole("link", { name: "‹ Back to Post-training" })).toHaveAttribute("href", /\/post-training/);
});

test("an uploaded file is imported and the new row and set appear", async ({ page }) => {
  const state = await mockApi(page);
  await page.goto("/your-data/");
  await expect(page.getByTestId("row-c1")).toBeVisible();
  await page.getByRole("button", { name: "Import files" }).click();
  await page.getByLabel("Choose files").setInputFiles({ name: "notes.txt", mimeType: "text/plain", buffer: Buffer.from("hello world") });
  await page.getByRole("button", { name: "Import 1 file" }).click();
  await expect(page.getByTestId("import-report")).toContainText("notes.txt: 1 added");
  expect(state.uploads[0]).toContain('filename="notes.txt"');
  expect(state.uploads[0]).toContain("hello world");
  // the rows and the sets are read again: the imported ones are listed
  await expect(page.getByTestId("row-c3")).toContainText("Imported prompt");
  await expect(page.getByTestId("sets-line")).toHaveText("Your conversations (3) · Imported set (1)");
});

test("a file in a folder on the server is picked and imported", async ({ page }) => {
  const state = await mockApi(page);
  await page.goto("/your-data/");
  await page.getByRole("button", { name: "Import files" }).click();
  await page.getByRole("button", { name: "/home/me/" }).click();
  await page.getByRole("checkbox").check();
  await page.getByRole("button", { name: "Import 1 picked file" }).click();
  await expect(page.getByTestId("import-report")).toContainText("corpus.txt: 4 added");
  expect(state.paths).toEqual([["/home/me/corpus.txt"]]);
});

test("download progress changes one row in place and the other rows keep their nodes", async ({ page }) => {
  const state = await mockApi(page);
  await page.goto("/your-data/");
  await page.getByRole("tab", { name: /Text it reads/ }).click();
  const first = page.getByTestId("source-tinystories");
  const second = page.getByTestId("source-wikitext2");
  await expect(first).toContainText("TinyStories");
  await expect(page.getByTestId("source-tinystories-status")).toHaveText("Downloading");
  await first.evaluate((node) => node.setAttribute("data-mark", "same"));
  await second.evaluate((node) => node.setAttribute("data-mark", "same"));

  state.gate();
  await expect(page.getByTestId("source-tinystories-status")).toHaveText("Downloading · 42 MB of 64 MB");
  await expect(page.getByRole("progressbar", { name: "TinyStories progress" })).toHaveAttribute("aria-valuenow", "66");
  await expect(first).toHaveAttribute("data-mark", "same");
  await expect(second).toHaveAttribute("data-mark", "same");

  await page.getByRole("button", { name: "Download and add", exact: true }).click();
  await expect.poll(() => state.downloads).toEqual([{ ids: ["wikitext2"], install: true }]);
  await page.getByRole("button", { name: "Download and add all" }).click();
  await expect.poll(() => state.downloads.length).toBe(2);
  expect(state.downloads[1]).toEqual({ ids: null, install: true });
});
