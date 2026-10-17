import { expect, test, type APIRequestContext, type Page } from "@playwright/test";

// Real fake services (python -m studio_api --fake), no intercepted responses: the rest of the Builder's learning loop around
// the tower. Where the size goes, the example sentence in pieces, the unsaved draft staged for Quick test, the learning-rate
// finder and the "when answering" card. The draft is seeded as live.builder-guess.spec.ts does.
async function seed(page: Page, request: APIRequestContext) {
  const presets = await (await request.get("/design/presets")).json();
  const spec = presets[0].spec;
  const base = await (await request.post("/versions", { data: { spec, change: "Builder loop fixture" } })).json();
  const cur = await (await request.post("/design/read", { data: { spec } })).json();
  const draft = await (await request.post("/design/resize", { data: { choices: cur, n: cur.L + 1 } })).json();
  await page.addInitScript((stored) => window.localStorage.setItem("builder.draft", JSON.stringify(stored)), { base: base.number, draft, anchor: null });
  return base.number as number;
}

const staged = async (request: APIRequestContext) => (await (await request.get("/quicktest")).json()).draft as { changes: string[] } | null;

test("the size splits into its four parts, the draft bar longer than the current one", async ({ page, request }) => {
  await seed(page, request);
  await page.goto("/builder/");
  const bar = page.getByTestId("param-bar");
  await expect(bar).toContainText("Where the size goes");
  await expect(bar).toContainText("word table");
  await expect(bar).toContainText("blocks");
  const width = (id: string) =>
    page.getByTestId(id).evaluate((row) => [...row.children].reduce((sum, part) => sum + (part as HTMLElement).getBoundingClientRect().width, 0));
  expect(await width("param-next")).toBeGreaterThan(await width("param-now")); // one more block: the draft is the bigger model
});

test("the example sentence is cut into pieces and every word list is counted", async ({ page, request }) => {
  await seed(page, request);
  await page.goto("/builder/");
  const card = page.getByTestId("pieces");
  await expect(card.getByTestId("pieces-count")).toContainText("pieces");
  await expect(card.getByRole("list").getByRole("listitem").first()).toBeVisible();
});

test("an unsaved draft is staged for Quick test and taken back when it is discarded", async ({ page, request }) => {
  await seed(page, request);
  await page.goto("/builder/");
  await expect.poll(async () => (await staged(request))?.changes.length ?? 0).toBeGreaterThan(0);
  await page.getByTestId("tray").getByRole("button", { name: "Discard" }).click();
  await expect.poll(() => staged(request)).toBeNull();
});

test("Quick test and Compare runs are one click from the tray and the page", async ({ page, request }) => {
  await seed(page, request);
  await page.goto("/builder/");
  await expect(page.getByTestId("quick-test-first")).toHaveAttribute("href", /\/quick-test\/$/);
  const links = page.getByTestId("try-it");
  await expect(links.getByRole("link", { name: "Compare runs" })).toHaveAttribute("href", /\/compare\/$/);
  await expect(links.getByRole("link", { name: "Quick test" })).toHaveAttribute("href", /\/quick-test\/$/);
});

test("the learning-rate finder reads a suggestion off the draft", async ({ page, request }) => {
  await seed(page, request);
  await page.goto("/builder/");
  await page.getByRole("button", { name: "Find a learning rate" }).click();
  await expect(page.getByTestId("lr-text")).toContainText("learning rate");
});

test("nothing can answer until a version has trained", async ({ page, request }) => {
  await seed(page, request);
  await page.goto("/builder/");
  await expect(page.getByTestId("answers")).toContainText("has no trained weights yet");
  await expect(page.getByTestId("ask-both")).toHaveCount(0);
});
