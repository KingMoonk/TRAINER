import { expect, test, type APIRequestContext, type Page } from "@playwright/test";

// Real fake services (python -m studio_api --fake), no intercepted responses: the Builder's lab notebook lists each saved
// version with its change and the question it was made to ask; the explain panel works a part through the
// example sentence; a starting question sets up one change.
// The number the Builder's own save was given, read off its POST /versions response, so a row is never guessed from the seed's number.
const savedNumber = (page: Page) =>
  page
    .waitForResponse((r) => r.request().method() === "POST" && new URL(r.url()).pathname === "/versions")
    .then(async (r) => (await r.json()).number as number);

async function seed(page: Page, request: APIRequestContext, more = 1) {
  const presets = await (await request.get("/design/presets")).json();
  const spec = presets[0].spec;
  const base = await (await request.post("/versions", { data: { spec, change: "Builder notebook fixture" } })).json();
  const cur = await (await request.post("/design/read", { data: { spec } })).json();
  const draft = await (await request.post("/design/resize", { data: { choices: cur, n: cur.L + more } })).json();
  await page.addInitScript((stored) => window.localStorage.setItem("builder.draft", JSON.stringify(stored)), { base: base.number, draft, anchor: null });
  return { base: base.number as number, blocks: cur.L as number };
}

test("the notebook lists each change and updates its rows in place", async ({ page, request }) => {
  const { base } = await seed(page, request);
  await page.goto("/builder/");
  await page.locator("summary").filter({ hasText: "Lab notebook · one line per version" }).click();
  const first = page.getByTestId(`notebook-${base}`);
  await expect(first).toContainText(`Version ${base} · Builder notebook fixture`);
  await first.evaluate((element) => element.setAttribute("data-original-row", "yes"));

  const saved = savedNumber(page);
  await page.getByTestId("save").click();
  const number = await saved;

  const row = page.getByTestId(`notebook-${number}`);
  await expect(row).toContainText(`Version ${number} ·`);
  await expect(row.getByTestId("notebook-effects")).toContainText("not trained yet");
  await expect(first).toHaveAttribute("data-original-row", "yes"); // the earlier row was kept, not rebuilt
  await expect(page.getByRole("list", { name: "Lab notebook · one line per version" }).getByRole("listitem").last()).toHaveAttribute(
    "data-testid",
    `notebook-${number}`,
  );
});

test("clicking a part explains it on the example sentence", async ({ page, request }) => {
  await seed(page, request);
  await page.goto("/builder/");
  const panel = page.getByTestId("explain");
  await expect(panel).not.toBeVisible();
  await expect(panel).toContainText("Click any part of the model");
  await page.getByTestId("explain-position").click();
  await expect(panel).toBeVisible();
  await expect(panel.getByTestId("explain-title")).toHaveText("Position");
  await expect(panel.getByTestId("explain-example")).toContainText("pieces are read at once");
  await expect(page.getByTestId("explain-position")).toHaveAttribute("aria-pressed", "true");
  await page.getByTestId("explain-embedding").click();
  await expect(panel.getByTestId("explain-title")).toHaveText("Word table");
  await expect(panel.getByTestId("explain-example")).toContainText("The lamp was lit before the tide came in.");
  await panel.getByTestId("explain-close").click();
  await expect(panel).toContainText("Click any part of the model");
});

test("a starting question sets up one change and goes with the version", async ({ page, request }) => {
  const { blocks } = await seed(page, request, 0);
  await page.goto("/builder/");
  await page.locator("summary").filter({ hasText: "Experiments & diagnostics" }).click();
  await page.getByTestId("try-depth").click();
  await expect(page.getByTestId("block-count")).toHaveText(String(blocks + 1));
  await expect(page.getByTestId("experiment-asking")).toContainText("Is depth or width better?");
  await page.getByTestId("save").click();
  const saved = savedNumber(page);
  await page.getByTestId("save").click();
  const number = await saved;
  const row = page.getByTestId(`notebook-${number}`);
  await expect(row.getByTestId("notebook-question")).toHaveText("asked “Is depth or width better?”");
  const list = await (await request.get("/versions")).json();
  expect(list.find((v: { number: number }) => v.number === number).prediction).toEqual({ question: "Is depth or width better?" });
});
