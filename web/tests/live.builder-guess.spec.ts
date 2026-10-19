import { expect, test, type APIRequestContext, type Page } from "@playwright/test";

// Real fake services (python -m studio_api --fake), no intercepted responses: saving a Builder draft asks for a guess
// first (or Skip), stores it with the version, and checks it next to the version before. The draft is seeded in
// localStorage ("builder.draft", as the page keeps it) with one more block than the version it is saved on.
async function seedDraft(page: Page, request: APIRequestContext) {
  const presets = await (await request.get("/design/presets")).json();
  const spec = presets[0].spec;
  const base = await (await request.post("/versions", { data: { spec, change: "Builder guess fixture" } })).json();
  const cur = await (await request.post("/design/read", { data: { spec } })).json();
  const draft = await (await request.post("/design/resize", { data: { choices: cur, n: cur.L + 1 } })).json();
  await page.addInitScript((stored) => window.localStorage.setItem("builder.draft", JSON.stringify(stored)), {
    base: base.number, draft, anchor: null,
  });
  return base.number as number;
}

async function lastVersion(request: APIRequestContext) {
  const list = await (await request.get("/versions")).json();
  return list[list.length - 1];
}

test("guess before saving: the guess is stored with the version and checked next to the one before", async ({ page, request }) => {
  const base = await seedDraft(page, request);
  await page.goto("/builder/");
  await page.getByTestId("save").click();
  const panel = page.getByTestId("guess-panel");
  await expect(panel).toBeFocused();
  await expect(panel.getByTestId("guess-error").getByRole("radio")).toHaveText(["Lower", "Higher"]);
  // By keyboard: focus the choice and press Space.
  await panel.getByTestId("guess-error").getByRole("radio", { name: "Lower" }).focus();
  await page.keyboard.press("Space");
  await expect(panel.getByTestId("guess-error").getByRole("radio", { name: "Lower" })).toBeChecked();
  await panel.getByTestId("guess-save").click();
  await expect(page.getByText(`Saved version ${base + 1}.`)).toBeVisible();
  const saved = await lastVersion(request);
  expect(saved.number).toBe(base + 1);
  expect(saved.prediction).toEqual({ error: "lower" });
  await expect(page.getByTestId("guess-check")).toContainText("You expected lower error.");
});

test("skipping the guess saves the version without one", async ({ page, request }) => {
  const base = await seedDraft(page, request);
  await page.goto("/builder/");
  await page.getByTestId("save").click();
  await page.getByTestId("guess-error").getByRole("radio", { name: "Higher" }).click();
  await page.getByTestId("guess-skip").click();
  await expect(page.getByText(`Saved version ${base + 1}.`)).toBeVisible();
  const saved = await lastVersion(request);
  expect(saved.number).toBe(base + 1);
  expect(saved.prediction).toEqual({});
  await expect(page.getByTestId("guess-check")).toHaveCount(0);
});

test("back to editing leaves the draft unsaved", async ({ page, request }) => {
  const base = await seedDraft(page, request);
  await page.goto("/builder/");
  await page.getByTestId("save").click();
  await page.getByTestId("guess-back").click();
  await expect(page.getByTestId("guess-panel")).toHaveCount(0);
  await expect(page.getByTestId("save")).toBeVisible();
  expect((await lastVersion(request)).number).toBe(base);
});
