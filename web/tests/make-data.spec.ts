import { expect, test, type Page } from "@playwright/test";

for (const [kind, label] of [["text", /Generate prose/], ["conversations", /Rewrite/], ["problems", /Problems/], ["invalid", /Problems/]] as const) {
  test(`a ${kind} link opens the matching generator`, async ({ page }) => {
    await setup(page);
    await page.goto(`/make-data/?kind=${kind}`);
    await expect(page.getByRole("group", { name: "What to make" }).getByRole("button", { name: label })).toHaveAttribute("aria-pressed", "true");
  });
}

async function setup(page: Page) {
  const state = { drafts: [] as { id: string; a: string; b: string; status: string; dataset: string }[], requests: [] as object[], reviews: [] as boolean[], fail: false };
  await page.route("**/events", (route) => route.abort());
  await page.route("**/data/problems/rules", (route) => route.fulfill({ json: { rules: [{ id: "reverse", label: "Reverse" }], max_problems: 5000, max_drafts: 200 } }));
  await page.route("**/data/drafts", (route) => route.fulfill({ json: { count: state.drafts.length } }));
  await page.route("**/data/collections/*", (route) => route.fulfill({ json: [] }));
  await page.route("**/data/problems/generate", (route) => {
    const body = route.request().postDataJSON(); state.requests.push(body);
    return route.fulfill({ json: { items: [{ prompt: "Reverse: abc", answer: "cba" }], added: body.store ? body.n : 0, duplicates: 0, checked: body.n, passed: body.n } });
  });
  await page.route("**/data/conversations/rows?*", (route) => {
    expect(new URL(route.request().url()).searchParams.get("review_only")).toBe("true");
    return route.fulfill({ json: { rows: state.drafts, next_cursor: null } });
  });
  await page.route("**/data/conversations/drafts", (route) => {
    if (!state.fail) state.drafts = [1, 2].map((id) => ({ id: `d${id}`, a: `Draft ${id}`, b: `Answer ${id}`, status: "Review", dataset: "Drafts" }));
    return route.fulfill({ status: 202, json: { id: "make-1", kind: "make_drafts", status: state.fail ? "failed" : "done", result: { count: 2 }, error: state.fail ? "No local model connected." : "", created_at: 0 } });
  });
  await page.route("**/data/examples/*/fields", (route) => route.fulfill({ json: { fields: { user: "Full prompt", assistant: "Full answer to review" } } }));
  await page.route("**/data/examples/*/review", (route) => {
    state.reviews.push(route.request().postDataJSON().keep);
    const id = new URL(route.request().url()).pathname.split("/").at(-2);
    state.drafts = state.drafts.filter((row) => row.id !== id);
    return route.fulfill({ status: 204 });
  });
  return state;
}

test("rules come from the server; preview never stores and generation supports one or all rules", async ({ page }) => {
  const state = await setup(page);
  await page.goto("/make-data/");
  await page.getByLabel("Rule", { exact: true }).selectOption("reverse");
  await page.getByRole("button", { name: "Try 5 first" }).click();
  await expect(page.getByText("Preview, nothing saved")).toBeVisible();
  expect(state.requests[0]).toEqual({ rule: "reverse", n: 5, store: false });
  await page.getByLabel("How many", { exact: true }).fill("3");
  await page.getByRole("button", { name: "Generate problems", exact: true }).click();
  await expect(page.getByText(/Added 3 problems/)).toBeVisible();
  expect(state.requests[1]).toEqual({ rule: "reverse", n: 3, store: true });
  await page.getByLabel("Rule", { exact: true }).selectOption("");
  await page.getByRole("button", { name: "Generate problems", exact: true }).click();
  await expect.poll(() => state.requests.length).toBe(3);
  expect(state.requests[2]).toEqual({ rule: null, n: 3, store: true });
  await page.getByLabel("How many", { exact: true }).fill("5001");
  await expect(page.getByRole("button", { name: "Generate problems", exact: true })).toBeDisabled();
  await expect(page.getByRole("main").getByRole("alert")).toContainText("5,000");
});

test("generated drafts require reading, and accept/discard remove only the chosen row", async ({ page }) => {
  const state = await setup(page);
  await page.goto("/make-data/");
  await page.getByRole("button", { name: /Rewrite with your local model/ }).click();
  await page.getByRole("button", { name: "Make drafts", exact: true }).click();
  await expect(page.getByTestId("draft-d1")).toBeVisible();
  await expect(page.getByRole("button", { name: "Mark Good" })).toHaveCount(0);
  await page.getByTestId("draft-d1").getByRole("button", { name: "Read draft" }).click();
  await expect(page.getByText("Full answer to review")).toBeVisible();
  await page.getByTestId("draft-d1").getByRole("button", { name: "Mark Good" }).click();
  await expect(page.getByTestId("draft-d1")).toHaveCount(0);
  await expect(page.getByTestId("draft-d2")).toBeVisible();
  await page.getByTestId("draft-d2").getByRole("button", { name: "Read draft" }).click();
  await page.getByTestId("draft-d2").getByRole("button", { name: "Discard" }).click();
  await expect(page.getByText("No drafts waiting for review in this category.")).toBeVisible();
  expect(state.reviews).toEqual([true, false]);
});

test("a missing local model is an error, never a fabricated draft; phone layout stays inside the viewport", async ({ page }) => {
  const state = await setup(page); state.fail = true;
  await page.setViewportSize({ width: 375, height: 812 });
  await page.goto("/make-data/");
  await page.getByRole("button", { name: /Rewrite with your local model/ }).click();
  await page.getByRole("button", { name: "Make drafts", exact: true }).click();
  await expect(page.getByRole("main").getByRole("alert")).toContainText("No local model connected.");
  await expect(page.getByTestId("draft-d1")).toHaveCount(0);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});
