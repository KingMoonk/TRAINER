import { expect, test, type Page } from "@playwright/test";

const frame = (kind: string, payload: object) => `data: ${JSON.stringify({ kind, payload })}\n\n`;
const version = (number: number, extra: object = {}) => ({
  number,
  name: `version ${number}`,
  change: `change ${number}`,
  status: "trained",
  verdict: "",
  best_checkpoint: `ckpt-${number}`,
  parent: number > 1 ? number - 1 : null,
  scorecard: {},
  ...extra,
});

async function mockVersions(page: Page, versions: object[]) {
  await page.route("**/versions", (route) => route.fulfill({ json: versions }));
  const trained = versions.filter((v) => "best_checkpoint" in v && v.best_checkpoint);
  const job = (result: unknown) => ({ id: "metadata", kind: "versions", status: "done", result, progress: "", error: "", created_at: 0 });
  await page.route("**/versions/best", (route) => route.fulfill({ json: job(trained.at(-1) ?? null) }));
  await page.route("**/versions/*/checkpoint-step", (route) => route.fulfill({ json: job(100) }));
}

test("a reply streams in, joins the conversation, and the request carries the whole thread", async ({ page }) => {
  await mockVersions(page, [version(1), version(2)]);
  const bodies: { version: number | null; messages: { role: string; content: string }[] }[] = [];
  await page.route("**/chat/reply", (route) => {
    bodies.push(route.request().postDataJSON());
    route.fulfill({
      contentType: "text/event-stream",
      body: frame("text", { text: "Hel" }) + frame("text", { text: "Hello there" }) + frame("done", { text: "Hello there" }),
    });
  });
  await page.goto("/chat/");
  await page.getByRole("button", { name: /Version 2/ }).click();
  await page.getByLabel("Your message").fill("hi");
  await page.getByRole("button", { name: "Send" }).click();
  await expect(page.getByText("Hello there")).toBeVisible();
  await page.getByLabel("Your message").fill("and again");
  await page.getByRole("button", { name: "Send" }).click();
  await expect.poll(() => bodies.length).toBe(2);
  expect(bodies[0]).toEqual({ version: 2, messages: [{ role: "user", content: "hi" }] });
  expect(bodies[1].messages.map((m) => m.content)).toEqual(["hi", "Hello there", "and again"]);
});

test("Stop aborts the request and the page is ready to send again", async ({ page }) => {
  await mockVersions(page, [version(1)]);
  let aborted = false;
  page.on("requestfailed", (request) => {
    if (request.url().endsWith("/chat/reply")) aborted = true;
  });
  await page.route("**/chat/reply", () => {
    // never answered: the reply is still streaming until the page gives up on it
  });
  await page.goto("/chat/");
  await page.getByLabel("Your message").fill("hi");
  await page.getByRole("button", { name: "Send" }).click();
  await page.getByRole("button", { name: "Stop" }).click();
  await expect(page.getByRole("button", { name: "Send" })).toBeVisible();
  await expect.poll(() => aborted).toBe(true);
});
