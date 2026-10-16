import { expect, test, type Page } from "@playwright/test";
import { SseParser } from "../src/lib/sse-post";

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

test("SseParser joins frames split across chunks, and skips comments and malformed frames", () => {
  const parser = new SseParser();
  const whole = frame("text", { text: "Hel" }) + ": ping\n\n" + "data: not json\n\n" + frame("done", { text: "Hello" });
  const events = [...parser.push(whole.slice(0, 20)), ...parser.push(whole.slice(20, 70)), ...parser.push(whole.slice(70))];
  expect(events).toEqual([
    { kind: "text", payload: { text: "Hel" } },
    { kind: "done", payload: { text: "Hello" } },
  ]);
  expect(new SseParser().push(frame("text", { text: "a" }).replace(/\n/g, "\r\n"))).toEqual([{ kind: "text", payload: { text: "a" } }]);
});

test("with no trained version the page says so and the message box is off", async ({ page }) => {
  await mockVersions(page, [version(1, { best_checkpoint: null, status: "draft" })]);
  await page.goto("/chat/");
  await expect(page.getByText("No trained version yet.")).toBeVisible();
  await expect(page.getByRole("link", { name: "Train one from Build your model." })).toBeVisible();
  await expect(page.getByLabel("Your message")).toBeDisabled();
});

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

test("an error event shows a toast and the reply says why", async ({ page }) => {
  await mockVersions(page, [version(1)]);
  await page.route("**/chat/reply", (route) =>
    route.fulfill({ contentType: "text/event-stream", body: frame("error", { message: "The model ran out of memory." }) }),
  );
  await page.goto("/chat/");
  await page.getByLabel("Your message").fill("hi");
  await page.getByRole("button", { name: "Send" }).click();
  await expect(page.getByText("The model ran out of memory.").first()).toBeVisible();
});

test("a correction is saved with the conversation up to the last message", async ({ page }) => {
  await mockVersions(page, [version(1)]);
  await page.route("**/chat/reply", (route) =>
    route.fulfill({ contentType: "text/event-stream", body: frame("done", { text: "Paris is in Italy." }) }),
  );
  let saved: unknown;
  await page.route("**/chat/corrections", (route) => {
    saved = route.request().postDataJSON();
    route.fulfill({ status: 204 });
  });
  await page.goto("/chat/");
  await expect(page.getByLabel("Your message")).toBeEnabled(); // the page has loaded its versions and hydrated
  await page.getByRole("button", { name: "Teach a correction" }).click();
  await expect(page.getByText("Send a message first")).toBeVisible();
  await page.getByLabel("Your message").fill("Where is Paris?");
  await page.getByRole("button", { name: "Send" }).click();
  await expect(page.getByText("Paris is in Italy.")).toBeVisible();
  await page.getByLabel("Correction").fill("Paris is in France.");
  await page.getByRole("button", { name: "Save", exact: true }).click();
  await expect(page.getByText("Saved for Post-training round 3.")).toBeVisible();
  expect(saved).toEqual({
    version: 1,
    messages: [{ role: "user", content: "Where is Paris?" }],
    correction: "Paris is in France.",
    whole_answer: false,
  });
});

test("compare asks both versions, shows a verdict, and saves the better reply as a rated pair", async ({ page }) => {
  await mockVersions(page, [version(1), version(2)]);
  await page.route("**/benchmark/verdict**", (route) => route.fulfill({ json: { comparable: true, percent: 80, reason: "" } }));
  await page.route("**/chat/reply", (route) => {
    const { version: v } = route.request().postDataJSON();
    route.fulfill({ contentType: "text/event-stream", body: frame("done", { text: `answer from ${v}` }) });
  });
  let pair: unknown;
  await page.route("**/chat/pairs", (route) => {
    pair = route.request().postDataJSON();
    route.fulfill({ json: { record: null } });
  });
  await page.goto("/chat/");
  await page.getByRole("button", { name: "Compare two versions" }).click();
  await expect(page.getByText("To reach the same error, Version 2 needs 20% of the training Version 1 needed.")).toBeVisible();
  await page.getByLabel("Your message").fill("hi");
  await page.getByRole("button", { name: "Send" }).click();
  await expect(page.getByText("answer from 2")).toBeVisible();
  await expect(page.getByText("answer from 1")).toBeVisible();
  await page.getByRole("button", { name: "This one is better" }).nth(1).click();
  await expect(page.getByText("Saved: Version 1 was better")).toBeVisible();
  expect(pair).toEqual({
    messages: [{ role: "user", content: "hi" }],
    chosen: { version: 1, reply: "answer from 1" },
    rejected: { version: 2, reply: "answer from 2" },
  });
});
