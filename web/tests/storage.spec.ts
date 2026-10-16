import { expect, test, type Page } from "@playwright/test";
import { downloadName, errorText, sizeText } from "../src/app/storage/logic";
import { ApiError } from "../src/lib/api";

const is = (pathname: string) => (url: URL) => url.pathname === pathname;

type Use = { items: number; bytes: number };

/** Mock the Storage routes: `use` answers GET /storage by the `days` and `data` asked, `cleaned` records every POST /storage/clean. */
async function mockStorage(page: Page, use: (days: number | null, data: boolean) => Use, exportsList: object[] = []) {
  const cleaned: object[] = [];
  const state = { use, refuse: "" };
  await page.route(is("/storage"), (route) => {
    const q = new URL(route.request().url()).searchParams;
    const days = q.has("days") ? Number(q.get("days")) : null;
    return route.fulfill({ json: state.use(days, q.get("data") === "true") });
  });
  await page.route(is("/storage/free"), (route) => route.fulfill({ json: { bytes: 120e9 } }));
  await page.route(is("/storage/exports"), (route) => route.fulfill({ json: exportsList }));
  await page.route(is("/storage/clean"), (route) => {
    cleaned.push(route.request().postDataJSON());
    if (state.refuse) return route.fulfill({ status: 409, json: { detail: state.refuse } });
    const body = route.request().postDataJSON() as { days: number | null; data: boolean };
    const done = state.use(body.days, body.data);
    state.use = () => ({ items: 0, bytes: 0 });
    return route.fulfill({ json: done });
  });
  return { cleaned, state };
}

const bySize = (days: number | null, data: boolean): Use =>
  days === null ? (data ? { items: 9, bytes: 4.5e9 } : { items: 5, bytes: 2.5e9 }) : days <= 7 ? { items: 2, bytes: 3e6 } : { items: 1, bytes: 1e6 };

test("sizes, server sentences and file names are read the way the Tk page shows them", () => {
  expect(sizeText(1.5e9)).toBe("1.5 GB");
  expect(sizeText(2.5e6)).toBe("2.5 MB");
  expect(errorText(new ApiError(409, "/storage/clean", JSON.stringify({ detail: "Stop the running job first." })))).toBe("Stop the running job first.");
  expect(downloadName('attachment; filename="my-model-hf.zip"', "x")).toBe("my-model-hf.zip");
  expect(downloadName(null, "fallback")).toBe("fallback");
});

test("the page shows free space and what a clean-up of each age would free, in place", async ({ page }) => {
  await mockStorage(page, bySize);
  await page.goto("/storage/");
  await expect(page.getByRole("heading", { name: "Storage" })).toBeVisible();
  await expect(page.getByText("120.0 GB free on this computer's disk")).toBeVisible();
  await expect(page.getByText("This frees 1.0 MB in 1 item")).toBeVisible();
  await page.getByLabel("Older than 7 days").check();
  await expect(page.getByText("This frees 3.0 MB in 2 items")).toBeVisible();
  const data = page.getByLabel("Also delete my datasets and downloads");
  await expect(data).toBeDisabled();
  await page.getByLabel("All history").check();
  await expect(page.getByText("This frees 2.5 GB in 5 items")).toBeVisible();
  await data.check();
  await expect(page.getByText("This frees 4.5 GB in 9 items")).toBeVisible();
  await page.getByLabel("Older than 1 day").check();
  await expect(data).toBeDisabled();
  await expect(data).not.toBeChecked();
});

test("a clean asks first, then frees the space and the numbers update", async ({ page }) => {
  const { cleaned } = await mockStorage(page, bySize);
  await page.goto("/storage/");
  await page.getByRole("button", { name: "Delete…" }).click();
  await expect(page.getByText("Delete 1 item and free 1.0 MB? This cannot be undone.")).toBeVisible();
  await page.getByRole("button", { name: "Cancel" }).click();
  expect(cleaned).toEqual([]);
  await page.getByRole("button", { name: "Delete…" }).click();
  await page.getByRole("alertdialog").getByRole("button", { name: "Delete" }).click();
  await expect(page.getByText("Freed 1.0 MB")).toBeVisible();
  expect(cleaned).toEqual([{ days: 30, data: false }]);
  await expect(page.getByText("Nothing to delete")).toBeVisible();
  await expect(page.getByRole("button", { name: "Delete…" })).toBeDisabled();
});

test("a clean the server refuses shows its sentence and changes nothing", async ({ page }) => {
  const { state } = await mockStorage(page, bySize);
  state.refuse = "Stop the running job first.";
  await page.goto("/storage/");
  await page.getByRole("button", { name: "Delete…" }).click();
  await page.getByRole("alertdialog").getByRole("button", { name: "Delete" }).click();
  await expect(page.getByText("Stop the running job first.")).toBeVisible();
  await expect(page.getByText("This frees 1.0 MB in 1 item")).toBeVisible();
});

test("nothing trained: the export card says so", async ({ page }) => {
  await mockStorage(page, () => ({ items: 0, bytes: 0 }));
  await page.goto("/storage/");
  await expect(page.getByText("Nothing to export yet.")).toBeVisible();
  await expect(page.getByText("Nothing to delete")).toBeVisible();
});

test("a version GGUF cannot hold greys out its button and says why; the others download", async ({ page }) => {
  const why = "GGUF export covers GPT-2's architecture only.";
  await mockStorage(page, bySize, [
    { number: 1, name: "plain", gguf_problem: "" },
    { number: 2, name: "exotic", gguf_problem: why },
  ]);
  const asked: string[] = [];
  await page.route(is("/versions/1/export"), (route) => {
    asked.push(route.request().url());
    return route.fulfill({ body: "zip", headers: { "content-disposition": 'attachment; filename="plain-hf.zip"' } });
  });
  await page.route(is("/versions/2/export"), (route) => route.fulfill({ status: 409, json: { detail: "Version 2 has no checkpoint yet." } }));
  await page.goto("/storage/");
  const rows = page.getByRole("listitem");
  await expect(rows).toHaveCount(2);
  await expect(rows.nth(1).getByRole("button", { name: "GGUF" })).toBeDisabled();
  await expect(rows.nth(1).getByText(why)).toBeVisible();
  await expect(rows.nth(0).getByRole("button", { name: "GGUF" })).toBeEnabled();

  const download = page.waitForEvent("download");
  await rows.nth(0).getByRole("button", { name: "GGUF" }).click();
  expect((await download).suggestedFilename()).toBe("plain-hf.zip");
  expect(asked[0]).toContain("kind=gguf");

  await rows.nth(1).getByRole("button", { name: "Hugging Face" }).click();
  await expect(page.getByText("Version 2 has no checkpoint yet.")).toBeVisible();
});
