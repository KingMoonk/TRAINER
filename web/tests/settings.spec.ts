import { expect, test } from "@playwright/test";

test("the gear opens Settings, and the mode and accent choices set <html> and survive a reload", async ({ page }) => {
  await page.goto("/builder/");
  const html = page.locator("html");
  await expect(html).toHaveClass(/\bdark\b/); // dark is the default
  await expect(html).not.toHaveAttribute("data-theme", /.+/);

  await page.getByRole("link", { name: "Settings", exact: true }).click();
  await expect(page).toHaveURL(/\/settings\/$/);
  await expect(page.getByRole("heading", { level: 1, name: "Settings" })).toBeVisible();
  await expect(page.getByRole("heading", { level: 2, name: "Appearance" })).toBeVisible();

  const mode = page.getByTestId("appearance-mode");
  const accent = page.getByTestId("appearance-accent");
  await mode.getByText("Light", { exact: true }).click();
  await accent.getByText("Forest", { exact: true }).click();
  await expect(html).toHaveClass(/\blight\b/);
  await expect(html).not.toHaveClass(/\bdark\b/);
  await expect(html).toHaveAttribute("data-theme", "forest");

  await page.reload();
  await expect(html).toHaveClass(/\blight\b/);
  await expect(html).toHaveAttribute("data-theme", "forest");
  await expect(mode.getByText("Light", { exact: true })).toHaveAttribute("data-state", "on");
  await expect(accent.getByText("Forest", { exact: true })).toHaveAttribute("data-state", "on");

  await accent.getByText("Neutral", { exact: true }).click();
  await expect(html).not.toHaveAttribute("data-theme", /.+/);
});

test.describe("on a phone", () => {
  test.use({ viewport: { width: 375, height: 812 } });
  test("the gear is reachable in the top row", async ({ page }) => {
    await page.goto("/queue/");
    await page.getByRole("link", { name: "Settings", exact: true }).click();
    await expect(page).toHaveURL(/\/settings\/$/);
  });
});

test("Settings links to the Cluster page instead of managing machines", async ({ page }) => {
  await page.goto("/settings/");
  await expect(page.getByTestId("settings-cluster").getByRole("link", { name: "Open the Cluster page" })).toHaveAttribute("href", "/cluster/");
  await expect(page.getByLabel("Machine to add")).toHaveCount(0);
});

test("the sidebar has a Cluster entry that opens /cluster/", async ({ page }) => {
  await page.goto("/builder/");
  await page.getByRole("link", { name: "Cluster", exact: true }).first().click();
  await expect(page).toHaveURL(/\/cluster\/$/);
  await expect(page.getByRole("heading", { level: 1, name: "Cluster" })).toBeVisible();
});
