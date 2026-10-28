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

test("the Developer tools switch is off by default, shows the traffic console button when on, and survives a reload", async ({ page }) => {
  await page.goto("/settings/");
  const consoleButton = page.getByRole("button", { name: "API traffic console" });
  const toggle = page.getByRole("switch", { name: "Show the API traffic console button" });
  await expect(toggle).not.toBeChecked();
  await expect(consoleButton).toHaveCount(0);

  await toggle.click();
  await expect(toggle).toBeChecked();
  await expect(consoleButton.locator("visible=true")).toBeVisible();

  await page.reload();
  await expect(toggle).toBeChecked();
  await expect(consoleButton.locator("visible=true")).toBeVisible();

  await toggle.click();
  await expect(consoleButton).toHaveCount(0);
});

test.describe("on a phone", () => {
  test.use({ viewport: { width: 375, height: 812 } });
  test("the gear is reachable in the top row", async ({ page }) => {
    await page.goto("/queue/");
    await page.getByRole("link", { name: "Settings", exact: true }).click();
    await expect(page).toHaveURL(/\/settings\/$/);
  });
});
