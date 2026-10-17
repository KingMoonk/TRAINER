import { expect, test } from "@playwright/test";

// The Cluster page (#319, the machines list from #311) against the real --fake server: no intercepted responses. The machine added here
// is a port nothing listens on, so it is listed as not answering, with the server's own reason.

const NOBODY = "127.0.0.1:9";

test.afterEach(async ({ request }) => {
  expect((await request.put("/cluster/nodes", { data: { addresses: [] } })).ok()).toBeTruthy(); // the specs share one workspace
});

test("adds a machine, saves, shows why it does not answer, then removes it", async ({ page, request }) => {
  expect((await request.put("/cluster/nodes", { data: { addresses: [] } })).ok()).toBeTruthy();
  const hostname = (await (await request.get("/node")).json()).hostname as string;
  await page.goto("/cluster/");
  await expect(page.getByTestId("machine-local")).toContainText(hostname);

  await page.getByLabel("Machine to add").fill("not a machine");
  await page.getByRole("button", { name: "Add", exact: true }).first().click();
  await page.getByRole("button", { name: "Save", exact: true }).click();
  await expect(page.getByTestId("machines-problem")).toContainText('"not a machine" is not a machine address');
  await page.getByRole("button", { name: "Undo changes" }).first().click();

  await page.getByLabel("Machine to add").fill(NOBODY);
  await page.getByRole("button", { name: "Add", exact: true }).first().click();
  await page.getByRole("button", { name: "Save", exact: true }).click();
  await expect(page.getByTestId(`machine-${NOBODY}`)).toHaveAttribute("data-online", "false", { timeout: 10_000 });
  await expect(page.getByTestId(`machine-error-${NOBODY}`)).toContainText(NOBODY);
  expect((await (await request.get("/cluster/nodes")).json()).map((n: { address: string }) => n.address)).toEqual(["local", NOBODY]);

  await page.reload();
  await expect(page.getByTestId(`machine-${NOBODY}`)).toBeVisible({ timeout: 10_000 });
  await page.getByRole("button", { name: `Remove ${NOBODY}` }).click();
  await page.getByRole("button", { name: "Save", exact: true }).click();
  await expect(page.getByText("Saved the list of machines.")).toBeVisible();
  expect((await (await request.get("/cluster/nodes")).json()).map((n: { address: string }) => n.address)).toEqual(["local"]);
});
