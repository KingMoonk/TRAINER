import { expect, test, type Page } from "@playwright/test";
import { changed, cleanList, deviceText, refusedUs } from "../src/app/settings/machines-logic";
import { fakeEvents, mockApi } from "./qp-mock";

// The Settings page's Machines section (#311) against a mocked GET and PUT /cluster/nodes. The live spec (live.machines.spec.ts)
// runs it against a real --fake server.

type Node = {
  address: string; local: boolean; hostname: string; version: string; device: string; device_name: string; memory_bytes: number;
  busy: boolean; dtypes: string[]; online: boolean; error: string; tailnet_name: string;
};
const node = (address: string, over: Partial<Node> = {}): Node => ({
  address, local: false, hostname: "", version: "", device: "", device_name: "", memory_bytes: 0, busy: false, dtypes: [], online: false,
  error: "", tailnet_name: "", ...over,
});
const ME = node("local", {
  local: true, hostname: "mac-mini", version: "1.4.0", device: "mps", device_name: "Apple M2", memory_bytes: 16 * 2 ** 30, online: true,
  dtypes: ["float32"], tailnet_name: "mac-mini.tail1234.ts.net",
});
const GPU = node("gpu-box", { hostname: "gpu-box", version: "1.4.0", device: "cuda", device_name: "RTX 4090", memory_bytes: 24 * 2 ** 30, online: true, busy: true });
const DOWN = node("old-laptop", { error: "old-laptop did not answer within 2 seconds." });
const LOCKED = node("brother-pc", { error: "brother-pc answered 401 to GET /node: This node does not know the caller." });

/** The server: the list it keeps, what each address answers, and every PUT body it was sent. */
async function setup(page: Page, addresses: string[], known: Node[] = [GPU, DOWN, LOCKED], me = ME) {
  await fakeEvents(page);
  const state = { addresses: [...addresses], puts: [] as string[][] };
  await mockApi(page, {
    "GET /cluster/nodes": () => ({ body: [me, ...state.addresses.map((a) => known.find((n) => n.address === a) ?? node(a, { error: `${a} refused the connection.` }))] }),
    "PUT /cluster/nodes": (route) => {
      const { addresses: sent } = route.request().postDataJSON() as { addresses: string[] };
      state.puts.push(sent);
      const bad = sent.find((a) => a.includes(" "));
      if (bad) return { status: 400, body: { detail: `"${bad}" is not a machine address: it has spaces.` } };
      state.addresses = sent;
      return { body: { addresses: sent } };
    },
  });
  await page.goto("/settings/");
  return state;
}

test("lists this computer and every listed machine, with the reason one does not answer", async ({ page }) => {
  await setup(page, ["gpu-box", "old-laptop"]);
  const list = page.getByTestId("machines");
  await expect(list.getByTestId("machine-local")).toContainText("mac-mini · This computer");
  await expect(list.getByTestId("machine-local")).toContainText("Apple M2 (mps, 16 GB) · version 1.4.0");
  await expect(list.getByTestId("machine-local")).not.toContainText("Remove");
  await expect(list.getByTestId("machine-gpu-box")).toContainText("RTX 4090 (cuda, 24 GB)");
  await expect(list.getByTestId("machine-gpu-box")).toContainText("Online");
  await expect(list.getByTestId("machine-gpu-box")).toContainText("Busy");
  await expect(list.getByTestId("machine-old-laptop")).toHaveAttribute("data-online", "false");
  await expect(list.getByTestId("machine-error-old-laptop")).toHaveText("old-laptop did not answer within 2 seconds.");
  await expect(page.getByTestId("machines-refused")).toHaveCount(0);
});

test("shows this computer's tailnet name and copies it", async ({ page, context }) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  await setup(page, []);
  await expect(page.getByTestId("tailnet-name")).toHaveText("mac-mini.tail1234.ts.net");
  await page.getByRole("button", { name: "Copy" }).click();
  await expect(page.getByText("Copied this computer's name.")).toBeVisible();
  expect(await page.evaluate(() => navigator.clipboard.readText())).toBe("mac-mini.tail1234.ts.net");
});

test("says so when Tailscale has not named this computer", async ({ page }) => {
  await setup(page, [], undefined, { ...ME, tailnet_name: "" });
  await expect(page.getByTestId("tailnet-name-missing")).toContainText("MagicDNS");
  await expect(page.getByRole("button", { name: "Copy" })).toHaveCount(0);
});

test("adds and removes machines, and saves the list", async ({ page }) => {
  const server = await setup(page, ["gpu-box", "old-laptop"]);
  const save = page.getByRole("button", { name: "Save" });
  await expect(page.getByTestId("machine-gpu-box")).toBeVisible();
  await expect(save).toBeDisabled();

  await page.getByRole("button", { name: "Remove old-laptop" }).click();
  await expect(page.getByTestId("machine-old-laptop")).toHaveCount(0);
  await page.getByLabel("Machine to add").fill("  windows-box ");
  await page.getByRole("button", { name: "Add" }).click();
  await expect(page.getByTestId("machine-windows-box")).toContainText("Not saved yet");
  await page.getByLabel("Machine to add").fill("gpu-box");
  await page.getByRole("button", { name: "Add" }).click();
  await expect(page.getByTestId("machines-problem")).toHaveText("That machine is already in the list.");

  await save.click();
  await expect(page.getByText("Saved the list of machines.")).toBeVisible();
  expect(server.puts).toEqual([["gpu-box", "windows-box"]]);
  await expect(page.getByTestId("machine-error-windows-box")).toHaveText("windows-box refused the connection.");
  await expect(save).toBeDisabled();
});

test("a refused list says why inline and saves nothing", async ({ page }) => {
  const server = await setup(page, ["gpu-box"]);
  await page.getByLabel("Machine to add").fill("my box");
  await page.getByRole("button", { name: "Add" }).click();
  await page.getByRole("button", { name: "Save" }).click();
  await expect(page.getByTestId("machines-problem")).toHaveText('"my box" is not a machine address: it has spaces.');
  expect(server.addresses).toEqual(["gpu-box"]);
  await page.getByRole("button", { name: "Undo changes" }).click();
  await expect(page.getByTestId("machine-my box")).toHaveCount(0);
  await expect(page.getByTestId("machines-problem")).toHaveCount(0);
});

test("a machine that answers 401 brings the tag:trainer and allowUsers hint", async ({ page }) => {
  await setup(page, ["gpu-box", "brother-pc"]);
  const hint = page.getByTestId("machines-refused");
  await expect(hint).toContainText("tag:trainer");
  await expect(hint).toContainText("studioConfig.cluster.allowUsers");
  await expect(page.getByTestId("machine-error-brother-pc")).toContainText("answered 401");
});

test("machines logic: device text, 401, cleaning and changes", () => {
  expect(deviceText(ME)).toBe("Apple M2 (mps, 16 GB)");
  expect(deviceText(node("x", { device: "cpu" }))).toBe("cpu");
  expect(deviceText(DOWN)).toBe("");
  expect([GPU, DOWN, LOCKED].map(refusedUs)).toEqual([false, false, true]);
  expect(cleanList([" a ", "", "b", "a", "A"])).toEqual(["a", "b", "A"]);
  expect(changed(["a", "b"], ["a", "b"])).toBe(false);
  expect(changed(["b", "a"], ["a", "b"])).toBe(true);
});
