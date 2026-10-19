import { spawn, type ChildProcess } from "node:child_process";
import path from "node:path";
import { expect, test } from "@playwright/test";

// Two real --fake servers, no intercepted responses: the primary (the "live" project's server, port 4174) lists a second one
// (started here, port 4175) as its other machine. Both are on this computer, so they share a hostname; the second is told apart
// by its address, which is what the Queue page pins a job to and what its name then carries. Versions cannot be covered here
// (FakeServices has no version store to publish to), so merged versions are covered against a mocked API in cluster-queue.spec.ts.

const PEER_PORT = 4175;
const PEER = `127.0.0.1:${PEER_PORT}`;
const peerUrl = `http://${PEER}`;
let peer: ChildProcess | undefined;

async function answers(url: string): Promise<boolean> {
  try {
    return (await fetch(url)).ok;
  } catch {
    return false;
  }
}

test.beforeAll(async () => {
  test.setTimeout(90_000);
  peer = spawn(process.env.API_PYTHON ?? "python", ["-m", "studio_api", "--fake", "--fake-step-seconds", "0.25", "--port", String(PEER_PORT)], {
    cwd: path.resolve(__dirname, "..", ".."), // the repository root, where studio_api lives
    stdio: "ignore",
  });
  const deadline = Date.now() + 60_000;
  while (!(await answers(`${peerUrl}/node`))) {
    if (Date.now() > deadline) throw new Error("the second --fake server did not start");
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
});

test.afterAll(() => {
  peer?.kill();
});

test("jobs of both machines appear with their node, a pin moves a row, and a stopped machine is greyed", async ({ page, request }) => {
  test.setTimeout(90_000);
  const created: string[] = [];
  const enqueue = async (api: typeof request, base: string, label: string) => {
    const made = await api.post(`${base}/queue/jobs`, { data: { kind: "pretrain", label, params: {} } });
    expect(made.status()).toBe(201);
    return (await made.json()).id as string;
  };

  // Hold both queues so the jobs stay waiting.
  expect((await request.post("/queue/pause")).ok()).toBeTruthy();
  expect((await request.post(`${peerUrl}/queue/pause`)).ok()).toBeTruthy();
  created.push(await enqueue(request, "", "Cluster live local A"), await enqueue(request, "", "Cluster live local B"));
  const peerJob = await enqueue(request, peerUrl, "Cluster live peer job");
  const hostname = (await (await request.get(`${peerUrl}/node`)).json()).hostname as string;
  const peerName = `${hostname} (${PEER})`; // the primary has the same hostname, so the address is added

  try {
    // A cluster of one: nothing about machines anywhere.
    expect((await request.put("/cluster/nodes", { data: { addresses: [] } })).ok()).toBeTruthy();
    await page.goto("/queue/");
    await expect(page.getByText("Cluster live local A")).toBeVisible();
    await expect(page.getByTestId("other-machines")).toHaveCount(0);
    await expect(page.getByRole("combobox")).toHaveCount(0);
    await expect(page.getByRole("status", { name: "Status" })).not.toContainText("Machines");

    // List the second server: its job appears under the other machines, every local row says where it runs.
    expect((await request.put("/cluster/nodes", { data: { addresses: [PEER] } })).ok()).toBeTruthy();
    await page.reload();
    const other = page.getByTestId("other-machines");
    await expect(other.getByTestId(`other-${peerJob}`)).toContainText("Cluster live peer job");
    await expect(other.getByTestId(`other-${peerJob}`)).toContainText("Runs on any machine that fits it");
    await expect(other.getByTestId(`node-${PEER}`)).toContainText(peerName);
    await expect(page.getByTestId(`place-${created[0]}`)).toHaveText("Runs on any machine that fits it");
    await expect(page.getByRole("status", { name: "Status" })).toContainText("Machines");
    await expect(page.getByRole("status", { name: "Status" })).toContainText("1 job on other machines");

    // Pin the first local job to the other machine: its row now says it runs there.
    const select = page.getByRole("combobox", { name: "Run on: Cluster live local A" });
    await select.selectOption(PEER);
    await expect(page.getByTestId(`place-${created[0]}`)).toHaveText(`Pinned to ${peerName}: it runs there or waits`);
    await expect(select).toHaveValue(PEER);
    await expect(page.getByTestId(`place-${created[1]}`)).toHaveText("Runs on any machine that fits it");
    // The pin is kept by the server, not just the page.
    await page.reload();
    await expect(page.getByTestId(`place-${created[0]}`)).toHaveText(`Pinned to ${peerName}: it runs there or waits`);

    // Stop the other machine: its rows stay, greyed and labelled, and the page says it is unreachable.
    peer?.kill();
    await expect(other.getByTestId(`other-${peerJob}`)).toHaveAttribute("data-stale", "true", { timeout: 30_000 });
    await expect(other.getByTestId(`other-${peerJob}`)).toContainText("Cluster live peer job");
    await expect(other.getByTestId(`other-${peerJob}`)).toContainText("last known");
    await expect(other.getByRole("status")).toContainText("is unreachable");
    await expect(other.getByTestId(`node-${PEER}`)).toHaveClass(/opacity-60/);
  } finally {
    await request.put("/cluster/nodes", { data: { addresses: [] } });
    for (const id of created) await request.delete(`/queue/jobs/${id}`);
    await request.post("/queue/resume");
  }
});
