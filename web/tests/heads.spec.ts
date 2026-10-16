import { expect, test, type Page } from "@playwright/test";

// Head lab and Inside a head against a mocked API: every job-starting route answers with a finished job.

const T = 5;
const D = 4;
const ids = [1, 7, 8, 7, 8];
const tokens = ids.map((i) => `t${i}`);
const version = { number: 3, name: "v3", change: "", status: "trained", best_checkpoint: "runs/v3/best.npz", spec: {}, scorecard: {}, verdict: "", prediction: {} };

/** Forward hooks for `blocks` x `heads` heads: scores differ per head, logits = scores / sqrt(d), pattern = causal softmax. */
function forwardResult(blocks: number, heads: number) {
  const hooks: Record<string, unknown> = { embed: [Array.from({ length: T }, () => new Array(16).fill(0))] };
  for (let b = 0; b < blocks; b++) {
    const scores = [Array.from({ length: heads }, (_, h) => Array.from({ length: T }, (_, q) => Array.from({ length: T }, (_, k) => Math.sin(1 + b + 2 * h + q * 3 + k))))];
    const logits = scores.map((x) => x.map((m) => m.map((r) => r.map((v) => v / Math.sqrt(D)))));
    const pattern = logits.map((x) =>
      x.map((m) =>
        m.map((row, q) => {
          const e = row.map((v, k) => (k <= q ? Math.exp(v) : 0));
          const sum = e.reduce((a, c) => a + c, 0);
          return e.map((v) => v / sum);
        }),
      ),
    );
    const vec = [Array.from({ length: heads }, () => Array.from({ length: T }, () => new Array(D).fill(0.1)))];
    Object.assign(hooks, { [`block${b}.attn.scores`]: scores, [`block${b}.attn.logits`]: logits, [`block${b}.attn.pattern`]: pattern, [`block${b}.attn.q`]: vec, [`block${b}.attn.v`]: vec });
  }
  return { ids: [ids], tokens: [tokens], logits: [], hooks };
}

const job = (result: unknown) => ({ id: "j1", kind: "x", status: "done", progress: "", result, error: "", created_at: 0 });

async function mockApi(page: Page, options: { trained?: boolean } = {}) {
  const calls: Record<string, unknown[]> = {};
  const answer = (path: string, make: (body: { off?: unknown[] }) => unknown) =>
    page.route(`**${path}`, (route) => {
      const body = route.request().postDataJSON();
      (calls[path] ??= []).push(body);
      return route.fulfill({ status: path === "/inspect/head-stats" ? 200 : 202, json: make(body) });
    });
  await answer("/versions/best", () => job(options.trained === false ? { ...version, best_checkpoint: null } : version));
  await page.route("**/versions/current", (route) => route.fulfill({ json: options.trained === false ? { ...version, best_checkpoint: null } : version }));
  await answer("/inspect/prepared", () => job({ ids, tokens }));
  await answer("/inspect/forward", () => job(forwardResult(2, 3)));
  await answer("/inspect/reply", (body) => job(body.off?.length ? "off-reply" : "on-reply"));
  await answer("/inspect/heads-off", () => job({ before: 2, after: 2.5 }));
  await answer("/inspect/rank-heads", () =>
    job(
      [
        { block: 1, head: 2, rise: 0.4 },
        { block: 0, head: 1, rise: 0.2 },
        { block: 0, head: 0, rise: 0.1 },
        { block: 0, head: 2, rise: 0 },
        { block: 1, head: 0, rise: 0 },
        { block: 1, head: 1, rise: 0 },
      ],
    ),
  );
  await answer("/inspect/head-stats", () => ({ entropy: [1.11, 1.22, 1.33], mean_distance: [0.5, 0.6, 0.7], first_token_mass: [0.1, 0.2, 0.3], previous_token_mass: [0.4, 0.5, 0.6] }));
  return calls;
}

test("Head lab: shows the heads, switches one off, and ranks them", async ({ page }) => {
  const calls = await mockApi(page);
  await page.goto("/head-lab/");
  await expect(page.getByRole("heading", { level: 1, name: "Head lab" })).toBeVisible();
  await expect(page.getByText("Experiments on version 3 · checkpoint best.npz.")).toBeVisible();
  await expect(page.getByText("6 HEADS · CLICK ONE")).toBeVisible();
  expect(calls["/inspect/forward"]).toEqual([{ version: 3, ids, hooks: true }]);

  await page.getByRole("button", { name: "Block 2 head 3" }).click();
  await expect(page.getByRole("heading", { name: "Block 2 · head 3" })).toBeVisible();
  await page.getByRole("button", { name: "Show the three tests" }).click();
  await expect(page.getByRole("group", { name: "The three pattern tests" })).toContainText("Previous word");
  await expect(page.getByRole("link", { name: /Step through one head/ })).toHaveAttribute("href", "/head-zoom/?block=1&head=2");

  await page.getByRole("button", { name: "Switch it off" }).click();
  await expect(page.getByText("switched off", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("2.00 → 2.50 bits/char")).toBeVisible();
  await expect(page.getByText("on-reply")).toBeVisible();
  await expect(page.getByText("off-reply")).toBeVisible();
  expect(calls["/inspect/heads-off"]).toEqual([{ version: 3, off: [[1, 2]], mode: "mean" }]);

  await page.getByRole("radio", { name: "Zero (harsher)" }).click();
  await expect.poll(() => calls["/inspect/heads-off"].length).toBe(2);
  expect(calls["/inspect/heads-off"][1]).toEqual({ version: 3, off: [[1, 2]], mode: "zero" });

  await page.getByRole("button", { name: "Rank heads by importance" }).click();
  await expect(page.getByRole("button", { name: "Hide importance" })).toBeVisible();
  await expect(page.getByText("Block 1 · head 1 error +0.10 bits/char")).toBeVisible();
  expect(calls["/inspect/rank-heads"]).toEqual([{ version: 3, mode: "zero" }]);

  await page.getByRole("button", { name: "Switch all back on" }).click();
  await expect(page.getByText("Switch a head off to see how the answer")).toBeVisible();
});

test("Head lab: without a trained checkpoint it says so", async ({ page }) => {
  await mockApi(page, { trained: false });
  await page.goto("/head-lab/");
  await expect(page.getByText("No trained checkpoint yet")).toBeVisible();
});

test("Inside a head: a link lands on its head, the steps change the grid, and the head's stats show", async ({ page }) => {
  const calls = await mockApi(page);
  await page.goto("/head-zoom/?block=1&head=2");
  await expect(page.getByRole("heading", { level: 1, name: "Inside a head" })).toBeVisible();
  await expect(page.getByText("Block 2 · head 3 · 5 tokens")).toBeVisible();
  await expect(page.getByText(/differ from the model/)).toHaveCount(0); // the derived softmax matches the model's pattern
  await expect(page.getByRole("gridcell", { name: /^k1 q1: -?\d\.\d$/ })).toBeVisible();

  await page.getByRole("button", { name: "3. Causal mask" }).click();
  await expect(page.getByRole("gridcell", { name: "k5 q1: −∞" })).toBeVisible();
  await page.getByRole("button", { name: "5. Weighted sum of values" }).click();
  await expect(page.getByRole("list", { name: "Biggest contributors" }).getByRole("listitem")).toHaveCount(5);
  await expect(page.getByRole("link", { name: /Where V and W_O meet/ })).toBeVisible();

  await page.getByRole("gridcell", { name: /^k2 q4/ }).click();
  await expect(page.getByText("k2 “t7” · q4 “t7”")).toBeVisible();

  await expect(page.getByRole("term").filter({ hasText: "Entropy" })).toBeVisible();
  await expect(page.getByText("1.33 nats")).toBeVisible();
  expect((calls["/inspect/head-stats"][0] as { pattern: number[][][][] }).pattern[0]).toHaveLength(3);

  await page.getByRole("radio", { name: "Head 1", exact: true }).click();
  await expect(page.getByText("1.11 nats")).toBeVisible();
  await expect(page).toHaveURL(/block=1&head=0/);
});

test("Inside a head: a block or head the model does not have falls back to the first", async ({ page }) => {
  await mockApi(page);
  await page.goto("/head-zoom/?block=9&head=9");
  await expect(page.getByText("Block 1 · head 1 · 5 tokens")).toBeVisible();
});
