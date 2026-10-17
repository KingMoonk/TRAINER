import { mockApi } from "@/test/network";

// A mocked inspection API shared by the Head lab and Inside a head tests: every job-starting route answers with a finished job.

const T = 5;
const D = 4;
export const ids = [1, 7, 8, 7, 8];
export const tokens = ids.map((i) => `t${i}`);
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

export function mockInspect(options: { trained?: boolean } = {}) {
  const calls = mockApi({
    "POST /versions/best": () => ({ status: 202, body: job(options.trained === false ? { ...version, best_checkpoint: null } : version) }),
    "GET /versions": { body: [version, { ...version, number: 7, name: "v7" }] },
    "GET /versions/current": { body: options.trained === false ? { ...version, best_checkpoint: null } : version },
    "POST /inspect/prepared": { status: 202, body: job({ ids, tokens }) },
    "POST /inspect/forward": { status: 202, body: job(forwardResult(2, 3)) },
    "POST /inspect/reply": ({ body }) => ({ status: 202, body: job((body as { off?: unknown[] }).off?.length ? "off-reply" : "on-reply") }),
    "POST /inspect/heads-off": { status: 202, body: job({ before: 2, after: 2.5 }) },
    "POST /inspect/rank-heads": {
      status: 202,
      body: job([
        { block: 1, head: 2, rise: 0.4 },
        { block: 0, head: 1, rise: 0.2 },
        { block: 0, head: 0, rise: 0.1 },
        { block: 0, head: 2, rise: 0 },
        { block: 1, head: 0, rise: 0 },
        { block: 1, head: 1, rise: 0 },
      ]),
    },
    "POST /inspect/head-stats": { body: { entropy: [1.11, 1.22, 1.33], mean_distance: [0.5, 0.6, 0.7], first_token_mass: [0.1, 0.2, 0.3], previous_token_mass: [0.4, 0.5, 0.6] } },
  });
  /** The request bodies sent to one route, in order. */
  const bodies = (method: string, path: string) => calls.filter((call) => call.method === method && call.path === path).map((call) => call.body);
  return { calls, bodies };
}
