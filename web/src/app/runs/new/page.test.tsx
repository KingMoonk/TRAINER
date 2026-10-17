import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, test } from "vitest";
import { json, mockApi, type Call } from "@/test/network";
import { nav, resetNav } from "@/test/navigation";
import { renderPage } from "@/test/render";
import NewRunPage from "./page";

const node = (address: string, over: Record<string, unknown> = {}) => ({
  address, busy: false, device: "cuda", device_name: "RTX 4060", dtypes: [], error: "", hostname: address, local: false, memory_bytes: 8 * 2 ** 30, online: true, tailnet_name: "", version: "", ...over,
});
const NODES = [node("local", { local: true, hostname: "mac", device: "mps", device_name: "Apple silicon" }), node("beast"), node("lab", { online: false, error: "timed out" })];
const MORE = [...NODES.slice(0, 2), node("lab")];

const job = { kind: "pretrain", label: "Version 2 from scratch", params: { version: 2, datasets: ["stories"], mode: "from_scratch" } };

function server({ outlook = "Starts after Version 1 (about 4 min)", nodes = NODES, failRun = false } = {}) {
  const calls = mockApi({
    "GET /cluster/nodes": { body: nodes },
    "GET /distributed/defaults": { body: { delta_dtype: "fp16", inner_steps: 100, outer_lr: 0.7, outer_momentum: 0.9, rounds: 40 } },
    "GET /pretraining/text_sets": {
      body: [
        { id: "stories", name: "Short stories", count: 1200, held: false, library: false, on: true },
        { id: "wiki", name: "Wiki", count: 300, held: false, library: true, on: false },
      ],
    },
    "GET /versions": {
      body: [
        { number: 1, name: "tiny", best_checkpoint: "ck1" },
        { number: 2, name: "", best_checkpoint: null },
      ],
    },
    "GET /queue/outlook": { body: { outlook } },
    "POST /pretraining/plan": ({ body }) => ({ body: { can_add: true, reason: "", job: { ...job, params: { ...job.params, mode: (body as { mode: string }).mode } }, mode: "from_scratch", outlook: "", pieces_per_step: 1, context: 256, estimating: false, can_continue: true } }),
    "POST /queue/jobs": { body: { id: "job-7", kind: "pretrain", label: "x", status: "waiting" } },
    "POST /distributed/runs": failRun ? { status: 400, body: { detail: "Rounds must be at most 1000." } } : { body: { id: "run-3" } },
    "POST /post-training/rounds": { body: { id: "job-8", label: "SFT", params: {}, round: 1 } },
    "POST /quicktest/run": { body: { job_id: "job-9", outlook: "x", started_at: 1 } },
    "POST /queue/jobs/*/pin": { body: {} },
  });
  const sent = (method: string, path: string): Call | undefined => calls.filter((c) => c.method === method && c.path === path).at(-1);
  return { calls, sent };
}

const cta = () => screen.findByRole("button", { name: /^(Add to queue|Start)$/ });

beforeEach(() => resetNav("/runs/new/"));

describe("New run", () => {
  test("one machine: no outer loop, the plan's job is queued, and the page goes to the run", async () => {
    const { sent } = server();
    const user = userEvent.setup();
    renderPage(<NewRunPage />, { route: "/runs/new/" });
    expect(await screen.findByRole("heading", { level: 1, name: "New run" })).toBeVisible();
    expect(screen.queryByTestId("outer-loop")).toBeNull();
    expect(await screen.findByRole("status")).toHaveTextContent("Local");
    expect(screen.getByTestId("coordinator")).toHaveTextContent("This computer");
    await waitFor(() => expect(screen.getByTestId("outlook")).toHaveTextContent("Starts after Version 1 (about 4 min)."));
    expect(screen.getByRole("link", { name: "Allowed hours" })).toHaveAttribute("href", "/queue");
    await waitFor(async () => expect(await cta()).toBeEnabled());
    await user.click(await cta());
    await waitFor(() => expect(nav.push).toHaveBeenCalledWith("/runs/view/?id=job-7"));
    expect(sent("POST", "/pretraining/plan")?.body).toMatchObject({ dataset_ids: ["stories"], mode: "from_scratch", version: null });
    expect(sent("POST", "/queue/jobs")?.body).toEqual({ ...job, note: "" });
  });

  test("the button says Start when the queue starts the job now", async () => {
    server({ outlook: "Starts now" });
    renderPage(<NewRunPage />, { route: "/runs/new/" });
    expect(await screen.findByRole("button", { name: "Start" })).toBeVisible();
  });

  test("Pretrain needs a Text set", async () => {
    server();
    const user = userEvent.setup();
    renderPage(<NewRunPage />, { route: "/runs/new/" });
    await user.click(await screen.findByRole("checkbox", { name: /Short stories/ }));
    expect(await cta()).toBeDisabled();
    expect(screen.getByTestId("blocker")).toHaveTextContent("Tick at least one text set.");
  });

  test("a machine that does not answer cannot be picked", async () => {
    server();
    renderPage(<NewRunPage />, { route: "/runs/new/" });
    expect(await screen.findByTestId("worker-lab")).toBeDisabled();
    expect(screen.getByTestId("worker-lab")).toHaveTextContent("Not answering");
  });

  test("a peer alone is pinned with params.pinned", async () => {
    const { sent } = server({ nodes: MORE });
    const user = userEvent.setup();
    renderPage(<NewRunPage />, { route: "/runs/new/" });
    await user.click(await screen.findByTestId("worker-beast"));
    await user.click(await screen.findByTestId("worker-local")); // local off: the peer is the only worker
    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent("Remote"));
    await waitFor(async () => expect(await cta()).toBeEnabled());
    await user.click(await cta());
    await waitFor(() => expect(nav.push).toHaveBeenCalledWith("/runs/view/?id=job-7"));
    expect(sent("POST", "/queue/jobs")?.body).toMatchObject({ params: { pinned: "beast" } });
  });

  test("several machines show the outer loop and start a DiLoCo run", async () => {
    const { sent } = server({ nodes: MORE });
    const user = userEvent.setup();
    renderPage(<NewRunPage />, { route: "/runs/new/" });
    await user.click(await screen.findByTestId("worker-beast"));
    expect(screen.getByRole("status")).toHaveTextContent("DiLoCo · |W| = 2 · c = this computer");
    expect(screen.getByLabelText(/H · inner steps/)).toHaveValue(100);
    expect(screen.getByLabelText(/R · rounds/)).toHaveValue(40);
    expect(screen.getByLabelText(/η_outer/)).toHaveValue(0.7);
    expect(screen.getByLabelText(/μ/)).toHaveValue(0.9);
    expect(screen.getByLabelText(/Δ dtype/)).toHaveValue("fp16");
    expect(screen.getByLabelText(/Δ dtype/)).toHaveAttribute("readonly");
    expect(screen.queryByText(/Straggler/)).toBeNull();
    expect(screen.getByLabelText(/Start from/)).toBeDisabled();

    const rounds = screen.getByLabelText(/R · rounds/);
    await user.clear(rounds);
    await user.type(rounds, "12");
    await user.click(await cta());
    await waitFor(() => expect(nav.push).toHaveBeenCalledWith("/runs/view/?id=run-3"));
    expect(sent("POST", "/distributed/runs")?.body).toEqual({
      version: null, dataset_ids: ["stories"], machines: ["beast"], rounds: 12, inner_steps: 100, outer_lr: 0.7, outer_momentum: 0.9,
    });
  });

  test("the server's sentence for a refused run is shown and the page stays", async () => {
    server({ nodes: MORE, failRun: true });
    const user = userEvent.setup();
    renderPage(<NewRunPage />, { route: "/runs/new/" });
    await user.click(await screen.findByTestId("worker-beast"));
    await user.click(await cta());
    expect(await screen.findByRole("alert")).toHaveTextContent("Could not start the run: Rounds must be at most 1000.");
    expect(nav.push).not.toHaveBeenCalled();
  });

  test("SFT queues the first post-training round on the chosen machine", async () => {
    const { sent } = server({ nodes: MORE });
    const user = userEvent.setup();
    renderPage(<NewRunPage />, { route: "/runs/new/" });
    await user.selectOptions(await screen.findByLabelText("Stage"), "sft");
    expect(screen.queryByText("Text")).toBeNull();
    await user.click(screen.getByTestId("worker-beast")); // one machine: replaces this computer
    expect(screen.getByTestId("worker-local")).toHaveAttribute("aria-pressed", "false");
    await user.click(await cta());
    await waitFor(() => expect(nav.push).toHaveBeenCalledWith("/runs/view/?id=job-8"));
    expect(sent("POST", "/post-training/rounds")?.body).toMatchObject({ round: 1, version: 1 });
    expect(sent("POST", "/queue/jobs/job-8/pin")?.body).toEqual({ pinned: "beast" });
  });

  test("Ablation queues the quick test once a guess is made", async () => {
    const { sent } = server();
    const user = userEvent.setup();
    renderPage(<NewRunPage />, { route: "/runs/new/" });
    await user.selectOptions(await screen.findByLabelText("Stage"), "ablation");
    expect(await cta()).toBeDisabled();
    await user.click(screen.getByRole("radio", { name: "Lower" }));
    await user.click(await cta());
    await waitFor(() => expect(nav.push).toHaveBeenCalledWith("/runs/view/?id=job-9"));
    expect(sent("POST", "/quicktest/run")?.body).toEqual({ dataset: "stories", minutes: 3, shrink: false, guess: "Lower" });
    expect(sent("POST", "/queue/jobs/job-9/pin")).toBeUndefined();
  });

  test("a failed read says so", async () => {
    mockApi({ "GET /cluster/nodes": { status: 500, body: { detail: "down" } }, "GET /distributed/defaults": json({}), "GET /pretraining/text_sets": json([]) });
    renderPage(<NewRunPage />, { route: "/runs/new/" });
    expect(await screen.findByRole("alert")).toHaveTextContent("Could not read the settings for a run");
  });
});
