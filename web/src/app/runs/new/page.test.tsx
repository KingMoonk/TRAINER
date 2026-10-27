import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { json, mockApi, type Call, type Handler } from "@/test/network";
import { nav, resetNav } from "@/test/navigation";
import { renderPage } from "@/test/render";
import NewRunPage from "./page";

const node = (address: string, over: Record<string, unknown> = {}) => ({
  address, busy: false, device: "cuda", device_name: "RTX 4060", dtypes: [], error: "", hostname: address, local: false, memory_bytes: 8 * 2 ** 30, online: true, tailnet_name: "", version: "", ...over,
});
const NODES = [node("local", { local: true, hostname: "mac", device: "mps", device_name: "Apple silicon" }), node("beast"), node("lab", { online: false, error: "timed out" })];
const MORE = [...NODES.slice(0, 2), node("lab")];

const PLAN = { hours: "any", start_hour: 22, end_hour: 7, memory: "most", idle_only: true, idle_after_s: 60, pause_on_battery: true, keep_awake: false, chat_follows_best: true };
const job = { kind: "pretrain", label: "Version 2 from scratch", params: { version: 2, datasets: ["stories"], mode: "from_scratch" } };

const row = (over: Record<string, unknown>) => ({
  id: "j1", kind: "post_training", label: "Post-training round 2", status: "running", node: "beast", hostname: "beast", queued_on: "beast", queued_here: false, queued_address: "beast",
  pinned: "", waiting_reason: "", stale: false, note: "", step: 10, total: 100, eta_s: 4.7 * 86_400, ...over,
});

const source = (version: number, over: Record<string, unknown> = {}) => ({
  version, name: "", change: "", status: "completed", finished_at: "2026-10-01T11:30:00", steps: 1200, steps_planned: 1500, seconds: 5400, tokens: 4_900_000,
  params: 812_000, best_val_loss: 2.25, train_loss: 2.5, device: "RTX 4090", rounds: null, inner_steps: null, same_design: true, parent: false, ...over,
});

function server({ outlook = "Starts after Version 1 (about 4 min)", nodes = NODES, failRun = false, jobs = [] as unknown[], sources = [] as unknown[], blocked = null as string | null, sizing = false, scaling = null as unknown, sized = null as unknown } = {}) {
  const calls = mockApi({
    "GET /cluster/merged": { body: { nodes: [], jobs } },
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
    "GET /queue": { body: { job: null, step: 0, total: 0, phase: "", lr: null, eta_s: null, paused: false, blocked: null, stopping: false, gate: { reason: null, supported: true }, up_next: [], plan: PLAN } },
    "POST /pretraining/plan": ({ body }) => {
      const { mode, from_version } = body as { mode: string; from_version: number | null };
      if (sizing) return { body: { can_add: false, reason: "The size of the text is still being worked out.", job: null, version: 2, mode: "from_scratch", outlook: "", pieces_per_step: 1, context: 256, estimating: true, can_continue: false, sources: [], continue_from: null, continue_blocked: null } };
      const from = sources.length ? (from_version ?? (sources[0] as { version: number }).version) : null;
      return { body: { can_add: true, reason: "", job: { ...job, params: { ...job.params, mode } }, mode: "from_scratch", outlook: "", pieces_per_step: 1, context: 256, estimating: false,
        can_continue: sources.length > 0, sources, continue_from: from, continue_blocked: sources.length ? null : blocked, scaling } };
    },
    "POST /pretraining/start": { status: 202, body: { id: "job-5", label: "Version 2, once the size of its text is known" } },
    "POST /distributed/sizing": { body: sized },
    "POST /distributed/runs": failRun ? { status: 400, body: { detail: "Rounds must be at most 1000." } } : { body: { id: "run-3" } },
    "POST /post-training/stages": { status: 202, body: { id: "job-8", label: "SFT", stage: "sft" } },
    "POST /quicktest/run": { body: { job_id: "job-9", outlook: "x", started_at: 1 } },
    "POST /queue/jobs/*/pin": { body: {} },
  });
  const sent = (method: string, path: string): Call | undefined => calls.filter((c) => c.method === method && c.path === path).at(-1);
  return { calls, sent };
}

const cta = () => screen.findByRole("button", { name: /^(Add to queue|Start)$/ });

beforeEach(() => resetNav("/runs/new/"));

describe("New run", () => {
  test("one machine: no outer loop, the run is queued through /pretraining/start, and the page goes to the run", async () => {
    const { sent, calls } = server();
    const user = userEvent.setup();
    renderPage(<NewRunPage />, { route: "/runs/new/" });
    expect(await screen.findByRole("heading", { level: 1, name: "New run" })).toBeVisible();
    expect(screen.queryByTestId("outer-loop")).toBeNull();
    expect(await screen.findByRole("status")).toHaveTextContent("Local");
    expect(screen.getByTestId("coordinator")).toHaveTextContent("This computer");
    await waitFor(() => expect(screen.getByTestId("outlook")).toHaveTextContent("Starts after Version 1 (about 4 min)."));
    await user.click(screen.getByRole("button", { name: "Allowed hours" }));
    expect(await screen.findByTestId("plan-card")).toBeInTheDocument();
    await user.keyboard("{Escape}");
    await waitFor(async () => expect(await cta()).toBeEnabled());
    await user.click(await cta());
    await waitFor(() => expect(nav.push).toHaveBeenCalledWith("/runs/view/?id=job-5"));
    expect(sent("POST", "/pretraining/plan")?.body).toMatchObject({ dataset_ids: ["stories"], mode: "from_scratch", version: null });
    expect(sent("POST", "/pretraining/start")?.body).toMatchObject({ dataset_ids: ["stories"], mode: "from_scratch", version: null });
    expect(calls.some((c) => c.method === "POST" && c.path === "/queue/jobs")).toBe(false);
  });

  test("a run the plan scaled to this computer lists every change before Start", async () => {
    const hardware = { device: "cuda", gpu_name: "Test GPU", vram_total: 8, vram_free: 7, ram_total: 32, ram_free: 24, disk_free: 300, cpu_cores: 8, tokens_per_second: null, measured_at: null };
    const notes = [{ kind: "batch", reason: "vram", text: "Batch 16 -> 3 x 6 accumulation to fit the 7.0 GB of GPU memory free.", advisory: false }];
    server({ scaling: { notes, scaled: true, can_keep_mine: false, kept_mine: false, window_hours: 72, hours: null, token_budget: null, offload: false, blocker: null, hardware } });
    renderPage(<NewRunPage />, { route: "/runs/new/" });
    expect(await screen.findByTestId("scaling-notes")).toHaveTextContent("Batch 16 -> 3 x 6 accumulation");
    expect(screen.getByTestId("scaling-notes")).toHaveTextContent("The run was scaled to fit this computer:");
  });

  test("Continue lists the earlier runs with what they did, and the one picked is the one the plan starts from", async () => {
    const { sent } = server({ sources: [source(3, { name: "wide", same_design: false, best_val_loss: 1.75, rounds: 5, inner_steps: 10, seconds: null }), source(1, { name: "tiny", parent: true })] });
    const user = userEvent.setup();
    renderPage(<NewRunPage />, { route: "/runs/new/" });
    await waitFor(async () => expect(await cta()).toBeEnabled());
    expect(screen.queryByText("Earlier training to continue from")).toBeNull();
    await user.selectOptions(screen.getByLabelText(/Start from/), "catch_up");

    const tiny = await screen.findByTestId("source-1");
    expect(tiny).toHaveTextContent("Version 1 · tiny · saved from it");
    expect(tiny).toHaveTextContent("Finished 1 Oct 2026");
    for (const fact of ["1,200 of 1,500 steps", "1 h 30 min", "Held-out loss 2.250", "Train loss 2.500", "812K params", "4.9M word pieces", "RTX 4090", "Same design"])
      expect(tiny).toHaveTextContent(fact);
    const wide = screen.getByTestId("source-3");
    expect(wide).toHaveTextContent("5 rounds × H 10");
    expect(wide).toHaveTextContent("Different design: only matching weights are copied");
    expect(within(wide).getByRole("radio")).toBeChecked();

    await user.click(within(tiny).getByRole("radio"));
    await waitFor(() => expect(within(screen.getByTestId("source-1")).getByRole("radio")).toBeChecked());
    expect(sent("POST", "/pretraining/plan")?.body).toMatchObject({ mode: "catch_up", from_version: 1 });
  });

  test("Continue with nothing to continue from says why", async () => {
    server({ blocked: "word_list_changed" });
    const user = userEvent.setup();
    renderPage(<NewRunPage />, { route: "/runs/new/" });
    await waitFor(async () => expect(await cta()).toBeEnabled());
    await user.selectOptions(screen.getByLabelText(/Start from/), "catch_up");
    expect(await screen.findByText(/this version changes the word list/)).toBeVisible();
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
    expect(screen.getByTestId("worker-lab")).toHaveTextContent("Unreachable");
  });

  test("while the size of the text is worked out Start is on: the run goes through /pretraining/start, is pinned and its page opens", async () => {
    const { sent, calls } = server({ nodes: MORE, sizing: true });
    const user = userEvent.setup();
    renderPage(<NewRunPage />, { route: "/runs/new/" });
    await user.click(await screen.findByTestId("worker-beast"));
    await user.click(await screen.findByTestId("worker-local"));
    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent("Remote"));
    await waitFor(async () => expect(await cta()).toBeEnabled());
    expect(screen.queryByTestId("blocker")).toBeNull();
    expect(await screen.findByTestId("size-first")).toHaveTextContent("it waits in the queue until the size is known");
    await user.click(await cta());
    await waitFor(() => expect(nav.push).toHaveBeenCalledWith("/runs/view/?id=job-5"));
    expect(sent("POST", "/pretraining/start")?.body).toMatchObject({ dataset_ids: ["stories"], mode: "from_scratch", version: 2 });
    expect(sent("POST", "/queue/jobs/job-5/pin")?.body).toEqual({ pinned: "beast" });
    expect(calls.some((c) => c.method === "POST" && c.path === "/queue/jobs")).toBe(false);
  });

  test("a peer alone gets the run pinned to it", async () => {
    const { sent } = server({ nodes: MORE });
    const user = userEvent.setup();
    renderPage(<NewRunPage />, { route: "/runs/new/" });
    await user.click(await screen.findByTestId("worker-beast"));
    await user.click(await screen.findByTestId("worker-local")); // local off: the peer is the only worker
    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent("Remote"));
    await waitFor(async () => expect(await cta()).toBeEnabled());
    await user.click(await cta());
    await waitFor(() => expect(nav.push).toHaveBeenCalledWith("/runs/view/?id=job-5"));
    expect(sent("POST", "/queue/jobs/job-5/pin")?.body).toEqual({ pinned: "beast" });
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
      version: null, dataset_ids: ["stories"], machines: ["beast"], rounds: 12, inner_steps: 100, outer_lr: 0.7, outer_momentum: 0.9, overlap: null, time_box: null, delta_dtype: "int8", delta_sparsity: 0, reviewed_only: false,
    });
  });

  test("a busy machine shows what it runs, and launching asks first: cancel sends nothing, launch anyway starts the run", async () => {
    const busy = [...NODES.slice(0, 1), node("beast", { busy: true })];
    const { sent } = server({ nodes: busy, jobs: [row({})] });
    const user = userEvent.setup();
    renderPage(<NewRunPage />, { route: "/runs/new/" });
    await user.click(await screen.findByTestId("worker-beast"));
    await waitFor(() => expect(screen.getByTestId("busy-beast")).toHaveTextContent("Running Post-training round 2 · about 4.7 days left"));
    await user.click(await cta());
    const dialog = await screen.findByRole("alertdialog");
    expect(dialog).toHaveTextContent("beast will not take part yet");
    expect(dialog).toHaveTextContent("beast is busy with another job, so the run starts without it. It will join a later round if it frees up.");
    expect(sent("POST", "/distributed/runs")).toBeUndefined();
    await user.click(within(dialog).getByRole("button", { name: "Cancel" }));
    await waitFor(() => expect(screen.queryByRole("alertdialog")).toBeNull());
    expect(sent("POST", "/distributed/runs")).toBeUndefined();

    await user.click(await cta());
    await user.click(within(await screen.findByRole("alertdialog")).getByRole("button", { name: "Launch anyway" }));
    await waitFor(() => expect(nav.push).toHaveBeenCalledWith("/runs/view/?id=run-3"));
    expect(sent("POST", "/distributed/runs")?.body).toMatchObject({ machines: ["beast"] });
  });

  describe("sizing per member", () => {
    const mem = (name: string, over: Record<string, unknown> = {}) => ({
      name, address: name, sized: true, fits: true, batch_size: 16, accumulation: 1, offload: false, notes: [], refusal: "", ...over,
    });
    const answer = (members: unknown[], over: Record<string, unknown> = {}) => ({
      members, flagged: [], tokens_per_second: 38000, token_budget: null, rounds: 40, rounds_cut: false, note: "", ...over,
    });

    test("each member's batch and notes are shown beside the form", async () => {
      server({ nodes: MORE, sized: answer([mem("mac"), mem("beast", { batch_size: 2, accumulation: 8, notes: ["beast: batch 16 -> 2 x 8 accumulation to fit 5.0 GB of GPU memory."] })]) });
      const user = userEvent.setup();
      renderPage(<NewRunPage />, { route: "/runs/new/" });
      await user.click(await screen.findByTestId("worker-beast"));
      const section = await screen.findByTestId("member-sizing");
      expect(section).toHaveTextContent("mac: batch 16, 1 accumulation steps");
      expect(section).toHaveTextContent("beast: batch 2, 8 accumulation steps");
      expect(section).toHaveTextContent("beast: batch 16 -> 2 x 8 accumulation to fit 5.0 GB of GPU memory.");
    });

    test("a member that cannot fit is named, Start waits, and ticking start-without sends the flagged names as leave_out", async () => {
      const { sent } = server({ nodes: MORE, sized: answer([mem("mac"), mem("beast", { fits: false, refusal: "beast cannot take part: it has 200 MB of GPU memory for this run and the design needs about 9 GB even at one row." })], { flagged: ["beast"] }) });
      const user = userEvent.setup();
      renderPage(<NewRunPage />, { route: "/runs/new/" });
      await user.click(await screen.findByTestId("worker-beast"));
      expect(await screen.findByTestId("member-sizing")).toHaveTextContent("beast cannot take part. it has 200 MB of GPU memory");
      await waitFor(() => expect(screen.getByTestId("blocker")).toHaveTextContent("beast cannot hold the design"));
      expect(await cta()).toBeDisabled();
      await user.click(screen.getByRole("checkbox", { name: "Start without beast" }));
      await waitFor(() => expect(screen.queryByTestId("blocker")).toBeNull());
      await user.click(await cta());
      await waitFor(() => expect(nav.push).toHaveBeenCalledWith("/runs/view/?id=run-3"));
      expect(sent("POST", "/distributed/runs")?.body).toMatchObject({ machines: ["beast"], leave_out: ["beast"] });
    });

    test("when this computer cannot fit, nothing offers to start without it", async () => {
      server({ nodes: MORE, sized: answer([mem("mac", { fits: false, refusal: "mac cannot take part: no room." }), mem("beast")], { flagged: ["mac"] }) });
      const user = userEvent.setup();
      renderPage(<NewRunPage />, { route: "/runs/new/" });
      await user.click(await screen.findByTestId("worker-beast"));
      await waitFor(() => expect(screen.getByTestId("blocker")).toHaveTextContent("This computer cannot hold the design"));
      expect(screen.queryByRole("checkbox", { name: /Start without/ })).toBeNull();
    });

    test("cut rounds are told, and a start with nothing flagged sends no leave_out", async () => {
      const { sent } = server({ nodes: MORE, sized: answer([mem("mac"), mem("beast")], { rounds: 12, rounds_cut: true, note: "Token budget 5B -> 100M to finish in about 1 h at the 38,000 tokens/s the members that fit measure together. Rounds 40 -> 12." }) });
      const user = userEvent.setup();
      renderPage(<NewRunPage />, { route: "/runs/new/" });
      await user.click(await screen.findByTestId("worker-beast"));
      expect(await screen.findByTestId("member-sizing")).toHaveTextContent("Rounds 40 -> 12.");
      await user.click(await cta());
      await waitFor(() => expect(nav.push).toHaveBeenCalledWith("/runs/view/?id=run-3"));
      expect(sent("POST", "/distributed/runs")?.body).not.toHaveProperty("leave_out");
    });
  });

  test("a busy machine with no job row says it runs another job", async () => {
    server({ nodes: [...NODES.slice(0, 1), node("beast", { busy: true })] });
    renderPage(<NewRunPage />, { route: "/runs/new/" });
    expect(await screen.findByTestId("busy-beast")).toHaveTextContent("Running another job");
  });

  test("free machines launch without a question", async () => {
    const { sent } = server({ nodes: MORE });
    const user = userEvent.setup();
    renderPage(<NewRunPage />, { route: "/runs/new/" });
    await user.click(await screen.findByTestId("worker-beast"));
    expect(screen.queryByTestId("busy-beast")).toBeNull();
    await user.click(await cta());
    await waitFor(() => expect(nav.push).toHaveBeenCalledWith("/runs/view/?id=run-3"));
    expect(screen.queryByRole("alertdialog")).toBeNull();
    expect(sent("POST", "/distributed/runs")).toBeDefined();
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

  test.each(["midtrain", "sft"] as const)("%s queues a stage from trained weights", async (stage) => {
    const { sent } = server({ nodes: MORE });
    const user = userEvent.setup();
    renderPage(<NewRunPage />, { route: "/runs/new/" });
    await user.selectOptions(await screen.findByLabelText("Stage"), stage);
    expect(screen.queryByText("Text")).toBeNull();
    expect(screen.getByTestId("worker-beast")).toBeDisabled();
    expect(screen.getByTestId("worker-local")).toHaveAttribute("aria-pressed", "true");
    await user.click(await cta());
    await waitFor(() => expect(nav.push).toHaveBeenCalledWith("/runs/view/?id=job-8"));
    expect(sent("POST", "/post-training/stages")?.body).toMatchObject({ stage, version: 1 });
    expect(sent("POST", "/queue/jobs/job-8/pin")).toBeUndefined();
  });

  test("Ablation queues the quick test with no guess", async () => {
    const { sent } = server();
    const user = userEvent.setup();
    renderPage(<NewRunPage />, { route: "/runs/new/" });
    await user.selectOptions(await screen.findByLabelText("Stage"), "ablation");
    await user.click(await cta());
    await waitFor(() => expect(nav.push).toHaveBeenCalledWith("/runs/view/?id=job-9"));
    expect(sent("POST", "/quicktest/run")?.body).toEqual({ dataset: "stories", minutes: 3, shrink: false });
    expect(sent("POST", "/queue/jobs/job-9/pin")).toBeUndefined();
  });

  test("a failed read says so in its own section and the rest of the page works", async () => {
    mockApi({
      "GET /cluster/nodes": { status: 500, body: { detail: "down" } },
      "GET /distributed/defaults": json({}),
      "GET /pretraining/text_sets": json([{ id: "stories", name: "Short stories", count: 1200, held: false, library: false, on: true }]),
    });
    renderPage(<NewRunPage />, { route: "/runs/new/" });
    expect(await screen.findByRole("alert")).toHaveTextContent("Could not list the machines: down");
    expect(screen.getByRole("heading", { level: 1, name: "New run" })).toBeVisible();
    expect(await screen.findByRole("checkbox", { name: /Short stories/ })).toBeChecked();
    expect(await screen.findByTestId("blocker")).toHaveTextContent("The machines could not be listed.");
  });
});

describe("New run while a read is slow", () => {
  const TEXT = [{ id: "stories", name: "Short stories", count: 1200, held: false, library: false, on: true }];
  // never answers; gives up when the page aborts the request
  const hang: Handler = (_call, request) =>
    new Promise((_resolve, reject) => request.signal.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError"))));

  afterEach(() => vi.useRealTimers());

  function slowServer(text: Handler) {
    return mockApi({
      "GET /cluster/nodes": { body: NODES },
      "GET /distributed/defaults": { body: { delta_dtype: "fp16", inner_steps: 100, outer_lr: 0.7, outer_momentum: 0.9, rounds: 40 } },
      "GET /pretraining/text_sets": text,
      "GET /versions": { body: [] },
      "GET /queue/outlook": { body: { outlook: "Starts now" } },
      "POST /pretraining/plan": { body: { can_add: true, reason: "", job, mode: "from_scratch", outlook: "", pieces_per_step: 1, context: 256, estimating: false, can_continue: true } },
    });
  }

  test("the page shows at once, the slow section says it is loading, and the other sections fill in", async () => {
    slowServer(hang);
    renderPage(<NewRunPage />, { route: "/runs/new/" });
    expect(screen.getByRole("heading", { level: 1, name: "New run" })).toBeVisible();
    expect(screen.getByRole("combobox", { name: /Stage/ })).toBeVisible();
    const loading = await screen.findByTestId("section-loading");
    expect(loading).toHaveTextContent("Reading your text sets...");
    expect(loading).toHaveAttribute("aria-busy", "true");
    expect(await screen.findByTestId("worker-beast")).toBeVisible();
    expect(await screen.findByTestId("blocker")).toHaveTextContent("Reading your text sets...");
    expect(await cta()).toBeDisabled();
  });

  // Fake timers stop findBy/waitFor from settling (see events.test.tsx), so these advance the clock inside act and read at once.
  const advance = (ms: number) => act(() => vi.advanceTimersByTimeAsync(ms));

  test("a read still going after a few seconds says it is still waiting", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    slowServer(hang);
    renderPage(<NewRunPage />, { route: "/runs/new/" });
    await advance(1);
    expect(screen.getByTestId("section-loading")).not.toHaveTextContent("taking longer");
    await advance(7_000);
    expect(within(screen.getByTestId("section-loading")).getByText(/taking longer than usual/)).toBeVisible();
  });

  test("a read that never answers ends in a plain sentence and can be tried again", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    let answer: Handler = hang;
    slowServer((call, request) => answer(call, request));
    renderPage(<NewRunPage />, { route: "/runs/new/" });
    await advance(1);
    expect(screen.getByTestId("section-loading")).toBeVisible();
    await advance(21_000);
    expect(screen.getByRole("alert")).toHaveTextContent("Could not read your text sets: The server did not answer within 20 seconds.");
    answer = () => ({ body: TEXT });
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    await advance(1);
    expect(screen.getByRole("checkbox", { name: /Short stories/ })).toBeChecked();
    expect(screen.queryByRole("alert")).toBeNull();
  });
});
