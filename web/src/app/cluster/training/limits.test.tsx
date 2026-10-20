import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test } from "vitest";
import { mockApi } from "@/test/network";
import { renderPage } from "@/test/render";
import TrainingPage from "./page";

// The start form says what each chosen machine's queue plan allows a member job, and where to change it.

const DEFAULTS = { rounds: 10, inner_steps: 100, remote_inner_steps: 1500, outer_lr: 0.7, outer_momentum: 0.9, delta_dtype: "float16", max_steps: 1_000_000 };
const SETS = [{ id: "stories", name: "Stories", count: 1200, on: true, held: false, library: false }];
const node = (address: string, over: Record<string, unknown> = {}) => ({
  address, local: false, hostname: address, version: "1", device: "cuda", device_name: "RTX 4090", memory_bytes: 0, busy: false, dtypes: [], online: true, error: "", tailnet_name: "", ...over,
});
const PLAN = { hours: "any", start_hour: 22, end_hour: 7, memory: "most", idle_only: false, idle_after_s: 60, pause_on_battery: true, keep_awake: false, chat_follows_best: true };
const queueNode = (address: string, limits: unknown) => ({
  address, hostname: address, pin: address, local: false, online: true, error: "", seen_at: 0, origin: "", age_s: null, origin_down: false, listed: true, limits,
});

function server() {
  mockApi({
    "GET /distributed/defaults": { body: DEFAULTS },
    "GET /distributed/runs": () => ({ body: [] }),
    "GET /pretraining/text_sets": { body: SETS },
    "GET /versions": { body: [] },
    "GET /cluster/nodes": { body: [node("local", { local: true, hostname: "mac-mini" }), node("laptop"), node("BEAST"), node("mystery")] },
    "GET /cluster/merged": {
      body: {
        jobs: [],
        nodes: [
          queueNode("laptop", { plan: { ...PLAN, memory: "half", keep_awake: true }, blocked: "on battery", paused: false }),
          queueNode("BEAST", { plan: { ...PLAN, memory: "all", pause_on_battery: false, idle_only: true, hours: "set", start_hour: 1, end_hour: 6 }, blocked: null, paused: false }),
          queueNode("mystery", null),
        ],
      },
    },
  });
}

describe("the limits of the chosen machines", () => {
  test("shows nothing until a machine is chosen, then each machine's memory, sleep, battery, idle and hours limits", async () => {
    server();
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    await screen.findByRole("checkbox", { name: /laptop/ });
    expect(screen.queryByTestId("limits")).toBeNull();
    const user = userEvent.setup();
    await user.click(screen.getByRole("checkbox", { name: /laptop/ }));
    await user.click(screen.getByRole("checkbox", { name: /BEAST/ }));
    const laptop = within(await screen.findByTestId("limits-laptop"));
    expect(laptop.getByText("Uses up to half of its memory")).toBeInTheDocument();
    expect(laptop.getByText("Keeps the computer awake while it trains")).toBeInTheDocument();
    expect(laptop.getByText("Pauses on battery")).toBeInTheDocument();
    expect(laptop.getByText("Holding its jobs right now: on battery.")).toBeInTheDocument();
    const beast = within(screen.getByTestId("limits-BEAST"));
    expect(beast.getByText("May use all of its memory")).toBeInTheDocument();
    expect(beast.getByText("Lets the computer sleep")).toBeInTheDocument();
    expect(beast.getByText("Keeps going on battery")).toBeInTheDocument();
    expect(beast.getByText("Trains only while the computer is idle")).toBeInTheDocument();
    expect(beast.getByText("From 01:00 to 06:00")).toBeInTheDocument();
    expect(beast.queryByText(/Holding its jobs/)).toBeNull();
  });

  test("links to the Queue page of each machine, where its plan is changed, and says when a machine's limits could not be read", async () => {
    server();
    renderPage(<TrainingPage />, { route: "/cluster/training" });
    await screen.findByRole("checkbox", { name: /laptop/ });
    const user = userEvent.setup();
    await user.click(screen.getByRole("checkbox", { name: /laptop/ }));
    await user.click(screen.getByRole("checkbox", { name: /mystery/ }));
    const link = within(await screen.findByTestId("limits-laptop")).getByRole("link", { name: "Change them on laptop's Queue page" });
    expect(link).toHaveAttribute("href", "https://laptop/queue/");
    expect(within(screen.getByTestId("limits-mystery")).getByText("mystery's limits could not be read.")).toBeInTheDocument();
  });
});
