import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test } from "vitest";
import { mockApi } from "@/test/network";
import { renderPage } from "@/test/render";
import { SelfTestCard } from "./selftest-card";
import type { SelfTestRun, SelfTestStep } from "./logic";

// The Test the cluster card against a mocked API: GET and POST /cluster/selftest (studio_selftest.py runs the real thing; the
// two-process proof is tests/e2e_cluster.py).

const step = (id: string, title: string, status: SelfTestStep["status"], over: Partial<SelfTestStep> = {}): SelfTestStep => ({
  id, title, machine: "", status, detail: "", fix: "", seconds: null, ...over,
});
const run = (state: SelfTestRun["state"], steps: SelfTestStep[], over: Partial<SelfTestRun> = {}): SelfTestRun => ({
  id: "selftest-1", machines: ["box"], started_at: 1_700_000_000, finished_at: state === "running" ? null : 1_700_000_060, state, steps, ...over,
});
const PEERS = [{ address: "box", hostname: "box" }, { address: "pc:8787", hostname: "" }];
const NEVER = run("never", [], { finished_at: null, started_at: 0, machines: [] });

describe("the Test the cluster card", () => {
  test("before any test it says so and offers every machine, ticked", async () => {
    mockApi({ "GET /cluster/selftest": { body: NEVER } });
    renderPage(<SelfTestCard peers={PEERS} />);
    expect(await screen.findByText("Not run yet.")).toBeInTheDocument();
    expect(screen.getAllByRole("checkbox")).toHaveLength(2);
    expect(screen.getAllByRole("checkbox").every((box) => (box as HTMLInputElement).checked)).toBe(true);
    expect(screen.queryByTestId("selftest-ran")).not.toBeInTheDocument();
  });

  test("with no other machine it asks for one and cannot start", async () => {
    mockApi({ "GET /cluster/selftest": { body: NEVER } });
    renderPage(<SelfTestCard peers={[]} />);
    expect(await screen.findByText(/Add another machine/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run the test" })).toBeDisabled();
  });

  test("it starts the test with the ticked machines and shows the steps as they pass, live", async () => {
    let reads = 0;
    const calls = mockApi({
      "GET /cluster/selftest": () => {
        reads += 1;
        return { body: reads >= 2 ? run("running", [step("reach:box", "Reach box", "pass", { detail: "Answered: box.", seconds: 0.05 }), step("launch", "Start a tiny run", "running")], { finished_at: null }) : NEVER };
      },
      "POST /cluster/selftest": { status: 202, body: run("running", [step("reach:box", "Reach box", "running")], { finished_at: null }) },
    });
    renderPage(<SelfTestCard peers={PEERS} />);
    await userEvent.click(await screen.findByRole("checkbox", { name: /pc:8787/ }));
    await userEvent.click(screen.getByRole("button", { name: "Run the test" }));
    await waitFor(() => expect(calls.find((c) => c.method === "POST")?.body).toEqual({ machines: ["box"] }));
    const reach = await screen.findByTestId("selftest-step-reach:box");
    expect(reach).toHaveAttribute("data-status", "pass");
    expect(screen.getByTestId("selftest-step-launch")).toHaveAttribute("data-status", "running");
    expect(screen.getByTestId("selftest-headline")).toHaveTextContent("Running now.");
    expect(screen.getByRole("button", { name: "Testing…" })).toBeDisabled();
  });

  test("a failed step names the machine and the fix, later steps read as not reached, and the last run shows when", async () => {
    mockApi({
      "GET /cluster/selftest": {
        body: run("failed", [
          step("reach:box", "Reach box", "fail", { machine: "box", detail: "box did not answer within 2 seconds.", fix: "Start the app on box, and check the Machines tab." }),
          step("launch", "Start a tiny run on these machines", "skipped", { detail: "Not reached: an earlier step failed." }),
        ]),
      },
    });
    renderPage(<SelfTestCard peers={PEERS} />);
    const failed = await screen.findByTestId("selftest-step-reach:box");
    expect(failed).toHaveAttribute("data-status", "fail");
    expect(within(failed).getByText("box did not answer within 2 seconds.")).toBeInTheDocument();
    expect(within(failed).getByTestId("selftest-fix")).toHaveTextContent("Fix: Start the app on box, and check the Machines tab.");
    expect(within(screen.getByTestId("selftest-step-launch")).getByRole("img", { name: "Not reached" })).toBeInTheDocument();
    expect(screen.getByTestId("selftest-headline")).toHaveTextContent("Failed");
    expect(screen.getByTestId("selftest-ran")).toHaveTextContent(/Last run .*with box\./);
  });

  test("a passed test says so with every step ticked", async () => {
    mockApi({ "GET /cluster/selftest": { body: run("passed", [step("reach:box", "Reach box", "pass"), step("loss", "The loss falls", "pass", { detail: "fell from 3.2 to 2.5" })]) } });
    renderPage(<SelfTestCard peers={PEERS} />);
    expect(await screen.findByText(/^Passed: a run across these machines works/)).toBeInTheDocument();
    expect(screen.getAllByRole("img", { name: "Passed" })).toHaveLength(2);
  });

  test("a refused start shows the sentence the server gave", async () => {
    mockApi({
      "GET /cluster/selftest": { body: NEVER },
      "POST /cluster/selftest": { status: 409, body: { detail: "This computer is training something. Test the cluster when it is free." } },
    });
    renderPage(<SelfTestCard peers={PEERS} />);
    await userEvent.click(await screen.findByRole("button", { name: "Run the test" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("This computer is training something");
  });
});
