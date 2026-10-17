import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test } from "vitest";
import { mockApi } from "@/test/network";
import { renderPage } from "@/test/render";
import Page from "./page";

// The Value map page against a mocked job endpoint (the component tier of tests/trace-value-map.spec.ts, which keeps the 375 px layout check).

const job = (result: unknown) => ({
  id: "j1",
  kind: "inspect.value_map",
  status: "done",
  progress: "",
  fraction: null,
  result,
  error: "",
  created_at: 0,
  started_at: null,
  finished_at: null,
});

const VALUE_MAP = { d_model: 256, heads: 4, blocks: { "0": 4, "1": 4 }, block: 0, head: 0, singular: [3, 1.5, 0.7, 0.1] };
const asked = (calls: ReturnType<typeof mockApi>) => calls.filter((call) => call.path === "/inspect/value-map").map((call) => call.body);

describe("Value map", () => {
  test("carries the linked version into the inspection request", async () => {
    const calls = mockApi({ "POST /inspect/value-map": { status: 202, body: job(VALUE_MAP) } });
    renderPage(<Page />, { route: "/value-map?version=7" });
    await waitFor(() => expect(asked(calls)[0]).toMatchObject({ version: 7 }));
  });

  test("shows a skeleton, then the funnel and the singular values", async () => {
    let release: () => void = () => {};
    const gate = new Promise<void>((resolve) => (release = resolve));
    const calls = mockApi({
      "POST /inspect/value-map": async () => {
        await gate;
        return { status: 202, body: job(VALUE_MAP) };
      },
    });
    renderPage(<Page />, { route: "/value-map" });
    expect(await screen.findByRole("status", { name: "Loading the value map" })).toBeVisible();
    release();
    expect(await screen.findByRole("heading", { level: 2 })).toHaveTextContent("head_dim = 256 ÷ 4 = 64");
    expect(asked(calls)).toEqual([{ heads: 4, block: 0, head: 0 }]);
    expect(screen.getByText(/d_model 256 · heads/)).toBeVisible();
    expect(screen.getByText("Per head W_V and W_O hold 32,768 numbers; all 4 heads together hold 131,072 = 2 · 256², whatever the head count.")).toBeVisible();
    expect(screen.getByText(/Measured on this model, block 1 head 1: the rank of W_V · W_O is 4, at most 64\./)).toBeVisible();
    expect(screen.getByRole("img", { name: "4 singular values" })).toBeVisible();
    expect(screen.getByRole("radio", { name: "4" })).toBeChecked();
  });

  test("choosing heads, a block and a head asks the server again", async () => {
    const user = userEvent.setup();
    const calls = mockApi({
      "POST /inspect/value-map": ({ body }) => {
        // the model has 4 heads: asking for another count gives no singular values, as in the Tk page
        const result = (body as { heads: number }).heads === 4 ? VALUE_MAP : { ...VALUE_MAP, singular: null };
        return { status: 202, body: job(result) };
      },
    });
    renderPage(<Page />, { route: "/value-map" });
    await user.click(await screen.findByRole("radio", { name: "Block 2" }));
    await user.click(await screen.findByRole("radio", { name: "Head 3" }));
    expect(await screen.findByText(/block 2 head 3/)).toBeVisible();
    await user.click(screen.getByRole("radio", { name: "8" }));
    expect(await screen.findByRole("heading", { level: 2 })).toHaveTextContent("head_dim = 256 ÷ 8 = 32");
    expect(await screen.findByText("Pick 4 heads to see this model's measured singular values; it has 4 heads per block.")).toBeVisible();
    expect(asked(calls)).toEqual([
      { heads: 4, block: 0, head: 0 },
      { heads: 4, block: 1, head: 0 },
      { heads: 4, block: 1, head: 2 },
      { heads: 8, block: 0, head: 0 },
    ]);
  });

  test("a head count that does not divide d_model is said so", async () => {
    mockApi({ "POST /inspect/value-map": { status: 202, body: job({ ...VALUE_MAP, d_model: 250 }) } });
    renderPage(<Page />, { route: "/value-map" });
    expect(await within(screen.getByRole("main")).findByRole("alert")).toHaveTextContent("250 does not split into 4 equal heads.");
  });
});
