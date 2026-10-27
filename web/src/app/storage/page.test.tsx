import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test } from "vitest";
import { mockApi } from "@/test/network";
import { renderPage } from "@/test/render";
import StoragePage from "./page";

// The Storage page against a mocked API (the component tier of tests/storage.spec.ts; the pure formatters are covered in logic.test.ts).

type Use = { items: number; bytes: number };

/** GET /storage answers by the `days` and `data` asked; every POST /storage/clean is recorded and, unless refused, empties the disk. */
function server(use: (days: number | null, data: boolean) => Use, extra: Parameters<typeof mockApi>[0] = {}) {
  const state = { use, refuse: "", cleaned: [] as unknown[] };
  mockApi({
    "GET /storage": ({ search }) => {
      const q = new URLSearchParams(search);
      return { body: state.use(q.has("days") ? Number(q.get("days")) : null, q.get("data") === "true") };
    },
    "GET /storage/free": { body: { bytes: 120e9 } },
    "GET /storage/leftovers": { body: [] },
    "POST /storage/clean": ({ body }) => {
      state.cleaned.push(body);
      if (state.refuse) return { status: 409, body: { detail: state.refuse } };
      const asked = body as { days: number | null; data: boolean };
      const done = state.use(asked.days, asked.data);
      state.use = () => ({ items: 0, bytes: 0 });
      return { body: done };
    },
    ...extra,
  });
  return state;
}

const bySize = (days: number | null, data: boolean): Use =>
  days === null ? (data ? { items: 9, bytes: 4.5e9 } : { items: 5, bytes: 2.5e9 }) : days <= 7 ? { items: 2, bytes: 3e6 } : { items: 1, bytes: 1e6 };

const deleteButton = () => screen.getByRole("button", { name: "Delete…" });

describe("Storage page", () => {
  test("the page shows free space and what a clean-up of each age would free, in place", async () => {
    server(bySize);
    const user = userEvent.setup();
    renderPage(<StoragePage />, { route: "/storage/" });
    expect(await screen.findByRole("heading", { name: "Storage" })).toBeVisible();
    expect(await screen.findByText("120.0 GB free on this computer's disk")).toBeVisible();
    expect(await screen.findByText("This frees 1.0 MB in 1 item")).toBeVisible();
    await user.click(screen.getByLabelText("Older than 7 days"));
    expect(await screen.findByText("This frees 3.0 MB in 2 items")).toBeVisible();
    const data = screen.getByLabelText("Also delete my datasets and downloads");
    expect(data).toBeDisabled();
    await user.click(screen.getByLabelText("All history"));
    expect(await screen.findByText("This frees 2.5 GB in 5 items")).toBeVisible();
    await user.click(data);
    expect(await screen.findByText("This frees 4.5 GB in 9 items")).toBeVisible();
    await user.click(screen.getByLabelText("Older than 1 days"));
    await waitFor(() => expect(data).toBeDisabled());
    expect(data).not.toBeChecked();
  });

  test("a clean asks first, then frees the space and the numbers update", async () => {
    const state = server(bySize);
    const user = userEvent.setup();
    renderPage(<StoragePage />, { route: "/storage/" });
    await screen.findByText("This frees 1.0 MB in 1 item");
    await user.click(deleteButton());
    expect(await screen.findByText("Delete 1 item and free 1.0 MB? This cannot be undone.")).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Cancel" }));
    expect(state.cleaned).toEqual([]);
    await user.click(deleteButton());
    await user.click(within(screen.getByRole("alertdialog")).getByRole("button", { name: "Delete" }));
    expect(await screen.findByText("Freed 1.0 MB")).toBeVisible();
    expect(state.cleaned).toEqual([{ days: 30, data: false }]);
    expect(await screen.findByText("Nothing to delete")).toBeVisible();
    expect(deleteButton()).toBeDisabled();
  });

  test("a clean the server refuses shows its sentence and changes nothing", async () => {
    const state = server(bySize);
    state.refuse = "Stop the running job first.";
    const user = userEvent.setup();
    renderPage(<StoragePage />, { route: "/storage/" });
    await screen.findByText("This frees 1.0 MB in 1 item");
    await user.click(deleteButton());
    await user.click(within(screen.getByRole("alertdialog")).getByRole("button", { name: "Delete" }));
    expect(await screen.findByText("Stop the running job first.")).toBeVisible();
    expect(state.cleaned).toHaveLength(1);
    expect(screen.getByText("This frees 1.0 MB in 1 item")).toBeVisible();
  });

  test("leftovers are listed with their sizes and only the ticked ones are removed, after asking", async () => {
    const rows = [
      { id: "runs/queue_a-before-restart", kind: "interrupted", name: "queue_a-before-restart", belongs_to: "Version 1 from scratch", bytes: 5e8, shared: 2e9 },
      { id: "runs/.snapshots/abc.jsonl", kind: "snapshot", name: "abc", belongs_to: "No run uses this training data", bytes: 2e9, shared: 0 },
    ];
    const removed: unknown[] = [];
    server(() => ({ items: 0, bytes: 0 }), {
      "GET /storage/leftovers": { body: rows },
      "POST /storage/leftovers/remove": ({ body }) => {
        removed.push(body);
        return { body: { items: 1, bytes: 5e8 } };
      },
    });
    const user = userEvent.setup();
    renderPage(<StoragePage />, { route: "/storage/" });
    expect(await screen.findByText(/Interrupted attempt: queue_a-before-restart/)).toBeVisible();
    expect(screen.getByText(/Version 1 from scratch · 500.0 MB · 2.0 GB is shared with other runs and stays/)).toBeVisible();
    expect(screen.getByText(/Unused training data: abc/)).toBeVisible();
    const button = screen.getByRole("button", { name: "Remove selected…" });
    expect(button).toBeDisabled();
    await user.click(screen.getByLabelText("Select queue_a-before-restart"));
    await user.click(button);
    expect(await screen.findByText("Remove 1 item and free 500.0 MB? This cannot be undone.")).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Cancel" }));
    expect(removed).toEqual([]);
    await user.click(screen.getByRole("button", { name: "Remove selected…" }));
    await user.click(screen.getByRole("button", { name: "Delete" }));
    await waitFor(() => expect(removed).toEqual([{ ids: ["runs/queue_a-before-restart"] }]));
    expect(await screen.findByText("Freed 500.0 MB")).toBeVisible();
  });

  test("nothing left over says so and offers nothing to remove", async () => {
    server(bySize);
    renderPage(<StoragePage />, { route: "/storage/" });
    expect(await screen.findByText("Nothing left over.")).toBeVisible();
    expect(screen.queryByRole("button", { name: "Remove selected…" })).toBeNull();
  });
});
