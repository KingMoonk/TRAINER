import { act, screen, waitFor } from "@testing-library/react";
import { describe, expect, test } from "vitest";
import { emit, mockApi } from "@/test/network";
import { renderPage } from "@/test/render";
import { BusyIndicator } from "./busy-indicator";

const busy = (...items: { id: string; label: string; done: number; total: number | null }[]) => act(() => emit("busy", { items }));

describe("busy indicator", () => {
  test("shows nothing while idle", () => {
    renderPage(<BusyIndicator />);
    expect(screen.queryByTestId("busy-indicator")).toBeNull();
  });

  test("a determinate bar is filled to its progress with the label under it", async () => {
    renderPage(<BusyIndicator />);
    busy({ id: "a", label: "Reading and counting your text", done: 1, total: 4 });
    const bar = await screen.findByRole("progressbar", { name: "Reading and counting your text" });
    expect(bar).toHaveAttribute("aria-valuenow", "25");
    expect(screen.getByText(/Reading and counting your text/)).toBeInTheDocument();
    expect(screen.queryByTestId("busy-indeterminate")).toBeNull();
  });

  test("work without steps shows an indeterminate bar", async () => {
    renderPage(<BusyIndicator />);
    busy({ id: "a", label: "Loading the model for chat", done: 0, total: null });
    const bar = await screen.findByRole("progressbar", { name: "Loading the model for chat" });
    expect(bar).not.toHaveAttribute("aria-valuenow");
    expect(screen.getByTestId("busy-indeterminate")).toBeInTheDocument();
  });

  test("several items stack, and progress updates in place", async () => {
    renderPage(<BusyIndicator />);
    busy({ id: "a", label: "First", done: 0, total: 2 }, { id: "b", label: "Second", done: 0, total: null });
    await waitFor(() => expect(screen.getAllByTestId("busy-item")).toHaveLength(2));
    busy({ id: "a", label: "First", done: 2, total: 2 }, { id: "b", label: "Second", done: 0, total: null });
    await waitFor(() => expect(screen.getByRole("progressbar", { name: "First" })).toHaveAttribute("aria-valuenow", "100"));
    expect(screen.getAllByTestId("busy-item")).toHaveLength(2);
  });

  test("it lifts the toasts while it shows, and fades out and lets go when idle", async () => {
    renderPage(<BusyIndicator />);
    busy({ id: "a", label: "Working", done: 0, total: null });
    await screen.findByTestId("busy-indicator");
    expect(document.documentElement.style.getPropertyValue("--busy-offset")).not.toBe("");
    busy();
    await waitFor(() => expect(screen.queryByTestId("busy-indicator")).toBeNull());
    expect(document.documentElement.style.getPropertyValue("--busy-offset")).toBe("");
  });

  test("a read still pending after a moment is shown, labelled from its query, and goes when it answers", async () => {
    let answer = () => {};
    mockApi({ "GET /distributed/runs": () => new Promise((resolve) => { answer = () => resolve({ body: [] }); }) });
    const { client } = renderPage(<BusyIndicator />);
    void client.fetchQuery({ queryKey: ["distributed", "runs"], queryFn: () => fetch("/distributed/runs").then((r) => r.json()) });
    expect(screen.queryByTestId("busy-indicator")).toBeNull();  // not yet: fast reads never flash a card
    expect(await screen.findByRole("progressbar", { name: "Loading runs" }, { timeout: 5_000 })).toBeInTheDocument(); // shown after SLOW_MS: room for a loaded CI runner
    expect(screen.getByTestId("busy-indeterminate")).toBeInTheDocument();
    answer();
    await waitFor(() => expect(screen.queryByTestId("busy-indicator")).toBeNull());
  });

  test("a read that answers at once never shows", async () => {
    mockApi({ "GET /distributed/runs": { body: [] } });
    const { client } = renderPage(<BusyIndicator />);
    await client.fetchQuery({ queryKey: ["distributed", "runs"], queryFn: () => fetch("/distributed/runs").then((r) => r.json()) });
    await new Promise((resolve) => setTimeout(resolve, 500));
    expect(screen.queryByTestId("busy-indicator")).toBeNull();
  });
});
