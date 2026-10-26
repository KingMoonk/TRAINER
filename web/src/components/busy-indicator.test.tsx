import { act, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, test, vi } from "vitest";
import { SLOW_MS } from "@/lib/busy";
import { emit, mockApi } from "@/test/network";
import { renderPage } from "@/test/render";
import { BusyIndicator } from "./busy-indicator";

const busy = (...items: { id: string; label: string; done: number; total: number | null }[]) => act(() => emit("busy", { items }));

describe("busy indicator", () => {
  afterEach(() => vi.useRealTimers());

  test("shows nothing while idle", () => {
    renderPage(<BusyIndicator />);
    expect(screen.queryByTestId("busy-indicator")).toBeNull();
  });

  test("a determinate bar is filled to its progress with its title under it", async () => {
    renderPage(<BusyIndicator />);
    busy({ id: "a", label: "Reading and counting your text", done: 1, total: 4 });
    const bar = await screen.findByRole("progressbar", { name: "Reading and counting your text" });
    expect(bar).toHaveAttribute("aria-valuenow", "25");
    expect(screen.getByTestId("busy-title")).toHaveTextContent("Reading and counting your text (1 of 4)");
    expect(screen.queryByTestId("busy-indeterminate")).toBeNull();
  });

  test("work without steps shows an indeterminate bar", async () => {
    renderPage(<BusyIndicator />);
    busy({ id: "a", label: "Loading the model for chat", done: 0, total: null });
    const bar = await screen.findByRole("progressbar", { name: "Loading the model for chat" });
    expect(bar).not.toHaveAttribute("aria-valuenow");
    expect(screen.getByTestId("busy-indeterminate")).toBeInTheDocument();
  });

  test("several items share one bar, titled by the first with a count of the rest, and progress updates in place", async () => {
    renderPage(<BusyIndicator />);
    busy({ id: "a", label: "First", done: 0, total: 2 }, { id: "b", label: "Second", done: 0, total: null });
    const bar = await screen.findByRole("progressbar", { name: "First, Second" });
    expect(screen.getAllByRole("progressbar")).toHaveLength(1);
    expect(bar).not.toHaveAttribute("aria-valuenow");
    expect(screen.getByTestId("busy-title")).toHaveTextContent("First (0 of 2) +1 more");
    busy({ id: "a", label: "First", done: 2, total: 2 }, { id: "b", label: "Second", done: 1, total: 2 });
    await waitFor(() => expect(screen.getByRole("progressbar", { name: "First, Second" })).toHaveAttribute("aria-valuenow", "75"));
  });

  test("it lifts the toasts and the traffic console while it shows, and fades out and lets go when idle", async () => {
    renderPage(<BusyIndicator />);
    busy({ id: "a", label: "Working", done: 0, total: null });
    await screen.findByTestId("busy-indicator");
    expect(document.documentElement.style.getPropertyValue("--busy-offset")).not.toBe("");
    expect(document.documentElement.style.getPropertyValue("--busy-bar")).not.toBe("");
    busy();
    await waitFor(() => expect(screen.queryByTestId("busy-indicator")).toBeNull());
    expect(document.documentElement.style.getPropertyValue("--busy-offset")).toBe("");
    expect(document.documentElement.style.getPropertyValue("--busy-bar")).toBe("");
  });

  test("a read still pending after a moment is shown, labelled from its query, and goes when it answers", async () => {
    // Fake clock: the delay is stepped past explicitly, so a slow runner cannot change what the test sees.
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "Date"], shouldAdvanceTime: false });
    let answer = () => {};
    mockApi({ "GET /distributed/runs": () => new Promise((resolve) => { answer = () => resolve({ body: [] }); }) });
    const { client } = renderPage(<BusyIndicator />);
    void client.fetchQuery({ queryKey: ["distributed", "runs"], queryFn: () => fetch("/distributed/runs").then((r) => r.json()) });
    await act(() => vi.advanceTimersByTimeAsync(SLOW_MS - 1));
    expect(screen.queryByTestId("busy-indicator")).toBeNull();  // not yet: fast reads never flash the bar
    await act(() => vi.advanceTimersByTimeAsync(1));
    expect(screen.getByRole("progressbar", { name: "Loading runs" })).toBeInTheDocument();
    expect(screen.getByTestId("busy-indeterminate")).toBeInTheDocument();
    answer();
    // The answer crosses the mocked fetch (real microtasks and macrotasks) before the query settles: step the fake clock until it has, then past the fade.
    for (let step = 0; step < 50 && client.isFetching(); step++) await act(() => vi.advanceTimersByTimeAsync(10));
    expect(client.isFetching()).toBe(0);
    await act(() => vi.advanceTimersByTimeAsync(1_000));
    expect(screen.queryByTestId("busy-indicator")).toBeNull();
  });

  test("a read is shown when its timer fires a millisecond before the clock has moved the full delay", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "Date"], shouldAdvanceTime: false });
    mockApi({ "GET /distributed/runs": () => new Promise(() => {}) });
    const { client } = renderPage(<BusyIndicator />);
    void client.fetchQuery({ queryKey: ["distributed", "runs"], queryFn: () => fetch("/distributed/runs").then((r) => r.json()) });
    const start = Date.now();
    vi.spyOn(Date, "now").mockReturnValue(start + SLOW_MS - 1);  // the clock is a millisecond behind when the timer fires
    await act(() => vi.advanceTimersByTimeAsync(SLOW_MS));
    expect(screen.getByRole("progressbar", { name: "Loading runs" })).toBeInTheDocument();
  });

  test("a read that answers at once never shows", async () => {
    mockApi({ "GET /distributed/runs": { body: [] } });
    const { client } = renderPage(<BusyIndicator />);
    await client.fetchQuery({ queryKey: ["distributed", "runs"], queryFn: () => fetch("/distributed/runs").then((r) => r.json()) });
    await new Promise((resolve) => setTimeout(resolve, 500));
    expect(screen.queryByTestId("busy-indicator")).toBeNull();
  });
});
