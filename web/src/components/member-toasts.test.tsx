import { waitFor } from "@testing-library/react";
import { toast } from "sonner";
import { afterEach, describe, expect, test, vi } from "vitest";
import { member, run } from "@/app/cluster/training/test-data";
import type { Invite } from "@/lib/join";
import { invite } from "@/test/invites";
import { emit, mockApi } from "@/test/network";
import { renderPage } from "@/test/render";
import { MemberToasts } from "./member-toasts";

const node = (hostname: string, local: boolean) => ({
  address: local ? "local" : hostname, local, hostname, version: "1", device: "cuda", device_name: "", memory_bytes: 0, busy: false, dtypes: [], online: true, error: "", tailnet_name: "",
});

function server(owner: string) {
  const state = { member: "invited" as "invited" | "training" | "behind" | "left", invites: [] as Invite[] };
  const calls = mockApi({
    "GET /cluster/nodes": { body: [node("mac-mini", true), node("gpu-box", false)] },
    "GET /distributed/runs": () => ({
      body: [run({ owner, members: [member("mac-mini", { local: true, can_stop: false }), member("gpu-box", { state: state.member })] })],
    }),
    "GET /distributed/invites": () => ({ body: { invites: state.invites, pause_requests: [] } }),
  });
  return { state, calls };
}

const spies = () => ({ success: vi.spyOn(toast, "success"), warning: vi.spyOn(toast, "warning"), info: vi.spyOn(toast, "info") });
const said = (spy: { mock: { calls: unknown[][] } }) => spy.mock.calls.map((call) => call[0]);

/** Let the first reads land (they only record), then change the server's answer and announce it the way the shell's stream does. */
async function readThen(calls: { path: string }[], apply: () => void) {
  await waitFor(() => {
    for (const path of ["/cluster/nodes", "/distributed/runs", "/distributed/invites"]) expect(calls.some((c) => c.path === path)).toBe(true);
  });
  await new Promise((resolve) => setTimeout(resolve, 50));
  apply();
  emit("queue", {});
}

afterEach(() => vi.restoreAllMocks());

describe("member toasts", () => {
  test("the coordinator is told when a machine joins, falls behind and leaves", async () => {
    const { state, calls } = server("mac-mini");
    const toasts = spies();
    renderPage(<MemberToasts />);
    await readThen(calls, () => { state.member = "training"; });
    await waitFor(() => expect(said(toasts.success)).toEqual(["gpu-box joined 'Version 2 across machines'."]));
    state.member = "behind";
    emit("queue", {});
    await waitFor(() => expect(said(toasts.warning)).toEqual(["gpu-box fell behind in 'Version 2 across machines'."]));
    state.member = "left";
    emit("queue", {});
    await waitFor(() => expect(said(toasts.info)).toEqual(["gpu-box left 'Version 2 across machines'."]));
  });

  test("a run another computer coordinates toasts nothing from its member list", async () => {
    const { state, calls } = server("someone-else");
    const toasts = spies();
    renderPage(<MemberToasts />);
    await readThen(calls, () => { state.member = "training"; });
    await new Promise((resolve) => setTimeout(resolve, 200));
    expect(said(toasts.success)).toEqual([]);
  });

  test("the machine that was added is told when it joins and when it leaves", async () => {
    const { state, calls } = server("someone-else");
    state.invites = [invite({ state: "joining" })];
    const toasts = spies();
    renderPage(<MemberToasts />);
    await readThen(calls, () => { state.invites = [invite({ state: "training", round: 1, rounds: 5 })]; });
    await waitFor(() => expect(said(toasts.success)).toEqual(["This machine joined 'Version 7'."]));
    state.invites = [];
    emit("queue", {});
    await waitFor(() => expect(said(toasts.info)).toEqual(["This machine left 'Version 7'."]));
  });

  test("opening the page with machines already training toasts nothing", async () => {
    const { state, calls } = server("mac-mini");
    state.member = "training";
    const toasts = spies();
    renderPage(<MemberToasts />);
    await readThen(calls, () => undefined);
    await new Promise((resolve) => setTimeout(resolve, 200));
    expect(said(toasts.success)).toEqual([]);
    expect(said(toasts.warning)).toEqual([]);
  });
});
