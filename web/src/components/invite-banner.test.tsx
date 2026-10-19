import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test } from "vitest";
import { invite, pauseRequest as request } from "@/test/invites";
import type { Invite, Invites } from "@/lib/join";
import { mockApi } from "@/test/network";
import { renderPage } from "@/test/render";
import { InviteBanner } from "./invite-banner";


function server(start: Invites) {
  const state = { ...start, left: [] as string[], paused: [] as string[], accepted: [] as string[], declined: [] as string[] };
  const calls = mockApi({
    "GET /distributed/invites": () => ({ body: { invites: state.invites, pause_requests: state.pause_requests } }),
    "POST /distributed/invites/*/leave": ({ path }) => {
      state.left.push(path.split("/")[3]);
      state.invites = [];
      return { body: { invites: [], pause_requests: state.pause_requests } };
    },
    "POST /distributed/invites/*/pause-and-join": ({ path }) => {
      state.paused.push(path.split("/")[3]);
      state.invites = state.invites.map((i) => ({ ...i, state: "joining" as const, can_pause_and_join: false, blocking_job: null }));
      return { body: { invites: state.invites, pause_requests: state.pause_requests } };
    },
    "POST /distributed/pause-requests/*/accept": ({ path }) => {
      state.accepted.push(path.split("/")[3]);
      state.pause_requests = [];
      return { body: { invites: state.invites, pause_requests: [] } };
    },
    "DELETE /distributed/pause-requests/*": ({ path }) => {
      state.declined.push(path.split("/")[3]);
      state.pause_requests = [];
      return { body: { invites: state.invites, pause_requests: [] } };
    },
  });
  return { state, calls };
}

const banner = (id: string) => within(screen.getByTestId(`invite-${id}`));

describe("the invite banner", () => {
  test("shows nothing when this machine was not added to a run", async () => {
    const { calls } = server({ invites: [], pause_requests: [] });
    renderPage(<InviteBanner />);
    await waitFor(() => expect(calls.some((c) => c.path === "/distributed/invites")).toBe(true));
    expect(screen.queryByTestId("invite-banner")).toBeNull();
  });

  test("a machine that is joining says so and offers only Leave, with no accept step", async () => {
    server({ invites: [invite({ state: "joining", round: 4 })], pause_requests: [] });
    renderPage(<InviteBanner />);
    expect(await screen.findByText("Michael's Mac added this machine to 'Version 7'")).toBeInTheDocument();
    expect(screen.getByTestId("invite-line-j1")).toHaveTextContent("Joining at round 4");
    expect(banner("j1").getByText("Joining", { selector: "[data-slot=badge]" })).toBeInTheDocument();
    expect(banner("j1").getAllByRole("button").map((b) => b.textContent)).toEqual(["Leave"]);
    expect(screen.queryByRole("button", { name: /accept/i })).toBeNull();
  });

  test("a training machine shows the round and speed", async () => {
    server({ invites: [invite({ state: "training", round: 6, rounds: 50, tokens_per_second: 1200 })], pause_requests: [] });
    renderPage(<InviteBanner />);
    expect(await screen.findByTestId("invite-line-j1")).toHaveTextContent("Training: round 6 of 50, 1,200 tok/s");
    expect(banner("j1").getAllByRole("button").map((b) => b.textContent)).toEqual(["Leave"]);
  });

  test("a machine waiting behind a local job can pause it and join now", async () => {
    const { state } = server({ invites: [invite({ state: "waiting", can_pause_and_join: true, blocking_job: { label: "Version 1", eta_s: 720 } })], pause_requests: [] });
    renderPage(<InviteBanner />);
    expect(await screen.findByTestId("invite-line-j1")).toHaveTextContent("Waiting: busy with 'Version 1', about 12 min left, joins when it finishes");
    expect(banner("j1").getAllByRole("button").map((b) => b.textContent)).toEqual(["Pause 'Version 1' and join now", "Leave"]);
    await userEvent.setup().click(banner("j1").getByRole("button", { name: "Pause 'Version 1' and join now" }));
    await waitFor(() => expect(state.paused).toEqual(["j1"]));
    expect(await screen.findByTestId("invite-line-j1")).toHaveTextContent("Joining");
  });

  test("a waiting machine that cannot pause the job offers only Leave", async () => {
    server({ invites: [invite({ state: "waiting", can_pause_and_join: false, blocking_job: { label: "Version 1", eta_s: null } })], pause_requests: [] });
    renderPage(<InviteBanner />);
    await screen.findByTestId("invite-line-j1");
    expect(banner("j1").getAllByRole("button").map((b) => b.textContent)).toEqual(["Leave"]);
  });

  test("an invited, behind or unreachable machine shows its line and Leave", async () => {
    const cases: [Invite, string][] = [
      [invite({ state: "invited" }), "Invited: it joins by itself when this machine is free"],
      [invite({ state: "behind", round: 3, rounds: 10 }), "Behind: it missed a round"],
      [invite({ state: "unreachable", reason: "Michael's Mac did not answer." }), "Unreachable: Michael's Mac did not answer."],
    ];
    for (const [one, line] of cases) {
      const { unmount } = (() => {
        server({ invites: [one], pause_requests: [] });
        return renderPage(<InviteBanner />);
      })();
      expect(await screen.findByTestId("invite-line-j1")).toHaveTextContent(line);
      expect(banner("j1").getAllByRole("button").map((b) => b.textContent)).toEqual(["Leave"]);
      unmount();
    }
  });

  test("Leave tells the owner and removes the banner", async () => {
    const { state } = server({ invites: [invite({ state: "training", round: 1, rounds: 5 })], pause_requests: [] });
    renderPage(<InviteBanner />);
    await userEvent.setup().click(await screen.findByRole("button", { name: "Leave" }));
    await waitFor(() => expect(state.left).toEqual(["j1"]));
    await waitFor(() => expect(screen.queryByTestId("invite-banner")).toBeNull());
  });

  test("a pause request is an alert a person answers, and nothing is paused until they do", async () => {
    const { state } = server({ invites: [], pause_requests: [request()] });
    renderPage(<InviteBanner />);
    const alert = within(await screen.findByTestId("pause-request-q1"));
    expect(alert.getByText("Michael's Mac asks this machine to pause 'Version 1' so 'Version 7' can use it")).toBeInTheDocument();
    expect(state.accepted).toEqual([]);
    await userEvent.setup().click(alert.getByRole("button", { name: "Pause 'Version 1'" }));
    await waitFor(() => expect(state.accepted).toEqual(["q1"]));
    await waitFor(() => expect(screen.queryByTestId("pause-request-q1")).toBeNull());
  });

  test("Not now declines a pause request", async () => {
    const { state } = server({ invites: [], pause_requests: [request()] });
    renderPage(<InviteBanner />);
    await userEvent.setup().click(await screen.findByRole("button", { name: "Not now" }));
    await waitFor(() => expect(state.declined).toEqual(["q1"]));
    expect(state.accepted).toEqual([]);
  });

  test("a pause request for a job that already ended has nothing to pause, only Not now", async () => {
    server({ invites: [], pause_requests: [request({ blocking_job: null })] });
    renderPage(<InviteBanner />);
    const alert = within(await screen.findByTestId("pause-request-q1"));
    expect(alert.getAllByRole("button").map((b) => b.textContent)).toEqual(["Not now"]);
  });
});
