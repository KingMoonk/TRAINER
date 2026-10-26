import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test } from "vitest";
import { mockApi } from "@/test/network";
import { renderPage } from "@/test/render";
import { LiveCheckpoints } from "./live";

// Score now follows a server job; leaving the page must end that wait and cancel the job (issue #755).

describe("Live checkpoints", () => {
  test("leaving the page while Score now runs cancels its job", async () => {
    const running = { id: "j1", kind: "versions.score_live", status: "running", result: null, progress: "", error: "", created_at: 0 };
    const calls = mockApi({
      "POST /versions/1/score-live": { status: 202, body: running },
      "GET /jobs/j1": { body: running },
      "DELETE /jobs/j1": { body: { ...running, status: "cancelled" } },
    });
    const live = [{ version: 1, step: 12000, total: 50000, saved_at: Date.now() / 1000, file: "best.npz" }];
    const user = userEvent.setup();
    const { unmount } = renderPage(<LiveCheckpoints live={live} answering={null} busy={false} onPick={() => {}} />, { route: "/chat" });
    await user.click(screen.getByRole("button", { name: "Score now" }));
    await waitFor(() => expect(calls.some((c) => c.method === "GET" && c.path === "/jobs/j1")).toBe(true));
    expect(calls.some((c) => c.method === "DELETE")).toBe(false);
    unmount();
    await waitFor(() => expect(calls.some((c) => c.method === "DELETE" && c.path === "/jobs/j1")).toBe(true));
  });
});
