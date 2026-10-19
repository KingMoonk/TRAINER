import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test } from "vitest";
import { mockApi } from "@/test/network";
import { renderPage } from "@/test/render";
import { MakeExtraData } from "./make-extra";

// The Make extra data section of the Your data page against a mocked API (the component tier of tests/live.spec.ts; its phone-width check is layout and stays out of jsdom).

type Draft = { id: string; a: string; b: string; status: string; dataset: string };

function server({ fail = false }: { fail?: boolean } = {}) {
  const state = { drafts: [] as Draft[], requests: [] as { rule: string | null; n: number; store: boolean }[], reviews: [] as boolean[], reviewOnly: [] as (string | null)[] };
  const calls = mockApi({
    "GET /data/problems/rules": { body: { rules: [{ id: "reverse", label: "Reverse" }], max_problems: 5000, max_drafts: 200 } },
    "GET /data/drafts": () => ({ body: { count: state.drafts.length } }),
    "GET /data/collections/*": { body: [] },
    "POST /data/problems/generate": ({ body }) => {
      const sent = body as { rule: string | null; n: number; store: boolean };
      state.requests.push(sent);
      return { body: { items: [{ prompt: "Reverse: abc", answer: "cba" }], added: sent.store ? sent.n : 0, duplicates: 0, checked: sent.n, passed: sent.n } };
    },
    "GET /data/conversations/rows": ({ search }) => {
      state.reviewOnly.push(new URLSearchParams(search).get("review_only"));
      return { body: { rows: state.drafts, next_cursor: null } };
    },
    "POST /data/conversations/drafts": () => {
      if (!fail) state.drafts = [1, 2].map((id) => ({ id: `d${id}`, a: `Draft ${id}`, b: `Answer ${id}`, status: "Review", dataset: "Drafts" }));
      return { status: 202, body: { id: "make-1", kind: "make_drafts", status: fail ? "failed" : "done", result: { count: 2 }, error: fail ? "No local model connected." : "", created_at: 0 } };
    },
    "GET /data/examples/*/fields": { body: { fields: { user: "Full prompt", assistant: "Full answer to review" } } },
    "POST /data/examples/*/review": ({ path, body }) => {
      state.reviews.push((body as { keep: boolean }).keep);
      const id = path.split("/").at(-2);
      state.drafts = state.drafts.filter((row) => row.id !== id);
      return { status: 204 };
    },
  });
  return { state, calls };
}

const generate = () => screen.getByRole("button", { name: "Generate problems" });

describe("Make extra data section", () => {
  test.each([
    ["text", /Generate prose/],
    ["conversations", /Rewrite/],
    ["problems", /Problems/],
  ] as const)("the %s data kind opens the matching generator", async (kind, label) => {
    server();
    renderPage(<MakeExtraData initialKind={kind} />, { route: "/your-data/" });
    expect(await screen.findByRole("button", { name: label })).toHaveAttribute("aria-pressed", "true");
    expect(within(screen.getByRole("group", { name: "What to make" })).getAllByRole("button", { pressed: true })).toHaveLength(1);
  });

  test("rules come from the server; preview never stores and generation supports one or all rules", async () => {
    const { state } = server();
    const user = userEvent.setup();
    renderPage(<MakeExtraData initialKind="problems" />, { route: "/your-data/" });
    const rule = await screen.findByLabelText("Rule");
    await user.selectOptions(rule, "reverse");
    await user.click(screen.getByRole("button", { name: "Try 5 first" }));
    expect(await screen.findByText("Preview, nothing saved")).toBeVisible();
    expect(state.requests[0]).toEqual({ rule: "reverse", n: 5, store: false });

    const many = screen.getByLabelText("How many");
    await user.clear(many);
    await user.type(many, "3");
    await user.click(generate());
    expect(await screen.findByText(/Added 3 problems/)).toBeVisible();
    expect(state.requests[1]).toEqual({ rule: "reverse", n: 3, store: true });

    await user.selectOptions(screen.getByLabelText("Rule"), "");
    await user.click(generate());
    await waitFor(() => expect(state.requests).toHaveLength(3));
    expect(state.requests[2]).toEqual({ rule: null, n: 3, store: true });

    await user.clear(screen.getByLabelText("How many"));
    await user.type(screen.getByLabelText("How many"), "5001");
    expect(generate()).toBeDisabled();
    expect(screen.getByRole("alert")).toHaveTextContent("5,000");
  });

  test("generated drafts require reading, and accept/discard remove only the chosen row", async () => {
    const { state } = server();
    const user = userEvent.setup();
    renderPage(<MakeExtraData initialKind="problems" />, { route: "/your-data/" });
    await user.click(await screen.findByRole("button", { name: /Rewrite with your local model/ }));
    await user.click(await screen.findByRole("button", { name: "Make drafts" }));
    const first = await screen.findByTestId("draft-d1");
    expect(screen.queryByRole("button", { name: "Mark Good" })).not.toBeInTheDocument();

    await user.click(within(first).getByRole("button", { name: "Read draft" }));
    expect(await screen.findByText("Full answer to review")).toBeVisible();
    await user.click(within(first).getByRole("button", { name: "Mark Good" }));
    await waitFor(() => expect(screen.queryByTestId("draft-d1")).not.toBeInTheDocument());
    const second = screen.getByTestId("draft-d2");
    expect(second).toBeVisible();

    await user.click(within(second).getByRole("button", { name: "Read draft" }));
    await user.click(await within(second).findByRole("button", { name: "Discard" }));
    expect(await screen.findByText("No drafts waiting for review in this category.")).toBeVisible();
    expect(state.reviews).toEqual([true, false]);
    expect(new Set(state.reviewOnly)).toEqual(new Set(["true"]));
  });

  test("a missing local model is an error, never a fabricated draft", async () => {
    server({ fail: true });
    const user = userEvent.setup();
    renderPage(<MakeExtraData initialKind="problems" />, { route: "/your-data/" });
    await user.click(await screen.findByRole("button", { name: /Rewrite with your local model/ }));
    await user.click(await screen.findByRole("button", { name: "Make drafts" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("No local model connected.");
    expect(screen.queryByTestId("draft-d1")).not.toBeInTheDocument();
  });
});
