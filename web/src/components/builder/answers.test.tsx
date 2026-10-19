import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test } from "vitest";
import { BUILDER } from "@/lib/copy/builder";
import type { Version } from "@/lib/builder";
import { mockApi } from "@/test/network";
import { renderPage } from "@/test/render";
import { Answers } from "./answers";

const version = (number: number, trained: boolean) => ({ number, name: `version ${number}`, best_checkpoint: trained ? `ckpt-${number}` : null }) as unknown as Version;

function sse(...frames: { kind: string; payload: object }[]): Response {
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const f of frames) controller.enqueue(encoder.encode(`data: ${JSON.stringify(f)}\n\n`));
      controller.close();
    },
  });
  return new Response(body, { headers: { "Content-Type": "text/event-stream" } });
}

describe("Answers", () => {
  test("says nothing can answer while no version has trained", () => {
    renderPage(<Answers versions={[version(1, false)]} current={1} />);
    expect(screen.getByText(/Version 1 has no trained weights yet/)).toBeInTheDocument();
    expect(screen.queryByTestId("ask-both")).toBeNull();
  });

  test("asks every trained version the same question and fills each row from its stream", async () => {
    const calls = mockApi({
      "POST /chat/reply": ({ body }) => {
        const { version: v } = body as { version: number };
        return sse({ kind: "text", payload: { text: `v${v} says` } }, { kind: "done", payload: { text: `v${v} says Jupiter` } });
      },
    });
    renderPage(<Answers versions={[version(1, true), version(2, true)]} current={1} />);
    await userEvent.click(screen.getByRole("button", { name: BUILDER.ask_both }));
    await waitFor(() => expect(screen.getByTestId("answer-1")).toHaveTextContent("v1 says Jupiter"));
    expect(screen.getByTestId("answer-2")).toHaveTextContent("v2 says Jupiter");
    expect(calls.filter((c) => c.path === "/chat/reply").map((c) => (c.body as { messages: unknown }).messages)).toEqual([
      [{ role: "user", content: BUILDER.example_prompt }],
      [{ role: "user", content: BUILDER.example_prompt }],
    ]);
  });

  test("a version without weights says so and only the other is asked", async () => {
    mockApi({ "POST /chat/reply": () => sse({ kind: "done", payload: { text: "ok" } }) });
    renderPage(<Answers versions={[version(1, true)]} current={1} />);
    expect(screen.getByTestId("answer-2")).toHaveTextContent(BUILDER.compare_later);
    await userEvent.click(screen.getByRole("button", { name: "Ask version 1" }));
    await waitFor(() => expect(screen.getByTestId("answer-1")).toHaveTextContent("ok"));
  });
});
