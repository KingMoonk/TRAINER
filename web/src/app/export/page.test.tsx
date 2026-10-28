import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test, vi } from "vitest";
import { mockApi } from "@/test/network";
import { renderPage } from "@/test/render";
import ExportPage from "./page";

// The Export page against a mocked API (moved from the Storage page tests).

describe("Export page", () => {
  test("nothing trained: the page says so", async () => {
    mockApi({ "GET /storage/exports": { body: [] } });
    renderPage(<ExportPage />, { route: "/export/" });
    expect(await screen.findByRole("heading", { level: 1, name: "Export" })).toBeVisible();
    expect(await screen.findByText(/Nothing to export yet\./)).toBeVisible();
  });

  test("a version GGUF cannot hold greys out its button and says why; the others download", async () => {
    const why = "GGUF export covers GPT-2's architecture only.";
    const asked: string[] = [];
    mockApi({
      "GET /storage/exports": {
        body: [
          { number: 1, name: "plain", gguf_problem: "" },
          { number: 2, name: "exotic", gguf_problem: why },
        ],
      },
      "GET /versions/1/export": ({ search }) => {
        asked.push(search);
        return { text: "zip", headers: { "content-disposition": 'attachment; filename="plain-hf.zip"' } };
      },
      "GET /versions/2/export": { status: 409, body: { detail: "Version 2 has no checkpoint yet." } },
    });
    // jsdom cannot make object URLs or follow a download link: record the link the page clicks.
    URL.createObjectURL = vi.fn(() => "blob:export");
    URL.revokeObjectURL = vi.fn();
    const downloads: string[] = [];
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (this: HTMLAnchorElement) {
      downloads.push(this.download);
    });

    const user = userEvent.setup();
    renderPage(<ExportPage />, { route: "/export/" });
    const list = await within(screen.getByRole("main")).findByRole("list"); // the toaster is a list too
    const rows = within(list).getAllByRole("listitem");
    expect(rows).toHaveLength(2);
    expect(within(rows[1]).getByRole("button", { name: "GGUF" })).toBeDisabled();
    expect(within(rows[1]).getByText(why)).toBeVisible();
    expect(within(rows[0]).getByRole("button", { name: "GGUF" })).toBeEnabled();

    await user.click(within(rows[0]).getByRole("button", { name: "GGUF" }));
    await waitFor(() => expect(downloads).toEqual(["plain-hf.zip"]));
    expect(asked[0]).toContain("kind=gguf");

    await user.click(within(rows[1]).getByRole("button", { name: "Hugging Face" }));
    expect(await screen.findByText("Version 2 has no checkpoint yet.")).toBeVisible();
    expect(downloads).toHaveLength(1);
  });
});
