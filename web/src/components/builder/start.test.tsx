import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test, vi } from "vitest";
import { useQuery } from "@tanstack/react-query";
import { BUILDER } from "@/lib/copy/builder";
import { versions, type Preset } from "@/lib/builder";
import { mockApi } from "@/test/network";
import { renderPage } from "@/test/render";
import { Builder } from "./builder";
import { StartFrom } from "./side";

Element.prototype.hasPointerCapture ??= () => false;

// A new design starts from the modern preset, and the classic recipe stays one pick away.

const MODERN = { schema: 1, name: "My network", position: { id: "rope" }, block: { norm: { id: "rmsnorm" } } };
const preset = (key: string, name: string): Preset => ({ key, term: `${key}_model`, name, spec: { name } });

describe("Builder with no saved version", () => {
  test("reads the modern preset from the server as the design to edit, not an empty spec", async () => {
    const calls = mockApi({
      "GET /design/registry": { body: { part_slots: [], block_slots: [], slots: [], words: [], widths: [], heads: [] } },
      "GET /design/presets": { body: [] },
      "GET /versions": { body: [] },
      "GET /versions/current": { body: null },
      "GET /design/start": { body: { spec: MODERN, note: "" } },
      "POST /design/read": { status: 500, body: { detail: "stop here" } },
    });
    renderPage(<Builder />);
    await waitFor(() => expect(calls.some((c) => c.path === "/design/read")).toBe(true));
    const reads = calls.filter((c) => c.path === "/design/read").map((c) => (c.body as { spec: unknown }).spec);
    expect(reads).toEqual([MODERN]);
  });
});

describe("StartFrom", () => {
  test("offers the modern preset first and the classic recipe, and loads the one picked", async () => {
    mockApi({ "POST /design/preset-cost": { body: { params: 1, need: 1, fits: true, rows: 1, step: 1, slow: false, lines: [], recipe: "r" } } });
    const onLoad = vi.fn();
    renderPage(
      <StartFrom presets={[preset("modern", "Modern preset"), preset("classic", "Classic preset")]} versions={[]} onLoad={onLoad} />,
    );
    await userEvent.click(screen.getByTestId("preset-pick"));
    const options = await screen.findAllByRole("option");
    expect(options.map((o) => o.textContent)).toEqual(["Modern preset", "Classic preset"]);
    await userEvent.click(options[1]);
    await userEvent.click(await screen.findByTestId("preset-load"));
    expect(onLoad).toHaveBeenCalledWith({ name: "Classic preset" });
    expect(screen.getByText(BUILDER.preset_classic)).toBeInTheDocument();
  });
});

describe("StartFrom pretrained import", () => {
  const models = [{ id: "gpt2", name: "GPT-2 (124M)", size: 548_000_000 }];

  test("importing GPT-2 posts the form and the new version shows in the saved list", async () => {
    const saved: unknown[] = [];
    let form: FormData | undefined;
    mockApi({
      "GET /post-training/import-models": { body: models },
      "GET /post-training/options": { body: { lora_ranks: [4, 8] } },
      "GET /versions": () => ({ body: saved }),
      "POST /post-training/imports": async (_, request) => {
        form = await request.formData();
        saved.push({ number: 2, name: "version 2", change: "Started from GPT-2 (124M)", spec: { name: "GPT-2" } });
        return { body: saved[0] };
      },
    });
    function Probe() {
      const list = useQuery({ queryKey: ["versions"], queryFn: () => versions.list() });
      return <StartFrom presets={[]} versions={list.data ?? []} onLoad={() => {}} />;
    }
    renderPage(<Probe />);
    expect(screen.queryByTestId("version-pick")).toBeNull();
    await userEvent.click(screen.getByText(BUILDER.start_from_pretrained, { selector: "summary" }));
    expect(await screen.findByText(/0\.5 GB/)).toBeInTheDocument();
    await userEvent.click(await screen.findByRole("button", { name: "Import GPT-2 (124M)" }));
    expect(await screen.findByTestId("version-pick")).toBeInTheDocument();
    expect(form?.get("name")).toBe("gpt2");
  });

  test("a failed model list shows the retry", async () => {
    mockApi({
      "GET /post-training/import-models": { status: 500, body: { detail: "down" } },
      "GET /post-training/options": { body: { lora_ranks: [4, 8] } },
    });
    renderPage(<StartFrom presets={[]} versions={[]} onLoad={() => {}} />);
    expect(await screen.findByRole("button", { name: "Retry model list" })).toBeInTheDocument();
  });
});
