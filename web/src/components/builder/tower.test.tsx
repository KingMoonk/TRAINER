import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, test, vi } from "vitest";
import fixture from "@/contract/fixtures/design-registry.json";
import { design, type Choices, type Registry, type Tower as TowerShape } from "@/lib/builder";
import { BUILDER, fill } from "@/lib/copy/builder";
import { Tower } from "./tower";

vi.mock("./pieces", () => ({ Pieces: () => <div>Text pieces</div> }));
afterEach(() => vi.restoreAllMocks());

const registry = fixture.body as Registry;
const choices: Choices = { L: 2, H: 4, W: 1, V: 0, T: 32, d: 32, O: [], M: { head: "tied", embedding: "learned", position: "learned", mixer: "attention", channel: "mlp_gelu", norm: "layernorm" } };
const tower = (value: Choices): TowerShape => ({
  knobs: { H: true, W: true },
  blocks: Array.from({ length: value.L }, (_, block) => ({
    block,
    parts: { ...value.M, ...Object.fromEntries(value.O.filter((o) => o.block === block).map((o) => [o.slot, o.id])) },
    own: value.O.filter((o) => o.block === block).map((o) => o.slot),
  })),
});

function setup(draft = choices, cur = choices) {
  const onPick = vi.fn();
  const onResize = vi.fn();
  const onExplain = vi.fn();
  vi.spyOn(design, "partOptions").mockResolvedValue([
    { short: "attention", name: "Attention", blurb: "", why: "" },
    { short: "linear", name: "Linear", blurb: "", why: "" },
    { short: "gqa", name: "Grouped attention", blurb: "", why: "Needs more heads" },
  ]);
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><Tower registry={registry} cur={cur} draft={draft} curTower={tower(cur)} draftTower={tower(draft)} writeBase={{}} was={null} onPick={onPick} onResize={onResize} onWords={vi.fn()} explaining={null} onExplain={onExplain} /></QueryClientProvider>);
  return { user: userEvent.setup(), onPick, onResize, onExplain };
}

describe("Tower layer presentation", () => {
  test("shows flow, numbered layers and readable dimensions without module menus", async () => {
    const { user, onExplain } = setup();
    expect(screen.getByText(/Text flows from input/)).toBeInTheDocument();
    expect(screen.getByTestId("blocks").firstElementChild).toHaveAttribute("data-testid", "block-1");
    expect(within(screen.getByTestId("block-0")).getByText("4 heads")).toBeInTheDocument();
    expect(within(screen.getByTestId("block-0")).getByText("Width · Normal")).toBeInTheDocument();
    expect(screen.queryByTestId("picker-position")).toBeNull();
    await user.click(screen.getByRole("button", { name: "open block 1" }));
    await user.click(screen.getByTestId("explain-mixer-0"));
    expect(onExplain).toHaveBeenCalledWith("mixer", 0);
    expect(screen.getByRole("button", { name: "close block 1" })).toHaveAttribute("aria-expanded", "true");
  });

  test("makes all-layer scope explicit and lets a user target only the opened layer", async () => {
    const { user, onPick } = setup();
    await user.click(screen.getByRole("button", { name: "open block 2" }));
    await user.click(screen.getByRole("button", { name: "Edit layer modules" }));
    expect(screen.getByText("Editing scope · All layers")).toBeInTheDocument();
    expect(screen.getByText(/including custom choices/)).toBeInTheDocument();
    await user.click(screen.getByTestId("picker-mixer"));
    expect(await screen.findByRole("option", { name: /Needs more heads/ })).toHaveAttribute("aria-disabled", "true");
    await user.click(screen.getByRole("option", { name: "Linear", exact: true }));
    expect(onPick).toHaveBeenLastCalledWith("mixer", "linear", null);
    await user.click(screen.getByRole("radio", { name: fill(BUILDER.scope_one, { n: 2 }) }));
    expect(screen.getByText("Editing scope · Layer 2 only")).toBeInTheDocument();
    await user.click(screen.getByTestId("picker-mixer-1"));
    await user.click(await screen.findByRole("option", { name: "Linear", exact: true }));
    expect(onPick).toHaveBeenLastCalledWith("mixer", "linear", 1);
    await user.click(screen.getByRole("button", { name: "Advanced modules", exact: true }));
    expect(screen.queryByTestId("picker-position")).toBeNull();
    expect(screen.getByRole("button", { name: "close block 2" })).toHaveAttribute("aria-expanded", "true");
  });

  test("keeps removed layers visible with their original dimensions and disables editing", () => {
    setup({ ...choices, L: 1, H: 2, W: 0 });
    const removed = within(screen.getByTestId("block-1"));
    expect(removed.getByText(BUILDER.block_removed)).toBeInTheDocument();
    expect(removed.getByText("4 heads")).toBeInTheDocument();
    expect(removed.getByText("Width · Normal")).toBeInTheDocument();
    expect(removed.getByRole("button", { name: "open block 2" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "fewer blocks" })).toBeDisabled();
  });

  test("marks added and custom layers and keeps the depth limit available", async () => {
    const { user, onResize } = setup({ ...choices, L: 12, O: [{ block: 11, slot: "mixer", id: "linear" }] });
    const added = within(screen.getByTestId("block-11"));
    expect(added.getByText(BUILDER.block_new)).toBeInTheDocument();
    expect(added.getByText(BUILDER.own_parts)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "more blocks" })).toBeDisabled();
    await user.click(screen.getByRole("button", { name: "fewer blocks" }));
    expect(onResize).toHaveBeenCalledWith(11);
  });
});
