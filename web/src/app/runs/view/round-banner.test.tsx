import { screen } from "@testing-library/react";
import { describe, expect, test } from "vitest";
import { run } from "@/app/cluster/training/test-data";
import { renderPage } from "@/test/render";
import { RoundBanner } from "./round-banner";

// The banner before the first round: what the coordinator is preparing.

describe("the round banner while the run is prepared", () => {
  test("it says what is going on and shows the documents tokenized so far", () => {
    const prep = { phase: "tokenizing" as const, note: "Tokenizing the text: 1,200 of 4,800 documents.", done: 1200, total: 4800 };
    renderPage(<RoundBanner run={run({ state: "running", phase: "preparing", round: 0, prep })} />);
    expect(screen.getByTestId("prep-note")).toHaveTextContent("Tokenizing the text: 1,200 of 4,800 documents.");
    expect(screen.getByRole("progressbar", { name: "Preparing the data" })).toHaveAttribute("aria-valuenow", "1200");
  });

  test("a phase with no count shows the note and no bar", () => {
    const prep = { phase: "inviting" as const, note: "Sending the invitations to the other machines.", done: 0, total: 0 };
    renderPage(<RoundBanner run={run({ phase: "inviting", round: 0, prep })} />);
    expect(screen.getByTestId("prep-note")).toHaveTextContent("Sending the invitations");
    expect(screen.queryByRole("progressbar", { name: "Preparing the data" })).toBeNull();
  });

  test("without a preparation there is no prep line", () => {
    renderPage(<RoundBanner run={run({ prep: null })} />);
    expect(screen.queryByTestId("prep")).toBeNull();
  });
});
