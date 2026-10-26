import { render, screen } from "@testing-library/react";
import { describe, expect, test } from "vitest";
import { GateCard } from "./gate-card";
import { T } from "./copy";
import type { Gate } from "./logic";

const check = (name: string, ok: boolean | null) => ({ name, detail: "detail", ok });

function show(gate: Gate | undefined, status = "done") {
  render(<GateCard round={1} status={status} gate={gate} queue={undefined} checkpoint="ck1" overlap={undefined} />);
}

describe("GateCard", () => {
  test("a passed gate reads as success and says Chat is live", () => {
    show({ ok: true, verdict: "Answers now look like answers.", checks: [check("Held-back conversations: answer error fell", true)] });
    expect(screen.getByTestId("gate-state")).toHaveTextContent(T.gatePassed);
    expect(screen.getByText("Answers now look like answers.")).toHaveClass("text-success");
    expect(screen.getByText(/Live: Chat uses ck1/)).toBeVisible();
    expect(screen.queryByText(/Switch back/)).toBeNull();
  });

  test("a failed gate is never styled as success and says the round is not live", () => {
    show({
      ok: false,
      verdict: "Answers now look like answers.",
      checks: [
        check("Held-back conversations: answer error fell", false),
        check("Answers still stop: end-of-answer rate did not fall", true),
        check("Raw text: error stayed within the drift budget", null),
      ],
    });
    expect(screen.getByTestId("gate-state")).toHaveTextContent(T.gateFailed);
    expect(screen.getByText("Answers now look like answers.")).not.toHaveClass("text-success");
    expect(screen.getByText("Answers now look like answers.")).toHaveClass("text-destructive");
    expect(screen.getByText("Held-back conversations: answer error fell")).toBeVisible();
    expect(screen.getByText("Answers still stop: end-of-answer rate did not fall")).toBeVisible();
    expect(screen.getByText(/not live/)).toBeVisible();
  });

  test("an unmeasured gate is neutral, not success", () => {
    show({ ok: true, verdict: "", checks: [check("a", null), check("b", null)] });
    expect(screen.getByTestId("gate-state")).toHaveTextContent(T.gateUnmeasured);
    expect(screen.getByText(T.verdictUnmeasured)).not.toHaveClass("text-success");
    expect(screen.getByText(/not live/)).toBeVisible();
  });

  test("before the round ends, Round 1 lists the conversation and stop checks as pending", () => {
    show(undefined, "running");
    expect(screen.getByText(T.conversationsCheck)).toBeVisible();
    expect(screen.getByText(T.stopCheck)).toBeVisible();
    expect(screen.queryByTestId("gate-state")).toBeNull();
  });
});
