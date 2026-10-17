import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { InlineScript } from "./inline-script";

describe("InlineScript", () => {
  it("is a data block on the client, so React does not warn about it", () => {
    const { container } = render(<InlineScript code="window.__ran = 1" />);
    const script = container.querySelector("script");
    expect(script?.getAttribute("type")).toBe("text/plain");
    expect(script?.innerHTML).toBe("window.__ran = 1");
  });
});
