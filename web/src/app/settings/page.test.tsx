import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test } from "vitest";
import { mockApi } from "@/test/network";
import { renderPage } from "@/test/render";
import SettingsPage from "./page";

describe("Settings page", () => {
  test("links to the Cluster page and no longer edits the machines list (#319)", () => {
    mockApi({});
    renderPage(<SettingsPage />, { route: "/settings" });
    expect(screen.getByRole("link", { name: "Open the Cluster page" })).toHaveAttribute("href", "/cluster");
    expect(screen.queryByLabelText("Machine to add")).toBeNull();
  });

  test("remembers the selected spoken-reply voice in this browser", async () => {
    const user = userEvent.setup();
    mockApi({});
    renderPage(<SettingsPage />, { route: "/settings" });
    await user.click(screen.getByRole("combobox", { name: "Reply voice" }));
    await user.click(screen.getByRole("option", { name: /Ryan/ }));
    expect(localStorage.getItem("voice.ttsVoice")).toBe("ryan");
  });
});
