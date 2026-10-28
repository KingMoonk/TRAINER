import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test } from "vitest";
import { mockApi } from "@/test/network";
import { renderPage } from "@/test/render";
import SettingsPage from "./page";

Element.prototype.hasPointerCapture ??= () => false; // jsdom has none; Radix Select asks
Element.prototype.scrollIntoView ??= () => {};

describe("Settings page", () => {
  test("links to the Machines and Storage pages and no longer edits the machines list (#319)", () => {
    mockApi({});
    renderPage(<SettingsPage />, { route: "/settings" });
    expect(screen.getByRole("link", { name: "Open the Machines page" })).toHaveAttribute("href", "/cluster/machines");
    expect(screen.getByRole("link", { name: "Open the Storage page" })).toHaveAttribute("href", "/storage");
    expect(screen.queryByLabelText("Machine to add")).toBeNull();
  });

  test("Developer tools is off by default and the switch remembers it in this browser", async () => {
    const user = userEvent.setup();
    mockApi({});
    renderPage(<SettingsPage />, { route: "/settings" });
    const toggle = screen.getByRole("switch", { name: "Show the API traffic console button" });
    expect(toggle).not.toBeChecked();
    expect(localStorage.getItem("settings.developerTools")).toBeNull();
    await user.click(toggle);
    expect(toggle).toBeChecked();
    expect(localStorage.getItem("settings.developerTools")).toBe("on");
    await user.click(toggle);
    expect(toggle).not.toBeChecked();
    expect(localStorage.getItem("settings.developerTools")).toBe("off");
  });

  test("remembers the selected spoken-reply voice in this browser", async () => {
    const user = userEvent.setup();
    mockApi({});
    renderPage(<SettingsPage />, { route: "/settings" });
    await user.click(screen.getByRole("combobox", { name: "Reply voice" }));
    await user.click(within(await screen.findByRole("listbox")).getByRole("option", { name: /Ryan/ }));
    expect(localStorage.getItem("voice.ttsVoice")).toBe("ryan");
  });
});
