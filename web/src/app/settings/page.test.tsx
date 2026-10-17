import { screen } from "@testing-library/react";
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
});
