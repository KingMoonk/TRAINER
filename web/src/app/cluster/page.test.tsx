import { screen, waitFor } from "@testing-library/react";
import { describe, expect, test } from "vitest";
import { nav } from "@/test/navigation";
import { renderPage } from "@/test/render";
import ClusterHome from "./page";

describe("Cluster home", () => {
  test("/cluster/ opens Machines", async () => {
    renderPage(<ClusterHome />, { route: "/cluster" });
    await waitFor(() => expect(nav.replace).toHaveBeenCalledWith("/cluster/machines"));
    expect(screen.getByRole("link", { name: "Machines" })).toHaveAttribute("href", "/cluster/machines");
  });
});
