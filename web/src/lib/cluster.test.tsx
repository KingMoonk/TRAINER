import { act, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test, vi } from "vitest";
import { toast } from "sonner";
import { emit, mockApi } from "@/test/network";
import { renderPage } from "@/test/render";
import { ApiError } from "./api";
import { availableElsewhere, cameFrom, isCluster, isStoreUnset, nodeName, refetchPublished, seenText, useCluster, usePublished, usePull, type ClusterQueue, type PublishedVersion } from "./cluster";
import { nodeName as nodeNameDirect } from "./node-name";

const node = (address: string, hostname = "") => ({ address, hostname, local: address === "local" });
const clusterOf = (...addresses: string[]) => ({ nodes: addresses.map((a) => node(a, a)) }) as unknown as ClusterQueue;
const published = (over: Partial<PublishedVersion> = {}): PublishedVersion =>
  ({ number: 1, origin: "peer", local: false, mine: false, local_number: null, ...over }) as unknown as PublishedVersion;

describe("isCluster", () => {
  test("is false until a second machine is listed", () => {
    expect(isCluster(undefined)).toBe(false);
    expect(isCluster(clusterOf())).toBe(false);
    expect(isCluster(clusterOf("local"))).toBe(false);
    expect(isCluster(clusterOf("local", "100.64.0.2"))).toBe(true);
  });
});

describe("seenText", () => {
  test("is empty for a machine that never answered", () => {
    expect(seenText(0)).toBe("");
    expect(seenText(-5)).toBe("");
  });

  test("is a clock time with hours and minutes, no seconds", () => {
    const seconds = Date.UTC(2026, 0, 2, 13, 5, 42) / 1000;
    const text = seenText(seconds);
    expect(text).toBe(new Date(seconds * 1000).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }));
    expect(text).toMatch(/\d{1,2}[:.]\d{2}/);
    expect(text).not.toMatch(/\d[:.]\d{2}[:.]\d{2}/);
  });
});

describe("availableElsewhere", () => {
  test("keeps the versions this workspace does not have", () => {
    const list = [published({ number: 1, local: true }), published({ number: 2 }), published({ number: 3 })];
    expect(availableElsewhere(list).map((p) => p.number)).toEqual([2, 3]);
  });

  test("is empty for nothing", () => {
    expect(availableElsewhere(undefined)).toEqual([]);
    expect(availableElsewhere([])).toEqual([]);
  });
});

describe("cameFrom", () => {
  const list = [
    published({ number: 5, local_number: 2, mine: false, origin: "studio-b" }),
    published({ number: 6, local_number: 3, mine: true, origin: "local" }),
  ];

  test("names the machine that published a version pulled here", () => {
    expect(cameFrom(list, 2)).toBe("studio-b");
  });

  test("is empty for a version this machine published", () => {
    expect(cameFrom(list, 3)).toBe("");
  });

  test("is empty for a version that was never published, or with no list", () => {
    expect(cameFrom(list, 9)).toBe("");
    expect(cameFrom(undefined, 2)).toBe("");
  });
});

test("nodeName is the one from node-name.ts", () => {
  expect(nodeName).toBe(nodeNameDirect);
});

function ClusterView() {
  const { data, error, isPending } = useCluster();
  if (isPending) return <p>loading</p>;
  if (error) return <p>failed {error.message}</p>;
  return <p>{data.nodes.length} machines</p>;
}

describe("useCluster", () => {
  test("reads /cluster/merged", async () => {
    const calls = mockApi({ "GET /cluster/merged": { body: { nodes: [node("local")], rows: [] } } });
    renderPage(<ClusterView />);
    expect(await screen.findByText("1 machines")).toBeInTheDocument();
    expect(calls.map((c) => c.path)).toEqual(["/cluster/merged"]);
  });

  test("reads again on a queue event, and on a cluster event", async () => {
    let size = 1;
    const calls = mockApi({ "GET /cluster/merged": () => ({ body: { nodes: Array.from({ length: size }, (_, i) => node(i ? `p${i}` : "local")), rows: [] } }) });
    renderPage(<ClusterView />);
    await screen.findByText("1 machines");
    size = 2;
    act(() => emit("queue", {}));
    await screen.findByText("2 machines");
    size = 3;
    act(() => emit("cluster", {}));
    await screen.findByText("3 machines");
    act(() => emit("versions", {})); // not a cluster event
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(calls).toHaveLength(3);
  });

  test("a failed read shows its error without retrying", async () => {
    const calls = mockApi({ "GET /cluster/merged": { status: 500, body: { detail: "down" } } });
    renderPage(<ClusterView />);
    expect(await screen.findByText(/failed 500 from/)).toBeInTheDocument();
    expect(calls).toHaveLength(1);
  });
});

function PublishedView({ cluster }: { cluster: ClusterQueue | undefined }) {
  const { data, fetchStatus } = usePublished(cluster);
  return <p>{fetchStatus === "idle" && !data ? "idle" : `${data?.length ?? "?"} published`}</p>;
}

describe("usePublished", () => {
  test("does not read while only this machine is listed", async () => {
    const calls = mockApi({ "GET /versions/published": { body: [] } });
    renderPage(<PublishedView cluster={clusterOf("local")} />);
    expect(screen.getByText("idle")).toBeInTheDocument();
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(calls).toHaveLength(0);
  });

  test("does not read before the cluster is known", async () => {
    const calls = mockApi({ "GET /versions/published": { body: [] } });
    renderPage(<PublishedView cluster={undefined} />);
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(calls).toHaveLength(0);
  });

  test("reads the version store once other machines are listed", async () => {
    mockApi({ "GET /versions/published": { body: [published({ number: 4 })] } });
    renderPage(<PublishedView cluster={clusterOf("local", "100.64.0.2")} />);
    expect(await screen.findByText("1 published")).toBeInTheDocument();
  });
});

describe("an unset version store", () => {
  const unset = new ApiError(409, "/versions/published", "no store");

  test("is a 409 and nothing else", () => {
    expect(isStoreUnset(unset)).toBe(true);
    expect(isStoreUnset(new ApiError(500, "/versions/published", ""))).toBe(false);
    expect(isStoreUnset(new Error("offline"))).toBe(false);
  });

  test("stops the 15-second poll, and keeps it for every other state", () => {
    expect(refetchPublished({ state: { error: unset } })).toBe(false);
    expect(refetchPublished({ state: { error: null } })).toBe(15_000);
    expect(refetchPublished({ state: { error: new ApiError(500, "/versions/published", "") } })).toBe(15_000);
  });
});

function PullButton({ entry }: { entry: PublishedVersion }) {
  const pull = usePull();
  return <button onClick={() => pull.mutate(entry)}>Pull</button>;
}

describe("usePull", () => {
  const entry = published({ number: 4, origin: "studio-b" });
  const done = { id: "pull-1", kind: "pull", status: "done", created_at: 0, progress: "", error: "", result: null };

  test("posts the pull for the version and origin, follows the job, and says it worked", async () => {
    const calls = mockApi({ "POST /versions/4/pull": { body: done } });
    const success = vi.spyOn(toast, "success");
    const { client } = renderPage(<PullButton entry={entry} />);
    const invalidate = vi.spyOn(client, "invalidateQueries");
    await userEvent.setup().click(screen.getByRole("button", { name: "Pull" }));
    await waitFor(() => expect(success).toHaveBeenCalledWith("Pulled version 4 from studio-b."));
    expect(calls[0]).toMatchObject({ method: "POST", path: "/versions/4/pull", search: "?origin=studio-b" });
    expect(invalidate).toHaveBeenCalledWith({ queryKey: ["versions"] });
  });

  test("waits for the job to finish before it says so", async () => {
    let reads = 0;
    mockApi({
      "POST /versions/4/pull": { body: { ...done, status: "running" } },
      "GET /jobs/pull-1": () => ({ body: ++reads === 1 ? { ...done, status: "running" } : done }),
    });
    const success = vi.spyOn(toast, "success");
    renderPage(<PullButton entry={entry} />);
    await userEvent.setup().click(screen.getByRole("button", { name: "Pull" }));
    await waitFor(() => expect(reads).toBe(1));
    expect(success).not.toHaveBeenCalled();
    await waitFor(() => expect(success).toHaveBeenCalledWith("Pulled version 4 from studio-b."), { timeout: 3_000 });
    expect(reads).toBe(2);
  });

  test("a refusal with a detail sentence shows that sentence", async () => {
    mockApi({ "POST /versions/4/pull": { status: 409, body: { detail: "This node has no version store." } } });
    const error = vi.spyOn(toast, "error");
    renderPage(<PullButton entry={entry} />);
    await userEvent.setup().click(screen.getByRole("button", { name: "Pull" }));
    await waitFor(() => expect(error).toHaveBeenCalledWith("Could not pull version 4 from studio-b: This node has no version store."));
  });

  test.each([
    ["a body that is not JSON", { text: "<html>bad gateway</html>", status: 502 }, "The server answered 502."],
    ["JSON without a detail", { status: 500, body: { error: "x" } }, "The server answered 500."],
    ["a detail that is not a sentence", { status: 422, body: { detail: [{ msg: "bad" }] } }, "The server answered 422."],
  ])("%s falls back to the status", async (_, reply, sentence) => {
    mockApi({ "POST /versions/4/pull": reply });
    const error = vi.spyOn(toast, "error");
    renderPage(<PullButton entry={entry} />);
    await userEvent.setup().click(screen.getByRole("button", { name: "Pull" }));
    await waitFor(() => expect(error).toHaveBeenCalledWith(`Could not pull version 4 from studio-b: ${sentence}`));
  });

  test("a pull job that fails shows the job's error and does not report success", async () => {
    mockApi({ "POST /versions/4/pull": { body: { ...done, status: "failed", error: "Checksum mismatch" } } });
    const error = vi.spyOn(toast, "error");
    const success = vi.spyOn(toast, "success");
    renderPage(<PullButton entry={entry} />);
    await userEvent.setup().click(screen.getByRole("button", { name: "Pull" }));
    await waitFor(() => expect(error).toHaveBeenCalledWith("Could not pull version 4 from studio-b: Checksum mismatch"));
    expect(success).not.toHaveBeenCalled();
  });

  test("a network failure shows the error's own message", async () => {
    mockApi({
      "POST /versions/4/pull": () => {
        throw new TypeError("Failed to fetch");
      },
    });
    const error = vi.spyOn(toast, "error");
    renderPage(<PullButton entry={entry} />);
    await userEvent.setup().click(screen.getByRole("button", { name: "Pull" }));
    await waitFor(() => expect(error).toHaveBeenCalledWith("Could not pull version 4 from studio-b: Failed to fetch"));
  });
});
