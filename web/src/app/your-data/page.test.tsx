import { act, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test, vi } from "vitest";
import { emit, mockApi } from "@/test/network";
import { renderPage } from "@/test/render";
import YourDataPage from "./page";

// The Your data page against a mocked API (the component tier of tests/your-data.spec.ts): three summaries read as jobs, the rows a
// page at a time, the recommended collection with live download progress, an uploaded file imported as a job, a folder on the server picked.

const job = (id: string, kind: string, result: unknown) => ({ id, kind, status: "done", progress: "", fraction: null, result, error: "", created_at: 0, started_at: 0, finished_at: 1 });

const row = (id: string, a: string, b: string, status = "In use") => ({ id, a, b, status, dataset: "Your files" });
const summary = (kind: string, rows: ReturnType<typeof row>[], sets: { id: string; name: string; count: number }[]) => ({
  kind,
  count: rows.length,
  characters: 100,
  used_by: kind === "text" ? "used by Pretraining" : "used by Post-training, round 1",
  count_line: `${rows.length} ${kind === "text" ? "documents" : "conversations"} · ${sets.length} sets`,
  sets,
  rules: {},
  near_duplicates: 0,
  held_back: 0,
  scanned: rows.length,
  rows,
});

const source = (id: string, name: string, state: string, extra: object = {}) => ({ id, name, size: "64 MB", records: "77,941 stories", state, completed: 0, total: 0, error: "", ...extra });

function server() {
  const state = { uploads: [] as Blob[], paths: [] as string[][], downloads: [] as unknown[], imported: false };
  const conversations = () =>
    state.imported
      ? [row("c1", "Name a prime.", "7 is a prime number."), row("c2", "Say hello.", "Hello there.", "Review"), row("c3", "Imported prompt", "Imported answer")]
      : [row("c1", "Name a prime.", "7 is a prime number."), row("c2", "Say hello.", "Hello there.", "Review")];
  const kindOf = (path: string) => path.split("/")[2];

  mockApi({
    "POST /data/refresh_downloads": { status: 202, body: job("r1", "refresh_downloads", []) },
    "POST /data/*/summary": ({ path }) => {
      const kind = kindOf(path);
      const rows = kind === "conversations" ? conversations() : [];
      const sets = kind === "conversations" ? [{ id: "d1", name: "Your conversations", count: rows.length }, ...(state.imported ? [{ id: "d2", name: "Imported set", count: 1 }] : [])] : [];
      return { status: 202, body: job(`s-${kind}-${state.imported}`, "data", summary(kind, rows, sets)) };
    },
    "GET /data/*/rows": ({ path }) => {
      const kind = kindOf(path);
      const rows = kind === "conversations" ? conversations() : kind === "text" ? [row("t1", "story.txt", "A small boat crossed the lake.")] : [];
      return { body: { rows, next_cursor: null } };
    },
    "GET /data/collections/*": { body: [source("tinystories", "TinyStories", "downloading", { total: 100, completed: 10 }), source("wikitext2", "WikiText-2", "not downloaded")] },
    "POST /data/collections/*/download": ({ body }) => {
      state.downloads.push(body);
      return { body: { started: true } };
    },
    "POST /data/conversations/import": () => {
      state.imported = true;
      return { status: 202, body: job("i1", "import_data", ["notes.txt: 1 added"]) };
    },
    "POST /data/conversations/import_paths": ({ body }) => {
      state.paths.push((body as { paths: string[] }).paths);
      return { status: 202, body: job("i2", "import_data", ["corpus.txt: 4 added"]) };
    },
    "GET /data/browse": ({ search }) => {
      const path = new URLSearchParams(search).get("path");
      if (path === null) return { body: { path: null, parent: null, entries: [{ name: "/home/me", path: "/home/me", is_dir: true, size: null }] } };
      return { body: { path, parent: null, entries: [{ name: "corpus.txt", path: `${path}/corpus.txt`, is_dir: false, size: 2048 }] } };
    },
  });
  return state;
}

describe("Your data page", () => {
  test("on load the overview and toolbar show and no examples are read", async () => {
    server();
    renderPage(<YourDataPage />, { route: "/your-data/?kind=conversations" });
    const toolbar = await screen.findByRole("toolbar", { name: "Data actions" });
    for (const name of ["Write a conversation", "Import files", "Make extra data"]) expect(within(toolbar).getByRole("button", { name })).toBeVisible();
    expect(within(toolbar).getByRole("button", { name: /Near-duplicates/ })).toBeDisabled();
    await waitFor(() => expect(screen.getByTestId("stat-total")).toHaveTextContent("2"));
    expect(screen.queryByTestId("row-c1")).toBeNull();
    expect(screen.queryByTestId("detail")).toBeNull();
    expect(screen.getByRole("button", { name: "Browse examples" })).toHaveAttribute("aria-expanded", "false");
  });

  test("the header links to Collect and Storage", async () => {
    server();
    renderPage(<YourDataPage />, { route: "/your-data/?kind=conversations" });
    expect(await screen.findByRole("link", { name: "Collect in the background ›" })).toHaveAttribute("href", "/collect");
    expect(screen.getByRole("link", { name: /^Storage/ })).toHaveAttribute("href", "/storage");
  });

  test("the set cards show each set's name, count and share of the total", async () => {
    server();
    renderPage(<YourDataPage />, { route: "/your-data/?kind=conversations" });
    const card = await screen.findByTestId("set-d1");
    expect(card).toHaveTextContent("Your conversations");
    expect(card).toHaveTextContent("2");
    expect(within(card).getByRole("progressbar", { name: "Your conversations share of the total" })).toHaveAttribute("aria-valuenow", "100");
  });

  test("Make extra data opens its panel under the toolbar", async () => {
    server();
    const user = userEvent.setup();
    renderPage(<YourDataPage />, { route: "/your-data/" });
    await user.click(await screen.findByRole("button", { name: "Make extra data" }));
    expect(await screen.findByRole("group", { name: "What to make" })).toBeVisible();
  });

  test("Write a conversation opens the add form under the toolbar", async () => {
    server();
    const user = userEvent.setup();
    renderPage(<YourDataPage />, { route: "/your-data/?kind=conversations" });
    await user.click(await screen.findByRole("button", { name: "Write a conversation" }));
    expect(await screen.findByLabelText("Prompt")).toBeVisible();
  });

  test("searching opens the examples list without the Browse click", async () => {
    server();
    const user = userEvent.setup();
    renderPage(<YourDataPage />, { route: "/your-data/?kind=conversations" });
    await user.type(await screen.findByRole("textbox", { name: "Search conversations" }), "prime");
    expect(await screen.findByTestId("row-c1")).toBeVisible();
  });

  test("the three kinds show their counts; Browse examples lists one text column plus status, and a row opens in the detail", async () => {
    server();
    const user = userEvent.setup();
    renderPage(<YourDataPage />, { route: "/your-data/?kind=conversations" });
    expect(await screen.findByRole("heading", { level: 1, name: "Your data" })).toBeVisible();
    await waitFor(() => expect(screen.getByRole("tab", { name: /Conversations it practises/ })).toHaveTextContent("2 conversations · 1 sets"));
    await user.click(screen.getByRole("button", { name: "Browse examples" }));
    const first = await screen.findByTestId("row-c1");
    expect(first).toHaveTextContent("Name a prime.");
    expect(first).not.toHaveTextContent("7 is a prime number.");
    expect(first).toHaveTextContent("In use");
    expect(screen.getByTestId("detail")).toHaveTextContent("Select a row");
    await user.click(screen.getByTestId("row-c2"));
    expect(screen.getByTestId("detail")).toHaveTextContent("Hello there.");
    expect(screen.getByRole("button", { name: "Mark Good" })).toBeVisible();
    expect(screen.getByRole("link", { name: "‹ Back to Post-training" })).toHaveAttribute("href", expect.stringMatching(/\/post-training/));
  });

  test("a text row shows its first line beside its document, since the document alone does not tell rows apart", async () => {
    server();
    const user = userEvent.setup();
    renderPage(<YourDataPage />, { route: "/your-data/?kind=text" });
    await user.click(await screen.findByRole("button", { name: "Browse examples" }));
    const first = await screen.findByTestId("row-t1");
    expect(within(first).getByText("A small boat crossed the lake.", { exact: true })).toBeVisible();
    expect(first).toHaveTextContent("story.txt");
  });

  test("an uploaded file is imported and the new row and set appear", async () => {
    const state = server();
    // Node's fetch cannot serialise a jsdom File (it arrives as an anonymous empty blob), so the upload is read where the page builds it.
    const append = FormData.prototype.append;
    vi.spyOn(FormData.prototype, "append").mockImplementation(function (this: FormData, name: string, value: string | Blob, filename?: string) {
      if (value instanceof File) state.uploads.push(value);
      return (append as (...args: unknown[]) => void).call(this, name, value, filename);
    } as typeof FormData.prototype.append);
    const user = userEvent.setup();
    renderPage(<YourDataPage />, { route: "/your-data/?kind=conversations" });
    await user.click(await screen.findByRole("button", { name: "Browse examples" }));
    expect(await screen.findByTestId("row-c1")).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Import files" }));
    await user.upload(screen.getByLabelText("Choose files"), new File(["hello world"], "notes.txt", { type: "text/plain" }));
    await user.click(screen.getByRole("button", { name: "Import 1 file" }));
    expect(await screen.findByTestId("import-report")).toHaveTextContent("notes.txt: 1 added");
    expect(state.uploads.map((file) => (file as File).name)).toEqual(["notes.txt"]);
    const content = await new Promise<string>((resolve) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result));
      reader.readAsText(state.uploads[0]);
    });
    expect(content).toBe("hello world");
    // the rows and the sets are read again: the imported ones are listed
    expect(await screen.findByTestId("row-c3")).toHaveTextContent("Imported prompt");
    await waitFor(() => expect(screen.getByTestId("set-d2")).toHaveTextContent("Imported set"));
    expect(screen.getByTestId("set-d1")).toHaveTextContent("3");
  });

  test("a file in a folder on the server is picked and imported", async () => {
    const state = server();
    const user = userEvent.setup();
    renderPage(<YourDataPage />, { route: "/your-data/?kind=conversations" });
    await user.click(await screen.findByRole("button", { name: "Import files" }));
    await user.click(await screen.findByRole("button", { name: "/home/me/" }));
    await user.click(await screen.findByRole("checkbox"));
    await user.click(screen.getByRole("button", { name: "Import 1 picked file" }));
    expect(await screen.findByTestId("import-report")).toHaveTextContent("corpus.txt: 4 added");
    expect(state.paths).toEqual([["/home/me/corpus.txt"]]);
  });

  test("download progress changes one row in place and the other rows keep their nodes", async () => {
    const state = server();
    const user = userEvent.setup();
    renderPage(<YourDataPage />, { route: "/your-data/" });
    await user.click(await screen.findByRole("tab", { name: /Text it reads/ }));
    const first = await screen.findByTestId("source-tinystories");
    const second = await screen.findByTestId("source-wikitext2");
    expect(first).toHaveTextContent("TinyStories");
    expect(screen.getByTestId("source-tinystories-status")).toHaveTextContent(/^Downloading$/);

    act(() => emit("download", { id: "tinystories", text: "Downloading · 42 MB of 64 MB", completed: 42, total: 64 }));
    await waitFor(() => expect(screen.getByTestId("source-tinystories-status")).toHaveTextContent("Downloading · 42 MB of 64 MB"));
    expect(screen.getByRole("progressbar", { name: "TinyStories progress" })).toHaveAttribute("aria-valuenow", "66");
    expect(screen.getByTestId("source-tinystories")).toBe(first);
    expect(screen.getByTestId("source-wikitext2")).toBe(second);

    await user.click(screen.getByRole("button", { name: "Download and add" }));
    await waitFor(() => expect(state.downloads).toEqual([{ ids: ["wikitext2"], install: true }]));
    await user.click(screen.getByRole("button", { name: "Download and add all" }));
    await waitFor(() => expect(state.downloads).toHaveLength(2));
    expect(state.downloads[1]).toEqual({ ids: null, install: true });
  });
});
