// The strings the Your data page and the collection card show, moved from the YOURDATA and PRETRAINING tables in studio_ia/copy.py
// (which stay while the Tk pages use them).

export type Kind = "text" | "conversations" | "problems";
export const KINDS: Kind[] = ["text", "conversations", "problems"];
export const MAKE_LABELS: Record<Kind, string> = {
  text: "Generate prose",
  conversations: "Make extra conversations",
  problems: "Generate problems a rule can check",
};

export const T = {
  h1: "Your data",
  lead: "Three kinds, because each one trains a different stage.",
  storageLink: "Storage: free up disk space ›",
  passiveLink: "Passive training: see or stop it ›",
  backPretraining: "Back to Pretraining",
  backPost: "Back to Post-training",
  noSets: "No sets yet",
  search: "Search",
  status: "Status",
  loadMore: "Load more",
  pickRow: "Select a row to read it in full.",
  markGood: "Mark Good",
  discard: "Discard",
  edit: "Edit",
  save: "Save",
  cancel: "Cancel",
  editFailed: "Could not save the edit: {why}",
  already: "That is already in this set (or empty).",
  fillAll: "Fill in every field first.",
  unknownRule: "Pick one of: {rules}.",
  shown: "Showing the first {shown} of {count}.",
  shownMatches: "Showing the first {shown} matches.",
  nearDuplicates: "Near-duplicates ({n} found)",
  showing: " · showing",
  importFiles: "Import files",
  noRows: "Nothing here yet.",
  noMatches: "Nothing matches.",
} as const;

type KindCopy = {
  tab: string;
  used: string;
  lead: string;
  note: string;
  colA: string;
  colB: string;
  add: string;
  searchLabel: string;
  collectionDesc: string;
};

export const KIND: Record<Kind, KindCopy> = {
  text: {
    tab: "Text it reads",
    used: "used by Pretraining",
    lead: "Plain text. It learns by predicting the next piece.",
    note: "Held-back text is split off by document, after exact duplicates are removed, and never trained on.",
    colA: "Document",
    colB: "First line",
    add: "Add a document",
    searchLabel: "Search text",
    collectionDesc: "Plain text to download and check. Pretraining adds the ones you import to its sources.",
  },
  conversations: {
    tab: "Conversations it practises",
    used: "used by Post-training, round 1",
    lead: "A prompt and a good answer. Only the answer is trained on.",
    note: "Open one to read, edit or mark it Good. Corrections you teach in Chat appear here.",
    colA: "Prompt",
    colB: "Answer",
    add: "Write a conversation",
    searchLabel: "Search conversations",
    collectionDesc: "Conversations to download, check and add to your sets.",
  },
  problems: {
    tab: "Problems with a checkable answer",
    used: "used by Post-training, round 2",
    lead: "A prompt and a rule that decides if an answer is right: a built-in rule, or the answer key an imported set came with.",
    note: "Rule-made problems can be generated safely in any amount; the recommended collection adds real ones with an answer key.",
    colA: "Prompt",
    colB: "Rule",
    add: "Write a problem",
    searchLabel: "Search problems",
    collectionDesc: "Real problems with an answer key (math, science, reading) to download, check and add to your sets.",
  },
};

/** The fields each kind's form asks for (studio_api.routers.data FIELDS) and the prompts the Tk dialogs used. */
export const FIELDS: Record<Kind, { name: string; label: string; long?: boolean }[]> = {
  text: [{ name: "text", label: "Document text", long: true }],
  conversations: [
    { name: "user", label: "Prompt", long: true },
    { name: "assistant", label: "Good answer", long: true },
  ],
  problems: [
    { name: "prompt", label: "Problem prompt", long: true },
    { name: "rule", label: "Rule" },
  ],
};

/** studio_rules.LABELS: the rules a problem can name; the last two need the answer key an imported set came with. */
export const RULE_LABELS: Record<string, string> = {
  copy: "copy",
  reverse: "reverse",
  sort: "sort",
  count: "count",
  small_addition: "small addition",
  max_length: "stay under N characters",
  ends_cleanly: "end cleanly",
  final_number: "final number matches the key",
  choice: "picks the keyed option",
};
export const RULES = ["copy", "reverse", "sort", "count", "small_addition", "max_length", "ends_cleanly"];
export const KEYED = ["final_number", "choice"];

export const IMPORT = {
  title: "Import files",
  desc: "Files from this computer, or from a folder on the machine the app runs on.",
  fromHere: "From this computer",
  fromServer: "From a folder on the server",
  choose: "Choose files",
  upload: "Import {n} {files}",
  importing: "Importing your files…",
  places: "Places to start from",
  up: "Up one folder",
  noFiles: "No files in this folder.",
  pick: "Import {n} picked {files}",
  report: "Import report",
  close: "Close",
} as const;

/** The collection card: studio_ia/copy.py PRETRAINING. */
export const COLLECTION = {
  title: "Recommended collection",
  downloadAll: "Download and add all",
  download: "Download and add",
  import: "Add to my sources",
  stateNotDownloaded: "Not downloaded",
  stateDownloading: "Downloading",
  stateDownloaded: "Downloaded",
  stateImported: "In your sources",
  failed: "Could not finish: {error} Nothing was added; try again.",
  busy: "A download is already running.",
  empty: "Nothing to download here.",
} as const;

export function fill(text: string, values: Record<string, string | number>): string {
  return text.replace(/\{(\w+)\}/g, (_, key: string) => String(values[key] ?? ""));
}
