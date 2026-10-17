// The words of the API traffic console (components/traffic).

export const TRAFFIC = {
  button: "API traffic console",
  shortcut: "Ctrl+`",
  title: "API traffic",
  searchPlaceholder: "Search method, path, node or body",
  searchHint: "Searches the records in this buffer only (the newest 2,000 the node kept), not anything older.",
  empty: "No requests yet. They appear here as they happen.",
  emptyFiltered: "No request in the buffer matches the filters.",
  offline: "The traffic stream is not connected. Retrying.",
  binary: "Binary body: only its size is kept.",
  oversize: "Body too large to keep: only its size is kept.",
  invalid: "The body is not valid JSON.",
  rulesTitle: "Alert rules",
  rulesHint: "Shared by every browser on this node. Traffic rules need the console open; event rules also raise a toast while it is closed.",
  noAlerts: "No alerts yet.",
} as const;
