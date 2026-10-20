// The words of the loading bar (components/busy-indicator.tsx): heavy work on the server and slow reads in the browser.

export const BUSY = {
  region: "Work in progress",
  loading: "Loading {what}",
  stepOf: "{done} of {total}",
  more: " +{count} more",
  /** What a slow read is called, by its React Query key (joined with spaces). Any other key is spelled out from its own words. */
  queries: {
    queue: "the queue",
    "queue outlook": "when jobs start",
    versions: "versions",
    "versions current": "the current version",
    "distributed runs": "runs",
    "distributed invites": "invitations",
    "cluster nodes": "machines",
    "cluster queue": "the cluster queue",
    "cluster jobs": "cluster jobs",
    "cluster access": "who may connect",
    "pretraining preview": "the text preview",
    "pretraining text_sets": "text sets",
    "storage exports": "exports",
    "storage leftovers": "leftover files",
    "storage free": "free space",
  } as Record<string, string>,
  somethingElse: "data",
};
