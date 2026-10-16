// Builder history copy. The desktop strings stay in studio_ia/copy.py until
// its last Tk user is removed; core editor copy is owned by #148.
export const T = {
  lead: "Review saved designs, follow training, and inspect what changed.",
  desktopEditor: "Model editing is currently available in the desktop app.",
  versions: "Versions",
  versionsDescription: "Follow each saved design from the queue through training.",
  empty: "No saved versions yet. Save a model design to start its training history.",
  queueUnavailable: "Queue state is unavailable. Saved versions remain visible.",
  curvesFailed: "Could not read the training curves. They will be checked again when training updates.",
  noCurve: "No error curve yet: it appears once training logs its first steps.",
  inspectDescription: "Open an inspector on this same saved version.",
  openQueue: "Open the queue",
} as const;
