# Implementation status

The modular Network interface and its service layer are implemented in this checkout. Automated tests cover registry and module contracts, numerical service behavior, scene data limits, snapshot loading and isolated one-step replay. Passing CPU tests does not establish desktop visual quality, GPU behavior or release packaging.

Validation: the full suite passed with 459 tests in 73.3 seconds and 2 skips. The focused Network UI, visualization limits, snapshot, inspection and experiment run passed all 29 tests in 26.8 seconds. Ruff and `git diff --check` pass. The skipped tests do not establish GPU behavior.

Several validation tasks still require the target environments:

- **GPU:** run an end-to-end training and inspection flow on a supported CUDA device, including forward/backward comparisons, checkpoint resume, snapshot capture and actual memory use. No GPU validation result is claimed here.
- **Frozen builds:** build and launch the PyInstaller app on macOS and Windows. Verify module discovery, the separate Panda3D viewer, passive-training subprocesses, and user-data paths in each packaged build. The hook test checks the hidden-import collection logic; it is not a frozen-app smoke test.
- **Desktop review:** manually review the Network tabs, hook-based disabled states, revision/weight flows, 3D selection and camera behavior, replay, weight edits and probe-step interaction at normal window sizes.
- **3D performance:** measure the 60 fps target at 32 lanes across 8 blocks on available integrated graphics. No frame-rate result is claimed.

Experiments now plots mean validation loss with the observed min–max range across seeds, keeps a selectable row for each completed run, and can load that run's saved design or checkpoint. The Network scene tab now includes an in-app 2D canvas for Run, Model, Window, Head, and Weights levels; checkpoint summaries and bounded replay snapshots populate the view when available. The separate 3D viewer remains available. These additions have not yet had manual desktop visual review.

Remaining implementation limits: μP is the width-scaled initialization and hidden-matrix learning-rate multiplier described in the README, not a full transfer implementation. The GPT-2 tokenizer's uncommon Unicode cases have not been fully verified. CUDA behavior, frozen builds, desktop visual quality, and the 60 fps target remain unverified in their target environments. Experiment ranges are descriptive min–max summaries, not statistical significance estimates.

Visualization snapshots are written under each run's `viz/` directory every 50 steps for Network training. Each snapshot stores full parameter arrays and gradients, and stores full updates when available; disk use therefore grows with model size and snapshot count. Window and Head render at most 32 tokens at once. Use smaller models or fewer retained snapshots when storage is constrained.
