# Groundwork AI Studio

A native Python desktop studio for creating datasets and training your own language model from random weights.

**PyTorch runs the maths on the CPU, an NVIDIA GPU or an Apple GPU; NumPy and pandas handle data.** Python's built-in Tkinter supplies the interface. The transformer modules, optimizers, tokenizers, checkpoint format and text sampler are implemented in this repository; PyTorch autograd supplies the gradients. Everything runs locally.

## Quickstart

Requires Python 3.12 or newer with Tkinter (the pinned NumPy 2.5 and pandas 3.0 need 3.12).

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Windows (PowerShell):

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

On Windows you can instead double-click `start_studio.bat`. The details below cover GPU support, the workflow and the command line.


## Lint and git hooks

Run once per clone:

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
git config core.hooksPath .githooks
```

- **pre-commit** lints the staged Python files with [ruff](https://docs.astral.sh/ruff/) and applies its safe fixes. The rules are in `ruff.toml`: syntax errors, undefined names, unused imports and variables. Style is not enforced.
- **pre-push** runs `ruff check .` and the fast tests (about 3 seconds); a failure stops the push. The GUI, network and 3D viewer tests take about 40 seconds more and run in CI only. They open windows, so every `gui_*.py` skips itself unless `CI` is set (GitHub Actions does) or you opt in with `GROUNDWORK_GUI_TESTS=1` (see `tests/windows.py`).

## Launch

On Windows, double-click **start_studio.bat**. It creates a virtual environment if needed, installs the two dependencies, and opens the desktop window.

Or run from this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Use the `.venv` interpreter in PyCharm. Python must include Tkinter (included with the standard Windows installer). The old web server and JavaScript version have been removed.

## GPU acceleration

Choose **Model blueprint → Compute device**:

- `auto` (default): use a CUDA GPU, else an Apple GPU (Metal), when a small calculation check passes; otherwise the CPU, with the reason shown. Small models stay on the CPU because kernel launches cost more than the arithmetic.
- `cuda` / `mps`: require that device. If it is unavailable, report the error instead of silently changing devices.
- `cpu`: never probe a GPU.

float64 precision is not available on Apple GPUs; such runs fall back to the CPU.

Chat has a separate **Inference device** selector. For chat, reload the checkpoint after changing this setting. The model status and training log show the actual device. Resuming a run uses the currently selected training device, so CPU checkpoints can continue on GPU and vice versa.

For a machine with an NVIDIA driver and CUDA 13 toolkit already installed:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
.\.venv\Scripts\python.exe train.py --list-devices
```

PyTorch's default wheel from PyPI already includes Apple Metal (MPS) support. For an NVIDIA GPU install the CUDA build of PyTorch that matches your driver (see requirements-gpu.txt), then restart the studio. The backend targets one GPU (device 0).

Weights, activations and optimizer state stay on the GPU during GPU training. Initialization and dropout random draws remain NumPy-based so saved random states stay portable; batches and dropout masks transfer to the GPU. Checkpoints store NumPy arrays (`.npz`), so a checkpoint trained on one device loads on any other. GPU reductions may differ slightly from CPU floating-point results. Small models or batches run faster on the CPU because kernel launch and transfer overhead dominate; `auto` keeps them there.

## Workflow

1. **Dataset library:** create a dataset, choose one or multiple files with **Import files**, or write examples manually. JSONL, JSON, CSV, TSV, tagged TXT and plain text are supported. Pandas parses tables and computes dataset statistics. Edit prompt/response pairs, full conversations or raw text; mark examples reviewed; export JSONL or CSV. Exact content duplicates are skipped. Existing workspace datasets are retained.
2. **Model blueprint:** use the guided setup to choose data, tokenizer, dimensions, training settings and objective. **Network → Design** edits a canonical ModelSpec; **Sandbox** swaps registry modules and their settings, while **Variants** compares named presets and estimates. Incompatible modules show why they are unavailable. These presets describe architectures and do not include pretrained weights.
3. **Training console:** view actual training and validation loss, token throughput and optimizer steps. Pause, stop and save, inspect saved runs, resume checkpoints or export a complete run ZIP. Training runs on a worker thread so the UI stays responsive.
4. **Playground:** load a `.npz` checkpoint and sample with temperature, top-k and response-length controls. Use **This reply is wrong** to send the conversation to Improve with feedback, where you pick a better reply or write the right one; an answer you write is also saved to the selected dataset.

The file picker is the supported import method; operating-system drag-and-drop is not required or installed.

## The learning engine

`studio_model.py` drives the decoder-only training path on PyTorch tensors; `studio_compile.py` assembles a ModelSpec from the registry's modules and runs it with autograd:

- Learned token/position embeddings, pre-layer normalization, multi-head causal self-attention, residual paths, tanh-approximation GELU, dropout, final normalization and tied output embeddings, plus the alternative modules in `studio_modules/` (RoPE, ALiBi, RMSNorm, GQA, linear attention, SSM, GRU, LSTM, convolution, SwiGLU, mixture of experts, untied heads).
- Gradients by autograd (every module's forward pass is checked against finite differences in float64), stable cross-entropy, global gradient clipping, gradient accumulation, AdamW with bias correction (also SGD, Lion and Muon), warmup with cosine or warmup-stable-decay schedules.
- Character vocabularies fitted to the training split, or a fixed UTF-8 byte vocabulary with special conversation tokens. Character tokenizers use an unknown token for unseen characters; bytes cover all UTF-8 text.
- Next-token prediction on raw text. Conversation targets can include every token or only assistant content and end-of-turn markers.

This is a small-model experimentation studio. Training from scratch is suited to learning and narrow tasks; a generally capable chatbot needs far more data and compute. A falling training loss alone does not establish generalization. Short context windows can truncate conversational history. On GPU, batch size, context length and model width determine memory demand; reduce them if the device runs out of memory.

## Network engine and inspection services

The modular network engine represents a design as a validated, JSON-serializable `ModelSpec`. Its slots describe the tokenizer, embedding, position method, block norm/token mixer/channel mixer, output head, objective, optimizer, learning-rate schedule, initializer and precision. The compiler builds the residual-block template from that spec; registry metadata supplies module configuration fields, compatibility requirements, hooks and parameter/compute estimates. New modules live one per file under `studio_modules/<slot>/` and register themselves. The Network page now includes **Design**, **Sandbox**, **Variants**, **Experiments**, **Trace a token**, **Inside a head**, **Value map**, **Loss**, **Head lab**, **3D** and **Versions** tabs. Sandbox controls come from registry metadata; incompatible choices show their reason, and inspection controls are disabled when the selected block lacks required hooks.

The inspection tabs read saved checkpoints and report measured values: token attention and intermediate predictions, a head's scores and probabilities, value/output projection summaries, selected-token loss and gradient, and head ablation statistics. **Versions** manages architecture revisions, checkpoint weights and backups. These views describe the selected checkpoint; they do not explain a head's behavior beyond the measured quantities shown.

`Experiments` runs paired seeds and can hold tokens, estimated parameters, estimated compute or wall-clock time equal. It records width adjustments, achieved budgets and run metrics. Network training saves a snapshot every 50 steps; only the newest 20 snapshots per run are kept by default (`snapshot_keep`). Each snapshot includes full model weights and gradients, plus full updates when available. The 3D viewer is a separate window with five levels: Run, Model, Window, Head and Weights. Its menus choose level and Build/Train/Ride a token mode. Mouse drag orbits the camera, the wheel zooms between levels, and WASD/Q/E pans. Model shows binned attention and stream norms; Window and Head display at most 32 tokens. Replay controls play, pause and scrub saved snapshots; **1 real step** runs one optimizer step on a private copy from a checkpoint. At Weights, select weight/gradient/update and a matrix; keyboard `+`/`-` changes a weight by a small amount and saves a new checkpoint version.

The `init/mup` option implements μP (Yang et al. 2022) relative to an explicit base width, with m = d_model / base_width: hidden matrices are initialized at std/√m and the untied readout at std/m; learning-rate multipliers follow `optimizer_family` (Adam-like: hidden and readout 1/m; SGD-like: input-like parameters m, readout 1/m); attention scores get an extra 1/√m and tied-readout logits 1/m. Set `optimizer_family` to match the optimizer you train with. At m = 1 it changes nothing.

Exact duplicates are removed **before** the train/validation split. Validation is separated before character-vocabulary fitting. Each evaluation covers up to 20 batches. The chart's training curve uses baseline evaluation followed by sampled batch losses; validation uses dropout-free inference. With validation disabled or only one example, the validation curve is unavailable.

## Data formats

Conversation JSONL (one object per line):

```json
{"messages":[{"role":"user","content":"What is 2 + 3?"},{"role":"assistant","content":"5"}]}
```

Simple pairs and raw text are also accepted:

```json
{"user":"Say hello","assistant":"Hello!"}
{"text":"A paragraph of training prose."}
```

CSV/TSV supports `user,assistant`, `prompt,completion`, `question,answer`, or `instruction,input,output`. A `messages` column may contain a JSON conversation. Import files must be UTF-8 and below 100 MB. Invalid rows are counted in the import report; malformed JSON reports its location.

## Checkpoints and reproducibility

Every experiment writes a separate `runs/run_<id>/` directory:

- `last.npz`: latest weights, Adam state, tokenizer, configuration, random-generator state and metric history.
- `best.npz`: best checkpoint evaluated in this run, using validation loss when available and training batch loss otherwise.
- `dataset.jsonl`: original input snapshot, including held-out examples.
- `config.json`, `tokenizer.json`, `metrics.jsonl`, `run.json`: configuration, progress and run metadata.

NumPy archives contain numeric arrays and JSON metadata, and load with `allow_pickle=False`. The format is shared by CPU and GPU, including old CPU-only checkpoints. Resume requires the original snapshot, architecture, tokenizer, objective, split and batching settings; the compute device may change. The GUI creates a new run directory and restores optimizer and random state. Extending the step target also extends the cosine schedule; it is not a bit-for-bit replay of an originally longer run. A resumed run may not create a new `best.npz` if it never improves the inherited best score; the original best checkpoint remains in its original run folder.

Workspace datasets and drafts live in `data/workspace.json`. Existing exports are preserved. Dataset edits do not alter a run's snapshot. Exported run ZIPs include both weights and that training snapshot.

## Command line

```powershell
.\.venv\Scripts\python.exe train.py --data my-data.jsonl --steps 500
.\.venv\Scripts\python.exe train.py --data my-data.jsonl --steps 500 --device cuda
.\.venv\Scripts\python.exe train.py --resume runs\run_ID\last.npz --steps 1000 --device cpu
```

`--steps` is the total optimizer-step target, including resumed steps. Pass `--config config.json` for model/training settings or `--output path` for a run folder. Press Ctrl+C to stop after the current optimizer step and save.

## Passive training

Passive training keeps training on its own: it trains until the test error settles, saves a checkpoint, learns from the teacher model if one is set up, then goes again or steps back to the best checkpoint. It runs in a separate process, so it keeps going with Groundwork closed.

A small watchdog (`studio_passive_watch.py`) checks the trainer's heartbeat. After 60 seconds of silence, or if the process is gone, it starts the trainer again from the last saved step; after 5 restarts in 10 minutes it gives up and records why. It also registers itself to start at sign-in (a LaunchAgent on macOS, a Task Scheduler task on Windows) and removes that entry when the session ends. The plan can keep the computer awake (`caffeinate` on macOS, a keep-awake request on Windows) and pause on battery.

The record lives in `data/passive.sqlite` (plan, state, process ids, heartbeat) and `runs/<session id>/` (journal, checkpoints, `trainer.log`). To stop a session, press Stop on the Training page; the watchdog then exits by itself.

## Releases

Push a `v*` tag (for example `git tag v1.0.0 && git push origin v1.0.0`) and `.github/workflows/release.yml` builds the app with PyInstaller on macOS and Windows and attaches `Groundwork-macos.zip` and `Groundwork-windows.zip` to a GitHub Release for that tag. Running the workflow by hand (Actions, Release, Run workflow) builds and uploads the zips as workflow artifacts without creating a release.

The builds are unsigned, so macOS Gatekeeper (right-click the app, Open) and Windows SmartScreen (More info, Run anyway) will warn on first launch. A packaged app keeps `data/` and `runs/` in your user data folder (`~/Library/Application Support/Groundwork` on macOS, `%APPDATA%\Groundwork` on Windows) so they survive an upgrade.

## Codebase structure

Entry points and engine services sit in the repository root; swappable model/training modules live in `studio_modules/`, tests in `tests/`, and packaging hooks in `packaging/`.

### Layers

| Layer | Modules | Role |
|---|---|---|
| Entry points | `app.py`, `train.py`, `start_studio.bat` | Launch the desktop GUI, run training from the command line, or set up and launch on Windows. |
| GUI shell | `studio_ia/shell.py` | `Studio` (Tk window, sidebar, page router). |
| GUI pages | `studio_ia/pages/` | One page per route; pages talk only to `studio_ia/services.py`, which wraps the workspace, the version store and the queue. |
| GUI helpers | `studio_ia/widgets.py`, `studio_ia/theme.py`, `studio_ia/copy.py`, `studio_ux.py`, `studio_glossary.py` | Shared widgets, colours and fonts, every string on screen, time estimates, plain-English glossary. |
| Data | `studio_core.py`, `studio_catalog.py`, `studio_planning.py` | Workspace persistence, parsing and normalization, verified dataset downloads, bounded dataset mixtures. |
| Learning engine | `studio_model.py`, `studio_backend.py` | Tokenizers, transformer, handwritten gradients, AdamW, training loop, checkpoints, inference, CPU/CUDA backend selection. |
| Modular network engine | `studio_registry.py`, `studio_spec.py`, `studio_validate.py`, `studio_estimate.py`, `studio_compile.py`, `studio_modules/` | Registry-discovered modules, canonical ModelSpec, compatibility checks, module estimates and the compiler. |
| Network editing | `studio_arch.py`, `studio_archdb.py` | Desktop architecture editor, per-block designs, and a versioned SQLite store of revisions and weight files. |
| Curriculum and post-training | `studio_curriculum.py`, `studio_rlhf.py` | Stage-by-stage training across datasets; DPO/IPO preference training and the Ollama teacher client. |
| Inspection and experiments | `studio_inspect.py`, `studio_summary.py`, `studio_experiments.py`, `studio_snapshot.py`, `studio_scene.py` | Probe, summary, experiment, snapshot/replay and renderer-independent scene services. |
| 3D viewer | `studio_vizbridge.py`, `studio_brain3d.py` | `studio_brain3d.py` is a Panda3D viewer run as a separate process. `studio_vizbridge.py` connects it to the GUI over a loopback socket. |

### Dependency flow

```
app.py ──► studio_ia.shell ──► studio_ia.pages ──► studio_ia.services ──► engine modules, studio_queue
train.py ─┐
          ├──► studio_model ──► studio_backend, studio_arch, studio_core, studio_planning
          │        ▲
          │        └── studio_rlhf, studio_curriculum
          └──► studio_core, studio_backend
```

`studio_model.py` owns training and checkpoint compatibility. The modular compiler and its services sit alongside it; they do not import Tkinter. The Panda3D viewer stays in its separate process and communicates through `studio_vizbridge.py`.

### Runtime data

Created at run time and git-ignored:

- `data/`: `workspace.json` (datasets and drafts), `architectures.sqlite`, `weights/`, `backups/`, downloaded corpora.
- `runs/run_<id>/`: checkpoints (`last.npz`, `best.npz`), dataset snapshot, config, metrics.
- `exports/`, `artifacts/`: user exports.
- `.venv/`: local virtual environment.

### Tests

`tests/` contains focused `unittest` modules for the engine, modules, data, architecture, inspection services, viewer bridge and GUI integration. GPU and display-dependent tests may skip when that hardware or display is unavailable.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests cover data preservation, reproducible recipes, causal attention, numerical gradient checking, learning, inference, module estimates and checkpoint resumption. CUDA tests compare supported GPU operations against CPU and skip when CUDA is unavailable. The PyInstaller hook `packaging/hook-studio_modules.py` collects all discovered modules so registry lookup also works in packaged builds.
