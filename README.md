# Groundwork AI Studio

A native Python desktop studio for creating datasets, generating synthetic examples, and training your own language model from random weights.

**NumPy and pandas provide the CPU version; optional CuPy enables NVIDIA GPU acceleration.** Python's built-in Tkinter supplies the interface. The transformer, backward pass, optimizer, tokenizer, checkpoint format and text sampler are implemented in this repository. Everything runs locally; CPU and GPU use the same handwritten learning code.

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

- `auto` (default): use the first CUDA GPU if the driver, CuPy, and CUDA kernel checks pass; otherwise use NumPy on CPU and show the reason.
- `cuda`: require GPU execution. If it is unavailable, report the error instead of silently changing devices.
- `cpu`: use NumPy without requiring CuPy or probing CUDA in the training engine.

Chat and model-generated synthetic data share a separate **Inference device** selector. For chat, reload the checkpoint after changing this setting. The model status and training log show the actual device. Resuming a run uses the currently selected training device, so CPU checkpoints can continue on GPU and vice versa.

For a machine with an NVIDIA driver and CUDA 13 toolkit already installed:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
.\.venv\Scripts\python.exe train.py --list-devices
```

If the machine has a compatible NVIDIA driver but no CUDA toolkit, install the runtime components in the virtual environment using CuPy's optional `ctk` extra:

```powershell
.\.venv\Scripts\python.exe -m pip install "cupy-cuda13x[ctk]>=14.2,<15"
```

See the [official CuPy installation guide](https://docs.cupy.dev/en/stable/install.html) for CUDA compatibility and CUDA 12 alternatives. Install only one CuPy distribution in an environment. Restart the studio after installing GPU support. This backend currently targets a single NVIDIA GPU (device 0).

Weights, activations, attention, handwritten gradients, and Adam state stay on GPU during GPU training. Initialization and dropout random draws remain NumPy-based for compatibility with existing saved random states; batches and dropout masks transfer to the GPU. Checkpoints transfer numeric arrays back to NumPy for storage. Inference transfers the final vocabulary vector back for seeded token sampling. GPU reductions may differ slightly from CPU floating-point results. Small models or batches may run faster on CPU because kernel launch and transfer overhead can dominate. Initial CUDA kernel compilation can make the first step slower.

## Workflow

1. **Dataset library:** create a dataset, choose one or multiple files with **Import files**, or write examples manually. JSONL, JSON, CSV, TSV, tagged TXT and plain text are supported. Pandas parses tables and computes dataset statistics. Edit prompt/response pairs, full conversations or raw text; mark examples reviewed; export JSONL or CSV. Exact content duplicates are skipped. Existing workspace datasets are retained.
2. **Synthesis lab:** expand template variables, generate arithmetic examples with computed answers, vary seed prompts while preserving conversation targets, or sample one of your own trained checkpoints. Generation is seeded and deduplicated. Drafts persist across restarts; edit, discard or keep selected examples. Generation may return fewer drafts than requested when the recipe's unique combinations are exhausted. Model-generated answers are unverified and require review.
3. **Model blueprint:** set character/byte tokenization, embedding width, blocks, heads, context, feed-forward expansion and dropout. Configure learning rate, AdamW weight decay, batch size, gradient accumulation, optimizer steps, warmup, validation split and learning objective. Micro/Small/Medium presets specify dimensions; they contain no trained weights.
4. **Training console:** view actual training and validation loss, token throughput and optimizer steps. Pause, stop and save, inspect saved runs, resume checkpoints or export a complete run ZIP. Training runs on a worker thread so the UI stays responsive.
5. **Playground:** load a `.npz` checkpoint and sample with temperature, top-k and response-length controls. Use **Teach a correction** to edit the conversation and save it to the selected dataset. Train a new run on the revised data to incorporate corrections.

The file picker is the supported import method; operating-system drag-and-drop is not required or installed.

## The learning engine

`studio_model.py` implements a decoder-only transformer using NumPy or CuPy arrays:

- Learned token/position embeddings, pre-layer normalization, multi-head causal self-attention, residual paths, tanh-approximation GELU, dropout, final normalization and tied output embeddings.
- Explicit gradients for every operation, stable cross-entropy, global gradient clipping, gradient accumulation, AdamW with bias correction, warmup and cosine learning-rate decay.
- Character vocabularies fitted to the training split, or a fixed UTF-8 byte vocabulary with special conversation tokens. Character tokenizers use an unknown token for unseen characters; bytes cover all UTF-8 text.
- Next-token prediction on raw text. Conversation targets can include every token or only assistant content and end-of-turn markers.

This is a small-model experimentation studio. Training from scratch is suited to learning and narrow tasks; a generally capable chatbot needs far more data and compute. A falling training loss alone does not establish generalization. Short context windows can truncate conversational history. On GPU, batch size, context length and model width determine memory demand; reduce them if the device runs out of memory.

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

## Source and checks

- `studio_gui.py`: Tkinter pages, background tasks, plots and review tools.
- `studio_core.py`: pandas dataset import/audit, persistence, normalization and synthetic recipes.
- `studio_model.py`: NumPy model, handwritten gradients, optimizer, training and inference.
- `studio_backend.py`: optional CuPy loading, verified CUDA detection and CPU fallback.
- `app.py` and `train.py`: desktop and command-line entry points.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests cover data preservation, reproducible recipes, causal attention, numerical gradient checking, learning, inference, checkpoint resumption and cancellation. CUDA tests compare GPU forward/backward/AdamW results against CPU, verify actual learning and inference on GPU, and resume checkpoints across devices. CUDA tests skip when a working GPU installation is unavailable.
