# Training evaluation budget

New local training runs evaluate at most **64 windows per split**. Set
`eval_windows` in the training configuration to choose another positive budget
(up to 100,000); `400` restores the earlier budget. The chosen value is saved in
`config.json` and checkpoint metadata and reported in the training event log.

The step-zero baseline, periodic validation, best-checkpoint selection and final
budget-bound evaluation all use the same budget and deterministic, evenly spaced
window indices. Small splits still use every window. Loss remains token-weighted
cross-entropy. Evaluation does not draw training batches or dropout masks.

An ordinary resume uses the checkpoint's budget even if the caller supplies a
different value, because its historical losses and best checkpoint must remain
comparable. Checkpoints without `eval_windows` retain the legacy 400-window
budget. A rebase starts best-checkpoint selection afresh on a different dataset
and may select a new budget. Distributed coordinator evaluation retains its
existing independent budget.

This is a latency/measurement-precision tradeoff, not a model-quality improvement:
64 windows provide a less precise estimate on heterogeneous data than 400 and
can change which checkpoint is selected as best. Compare runs using the same
budget and dataset. Use a larger budget or a dedicated held-out evaluation when
making model-quality decisions. Training weights, sampler and dropout state do
not depend on the evaluation budget (subject to ordinary floating-point noise).

## Reproducing the startup measurement

The opt-in, window-free unittest
`tests.test_training_evaluation.EvaluationStartupBenchmark` runs real one-step
training attempts in temporary folders without submitting jobs to the runtime.
Set `GROUNDWORK_EVAL_BENCHMARK=1` and `GROUNDWORK_EVAL_DEVICE=cpu` or `cuda`, then run:

```text
python -m unittest tests.test_training_evaluation.EvaluationStartupBenchmark
```

It uses a byte tokenizer, 64-wide two-layer model, context 256, batch four,
dropout 0.1, a 10% validation split and 2.5 million characters of synthetic text.
Both 400- and 64-window settings warm their shard cache before three measured
attempts, alternating order. The timer includes preparation, cached token-window
loading, both baseline evaluations and the first optimizer step. Checkpoint I/O
after the first step is excluded. It also checks equal training RNG state and
first-step weights within tight floating-point tolerance. Run without concurrent
training or inference for interpretable timings; this is a startup benchmark,
not evidence of downstream language quality.

## October 19, 2026 measurement

On the development machine (16 PyTorch CPU threads; RTX 4060 CUDA), with other
agent performance measurements paused:

| Device | Legacy 400 windows | 64 windows | Median reduction |
| --- | ---: | ---: | ---: |
| CPU | 2.603 s | 0.608 s | 76.7% |
| CUDA | 2.944 s | 0.614 s | 79.1% |

These are medians of three repeated attempts after warmup. CPU samples were
2.871, 2.493, 2.603 seconds versus 0.608, 0.582, 0.612 seconds; CUDA samples were
2.944, 2.918, 3.017 seconds versus 0.614, 0.615, 0.577 seconds. The maximum
first-step weight difference was 7.45e-9 on CPU and zero on CUDA; both retained
identical sampler/model RNG states. Ordinary desktop applications were open,
so these are practical local measurements rather than an isolated hardware lab.
The synthetic data and small model explain why this current baseline need not
match the older 20–28 second measurements in issue #529.
