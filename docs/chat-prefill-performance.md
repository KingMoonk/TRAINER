# Chat output projection

Generation now projects only the final hidden position into vocabulary logits.
Transformer computation and KV-cache population still cover every input token.
Training and inspection retain full outputs; `last_only` rejects those modes.

This removes unused output work during prompt processing and sliding-window
recomputation. It does not remove transformer work or accelerate a one-token
cached decode by reducing its output size.

## Local measurement, October 19, 2026

The opt-in unittest uses the same weights and prompt, width 64, two layers,
context 256, batch one and fp32. It alternates full and final-position projection
for 12 pairs, discarding the first three, and synchronizes CUDA for timing.

| Device | Vocabulary | Full median | Final-only median | Speed ratio |
| --- | ---: | ---: | ---: | ---: |
| CPU | 264 | 5.583 ms | 5.538 ms | 1.01x |
| CPU | 32,000 | 18.516 ms | 5.692 ms | 3.25x |
| RTX 4060 CUDA | 264 | 185.706 ms | 200.287 ms | 0.93x |
| RTX 4060 CUDA | 32,000 | 228.785 ms | 205.816 ms | 1.11x |

Large-vocabulary CUDA incremental peak allocated memory fell from 67,438,592
to 1,116,160 bytes. Small-vocabulary peak stayed at 1,116,160 bytes.
These are local microbenchmarks with ordinary desktop applications open,
not end-to-end response timings. Small-vocabulary results show no reliable
speed benefit; CUDA latency in this small-model workload needs separate profiling.

Reproduce with `GROUNDWORK_CHAT_BENCHMARK=1` and:

```text
python -m unittest tests.test_last_logits.LastLogitsBenchmark
```

## Design findings and next measurements

The existing design already reuses loaded models and has KV caching, shared and
chunked prefill, weights-only checkpoint loads and memory recovery. Replacing
these mechanisms is not justified by this measurement.

The remaining questions need end-to-end measurements: time to first token versus
decode throughput; CPU chat routing while training occupies CUDA; full-window
recomputation after context fills; and small-model GPU overhead. A rolling cache
would require a deliberate change to position/context semantics rather than
blindly retaining shifted keys. Model quality requires held-out evaluation and
data investigation separately from execution speed.
