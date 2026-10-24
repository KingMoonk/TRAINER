# Matched upload and download exchange

Measured at clean revision `62c20fa5a118dd185d58dbdb384f1afec05a7d55`. [Raw samples](651-exchange.json.gz) include each transfer, control probe, missed dispatch and admitted-byte peak. This uses real loopback HTTP clients with one synthetic application byte pacer. It does not emulate Tailscale, TCP loss, physical network queues or GPU computation.

```sh
.venv/bin/python studio_transfer_discovery.py \
  --output artifacts/discovery-651-exchange.json \
  --payload-mib 1 --rate-mib 1 --latency-ms 2 --part-kib 256 \
  --members 2 --background-members 1 --idle-seconds .5 \
  --control-interval .02 --direction exchange --repeats 2 \
  --admission both --window-kib 512 --turn-kib 256
```

Each member uploads and downloads the same 1 MiB body. Baseline and admitted cases use 256 KiB multipart uploads and ranged downloads with stream ceilings of one or eight. One member's download is marked required and the other's background in admitted cases; these labels describe the synthetic transfer policy, not actual training stalls. One 1 MiB/s pacer serializes application reads/writes and control replies; 2 ms is a fixed handler delay. Policy and stream order alternate between repetitions. The admitted process has a 512 KiB shared ingress and egress budget and a 256 KiB upload turn, which are deliberately small experimental windows.

| Policy | Stream ceiling | Useful MiB/s | Bulk control p99 (ms) | Jain fairness | Missed control dispatches |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legacy | 1 | 0.990, 0.990 | 36.99, 37.04 | 0.999980, 0.999999 | 96, 99 |
| Legacy | 8 | 0.995, 0.993 | 127.81, 129.64 | 0.999964, 0.999968 | 167, 168 |
| Admission | 1 | 0.981, 0.980 | 37.78, 37.62 | 0.999996, 0.999982 | 100, 96 |
| Admission | 8 | 0.982, 0.964 | 36.51, 37.15 | 0.999997, 0.996953 | 97, 94 |

All eight cases verified every upload and download; no control call failed. Admitted ingress and egress peaks each reached, and never exceeded, 512 KiB. Legacy peaks are zero because legacy transfers do not use those counters. Control p99 uses 34 successful bulk probes in each legacy eight-stream repeat and 107/114 in the admitted eight-stream repeats; their tail estimates have different sample counts and cannot be read as stable percentiles. Raw missed dispatches provide the other side of control responsiveness.

At this constrained window, admission kept the synthetic eight-stream control tail near 37 ms instead of about 129 ms. Its useful throughput was 1–3% lower, and one admitted repeat had a lower per-member fairness score. That is a tradeoff to test on real machines, not a basis for default enablement. The first range must pin a version before later ranges can be sent; this and per-part credit introduce feedback work. Current windows, part size and stream ceiling are safety/test settings. The local experiment does not measure exposed GPU time, useful optimizer progress or validation loss.
