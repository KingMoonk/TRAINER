# Local shared-bottleneck measurements for #648

These are synthetic, application-paced loopback measurements on macOS arm64, not physical-node, Tailscale, GPU or convergence results. Two logical members use the real PeerClient against synthetic HTTP endpoints. A single FIFO service budget covers upload reads, download writes and control replies. The fixed 5 ms handler delay is not a TCP RTT or loss model.

Nearest-rank p99 is calculated per repetition on successful calls. The table gives the range across repetitions; sample counts and missed dispatches are summed. Missed dispatches are not included as successful low-latency samples. Each control probe is scheduled at 10 ms cadence with one outstanding probe; this does not model unlimited independent control arrivals. Each trial includes 3 seconds of idle probing. Finite samples, especially the eight-stream cases, do not establish stable production tail estimates.

| Profile | Streams/member | Useful MiB/s | Bulk control p99 ms | Bulk samples | Missed bulk probes | Idle control p99 ms | Transfer overlap s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Upload, 16 MiB/member, 256 KiB parts, 4 MiB/s, 3 repeats | 1 | 3.97–3.98 | 17.42–18.74 | 1205 | 1209 | 10.57–12.31 | 8.02–8.04 |
| Upload, 16 MiB/member, 256 KiB parts, 4 MiB/s, 3 repeats | 8 | 3.96–3.97 | 72.67–75.06 | 355 | 2067 | 10.11–11.08 | 8.05–8.06 |
| Download, 16 MiB/member, 256 KiB parts, 4 MiB/s, 3 repeats | 1 | 3.98–3.99 | 17.22–17.92 | 1206 | 1206 | 9.70–11.67 | 8.02–8.04 |
| Download, 16 MiB/member, 256 KiB parts, 4 MiB/s, 3 repeats | 8 | 3.98–3.99 | 73.00–74.85 | 352 | 2058 | 8.62–13.61 | 8.01–8.02 |
| Upload, 40 MiB/member, 4 MiB parts, 8 MiB/s, 2 repeats | 1 | 7.92–7.93 | 12.61–12.96 | 1095 | 925 | 9.75–10.86 | 10.06–10.08 |
| Upload, 40 MiB/member, 4 MiB parts, 8 MiB/s, 2 repeats | 8 | 7.89–7.90 | 42.26–46.78 | 597 | 1431 | 10.36–10.81 | 10.11–10.13 |

All transfers passed SHA-256 verification; no control errors were recorded. Transfer overlap is measured from client operation start to verified completion, not packet capture. Both members overlap for almost the entire transfer. Useful goodput nearly fills the configured synthetic service budget; this is not evidence of physical-link saturation.

Within this model, eight streams increase control delay without increasing useful throughput once the shared service budget is full. This supports testing bounded admission. It does not establish that one stream is optimal on a lossy WAN, measure a receiver-grant implementation, or justify changing production defaults. One stream uses whole-body requests while eight uses multipart/ranges, so the comparison includes framing differences.

## Reproduction

Run these sequentially so experiments do not contend with each other:

```sh
make network-bench BENCH_ARGS="--output artifacts/discovery-648-up.json --payload-mib 16 --rate-mib 4 --latency-ms 5 --part-kib 256 --members 2 --idle-seconds 3 --control-interval .01 --direction up --repeats 3"
make network-bench BENCH_ARGS="--output artifacts/discovery-648-down.json --payload-mib 16 --rate-mib 4 --latency-ms 5 --part-kib 256 --members 2 --idle-seconds 3 --control-interval .01 --direction down --repeats 3"
make network-bench BENCH_ARGS="--output artifacts/discovery-648-default-parts.json --payload-mib 40 --rate-mib 8 --latency-ms 5 --part-kib 4096 --members 2 --idle-seconds 3 --control-interval .01 --direction up --repeats 2"
```

Order alternates 1/8, 8/1 by repetition. The gzip JSON files beside this note contain settings, platform/Python, base revision, module hash, raw samples, missed probe timestamps, transfer intervals and summaries. They were recorded from the working tree (dirty=true); the module hash identifies the exact benchmark source. Read with `gzip -dc docs/measurements/648-loopback-up.json.gz` or Python gzip. No production networking behavior changed.

## Remaining evidence

Real Mac–BEAST upload overlap, physical interface saturation, production progress/view p99 during bulk, exposed per-GPU stalls, and convergence measurements remain outstanding. At discovery time BEAST returned HTTP 502 and the local API was not listening. Local nodes can validate behavior but cannot substitute for the required physical-machine run.
