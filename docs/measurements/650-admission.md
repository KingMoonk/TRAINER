# Matched upload admission comparison

Measured at clean revision `7bb015cc093fc97d4e06f7880c808b360ad99ab4`. This is a loopback experiment with real HTTP and one synthetic application byte pacer, not a physical WAN or GPU result. [Raw samples](650-admission.json.gz) contain every transfer, control sample, missed dispatch, admission event and environment setting.

```sh
.venv/bin/python studio_transfer_discovery.py \
  --output artifacts/discovery-650-compare.json \
  --payload-mib 1 --rate-mib 1 --latency-ms 2 --part-kib 256 \
  --members 2 --idle-seconds .5 --control-interval .02 \
  --direction up --repeats 2 --admission both \
  --window-kib 512 --turn-kib 256
```

Both policies use 256 KiB multipart uploads at both stream settings. Each member sends the same 1 MiB payload. The admitted receiver has a 512 KiB aggregate window and 256 KiB per-transfer window. These deliberately small experimental windows are not production defaults. Policy order and stream order alternate between repetitions. The pacer serializes upload reads, download writes and control replies under one 1 MiB/s budget; 2 ms is a handler delay, not an emulated network RTT.

| Policy | Stream ceiling | Useful MiB/s | Bulk control p99 (ms) | Control samples/repeat | Missed dispatches/repeat |
| --- | ---: | ---: | ---: | --- | --- |
| Legacy multipart | 1 | 0.979–0.982 | 39.20–39.58 | 53, 53 | 50, 49 |
| Legacy multipart | 8 | 0.988–0.989 | 127.38–128.79 | 17, 17 | 85, 85 |
| Receiver grants | 1 | 0.962–0.967 | 37.79–38.32 | 54, 55 | 50, 49 |
| Receiver grants | 8 | 0.967–0.968 | 37.68–39.26 | 53, 54 | 51, 50 |

All eight cases verified both payloads; control errors were zero. Jain fairness of each member's useful rate was at least 0.999994. The near-equal, simultaneously started senders make this an integrity/contention check, not a demonstration of fairness under heterogeneous arrivals or rates. Queue fairness and starvation/cancellation cases are tested separately with deterministic schedules.

At the eight-stream ceiling, bounded admission reduced the observed control tail while useful throughput fell about 2%. At one stream, both configurations already limit outstanding work similarly and their control tails are close. The grant window admits one part per member in this profile, so increasing the stream ceiling cannot create eight simultaneous parts. This experiment illustrates the cost of extra queued work once the shared service budget is full; it does not choose a production connection count, byte window or bandwidth target.

These are short runs with few tail samples; p99 is effectively the maximum in each repeat. Missed probe dispatches must remain visible: successful-call latency alone understates contention. No exposed GPU wait or convergence improvement was measured. A later reachability check to the configured BEAST URL timed out, so physical-machine acceptance remains open.
