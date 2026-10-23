# Receiver-driven transfers over the swarm network

Discovery and implementation notes for [#648](https://github.com/KingMoonk/TRAINER/issues/648). The opt-in delta upload path from [#650](https://github.com/KingMoonk/TRAINER/issues/650) is implemented as an experimental HTTP admission protocol; versioned downloads are also implemented under the same opt-in flag; default enablement remains gated on physical measurements. Secondary GPU work is deferred to [#649](https://github.com/KingMoonk/TRAINER/issues/649).

## Objective and constraints

Keep coordination responsive and reduce exposed communication stalls without changing the DiLoCo update rule, round membership, deadlines, permitted update age, or quantization. The eventual system objective is time to a target validation loss; GPU utilization and transfer goodput are explanatory measurements, not substitutes for convergence. First compare transport policies with the training algorithm held fixed.

The swarm has asymmetric access links, different GPUs, variable propagation delay and loss, and potentially a shared Tailscale relay. A node can simultaneously receive deltas, serve weights, download shards and replicate state. The bottleneck may be sender egress, receiver ingress, a shared intermediate link, CPU serialization, memory pressure or coordinator execution. Neither receiver goodput nor eight active sockets proves link saturation.

Homa makes message boundaries and receiver scheduling explicit. Its unscheduled initial packets hide grant latency; scheduled packets require receiver grants. Switch priorities allow short traffic to bypass bulk traffic, and an oldest-message share prevents starvation. Those mechanisms work together. HTTP application scheduling cannot reorder bytes already in TCP queues or provide switch priorities. The paper's datacenter latency results are not a prediction for this WAN.

Source: John Ousterhout, [It's Time to Replace TCP in the Datacenter](https://arxiv.org/pdf/2210.00714), §§3.3 and 5.1–5.5, especially PDF pages 3 and 5. The policy choices below are engineering proposals for TRAINER, not claims made by the paper.

## Current behavior

- `studio_peer.PeerClient` defaults to eight streams and 4 MiB parts. Each transfer creates its own executor, so concurrency multiplies across transfers and members. A full wave can contain 32 MiB per member.
- `_post_parts` probes support, hashes the encoded body, and schedules every part into a per-transfer pool. `studio_parts.Assembler` checks part and whole hashes. The collective route submits the assembled update through the existing coordinator.
- Downloads already use receiver-initiated ranged GETs. The missing control is a shared byte/concurrency budget and intentional ordering across transfers; replacing ranged GET with another request protocol alone adds no scheduling benefit.
- Single-stream mode currently also changes the transfer path: uploads use one full-body POST and downloads use one full-body GET. Comparing one versus eight streams measures that whole configuration change, not solely socket count.
- The overlap exchange encodes, uploads, waits for the next round, and downloads weights in the background. The next compute boundary calls `Member._join`; any exchange time beyond the available computation becomes an exposed stall.
- Progress reports have a bounded coordinator-lock wait and can return `accepted=false`. Round views use a separate long-poll executor but still acquire coordinator state. Measure both transport completion and application acceptance.
- Existing `node_rates` report application goodput. Timeline marks approximate transfer phases; they do not establish exact transfer overlap. Traffic records have a bounded ring; long experiments need continuous export.

## Implemented prototype (#650)

`PeerClient(receiver_grants=True)` or the member's `--receiver-grants` flag opts into experimental protocol version 1. For uploads, it first requests capabilities, announces the immutable encoded body and its run/round/term/generation/lease identity, then polls for exact byte ranges and sends only those parts. The sender clamps its part size to the receiver's advertised limit. A 404 on capability discovery disables the option for that client and falls back to the existing upload route; it is off by default.

The API process owns one admission manager shared by all its requests. The supported `studio_api` entrypoint starts one Uvicorn worker. Upload clients also share one process-wide 32 MiB outstanding egress budget, with FIFO waiting and injectable `SenderEgressBudget` configuration. Multiple processes on one physical host must partition budgets or use a shared service before claiming a node-wide limit. These are application byte windows, not physical-link rate caps; legacy traffic is explicitly outside them.

The receiver defaults to 32 MiB outstanding, 512 MiB reserved assembly memory, 4 MiB maximum grant and 4 MiB per-transfer window/turn. `--delta-outstanding-mib`, `--delta-assembly-mib`, `--delta-grant-mib` and `--delta-turn-mib` configure them. The hard transfer ceiling is 256 MiB; negotiated capacity is smaller when needed to fit the reservation of three encoded bodies plus three part buffers. There are at most 256 live/retained transfer records and eight active reservations per member. Completed receipts are evicted by age or record pressure, carry no payload, and do not consume active-member slots. These defaults are experimental safety bounds, not measured WAN settings.

FIFO scheduling reserves bounded batches for waiting transfers without requiring the queue's first sender to poll first. Serviced transfers move behind existing demand; older requests that cannot yet fit retain their place. A dormant sender holds only its per-transfer window until unused grants expire. The part size is clamped to fit that window. Grant responses replay unstarted credit and report accepted parts and completion, so a lost response does not require resending accepted data. Polls and announcements are rate limited per member and process. Temporary full-admission responses cause bounded sender backoff.

Unused grants expire after 30 seconds on the receiver's monotonic clock. A receiving request retains its outstanding and memory reservations through cancellation, fencing and expiry until its buffers drain. A separate 120-second receive deadline bounds stalled request bodies. Sender socket waits preserve the caller's timeout; `PeerClient(grant_deadline=600.)` separately bounds the whole admitted operation. Cancellation stops new dispatch, drains active sender requests, then sends a best-effort cancellation. Once the receiver claims application, it completes submission and its receipt together; cancellation cannot revoke that optimizer update. Decode/apply uses a separate worker limiter, and status polls can recover an assembled body whose last request disappeared before application.

A fresh receiver incarnation fences old transfer IDs after a process restart. At most two sender re-announcements reuse the same encoded bytes. Completion receipts are process-local: after recovery, existing coordinator fencing and duplicate/late-contribution rules remain authoritative, and the client may receive a stale error rather than a recovered success receipt. The grant path now accepts the same permitted one-round-late updates as the legacy path. Quantization and residual updates are unchanged.

Admission events include announcement, grant ranges, initial grant wait, expiry, retry, cancellation, accepted duration and current accounting. Legacy uploads emit a `bypass` event. Client HTTP observers expose admission mode and retry count; `PeerClient.admission_mode` identifies negotiated or legacy fallback. The experiment reports per-member useful rates, Jain fairness, control delay and raw admission events. It does not yet measure exact GPU stalls, physical bandwidth headroom or time to validation loss. The #651 implementation below adds shared download budgets and explicit observed-wait priority.

## Versioned downloads and observed waits (#651)

The same opt-in flag enables admitted ranged GETs for weights, shards, validation and replica state, even with one stream. Each process has a shared 32 MiB ingress budget for both incoming granted uploads and locally requested download parts. Its shared 32 MiB egress budget covers granted outgoing uploads and admitted serving responses. The upload-specific receiver window is a sub-limit of ingress credit. Stream count is only a ceiling within those windows. Separate API/member processes still need explicit partitioning or shared ownership before these can be described as physical-node limits.

`X-Transfer-Admission: 1` requests the protocol. An admitted response supplies a strong SHA-256 `ETag` and `X-Content-SHA256`; the receiver pins them before reading the first body, sends `If-Match` on subsequent ranges and retries, and verifies the complete body before decode or file success. A same-size version change returns 412. An unavailable old representation fails rather than combining versions. Peers without the response marker retain explicit `legacy-fallback` behavior, which does not claim admitted server traffic. Upload capability fallback does not disable independently negotiated downloads.

Serving byte credit remains held until the ASGI send completes, including disconnect/cancellation drain. Serialization, hashing and snapshot construction use a separate worker limiter for admitted requests, leaving the existing control worker pool available. Shard responses use immutable temporary snapshots; the cache retains at most eight files, and active plus cached snapshot storage shares a 512 MiB limit. Identical concurrent readers share a snapshot. Active responses keep their storage charged until they drain; temporary exhaustion returns 429 with bounded client retry. Individual admitted shards above 512 MiB are rejected. Byte digest caching retains at most sixteen immutable objects totaling 64 MiB. These limits are experimental guardrails.

Required requests get one turn followed by an older background turn when both wait; FIFO order within each class prevents a stream of urgent claims from starving background work. For overlapped exchanges, a local event becomes required only while the training thread is actually inside `_join`. It is evaluated on new grant/range requests, with distinct compute-round and exchange-round identifiers. Synchronous weights, required shards and their necessary exchange work are required immediately. Replica prefetch stays background and is cancelled on demotion; a foreground request for a shard already being prefetched promotes that existing transfer. Already queued TCP bytes cannot be preempted.

`diloco_transfer` events record encode, upload outcome, download outcome, weight decode and measured exchange-join wait. Join events carry `exposed_wait_s`, `compute_round` and `exchange_round`; background phases do not rewrite the training progress round. Existing coordinator history remains the source for accepted contributions, permitted late updates and missing rounds. The tests exercise dynamic priority and cancellation without changing local step counts, quantization/residuals, full-weight fallback or replica recovery rules. Time to validation loss and real GPU efficiency remain physical-machine gates.

## First-principles budgets

Each physical node owns an ingress budget, an egress budget, a bounded assembly-memory budget and bounded bulk execution capacity. All runs and transfer types on that node consume the relevant budgets. A per-transfer or per-process limit is insufficient when multiple processes share a physical link. Budget ownership needs a node-level service or an explicit partition across processes; local test nodes on different ports still share one host budget. Control requests must have independent dispatch capacity, bounded request bodies and service time; long polls must not occupy the grant/control execution pool.

Use both a rate target and outstanding-byte credit. A window alone can burst; a rate alone can admit unbounded buffered work. As a starting model, for useful rate R and grant feedback time T, the aggregate window must cover roughly R*T bytes to avoid grant stalls. This is a tuning model, not a fixed WAN guarantee: TCP buffering, loss, multiple paths and feedback timing affect the required headroom. Count granted but unacknowledged bytes, including requests in flight. Pace bytes admitted to socket writes, not merely request creation; count a part against credit until its receiver acknowledgement. Large writes already buffered by TCP cannot be preempted by a new urgent grant. Measure queue delay while adapting; low throughput alone must not trigger unlimited credit growth.

Part size sets how much already-admitted work can delay a priority change. At an illustrative 20 Mbit/s bottleneck, 4 MiB takes about 1.68 seconds to serialize. Two unscheduled parts per sender would therefore admit about 3.36 seconds of work per sender before scheduling, excluding overhead. Do not inherit "one or two parts" as an unmeasured constant. Start with metadata-only announcements, then evaluate a small bounded initial-byte allowance. Pipeline grants to avoid a stop-and-wait protocol.

These are endpoint admission limits, not replacement path congestion control. Independent endpoint budgets cannot enforce a shared relay bottleneck. If a relay or common access path is limiting, measure it separately and adapt conservatively; use a common aggregate budget where one authority actually controls the traffic. TCP congestion control remains active.

The sender also owns an egress budget. It must select which authorized bytes it can actually dispatch; a remote grant is permission, not an instruction to overrun local egress. Avoid holding large receiver reservations while queued indefinitely behind other sender work: acquire a small local dispatch slot before requesting short-lived remote credit, or explicitly return unused credit. Do not hold locks while making network calls.

## Proposed upload protocol

1. Capability negotiation advertises a versioned grant protocol. Legacy peers retain the current protocol with visible legacy-mode telemetry. They cannot be counted as obeying the new global bound.
2. After encoding once, announce a transfer with run, direction, round, term, generation, member identity, lease epoch, exact encoded length, immutable body hash and requested part geometry. The receiver validates membership and limits before admitting binary bytes. Announcements have per-peer and node-wide count/size limits.
3. The receiver assigns an opaque transfer ID bound to those fields and a manifest. A short control request retrieves a batch of grants naming exact byte ranges, grant IDs and expiry. No bulk data accompanies this request. Bound per-peer and node-wide control request rates, batch acknowledgements fairly, and back off with jitter on empty grants so polling cannot overload coordination. Do not keep a synchronous worker occupied waiting for a future grant.
4. A sender dispatches only granted ranges under its local egress budget. The receiver validates a grant before accepting the part into application assembly. HTTP headers and body may already reach kernel buffers; application validation cannot guarantee zero unauthorized wire traffic. Bound body lengths and reject excess rather than reading an arbitrary whole body.
5. Acknowledgement records each range once. A retry uses the same immutable encoded bytes and transfer identity. An ambiguous response triggers status/reconciliation; a duplicate must not release credit twice or apply a delta twice. Retain a bounded completion record so a lost final acknowledgement can be answered idempotently.
6. Completion requires the whole hash and existing coordinator admission checks. Admission to the network is not admission to the optimizer. A round may close during a transfer; preserve the existing late-contribution policy and report the outcome explicitly.
7. Stop, fencing, disconnect and expiry release unused reservations. The receiver is authoritative for expiry using its monotonic clock; sender-relative lifetimes are advisory and must not rely on synchronized wall clocks. Expired credit does not instantly mean already-sent bytes have disappeared: track draining requests separately before reusing capacity. Bind state to the coordinator term and a fresh admission-service incarnation so a restarted service cannot accidentally honor old credit. Restart and term changes require conservative drain accounting before issuing replacement capacity. Age out incomplete assembly under a separate memory limit.

Telemetry distinguishes announcement wait, granted-but-unsent time, transmission, retry, assembly, decode/apply and exposed GPU stall. Scheduling delay must not be misclassified as a dead member.

## Downloads and control

A member requests weights only within its ingress budget. The serving node admits those requests under its egress budget, including all other consumers, shards and replicas. Keep ranged GET and bounded prefetch within credit so propagation latency can be hidden. Pin an immutable weight version/hash across parts; reject a changed version rather than assembling mixed rounds.

Control includes membership/leases, progress, nonblocking views, cancellation, grants and acknowledgements. Give it independent application execution capacity and small bounded bodies. Leave measured bandwidth headroom for it by controlling bulk dispatch. Separate sockets avoid sharing one TCP byte stream but do not bypass a saturated link, Tailscale queue or coordinator lock. Do not promise "never queued" without network QoS that can enforce that property.

## Policy: preserve useful computation

Use fair service across runs and aging within a run. Within that envelope, prioritize transfers that are predicted to cause an exposed compute stall or that the coordinator identifies as necessary for round progress. Use remaining encoded bytes as a tie-breaker. Begin with observable urgency (already waiting, required contribution) before relying on uncertain predicted compute completion times. Clamp reported urgency; one peer must not monopolize the node by claiming an imminent deadline.

Pure SRPT can improve mean transfer completion yet leave the round's last large transfer unchanged or worse. If every contribution is required and all share one bottleneck, reordering cannot reduce the total bytes that must cross it. Expected gains come from less excess queueing, timely control, serving genuinely critical transfers, and hiding transfer time behind available computation. They are not a guaranteed increase in link capacity.

Preserve a minimum service opportunity for older bulk transfers. A stalled or slow sender must not reserve the whole receiver window. Use expiring small grants and redistribute unconsumed credit. Evaluate fairness by waiting time and accepted useful work, not equal connection counts.

## Interaction with existing changes

| Change | Consequence |
| --- | --- |
| #588, parallel streams | Keep multiple TCP connections where they overcome per-flow limitations. Eight becomes a ceiling under shared budgets, not an entitlement per transfer. Compare legacy one/eight settings and later hold chunk geometry fixed when isolating concurrency. |
| #586, overlap and int8 | Preserve the overlap algorithm and bit-identical quantized updates. Measure `_join` stall, full-weight fallback and encoding/apply time separately. Scheduling cannot silently grant extra local steps or relax update-age rules. |
| #593, sparse deltas | Announce the final encoded length after sparse selection and quantization. Uploads shrink; dense int8 downloads remain. Retries reuse bytes without advancing error feedback again. Separate sparsification CPU time from wire time. |

## Reproduce the local experiment

Run `make network-bench` from a checkout with its Python environment installed. Raw samples and summaries are written to `artifacts/network-discovery.json` (ignored by git). Use `make network-bench BENCH_ARGS="--help"` for explicit payload, rate, delay and sampling settings; supply `--output PATH` when overriding arguments. The experiment creates and closes its own loopback HTTP server, does not start a UI, and does not change the workspace's real runs or Tailscale routing.

The synthetic control endpoint is not the production progress/view route. The existing `tests.test_api_collective_parts` exercises real collective HTTP routes and integrity separately; it does not measure their loaded p99. This distinction must remain visible in any report.

## Evidence and decision gates

[Initial local results](measurements/648-loopback.md) cover contention and the current 4 MiB part size. [The matched admission comparison](measurements/650-admission.md) holds upload geometry fixed and records the throughput/control-delay tradeoff.

The local experiment is an application-paced shared-bottleneck model using real HTTP client transfers. It can validate instrumentation and demonstrate contention under its assumptions. It cannot establish physical link saturation, real Mac–BEAST behavior, GPU utilization, or convergence. Independent client connections on one machine are not three physical nodes.

For each configuration record code revision, body and part sizes, number of senders, stream ceiling, artificial rate/delay settings, repetitions and raw timestamps. Include idle and bulk control samples with counts, p50/p95/p99/max, errors and missed dispatches. Small samples do not support stable tail estimates. Report useful verified bytes, aggregate makespan, transfer overlap, per-transfer duration and retries; do not add per-member rates measured over different windows.

On real machines, additionally record Tailscale direct/relay path, RTT/loss observations, isolated directional capacity, aggregate interface counters and actual compute/communication phases. Run matched one/eight-stream trials with warmed shard caches and fixed model, dtype, sparsity, local steps, overlap and deadline settings. A third physical member is needed to study two independent remote senders competing at the coordinator; two local processes cannot substitute for its access link.

Proceed to an opt-in grant implementation only after a repeatable baseline identifies harmful contention and the experiment can detect improvement. Before enabling by default, require representative physical-node runs showing reduced control delay or exposed stalls without lost/duplicate contributions, broken fences, starvation or an unexplained useful-throughput regression. Report per-member exposed stalls, accepted useful contributions, contribution age, and late/missing rounds so aggregate gains cannot hide a repeatedly blocked GPU. Compare validation at matched token counts and time to target loss before any subsequent change to training coordination. Keep #648 open until its real-machine acceptance criteria are met.

## Implementation sequence

Proceed with an opt-in prototype, while retaining the real-machine gate before any default change:

1. [#650](https://github.com/KingMoonk/TRAINER/issues/650): receiver-granted delta uploads, shared budget ownership, retry/fence/expiry invariants and instrumentation.
2. [#651](https://github.com/KingMoonk/TRAINER/issues/651): budgeted ranged downloads, background replication and scheduling around exposed training stalls; depends on #650.

The local measurements justify testing admission under a controlled shared bottleneck. They do not establish that the current Mac–BEAST path is receiver-limited, select a production rate/window, or demonstrate a convergence benefit. #648 remains open for its physical-machine measurements.
