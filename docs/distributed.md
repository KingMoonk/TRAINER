# Distributed training across nodes

`docs/cluster.md` is the operator's view: the tailnet, the node list, the bucket. This page is the engineering design for
running work across those nodes: how nodes trust each other, how a job is owned and moved, and how one training run uses
several machines at once. Phase 6 of `docs/rewrite.md` (#98) is built to this design.

## Why one run now spans machines

The rewrite plan made "splitting one training run across machines" a non-goal, because data-parallel training as datacenters
do it (DDP, FSDP) all-reduces every gradient on every step. Over Starlink (10 to 20 Mbps up) a 25M-parameter model's fp32
gradients are 100 MB per step: about 40 seconds of upload for a step that computes in a fraction of a second. That argument
is right about DDP and wrong about distributed training in general.

The low-communication methods built for exactly this case are now how frontier-scale runs cross weak links:

- **DiLoCo** (Douillard et al. 2023, arXiv:2311.08105): each worker takes H local AdamW steps (H of 100 to 500), then the
  workers average the change in their weights (a *pseudo-gradient*) and an outer Nesterov optimizer applies it. It matches
  synchronous data parallelism at 8 workers while communicating H times less.
- **INTELLECT-1** (Prime Intellect 2024): a 10B-parameter model trained with DiLoCo across three continents on commodity
  links, with int8 pseudo-gradients, workers joining and leaving mid-run.
- **Streaming DiLoCo** (Douillard et al. 2025): syncs a fragment of the weights at a time and overlaps the transfer with
  compute, cutting peak bandwidth further.

So the goal changes: **one pretraining run can use every free node, with DiLoCo as the algorithm.** Everything a frontier
stack has that matters at this scale is here: authenticated transport, membership and failure detection, a scheduler with
leases and fencing, gang placement, an elastic training runtime with deterministic data sharding, and round-level
checkpoints. What is left out, on purpose: tensor and pipeline parallelism and ZeRO/FSDP sharding. They split a model that
does not fit on one device; every model this app trains fits on every node, and they need per-layer, per-step traffic that
a WAN cannot carry.

## Layers

| layer | job | datacenter analogue | module |
| --- | --- | --- | --- |
| identity and transport | who is calling, over an encrypted link | mTLS, service mesh | `studio_api/auth.py`, `studio_peer.py` (#126) |
| membership and health | which nodes are up, from heartbeats | gossip / failure detector | `studio_cluster.py` |
| scheduler | which node owns which job, leases, fencing, gang placement | Borg, Slurm | `studio_scheduler.py` (#127, #128) |
| training runtime | rendezvous, rounds, outer optimizer, elastic membership, data shards | torchelastic, DiLoCo | `studio_diloco.py` (#132) |
| storage | weights between machines | blob store | `studio_publish.py` (#92) |

Each layer only calls the one below it, and each is pure logic plus a thin I/O edge, so every rule here is a window-free test
with fake peers.

## Identity: node-to-node calls (#126)

The tailnet is the perimeter and WireGuard is the cryptography: a packet from a tailnet address really comes from the machine
that holds that address's key. What the API must not do is trust a header a caller can type. Today a node bound to its tailnet
address accepts any request that carries `Tailscale-User-Login`, and nothing stops a peer from sending one by hand.

- Requests that come through `tailscale serve` (loopback peer address) keep trusting the header serve adds: serve strips any
  value the client sent and writes its own.
- Requests that reach the API directly on its tailnet address are identified by asking the local `tailscaled` who owns the
  source address (`tailscale whois --json <ip>`, cached for a minute). The answer names the machine, its tags and its user.
  A caller is let in when it is tagged `tag:trainer`, or is a user in the configured allow-list. The header is ignored on
  this path.
- Node-to-node traffic uses the direct path: `http://<node>:8787` bound to the tailnet address, ACL'd to `tag:trainer`.
  Tagged nodes carry no user login, which is why the serve path alone cannot authenticate them.
- `studio_peer.PeerClient` is the one HTTP client every layer above uses: base URL resolution, timeouts, JSON and binary
  bodies, retries with jittered backoff on connection errors only (never on a 4xx), and one error type.

## Membership and health

`studio_cluster.survey()` already polls `GET /node`. The failure detector on top of it is a timeout, not a single missed poll:
a node is *suspect* after one failed poll and *down* after `DOWN_AFTER` seconds (default 30) with no answer. Only *down*
moves work. Every node runs its own detector; there is no shared view to agree on, because the scheduler below never needs
two nodes to agree that a third is dead: the lease does that.

## Scheduler: ownership, leases and fencing (#127, #128)

**The node a job was queued on owns it** for its whole life. There is no cluster-wide consensus store: with two to five
laptops, a Raft or etcd cluster would add a quorum that a home network loses more often than it loses a node. A job owner is
the single controller for its job, the same choice torchelastic makes with its rendezvous host and Pathways with its single
controller. If the owner is down, its jobs pause; they continue when it returns, from durable state on its disk.

A job runs on another node under a **lease**:

```
Lease(job_id, node, epoch, duration_s)
```

- The owner grants it when it dispatches (`POST /cluster/jobs` on the peer, with the job spec and the lease).
- The runner renews it every `duration_s / 3` (`POST /cluster/jobs/{id}/renew` on the owner). Renewal carries progress.
- **Self-fencing.** The runner measures the lease on its own monotonic clock from the moment it *sent* the renewal that
  extended it. If it cannot renew before the lease runs out it stops the job itself, saving a checkpoint. No clock
  agreement between machines is assumed: each side measures durations on its own clock, and the runner's count starts
  earlier than the owner's, so the runner always gives up first.
- **Fencing tokens.** Every reassignment increments `epoch`. Every write a runner makes back to the owner (renewal,
  progress, the finished version, a DiLoCo contribution) carries its epoch, and the owner rejects any epoch but the current
  one with `409`. A node that went dark and comes back with a job that has moved can never publish over its successor.
- **Reassignment.** When a lease expires, the owner waits `duration_s` more (the runner's self-fence has certainly fired by
  then), marks the job *interrupted* at its last checkpoint, and places it again. On a different node the job resumes from
  the last published checkpoint with a fresh optimizer, and its record says so.

Placement stays `studio_cluster.place()`. A job that needs several nodes (a distributed run) is **gang-placed**: the
scheduler takes leases on `min_nodes` to `max_nodes` free nodes in one decision, or none, so two half-placed runs never
deadlock each other waiting for the last node.

## Training runtime: DiLoCo (#132)

A distributed run has one **coordinator**, on the owner node, and one or more **members**, each a worker process on a node
(the owner usually runs a member too). The coordinator holds the global weights and the outer optimizer; members only train.

### One round

1. Every member holds the global weights `θ` of round `r`.
2. Each member runs inner AdamW steps on its own data shard, keeping its own AdamW state from round to round. Inner steps
   take the learning rate of the global schedule at `r * H + local step`, so warmup and decay span the whole run.
3. When it has done `H` steps, or the round's deadline arrives, a member sends its pseudo-gradient `Δk = θ - θk` and the
   number of target tokens it trained, `tk`.
4. The coordinator averages by tokens, `Δ = Σ tk Δk / Σ tk`, so a slower node counts for the work it did, then applies the
   outer Nesterov step: `v = μ v + Δ`, `θ = θ - η (Δ + μ v)`. Defaults `η = 0.7`, `μ = 0.9` (the DiLoCo paper's).
5. The coordinator evaluates `θ` on the validation windows, which never leave it, writes the round checkpoint, and opens
   round `r + 1`.

With one member, `η = 1` and `μ = 0`, a round is exactly `H` steps of ordinary training: that equivalence is a test.

### Heterogeneous nodes

An NVIDIA box and a MacBook differ several times in speed. A round has a deadline set from the fastest member's measured step
time times `H`, plus slack. Slower members send whatever they have done by then. A member that has not sent anything by the
deadline plus a grace period is dropped from the round, and its contribution would be refused if it arrived later; it rejoins
at the next round with the new `θ`.

### Elastic membership and rendezvous

- A member joins with `POST /collective/{run}/join`, carrying its scheduler lease. It gets its rank, the round, the run's
  config and tokenizer, and its data shards. Joins and leaves take effect at a round boundary, never mid-round.
- The run has a membership **generation** that increments on every change. Contributions carry the generation and the round;
  anything stale is refused with `409`.
- The run continues with fewer members while at least `min_members` are live, and pauses below that. A member that rejoins
  after a crash starts with fresh AdamW state, which DiLoCo tolerates.

### Data

The data has to reach the members, so a distributed run changes what `docs/cluster.md` says about datasets never leaving a
machine: the training windows of that run go to its members over the tailnet, nothing else does.

- The coordinator tokenizes once and cuts the training windows into `S` fixed shards (`S` well above the most members a run
  will have, for example 64), each a token-window archive in the `studio_shards` format.
- Each round, shards are dealt to live members by a deterministic rule of (seed, round, membership). A member samples only
  from its shards, with a sampler seeded by (seed, rank, round), so a rerun with the same membership history is reproducible.
- A member downloads a shard once (`GET /collective/{run}/shards/{s}`), caches it under `data/remote_shards/<run>/`, and
  deletes the cache when the run ends. Validation windows stay on the coordinator.

### Wire protocol

All routes are on the coordinator's API, behind node-to-node auth, under `/collective/{run}`:

| route | body | answer |
| --- | --- | --- |
| `POST /join` | node, lease epoch | rank, generation, round, config, tokenizer, shard ids |
| `GET /params?round=r` | | the round's `θ`, fp32 `.npz` (downloads are the fast direction on Starlink) |
| `GET /shards/{s}` | | one training shard |
| `POST /rounds/{r}/delta` | generation, epoch, tokens, steps; `Δk` as fp16, fp32 or int8 `.npz` (int8: per-tensor scale beside the values) | `202`, or `409` when stale |
| `GET /rounds/{r}` | | `open` or `closed` with the next round; a `pending` round long-polls up to 30 s |
| `POST /progress` | node, generation, epoch, round, phase (`downloading`, `training`, `uploading`, `waiting`), step, loss, tokens/s | `{accepted}`, or `409` for a stale generation or epoch, a non-member, or a closed round |
| `POST /leave` | generation, epoch | `204` |

Bytes per member per round for a 25M-parameter model: 100 MB down (fp32) and 50 MB up (fp16). At 20 Mbps up, the upload is
about 20 s against H = 500 steps of compute, so communication is a small share of the round. `delta_dtype: int8` sends a quarter of fp32 (25 MB up for that model). Each round logs the delta bytes and round seconds
(`diloco.json` history, and an info event on the member with its upload time). Streaming fragments come later, once
measurements show the transfer is the bottleneck.

### Live view

The Cluster › Training tab shows a run as it happens. Each member reports where it is every 2 s or 20 steps, and at once when its phase
changes (`POST /progress`, from a daemon thread; a failed report is dropped and never slows or stops training). The coordinator keeps the
latest report per member, a bounded run of loss and speed samples for each, and the moments of each round on its own clock: `θ` taken, first
step, `Δ` sent, `Δ` received. The last 8 rounds' timelines stay in memory and each finished round's timeline is appended to `metrics.jsonl`.
A bounded event log (200 entries) records joins, leaves, drops, late and refused (stale) changes, each outer step with its size, and each
checkpoint. `GET /distributed/runs` carries all of it on the run (phase, deadline, `θ` size, timeline, events) and on each member (phase, step,
loss, speed, seconds since the report, shards, samples); the page polls it once a second while a run is going. Taking the lock for a report or a
snapshot waits at most a quarter second, so a round that is closing (the held-back text is checked then) never blocks the page.

`studio_api --fake` opens on a simulated run (`demo-run`: one MPS and one CUDA machine) that advances over time, so the tab can be seen without a second machine.

### Checkpoints

After each round the coordinator writes the ordinary `last.npz` (the global weights, loadable by everything that loads a
checkpoint today) and `diloco.json` (round, generation, outer momentum file, shard deal, membership, per-round metrics).
Restarting the coordinator continues from the last completed round; a member's state is never needed to continue the run.

## Failures

| what fails | what happens |
| --- | --- |
| a member's node loses its network | its lease lapses; it self-fences and stops; the round closes without it at the deadline; the scheduler may give the slot to another free node, which joins next round |
| a member's process crashes | same as above, and the owner sees the lease lapse sooner (no renewals) |
| the node of a single-node job | lease lapses, the owner waits out the self-fence, marks it interrupted, places it again from the last checkpoint |
| the owner (coordinator) node | its jobs pause; members cannot renew and self-fence; when it returns it reloads `diloco.json` and reopens the last round |
| a node returns with a job that moved | its renewals and writes carry an old epoch and are refused; it stops its copy |
| a slow link | a member that misses the deadline counts for nothing that round and rejoins next round |
| a stale or forged contribution | refused: wrong generation, round or epoch is `409`; an unauthenticated caller is `401` |

## Testing

- Pure state machines (lease table, membership, round accounting, outer optimizer, shard deal) tested directly in
  `tests/test_*.py`, with fake clocks.
- The 1-member equivalence test: DiLoCo with `η = 1`, `μ = 0` gives the same weights as plain training, step for step.
- A 2-member run in one test process over the real HTTP routes (FastAPI `TestClient`), on CPU, that converges below the loss
  of one member alone after the same number of steps per member.
- The two-machine checks (a member on the Mac, one on the NVIDIA box; pulling a network cable mid-round) are manual and are
  written down in each PR.

## Issues

- #126 node-to-node auth and the peer client
- #127 dispatch with leases and fencing; #128 failover from the last checkpoint
- #132 DiLoCo runtime: coordinator, members, shards, wire protocol, round checkpoints
- #133 gang-place a distributed run through the scheduler (needs #127 and #132)
- #129 device records and the cross-device resume test
