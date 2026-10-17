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
| cluster state | what every node knows about every other: an eventually consistent replica on each node | Serf / Cassandra anti-entropy | `studio_state.py` |
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

A peer that does not answer is not asked at every poll: the wait between asks doubles (10 s, 20 s, ... up to 60 s) and is one
`GET /node` until it answers, so a node that is down shows as down instead of as a stream of failed requests.

## Cluster state: what every node knows (#357, #358)

Nodes join and leave, and any node's UI must show the whole cluster. State falls into two kinds with two consistency
models:

| kind | examples | writer | model |
| --- | --- | --- | --- |
| observations | node health and device, queue and run summaries, member progress, leases as their grantor sees them, events, round timelines, cached shard ids | the node the fact is about, its *origin* | eventually consistent, replicated to every node |
| training state | `θ`, outer momentum `v`, round, generation, term, shard deal, accepted `Δk` | the run's coordinator of the current term | single writer; its result replicated to `R` members each round |

Observations need no agreement: each fact has exactly one writer, so two replicas can only differ in how recent their copy
is, never in which value is right. Closing a round does need agreement (which contributions count, what `θ` becomes), and
no merge function can reach it after the fact. That stays with one coordinator, and what is replicated is its result
(Terms and coordinator failover, below).

### The observation store (`studio_state.py`)

- An entry is `(origin, kind, key) → value`, versioned `(incarnation, seq)`. `incarnation` is a counter the node keeps on
  its own disk and increments at every start; `seq` increments with every write in that process. Only the origin writes
  its entries.
- Merge keeps the higher version of each entry. The merge is commutative, associative and idempotent (a state-based CRDT:
  a map of single-writer registers), so replicas that have seen the same writes hold the same state, whatever order the
  writes came in and however often. A newer incarnation of an origin replaces all of that origin's older entries, so a
  restarted node's stale facts go in the same merge that brings its new ones.
- **Anti-entropy rides the poll every node already makes** (`studio_remote`, every `POLL_EVERY` = 5 s, to every peer):
  `POST /state/sync` sends the caller's version vector `{origin: (incarnation, seq)}`; the peer answers with every entry
  newer than it, from every origin, plus its own vector, and the caller pushes back what the peer lacks (push-pull).
  Entries are relayed, not only told by their origin, so a fact reaches a node that cannot reach its origin as long as a
  path of working links exists. A write reaches every connected node within one poll interval, two when relayed. Full mesh
  is right at two to five nodes; past that, sync with a few random peers per interval (gossip) instead.
- **Joining** is a sync with an empty vector: the answer is the whole state. Nothing in the store is persisted. Every fact
  is regenerated by its origin, so a restarted node's view refills on its first sync.
- **Ages, not timestamps.** No two clocks are compared. Each entry carries its age when sent; the receiver adds the time
  since it arrived, on its own monotonic clock, so a relayed entry's age includes the relay delay. Every view shows the
  age beside the value.
- **Departure.** No node deletes another's entries. An origin whose newest entry is older than `PRUNE_AFTER` (default 1 h)
  is pruned, leaving a tombstone `(origin, incarnation)` that refuses that incarnation's entries if a peer relays them
  back; the tombstone is dropped after a day. A node that returns comes with a new incarnation and is accepted.
- **Size.** Entries are summaries, bytes to a few KB: the latest progress per member, the last 200 events per origin, the
  last 8 round timelines. Weights and shards never go through the store.

### Reading and writing

- Every read of cluster-wide state is local. `GET /distributed/runs` and the cluster and queue views read the node's own
  replica, so any node's UI shows every run and every node, each fact labelled with its origin and age.
- Every command goes to the node that owns its target (stop a run: its coordinator; cancel a job: its owner) through
  `PeerClient`. A command never writes the replica; the owner writes the outcome and it arrives by sync.
- During a partition each side shows what it last heard from the other, with growing ages and the origin marked down. The
  latest state on the far side of a partition cannot be known (CAP); the UI says how old its copy is instead.

## Scheduler: ownership, leases and fencing (#127, #128)

**The node a job was queued on owns it** for its whole life. There is no cluster-wide consensus store: with two to five
laptops, a Raft or etcd cluster would add a quorum that a home network loses more often than it loses a node. A job owner is
the single controller for its job, the same choice torchelastic makes with its rendezvous host and Pathways with its single
controller. If the owner is down, its jobs pause; they continue when it returns, from durable state on its disk. A
distributed run is the exception: its coordinator role can move to a member (Terms and coordinator failover), and the
owner keeps only the queue record, which says where the run is now coordinated.

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
  The members of a distributed run hold leases granted by the run's coordinator; a new coordinator term grants them again,
  and the fence compares `(term, epoch)`.
- **Reassignment.** When a lease expires, the owner waits `duration_s` more (the runner's self-fence has certainly fired by
  then), marks the job *interrupted* at its last checkpoint, and places it again. On a different node the job resumes from
  the last published checkpoint with a fresh optimizer, and its record says so. Placed again on the node that wrote that
  checkpoint (#185), it continues from that node's own `last.npz` with the optimizer, sampler and RNG (a published
  checkpoint is the weights only), on the dataset snapshot saved beside it; if the file is gone or training refuses it, the
  node falls back to the published weights, and the record says which way it went.

Placement stays `studio_cluster.place()`. A job that needs several nodes (a distributed run) is **gang-placed**: the
scheduler takes leases on `min_members` to `max_members` free nodes in one decision (`studio_scheduler.decide_gang`,
`Gangs.place`), or none, so two half-placed runs never deadlock each other waiting for the last node. The nodes are the
ones the run was given, this one included (it coordinates); the run waits in the queue, with the reason, until enough of
them are free at once. Each member slot is a lease. The owner renews a slot while the coordinator hears from that member,
and a slot silent for longer than its lease (more for a member that has not joined yet) is freed: the coordinator lets
the member go at the round boundary (`Coordinator.lose`), its member job leaves the node's queue, and a node that is free
then is leased the slot and invited, joining at the round after. A node that frees up later fills an open slot the same way,
up to `max_members`. Two runs owned by different nodes decide on their own polls, so they can still pick the same free node
in the same few seconds; the second member job then waits on that node's queue.

## Training runtime: DiLoCo (#132)

A distributed run has one **coordinator**, on the owner node until a failover moves it, and one or more **members**, each a
worker process on a node (the coordinator's node usually runs a member too). The coordinator holds the global weights and
the outer optimizer and is the only writer of the run's training state; members train, and some of them keep a replica of
the last round so the run outlives its coordinator.

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
  deletes the cache when the run ends. Validation windows stay on the coordinator and its replicas, which never train on
  them.

### Wire protocol

All routes are on the coordinator's API, behind node-to-node auth, under `/collective/{run}`. Every call carries the run's
`term`; a lower term than the coordinator's is `409`, and an answer with a higher term tells the caller to follow it.

| route | body | answer |
| --- | --- | --- |
| `POST /join` | node, lease epoch | rank, generation, round, config, tokenizer, shard ids |
| `GET /params?round=r` | | the round's `θ`, fp32 `.npz` (downloads are the fast direction on Starlink) |
| `GET /shards/{s}` | | one training shard |
| `POST /rounds/{r}/delta` | generation, epoch, tokens, steps; `Δk` as fp16, fp32 or int8 `.npz` (int8: per-tensor scale beside the values) | `202`, or `409` when stale |
| `GET /rounds/{r}` | | `open` or `closed` with the next round; a `pending` round long-polls up to 30 s |
| `POST /progress` | node, generation, epoch, round, phase (`downloading`, `training`, `uploading`, `waiting`), step, loss, tokens/s | `{accepted}`, or `409` for a stale generation or epoch, a non-member, or a closed round |
| `POST /leave` | generation, epoch | `204` |
| `GET /state?round=r` | | a replica's copy of the closed round: outer momentum `v` (fp32 `.npz`) and `diloco.json` |
| `GET /validation` | | the validation windows, for a replica, once per run |

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
loss, speed, seconds since the report, shards, samples); the page polls it once a second while a run is going. The coordinator writes the run's
summary, events and timelines into the observation store and each member writes its own progress, so `GET /distributed/runs` answers from the
local replica on every node: the page is the same on any machine, with each part's age. Taking the lock for a report or a
snapshot waits at most a quarter second, so a round that is closing (the held-back text is checked then) never blocks the page.

`studio_api --fake` opens on a simulated run (`demo-run`: one MPS and one CUDA machine) that advances over time, so the tab can be seen without a second machine.

### Checkpoints

After each round the coordinator writes the ordinary `last.npz` (the global weights, loadable by everything that loads a
checkpoint today) and `diloco.json` (round, generation, term, outer momentum file, shard deal, membership, per-round
metrics). Restarting the coordinator continues from the last completed round, unless a failover has moved the run.

**Saving the weights as a Version (#333).** A `pretrain_distributed` job carries its design's version through queued, training and
trained like a local pretrain (`VERSION_JOBS`). When the job ends `done` and the run's `run.json` says `completed` (all rounds
done), `WorkspaceServices._record_distributed` records the run folder on that version, points its best checkpoint at the final
`last.npz`, scores it on the held-back text like a local pretrain, and adds a change note naming the run and its machines. The
run's view then names it (`saved_version`) and the Training tab links to it. A stopped or failed run saves nothing, even past
some rounds (the round weights stay in the run folder): the version goes back to `saved`.

**The Training tab's records (#334).** The runs the tab lists are kept in `data/distributed_runs.json` (`studio_distributed.persist`,
written atomically when a run is opened, after each round, when it stops and once the invitations are answered) and read back when
the services start (`load`, the newest `KEPT`). A missing or corrupt file gives no runs. A run whose last account still says it was
going loads as stopped with a note that the app restarted, unless the queue still holds its job (then it waits as queued and the
job rebuilds the coordinator from `diloco.json`).

**Durability.** The latest completed round (`θ`, `v`, `diloco.json`, term) is on the coordinator and its `R` replicas, so
it survives the loss of any `R` of those `R + 1` nodes without the bucket. Only the latest round is kept; runs are
experiments and there is no round history to restore.

### Terms and coordinator failover (#359, #360, #361, #362)

A run has a **term**, starting at 1, that orders its coordinators the way the lease epoch orders a job's runners. A
coordinator refuses a lower term with `409`; a node that sees a higher term, in an answer or in the observation store,
follows it: a member rejoins the new coordinator, an old coordinator stops closing rounds and joins as a member.

- **Replicas.** At each round close the `R` lowest-ranked live members (default `R = 2`) are the run's replicas. Besides
  `θ`, which every member already downloads, a replica downloads `v` and `diloco.json` (`GET /state`), and once per run
  the validation windows and every shard it does not hold. It keeps the latest round only, under `data/replicas/<run>/`,
  deletes it when the run ends, and writes `(run, term, round)` held into the observation store. `v` is the size of `θ`
  (100 MB fp32 for 25M parameters), on the fast direction.
- **Losing the coordinator.** Members cannot reach it, cannot renew their leases, and self-fence: they stop training but
  keep `θ` and, on replicas, the round state. The coordinator counts as gone once the failure detector has it *down*
  (`DOWN_AFTER`) and the members' leases have lapsed: the wait the scheduler takes before it reassigns a job.
- **The successor.** Every member applies one rule to its replica of the observation store: among live replicas, the one
  holding the highest completed round, ties to the lowest rank. That replica claims term `t + 1` at its round, writes the
  run summary (coordinator, term, round) into the store, grants leases and takes joins. The run continues from the last
  completed round; the round in flight is lost, as in a restart.
- **Conflicting claims.** While views differ (a partition, or a store that has not converged) two replicas can both claim.
  Claims order by `(term, round, lower rank)`; a coordinator that sees a higher claim steps down and joins it, and the
  rounds the loser closed are discarded. This is the price of staying available on two or three machines: a quorum of two
  is both of them, so Raft or Paxos would halt the run at any loss. Losing the smaller side's rounds after a partition is
  acceptable for an experiment, and the run never ends with two `θ`s.
- **The bucket breaks ties, it is never the source of truth.** When the bucket is configured and reachable, a claimant
  first writes `runs/<run>/term/<t+1>` with a conditional PUT (`If-None-Match: *`): the first writer wins and the others
  read the winner and follow it, which removes split brain whenever both sides can reach the bucket. When it cannot be
  reached, the rule above decides alone. No decision is ever only in the bucket.

## Failures

| what fails | what happens |
| --- | --- |
| a member's node loses its network | its lease lapses; it self-fences and stops; the round closes without it at the deadline; the scheduler may give the slot to another free node, which joins next round |
| a member's process crashes | same as above, and the owner sees the lease lapse sooner (no renewals) |
| the node of a single-node job | lease lapses, the owner waits out the self-fence, marks it interrupted, places it again from the last checkpoint |
| the coordinator's node | members self-fence; once it is down and the leases have lapsed, the replica with the latest round takes over at term + 1 and the members rejoin it; the round in flight is lost |
| the coordinator returns | its first sync shows a higher term; it steps down and joins as a member |
| a partition splits a run | each side with a replica continues under its own claim; on healing the higher `(term, round)` wins and the other side's rounds are discarded; with the bucket reachable only one claim succeeds |
| every replica and the coordinator | the run stops; it is an experiment, start it again |
| a node restarts | it starts a new incarnation; its old observations are replaced in the next sync, and it refills its view from a peer |
| a node returns with a job that moved | its renewals and writes carry an old epoch and are refused; it stops its copy |
| a slow link | a member that misses the deadline counts for nothing that round and rejoins next round |
| a stale or forged contribution | refused: wrong generation, round or epoch is `409`; an unauthenticated caller is `401` |

## Testing

- Pure state machines (lease table, membership, round accounting, outer optimizer, shard deal) tested directly in
  `tests/test_*.py`, with fake clocks.
- The 1-member equivalence test: DiLoCo with `η = 1`, `μ = 0` gives the same weights as plain training, step for step.
- A 2-member run in one test process over the real HTTP routes (FastAPI `TestClient`), on CPU, that converges below the loss
  of one member alone after the same number of steps per member.
- The observation store: N in-process replicas exchanging syncs in random order, duplicated, dropped and partitioned,
  hold identical state once the links heal; a restart and a prune behave as above.
- Failover: three in-process members over `TestClient`; stopping the coordinator makes the lowest-ranked replica take over
  at the last round with term + 1, and a returning coordinator steps down. Two conflicting claims settle on the higher one.
- The two-machine checks (a member on the Mac, one on the NVIDIA box; pulling a network cable mid-round) are manual and are
  written down in each PR.

## Issues

- #126 node-to-node auth and the peer client
- #127 dispatch with leases and fencing; #128 failover from the last checkpoint
- #132 DiLoCo runtime: coordinator, members, shards, wire protocol, round checkpoints
- #133 gang-place a distributed run through the scheduler (needs #127 and #132)
- #129 device records and the cross-device resume test
- #357 the observation store and its sync; #358 every node's facts in it, any node's UI shows every run
- #359 run term; #360 replicate the last round to `R` members; #361 coordinator failover; #362 the bucket as tie-breaker
