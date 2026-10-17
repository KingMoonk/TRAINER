# Cluster

Several machines can pool their compute. Each one runs the API as a node; a node alone is a cluster of one. Nodes find each
other over a Tailscale network (a *tailnet*), hand whole jobs to each other, and exchange trained versions through a bucket.
A distributed pretraining run uses several machines at once with DiLoCo; `docs/distributed.md` is its design.

Three things to set up, in this order: the tailnet, the node list in the workspace config, and the bucket.

## The tailnet

Tailscale gives every machine a private address and a name that work through home routers and CGNAT (Starlink) without opening
a port. The tailnet is the whole perimeter: the API is not exposed to the internet.

1. Make a Tailscale account (the free personal plan is enough) and install Tailscale on every machine that will be a node,
   then sign in to the same account on each.
2. Turn on **MagicDNS** in the admin console (**DNS** page). Every machine is then reachable by its name: `windows-box`, or the
   full `windows-box.tail1234.ts.net`. Both are shown on the **Machines** page.
3. Check the machines see each other: `tailscale ping windows-box` from the Mac. It says whether the path is direct or goes
   through a relay (DERP). A relayed path works but is slower; job placement does not depend on it, only the speed of copying
   files between the two.
4. Run the API on each node and bind it to the tailnet address (`--bind`, see `docs/node.md`), so the other nodes can call it. The default port is `8787`.

## Inviting a second user

A friend with their own machine joins in one of two ways.

- **Share one machine** (nothing else is exposed). In the admin console, **Machines**, the machine's menu, **Share...**, and
  send the link. They accept it from their own Tailscale account and see only that machine. Use this to lend one box.
- **Invite them to your tailnet** (everyone sees everyone, narrowed by the ACL below). **Users**, **Invite external users**
  (or **Invite users** for someone on your organization's domain). They install Tailscale, accept, and their machines join.
  Use this to pool several machines each way.

Either way, tell them the machine names to list in their config, and give them the bucket settings if they should publish
and pull versions (the keys give read and write on the bucket: only share them with people you would trust with your versions).

## ACL

By default every machine on a personal tailnet can reach every other on every port. Once a second person is in, restrict it to
the API port on the machines that are nodes. In the admin console, **Access controls**, tag the nodes
(`sudo tailscale up --advertise-tags=tag:trainer` on each, or tag them in the **Machines** menu) and use something like:

```json
{
  "tagOwners": {"tag:trainer": ["autogroup:admin"]},
  "groups": {"group:trainer": ["you@example.com", "friend@example.com"]},
  "acls": [
    // people in the group open the web UI that `tailscale serve` publishes on 443
    {"action": "accept", "src": ["group:trainer"], "dst": ["tag:trainer:443"]},
    // nodes call each other's API directly on 8787 (to poll /node and hand over jobs)
    {"action": "accept", "src": ["tag:trainer"], "dst": ["tag:trainer:8787"]}
  ]
}
```

Replace the emails with the Tailscale logins. Browsers use port `443` (`tailscale serve`); nodes use the API's own port `8787`, and the
API identifies them by asking Tailscale who owns the source address, so a node must carry `tag:trainer` (or a tag listed in
`studioConfig.cluster.allowTags`, `docs/node.md`). A person who should call a node's API directly on `8787` too needs their
login in `studioConfig.cluster.allowUsers` and a rule for `:8787`. Anything not accepted is denied, so
this also keeps the people in the group off the machines' other ports (SSH, file sharing). A tagged machine no longer
belongs to a person: its logins are managed through the tag, so tag the nodes only after you are sure you can still reach
them from your own devices (the first rule does that).

## Listing nodes in the workspace config

Nodes are listed by hand. Add a `nodes` list to `studioConfig` in the workspace file `data/workspace.json`, on every machine that
should see the others (the list names the *other* machines; the machine itself is always a node):

```json
{
  "studioConfig": {
    "nodes": ["windows-box", "mini.tail1234.ts.net", "http://100.64.0.7:8787"]
  }
}
```

Each entry is a tailnet name (port `8787` is assumed, the direct node-to-node path), or a full URL when the API is on another port
or behind `tailscale serve` (`https://windows-box.tail1234.ts.net`, port 443; a caller through serve is a person, so it needs a
login the ACL lets through, not a tagged node). Blank and repeated entries are ignored. With no `nodes` (or an empty
list) the machine works alone.

Each machine asks the others for `GET /node` with a two-second timeout and shows what answered: hostname, version, device
(`cuda`, `mps` or `cpu`), memory, and whether it is busy. Video memory is reported on CUDA; on Apple Silicon and CPU it is the
computer's memory. A machine that does not answer is shown offline with the reason, and is never given a job; the next poll
brings it back.

The API serves the same survey: `GET /cluster/nodes` lists this machine first and then each listed peer (`local` marks this machine;
a peer that does not answer has `online: false` and an `error` sentence), and `PUT /cluster/nodes` with `{"addresses": [...]}` saves
the list in `studioConfig.nodes` for you. Each entry is checked first: a name or IP, with an optional `:port`, or an `http://` or
`https://` URL; anything else is refused with a sentence, up to 32 entries. Peers are asked at the same time and the survey ends at
the two-second timeout, so one dead machine never holds up the rest.

## Where a job runs

Every job kind carries requirements, and the cluster picks the best free node that fits them (`studio_cluster.py`):

| job | prefers | notes |
| --- | --- | --- |
| pretraining, RLHF rounds | a CUDA node | a free node of another kind still runs it if no CUDA node is free |
| evals, scoring, benchmarks, data generation, LoRA, passive training | any free node | this machine first, on a tie |
| inspect jobs (memory-bound) | the node with the most memory | |

A job may also ask for a minimum of memory or a dtype (`float32`, `bfloat16`, `float16`); those are not preferences: a node
below the memory, or that does not train in the dtype, never gets the job. You can pin a job to one node by name instead: it
runs there or waits, and is never moved to a better one.

## The version bucket

Machines exchange trained versions through an S3-compatible bucket, because upload bandwidth is the limit on a home
connection and a bucket both ends can reach needs no open ports. Cloudflare R2 and Backblaze B2 both have a free tier of about
10 GB, enough for many small-model versions (check each provider's pricing page for the current limits). Any S3-compatible
service works the same way; only the endpoint and keys differ.

A shared folder (a LAN share or a synced drive) works too and needs no setup: `FolderStore(path)` in `studio_publish.py` writes
the same files described below. This section is about the bucket.

### What gets uploaded

Publishing a version uploads two small objects, never the checkpoint itself:

| object | what it is |
| --- | --- |
| `<prefix>/<origin>/v<number>-<hash>.npz` | The weights only, in half precision (float16): no optimizer state, about a sixth of the training checkpoint. LoRA adapters are kept whole. |
| `<prefix>/<origin>/v<number>-<hash>.json` | The version's number, name, spec and scorecard, the machine it came from, the device that trained it, the publish time, the file's size and its sha256. |

`<origin>` is the machine's hostname, so two machines can both have a "version 3" without clashing. `<hash>` is the first 12
characters of the file's sha256: publishing the same checkpoint twice uploads nothing the second time. The `.json` is uploaded
last, so other machines list a version only once its weights are complete, and they check the hash before using a download.

Pulling a version writes an ordinary float32 checkpoint on the local disk. It has no optimizer state, so a run resumed from it
restarts the optimizer. Nothing else is uploaded: datasets, runs, logs, tokenizers and the optimizer never leave the machine,
except that a distributed run sends its training windows to its own members over the tailnet (`docs/distributed.md`).

### Set up Cloudflare R2

1. In the Cloudflare dashboard open **R2 object storage** and **Create bucket** (for example `trainer-versions`). Leave the
   location on automatic.
2. On the R2 **Overview** page, under **Account Details**, choose **Manage** next to **API Tokens**, then **Create Account API
   token**. Give it **Object Read & Write** and scope it to the one bucket.
3. Copy the **Access Key ID** and **Secret Access Key** now: the secret is shown once.
4. The endpoint is `https://<ACCOUNT_ID>.r2.cloudflarestorage.com` (the account id is on the same Overview page). A bucket
   created in a jurisdiction (EU, FedRAMP) uses `<ACCOUNT_ID>.eu.r2.cloudflarestorage.com` and the like.
5. Leave `TRAINER_STORE_REGION` unset: R2 uses `auto`.
6. Keep the keys outside the repository. `docs/store.env.example` is the template: copy it to `~/.config/trainer/store.env`, fill in
   the endpoint, bucket, key id and secret, and source it from your shell profile (`~/.zshrc`):
   `[ -f ~/.config/trainer/store.env ] && . ~/.config/trainer/store.env`. Open a new shell, then run the check below.

### Set up Backblaze B2

1. In the Backblaze console open **B2 Cloud Storage**, **Buckets**, **Create a Bucket**. Keep it **Private**.
2. Note the bucket's **Endpoint** on its details (`s3.<region>.backblazeb2.com`, for example `s3.us-west-004.backblazeb2.com`).
3. Under **Application Keys** choose **Add a New Application Key**. Pick the one bucket (not *All*) and **Read and Write**.
   Copy the **keyID** and the **applicationKey** now: the applicationKey is shown once.
4. Set `TRAINER_STORE_REGION` to the region in the endpoint (`us-west-004`). B2 expects the region its endpoint names; `auto`
   is R2's value.

### Check the connection

```bash
python -m studio_publish --check            # reads the settings, lists the bucket, prints objects, bytes used, limit and room left
python -m studio_publish --check --write    # also writes, reads back and deletes a small test object
python -m studio_publish --check --queue data/queue.json   # (the default) which queue file tells the leftover job checkpoints
```

With no keys set it prints `Not configured` and says what to set; a half-set environment names what is missing. A refused key, a
wrong bucket or an unreachable endpoint ends with a sentence on what to check. It exits 0 when everything works and 1 otherwise,
and it never prints the key id or the secret. The write test goes through the same limit as any upload. It is a command to run in
a terminal, not something the app does on its own.

It also lists the **job checkpoints left over**: checkpoint objects in the bucket (recognised by their metadata, `job:<id>` and
`checkpoint at step N`) that no unfinished job in this machine's queue file can resume from, because the job is not in the queue or
has ended. Each line gives the key (with the bucket prefix; delete its `.npz` and `.json` by hand), the job, the size and when it was
published. It only lists and never deletes. Run it from the TRAINER folder, or pass `--queue` the path of `data/queue.json`; with no
readable queue it only counts the checkpoints. A checkpoint of a job another machine owns is not in this machine's queue, so it shows
up here too: look at that machine's queue before deleting.

### The free tier is a hard limit

A bucket store never takes an upload that would take the bucket past its limit. The limit is **9.5 GB** (9.5 x 10^9 bytes) unless
`TRAINER_STORE_LIMIT_GB` says otherwise: 0.5 GB under R2's 10 GB free storage, so that a second machine uploading at the same moment,
or a measurement that is a little behind, cannot carry the bucket past the free tier. Before the first byte of a publish goes up, the
store adds up the size of **every object in the bucket** (all pages of the listing, whatever the prefix, since the free tier counts
the whole bucket) and of **every unfinished multipart upload** (see below), adds the size of the weights and the metadata it is about
to put, and refuses when the total passes the limit. The refusal is a plain sentence with the sizes, for example:

> Not uploaded: this upload needs 412.30 MB and the bucket already holds 9.20 GB, which would make 9.61 GB, over the 9.50 GB
> limit (the default cap, 0.5 GB under the R2 free tier). Delete versions you no longer need from the bucket (the Cloudflare
> dashboard, R2, the bucket's Objects), then publish again.

- **An upload that cannot be checked is refused**, and says so. If the listing fails (no network, a refused key), or its answer does not add
  up (a page that says it is truncated but gives no continuation token, a token on a page that says it is the last, a repeated
  token, an object without a size), the upload is refused with "the bucket's usage could not be read", not with "limit reached". Try
  again; if it keeps happening, run `python -m studio_publish --check`.
- `TRAINER_STORE_LIMIT_GB` sets the limit. Anything up to 10 is allowed (`9.8` gives less headroom than the default; `5` gives more).
  A value above 10, or one that is not a positive number, is an error: every upload fails with a sentence naming the setting.
  It cannot raise the limit past the free tier.
- Publishing a version that is already in the bucket uploads nothing, so it is never refused. The check for "already there" is made
  while the upload slot is held, so two publishes of the same version at once put it up once and count it once.
- Each upload that goes ahead lists the whole bucket and its unfinished uploads first: one R2 Class A operation per 1,000 objects (the
  free tier allows a million a month).
- The refusal reaches you where the upload was asked for: `POST /versions/{n}/publish` is a job, and the sentence is the job's `error`
  (`GET /jobs/{id}`, and the `job` event on `/events`). The web app has no screen for that publish job, so read it there or in the
  terminal. A job running on another node is described under "Jobs on other nodes" below.
- Only a bucket store has the limit. A shared folder (`TRAINER_STORE_FOLDER`) is limited by its disk.

#### Unfinished multipart uploads count

An upload cut off part-way (a lost connection, a killed process) leaves its uploaded parts in the bucket, invisible in the object
listing, and R2 bills them as storage. The check counts them: it lists the bucket's in-progress multipart uploads and adds up their
parts, so a bucket with a pile of orphans refuses new uploads instead of drifting past the free tier. If that listing fails, the
upload is refused. `python -m studio_publish --check` prints `Unfinished uploads: N, holding X`.

To stop orphans collecting, add an **abort incomplete multipart uploads** lifecycle rule to the bucket (Cloudflare dashboard, R2, the
bucket, Settings, Object lifecycle rules, "Abort incomplete multipart uploads" after 1 day). R2 has a 7-day rule by default; a
shorter one frees the space sooner. Parts of an unfinished upload can also be aborted by hand from any S3 client.

What the limit does not see, so leave some room: a second machine uploading at the same moment (the check is made against what the
bucket holds when it is made; uploads from one machine are serialized, which is why the default sits under 10). R2 bills storage as the average of each day's peak over the month, so 10 GB at any moment is a conservative cap. Other
things in the same bucket do count, which is the point: use a bucket of its own, or a lower `TRAINER_STORE_LIMIT_GB`.

#### Jobs on other nodes

A job a peer runs for you publishes a checkpoint to the store about every time it saves one, and publishes its result when it ends.
Only one checkpoint per job is kept: **after the owner has taken a new checkpoint, the runner deletes that job's previous checkpoint**
(the weights and their metadata), and when the result is stored it deletes the rest. It deletes only objects its own upload created,
so a version someone else published, or one already in the bucket, is never touched. A delete that fails is logged and is not
fatal; the key is tried again with the next cleanup. A checkpoint is never deleted before the owner has the key of its replacement,
so the owner is never left pointing at something gone.

**The owner deletes what the runner could not.** A runner that is lost, or a job that is cancelled or finishes on another node, leaves
its last checkpoints in the bucket, where they count against the cap. The node that owns the job (the one whose queue holds it)
records the key of every checkpoint a runner reports, and deletes them once the job can no longer be resumed from them:

| the job | its checkpoints |
| --- | --- |
| running, or interrupted and waiting to be placed again | the newest is kept (a resume starts from it); the ones before it are deleted |
| done (on any node, including after being placed again), failed, or cancelled | all deleted, except the key of the result when it has the same bytes |
| removed from the queue while waiting | all deleted |

Only keys a runner reported for that job are deleted, never a prefix sweep of the bucket, and a key is first looked up: it is deleted
only if its metadata says it is a checkpoint of that job (a key that is a published version, another job's checkpoint, or the
result, or one a job that is still running counts on, is left alone). Deleting adds no bytes, so it does not go through the size
check, and nothing in this cleanup uploads. A delete that fails is logged and tried again about a minute later; it never stops
the job or the queue. The keys still to delete are kept in the job's record in `data/queue.json` (`params.cluster.checkpoints`), so
a restart carries on. What the owner never heard of stays: a checkpoint uploaded after the runner's last renewal, one from a runner
that kept going after the job ended, one for a job taken out of the queue whose delete failed before the app restarted, or whose record has
aged out of the queue's last 50 finished jobs. `python -m studio_publish --check` lists those. A folder store is cleaned the same way.

When the store refuses:

- **A checkpoint: the job keeps training.** The checkpoint is only a way to resume if the node is lost, so losing one is not worth
  stopping a run for. The runner does not offer that checkpoint again (it would be refused again, and each try lists the bucket); it
  tries the next one it saves. A refusal because the usage could not be read is tried again after a short wait instead. The refusal
  is reported, never dropped: the runner's row in `GET /cluster/jobs` says it in `waiting_reason`, and it travels to the owner on the
  next lease renewal, where it is added to the job's note (`Checkpoint at step N not shared: ...`, once per reason; the Queue page
  shows it under "Running on other machines") and sent as a `cluster` event with `state: "checkpoint_refused"`. When room is made,
  the next checkpoint goes through and the report clears. Until then, if the node is lost the job resumes from the last checkpoint
  that did make it, or starts again.
- **The result: the work is not lost.** The runner first deletes this job's checkpoints (they are the likeliest thing filling the
  bucket) and tries the result once more. If it is still refused, the job ends `failed` with the sentence, and the sentence ends
  with where the finished weights are kept: on the runner's disk, in its run folder, at the path it names. Free some room and publish
  from there, or copy the file by hand. The runner keeps its checkpoints in that case, and does not retry when the usage could not be
  read at all.
- **A catch-up job's starting version: it waits.** The job is not failed and the peer is not blamed or put on a cool-down; the job's
  waiting reason is the plain limit sentence (a `cluster` event with `state: "store_refused"`), and it is offered again after a
  short wait.

### Environment variables

Set these on every machine, with the same bucket and prefix, before starting the app. The first four are required and the last three optional; none
of the bucket variables set at all means no bucket (the machine works alone).

| variable | meaning | R2 | B2 |
| --- | --- | --- | --- |
| `TRAINER_STORE_ENDPOINT` | the S3 endpoint URL | `https://<account>.r2.cloudflarestorage.com` | `https://s3.<region>.backblazeb2.com` |
| `TRAINER_STORE_BUCKET` | the bucket name | `trainer-versions` | `trainer-versions` |
| `TRAINER_STORE_KEY_ID` | the access key id | Access Key ID | keyID |
| `TRAINER_STORE_SECRET` | the secret | Secret Access Key | applicationKey |
| `TRAINER_STORE_PREFIX` | a folder inside the bucket, so the bucket can hold other things (optional, default none) | `trainer` | `trainer` |
| `TRAINER_STORE_REGION` | the signing region (optional) | leave unset (`auto`) | e.g. `us-west-004` |
| `TRAINER_STORE_LIMIT_GB` | the most the whole bucket may hold, in GB (optional; default 9.5, maximum 10; see "The free tier is a hard limit") | `9.5` | `9.5` |

On macOS or Linux, put them in the shell profile of the account that starts the app:

```bash
export TRAINER_STORE_ENDPOINT=https://<account>.r2.cloudflarestorage.com
export TRAINER_STORE_BUCKET=trainer-versions
export TRAINER_STORE_KEY_ID=...
export TRAINER_STORE_SECRET=...
export TRAINER_STORE_PREFIX=trainer
```

On Windows, set them once from PowerShell (they apply to programs started afterwards):

```powershell
setx TRAINER_STORE_ENDPOINT "https://<account>.r2.cloudflarestorage.com"
setx TRAINER_STORE_BUCKET "trainer-versions"
setx TRAINER_STORE_KEY_ID "..."
setx TRAINER_STORE_SECRET "..."
setx TRAINER_STORE_PREFIX "trainer"
```

A half-set environment is an error that names the missing variables, not a silent fallback to working alone. The app never writes the
secret to the workspace or the logs; keep it out of anything you commit. If it leaks, delete the token or key in the
provider's console and make a new one: the bucket's contents are unaffected.

### Publishing and pulling from the API

With a store set up, the API shares versions:

| route | what it does |
| --- | --- |
| `POST /versions/{n}/publish` | A job: uploads version `n`'s weights and metadata. Its result is `{key, number, origin}`. |
| `GET /versions/published` | What the store holds, newest first: each entry has `origin` (the machine that published it), `mine`, and `local` / `local_number` (whether this workspace has it as a version). |
| `POST /versions/{n}/pull` | A job: downloads published version `n` and adds it here as a trained version. `?origin=<machine>` picks one when two machines published a version `n`, and asks that machine first (below). Pulling what is already here returns that version. |
| `GET /versions/{n}/weights/meta`, `GET /versions/{n}/weights` | What a peer calls on the machine that holds version `n`: the description `publish` would upload, and the half-precision weights file, streamed. No optimizer state, no run files. The copy is made on the first call and kept in `data/shared/` until the version changes. |

**Straight from the owner.** With `?origin=<machine>`, a pull asks the machine that published the version before it reads the store, when the
workspace lists it under `studioConfig.nodes` and it answers `GET /node` within two seconds. The owner is found by hostname, so a listed IP
address works. The weights then travel over the tailnet and the store is not read. If the owner is off, is not listed, or cannot send the
version, the pull falls back to the store. The job's progress says which source was used and why (`Copied version 3 straight from mac.`,
`mac did not answer, so version 3 came from the version store.`). Without `?origin=` the store is listed to learn who published the version (then that machine is asked the same way), so the store is read for the listing but the weights still come from the owner.

`n` in a pull is the number the publishing machine gave the version. A pulled version has no scorecard: the numbers measured on the other
machine used its held-back text, so they are not comparable here. Its file is kept under `data/pulled/`.

Without a store the routes answer `409` with a sentence on what to set up; a store that fails (an unmounted folder, a refused key, an
unreachable bucket) ends the job with a sentence on what to check, and running it again retries. The store is the shared folder in the
workspace config, `"studioConfig": {"store": {"folder": "/path/to/share"}}`, else the `TRAINER_STORE_FOLDER` or `TRAINER_STORE_*` variables
above. (The queue's own dispatch between nodes reads only the variables.)
