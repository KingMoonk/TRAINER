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

### Set up Backblaze B2

1. In the Backblaze console open **B2 Cloud Storage**, **Buckets**, **Create a Bucket**. Keep it **Private**.
2. Note the bucket's **Endpoint** on its details (`s3.<region>.backblazeb2.com`, for example `s3.us-west-004.backblazeb2.com`).
3. Under **Application Keys** choose **Add a New Application Key**. Pick the one bucket (not *All*) and **Read and Write**.
   Copy the **keyID** and the **applicationKey** now: the applicationKey is shown once.
4. Set `TRAINER_STORE_REGION` to the region in the endpoint (`us-west-004`). B2 expects the region its endpoint names; `auto`
   is R2's value.

### Environment variables

Set these on every machine, with the same bucket and prefix, before starting the app. All but the last two are required; none
set at all means no bucket (the machine works alone).

| variable | meaning | R2 | B2 |
| --- | --- | --- | --- |
| `TRAINER_STORE_ENDPOINT` | the S3 endpoint URL | `https://<account>.r2.cloudflarestorage.com` | `https://s3.<region>.backblazeb2.com` |
| `TRAINER_STORE_BUCKET` | the bucket name | `trainer-versions` | `trainer-versions` |
| `TRAINER_STORE_KEY_ID` | the access key id | Access Key ID | keyID |
| `TRAINER_STORE_SECRET` | the secret | Secret Access Key | applicationKey |
| `TRAINER_STORE_PREFIX` | a folder inside the bucket, so the bucket can hold other things (optional, default none) | `trainer` | `trainer` |
| `TRAINER_STORE_REGION` | the signing region (optional) | leave unset (`auto`) | e.g. `us-west-004` |

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
| `POST /versions/{n}/pull` | A job: downloads published version `n` and adds it here as a trained version. `?origin=<machine>` picks one when two machines published a version `n`. Pulling what is already here returns that version. |

`n` in a pull is the number the publishing machine gave the version. A pulled version has no scorecard: the numbers measured on the other
machine used its held-back text, so they are not comparable here. Its file is kept under `data/pulled/`.

Without a store the routes answer `409` with a sentence on what to set up; a store that fails (an unmounted folder, a refused key, an
unreachable bucket) ends the job with a sentence on what to check, and running it again retries. The store is the shared folder in the
workspace config, `"studioConfig": {"store": {"folder": "/path/to/share"}}`, else the `TRAINER_STORE_FOLDER` or `TRAINER_STORE_*` variables
above. (The queue's own dispatch between nodes reads only the variables.)
