# Cluster

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
restarts the optimizer. Nothing else is uploaded: datasets, runs, logs, tokenizers and the optimizer never leave the machine.

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
