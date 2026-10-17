---
name: setup
description: Get TRAINER running on this machine from scratch (Python venv, web build, API) and audit Tailscale, serve, tags, ACL hints and the workspace node list, reporting anything missing. Use when the user says /setup, "get it running", "set up this machine as a node", or a node will not load.
---

# /setup — zero to running, then audit the tailnet

Bring this machine to "the web UI loads at its tailnet URL", fixing what can be fixed from the shell and listing what only the
user can do. The README Quickstart, `docs/node.md` and `docs/cluster.md` are the reference; this skill runs them in order and
checks each step.

## Ground rules

- Never open an app window: no `app.py`, no Tk, no Panda3D viewer. The web UI is checked with `curl`.
- Never read, print or ask for secrets. For the version bucket, report only which `TRAINER_STORE_*` variable *names* are set
  (`[ -n "${TRAINER_STORE_SECRET:-}" ]`), never values. Do not read `~/.config/trainer` or shell profiles. Do not run
  `studio_publish --check` (it contacts the bucket); tell the user to run it themselves if they want to.
- `sudo` commands, the Tailscale login, the admin console and installers are the user's: give the exact command or click path
  and wait. Do not run `sudo` yourself.
- Run long-lived processes (the API) in a Terminal tab the user can see (`run_in_terminal`), not in the background Bash tool.
- Work from the repository root. Use `.venv/bin/python` (Windows: `.venv\Scripts\python.exe`), never a bare `python`/`python3`.

## 1. Toolchain

Check each and record ok / missing with the fix:

| check | command | fix if missing |
| --- | --- | --- |
| Python ≥ 3.12 for the venv | `.venv/bin/python --version`, else `python3 --version` | python.org or `brew install python@3.13`. Xcode's `python3` is 3.9 and will not do. |
| Node 24 | `node --version` | nodejs.org or `brew install node@24` |
| pnpm | `pnpm --version` | `npm install -g pnpm` |
| git | `git --version` | Xcode command line tools / git-scm.com |
| Tailscale CLI | `command -v tailscale`, else `/Applications/Tailscale.app/Contents/MacOS/Tailscale`, else `C:\Program Files\Tailscale\tailscale.exe` | tailscale.com/download (optional: only for other devices and clusters) |

Call the found Tailscale binary `$TS` below.

## 2. Python environment

- No `.venv`: create it with a ≥ 3.12 interpreter (`python3.13 -m venv .venv` or whichever passed step 1).
- `.venv/bin/python -c "import torch, fastapi, uvicorn, numpy, pandas"`; on failure
  `.venv/bin/python -m pip install -r requirements-dev.txt`, then re-check.
- Report the device: `.venv/bin/python -c "import torch; print('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')"`.
  On a machine with an NVIDIA GPU that reports `cpu`, point to `requirements-gpu.txt` (the CUDA wheel) — do not reinstall torch
  without asking.

## 3. Web build

- `cd web && pnpm install` if `web/node_modules` is missing.
- If `web/out/index.html` is missing or older than the newest file under `web/src`, run `.venv/bin/python tasks.py web:build`.
- Remember whether you rebuilt: a running API only picks up `web/out` when it starts.

## 4. The API

- Is something listening on 8787? macOS/Linux `lsof -nP -iTCP:8787 -sTCP:LISTEN`; Windows `Get-NetTCPConnection -LocalPort 8787 -State Listen`.
- Is it ours? `curl -s --max-time 5 http://127.0.0.1:8787/node` returns JSON with `hostname`, `device`, `version` when the API is
  bound to loopback. A `401` means it is bound wider (`0.0.0.0` or the tailnet address) and loopback needs the serve header — still
  ours. Anything else on the port: report the process (`lsof` line) and stop.
- Not running: start it in a Terminal tab titled `trainer api`:
  - tailnet use (Tailscale is installed and logged in): `.venv/bin/python -m studio_api --bind 0.0.0.0 --port 8787`
  - local only: `.venv/bin/python -m studio_api --bind 127.0.0.1 --port 8787`
  Read the tab after a few seconds: `Uvicorn running on` is success; `address already in use` means another copy holds the port.
- Running but you rebuilt `web/out` in step 3: tell the user to restart it (Ctrl+C in its tab, run again) rather than killing it.

## 5. Tailscale audit

Skip with a one-line note if Tailscale is not installed. Otherwise gather, without sudo:

```bash
$TS status --json | jq '{state: .BackendState, auth: .AuthURL, online: .Self.Online, name: .Self.DNSName,
  tags: .Self.Tags, magicdns: .CurrentTailnet.MagicDNSEnabled, certs: .CertDomains, user: .User[(.Self.UserID|tostring)].LoginName,
  peers: [(.Peer // {})[] | {name: .DNSName, online: .Online, tags: .Tags, os: .OS}]}'
$TS debug prefs | jq '{AdvertiseTags, WantRunning, CorpDNS}'
$TS serve status
```

Flag each problem with its fix:

| finding | meaning | fix (user runs) |
| --- | --- | --- |
| `state` is `NeedsLogin` and `AdvertiseTags` is non-empty | a tag request the policy refused; the machine stays logged out and its `.ts.net` name does not resolve | `sudo tailscale up --reset`, open the link, sign in. Add the tag later only after `tagOwners` is saved. |
| `state` is `NeedsLogin`, no tags | not signed in | open `auth` URL, or `sudo tailscale up` |
| `state` is `Stopped` | Tailscale is off | `tailscale up` or the menu-bar toggle |
| `online` false while `Running` | no connection to the control server | check the network, then `$TS netcheck` |
| `magicdns` false | names like `windows-box` will not resolve | admin console → **DNS** → enable MagicDNS |
| `certs` null/empty | HTTPS certificates off; `serve` cannot get a certificate | admin console → **DNS** → **HTTPS Certificates**, or follow the link the first `serve` prints |
| `serve status` has no proxy to `127.0.0.1:8787` | the UI is not published | `tailscale serve --bg 8787` (no sudo needed on macOS app installs) |
| serve proxies but the API is bound to `127.0.0.1` | fine for browsers; other nodes cannot call it directly | restart with `--bind 0.0.0.0` if this machine is a cluster node |
| `tags` null | fine while the tailnet is only the owner; needed once other people join | see `docs/cluster.md` ACL. `tailscale set` has no `--advertise-tags`; use the admin console **Machines** → **Edit ACL tags** or `sudo tailscale up --advertise-tags=tag:trainer` after `tagOwners` lists `tag:trainer` |
| a tag other than `tag:trainer` | the API only admits node callers with an allowed tag | add it to `studioConfig.cluster.allowTags` in `data/workspace.json` |

Then the end-to-end check: `curl -s -o /dev/null -w '%{http_code}' --max-time 10 https://<name>/` and
`curl -s --max-time 10 https://<name>/node`, where `<name>` is `Self.DNSName` without the trailing dot. `200` and the node JSON
mean done. `502`: the API is not running (step 4). `000`/DNS failure: the machine is logged out or MagicDNS is off (above).
`421`: the Host name is not one the API knows; pass `--allow-host <name>`.

## 6. Workspace and cluster config

Read `data/workspace.json` (create nothing without asking). Report:

- `studioConfig.nodes`: the peers this machine polls. For each, whether it appears in the Tailscale peer list and is online. A
  peer that is online but untagged is only admitted if the user's login is in `allowUsers`.
- `studioConfig.cluster.allowUsers` / `allowTags`. If there are peers, the peers are untagged and `allowUsers` does not contain
  this machine's Tailscale login (`user` above), say node-to-node calls will get `401` and show the JSON to add.
- If the API is up, `GET https://<name>/cluster/nodes` lists what this machine sees; report each peer's `online` and `error`.
- Version bucket: which of `TRAINER_STORE_ENDPOINT`, `_BUCKET`, `_KEY_ID`, `_SECRET` are set in this shell (names only). None
  set = no bucket, the machine works alone (fine). Some but not all = name the missing ones. All set = suggest the user run
  `.venv/bin/python -m studio_publish --check` themselves.

## 7. Report

End with one short table — step, status (ok / fixed / needs you), and for "needs you" the exact command or click path — then
the URL the UI is on (`http://127.0.0.1:8787` and, if published, `https://<name>/`). Lead with whether the UI loads. Do not
claim a step is ok without the command output that shows it.
