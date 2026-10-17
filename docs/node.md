# Running a node

A *node* is a machine that runs the Groundwork API (`studio_api/`). The API serves the web UI (`web/out`, one URL), the
`Services` routes and the `GET /events` stream, and it is the process that owns the workspace and the worker. Run it as a
service on every machine with compute, so the UI is there whenever a browser on the tailnet asks. `docs/cluster.md` covers
the tailnet itself and listing nodes; this page covers one machine.

## The command

```bash
.venv/bin/python -m studio_api --port 8787          # macOS / Linux
.venv\Scripts\python.exe -m studio_api --port 8787   # Windows
```

| Option | Meaning |
| --- | --- |
| `--bind HOST` | The address to listen on. Default `127.0.0.1`: only this machine, and `tailscale serve` (below) reaches it from there. |
| `--port N` | The port. Use `8787`, the port `studio_cluster.DEFAULT_PORT` assumes when a node is listed as a bare name. |
| `--allow-origin ORIGIN` | A web origin (the Vercel site, `https://trainer.example.vercel.app`) allowed to call the API from a browser. Repeat it for several, or list them comma-separated in the `STUDIO_API_ALLOW_ORIGINS` environment variable. Default: none (below). |
| `--allow-host NAME` | A host name the API answers to besides the defaults (below): a custom DNS name or a reverse proxy's name. Repeat it for several. |
| `--fake` | Serve `FakeServices` (empty, in-memory data) instead of the workspace, for frontend work. Needs no workspace and is always open. |
| `--allow-cpu` | Run on the CPU although this machine has a GPU. Without it the API prints its device at startup and refuses to start when the machine has a GPU (an Apple Silicon Mac, or `nvidia-smi` listing one) but torch reports `cpu`. `--fake` is never checked. |
| `--dump-openapi FILE` | Write the OpenAPI schema to `FILE` and exit. This is how `web/openapi.json` is made (see `web/README.md`). |

Install the requirements first (`python -m pip install -r requirements.txt`, as in the README); the API needs FastAPI,
uvicorn and the training engine. The web UI is served from `web/out`: build it once with `python tasks.py web:build` (Node and
pnpm needed; the API starts without it and then serves only the routes).

### Who is allowed in

The tailnet is the perimeter, but the API does not trust a header a caller can type. How a request is identified depends on
where it came from:

- **Open.** Bound to `127.0.0.1` (or `localhost`, `::1`) or run with `--fake`, the server is open: any process on this machine
  can use it, with or without a header.
- **Through `tailscale serve`** (a browser): the request arrives from the loopback address with the `Tailscale-User-Login`
  header. `tailscale serve` deletes any `Tailscale-User-*` value the client sent and writes its own, so the API trusts it on
  that path. A loopback request without the header gets `401`.
- **Straight to the node's tailnet address** (another node, `http://<node>:8787`): the header is ignored. The API asks the local
  `tailscaled` who owns the source address (`tailscale whois --json <ip>`, answers cached for a minute) and lets the caller in
  when it is a node carrying an allowed tag (default `tag:trainer`; recorded as `node:<hostname>`) or a user listed in the
  workspace config. Anyone else gets `401`. This needs the `tailscale` command on the node: it is found on `PATH`, at
  `C:\Program Files\Tailscale\tailscale.exe` or at `/Applications/Tailscale.app/Contents/MacOS/Tailscale`.

The allow-list lives in `data/workspace.json` and is read when the API starts:

```json
{"studioConfig": {"cluster": {"allowTags": ["tag:trainer"], "allowUsers": ["friend@example.com"]}}}
```

Both keys are optional (`allowTags` defaults to `["tag:trainer"]`, `allowUsers` to nobody). A bad value stops the API at start
with a sentence saying which key to fix.

### Host names

Every request and websocket must carry a `Host` header naming this node; anything else gets `421` (with the CORS header for a
configured `--allow-origin`, so a page can show the message). This stops DNS rebinding: a web page that rebinds its own name to
`127.0.0.1` could otherwise call a loopback node and forge `Tailscale-User-Login`. The header must be plainly `name`, `ipv4` or
`[ipv6]` with an optional `:port`; userinfo or text after the brackets is refused. Allowed, with any port: `localhost`, any IP
address (`127.0.0.1`, `[::1]`, the tailnet address other nodes may use), this machine's name (full and short), the `--bind`
address, each `--allow-host`, and the node's Tailscale names: its MagicDNS name, that name's first label, its host name and its
certificate domains, from `tailscale status --json`. They are read at start and again by a background thread (every minute while
none is known, every ten minutes after; a failed read keeps the last good names), never while a request waits. If none is found
the API logs a warning: a node behind `tailscale serve` or Funnel then refuses those requests until tailscale answers or you
pass `--allow-host`. Other nodes address this one by the name in their `studioConfig.nodes` entry, so list it under its machine
name, Tailscale name or IP; any other name needs `--allow-host`.

`--allow-host` takes a name, or a pattern where one `*` stands for exactly one label (`node.*.ts.net`; at least two literal labels
are required). **Put a wildcard only under a domain you control**: `*.co.uk` or `*.duckdns.org` would let any stranger's
name through and re-open DNS rebinding. A node without the `tailscale` command, such as the API container in `compose.yaml`, needs it: compose passes
`--allow-host ${TS_HOSTNAME:-groundwork-dev}` and `--allow-host ${TS_HOSTNAME:-groundwork-dev}.*.ts.net`, which covers the
`groundwork-dev.<tailnet>.ts.net` name `tailscale serve` forwards. Add others through `API_ARGS`.

### Funnel and a page on another origin (optional)

Everything in this section is off until you configure it. With nothing set, the API behaves as described above.

`tailscale funnel` publishes the node to the public internet. Its requests reach the API from the loopback address like
`tailscale serve`'s, but with no `Tailscale-User-Login` header; Tailscale adds `Tailscale-Funnel-Request: ?1` to them (and strips
any copy a client sent). **Never run Funnel without a token.** The API refuses a request carrying that header unless it sends the
token, even on an open node, and with no token configured every Funnel request gets `401`. Funnel's TCP-forward mode is the
exception: it carries no HTTP headers at all, so the API cannot tell such a request from a local one and it is unprotected whenever
the node is open (a loopback bind with no token). Use the HTTPS mode (`tailscale funnel --bg 8787`), and give the node a token.

To let a page hosted elsewhere (the Vercel site) use the node this way:

1. Put a shared token in the workspace config, `data/workspace.json`, next to the allow-list (read when the API starts). It must
   be at least 32 printable ASCII characters; make it random, for example the output of `openssl rand -hex 32`:

   ```json
   {"studioConfig": {"cluster": {"funnelToken": "<the token>"}}}
   ```

   The file stays on the node; keep the token out of the repository. A shorter, empty or non-ASCII value stops the API at start
   with a sentence naming the setting (never the value), and no token means no Funnel access.
2. Start the API with the page's origin (scheme, host, optional port; no path, no `*`; a default port such as `:443` is dropped):

   ```bash
   .venv/bin/python -m studio_api --port 8787 --allow-origin https://trainer.example.vercel.app
   # or: STUDIO_API_ALLOW_ORIGINS=https://trainer.example.vercel.app,https://other.example.com
   ```

3. Publish the port: `tailscale funnel --bg 8787` (`tailscale funnel status` shows the public name).

Then:

- A Funnel request is let in only if it sends `X-Groundwork-Token: <the token>`, and is recorded as the user `funnel`; any
  `Tailscale-User-Login` on it is ignored. Without the token, or with a wrong one, the answer is `401`. A loopback request that
  has neither header (not Funnel, not `serve`) is let in the same way, by the token. The token is compared in constant time and is
  never logged or echoed. Only loopback requests can use it; a node calling directly is still identified by `tailscale whois`.
- **A node with a token is no longer open.** The loopback bind normally lets everything on the machine in without a header; with a
  token set a loopback request needs the header or the token instead. Local use with `curl` then needs
  `-H "X-Groundwork-Token: ..."`. Leave the token out of the config of a node you also use with `pnpm dev`.
- A browser on an `--allow-origin` origin gets the CORS headers on its preflight (`OPTIONS`, which carries no token) and on
  every response that was let in. A response the API refused with `401` carries none, and an origin that is not listed gets
  none at all. Cookies are not allowed cross-origin; the page sends the token header. On a node that is open and has no token,
  `http://localhost:*` pages keep their CORS access too, as before. With a token set they get none, so a page on another local
  port cannot forge `Tailscale-User-Login` and read the answer.
- The web app sends the token as `X-Groundwork-Token` on every request, and reads the live `/events` stream with `fetch` so the
  header can go on it (an `EventSource` cannot send headers, and the token never goes in a URL). The page takes the token from
  the browser's `localStorage` key `groundwork-token`; set it once in the page's console on the Funnel site:
  `localStorage.setItem("groundwork-token", "<the token>")`, then reload the page (the live stream is chosen when the page loads).
  If the node answers `401` to the stored token, the stream stops, a notice says so, and it does not retry; fix the token and reload. It stays in that browser only and is never baked into the build.
- **No Host check.** The API does not compare the `Host` header with a list of names, so a DNS-rebinding page can reach a
  loopback node under its own hostname and the browser then treats the API as that page's own origin. **A token does not help
  here**: a loopback request that carries `Tailscale-User-Login` is still trusted as that user even on a node with a token, and a
  rebinding page can send that header itself, same-origin and with no CORS involved. Only a Host check would close this. Until
  there is one, do not browse untrusted sites on a machine running the node (the same holds for `--bind 127.0.0.1` without a token).
- Anyone who holds the token can use the node, as anyone on the tailnet can with `tailscale serve`: keep it out of pages and
  logs you do not control, and change it in the config (and restart) to revoke it. The API does not limit how often a caller may
  try tokens; a long random token is the protection.

### Binding

`--bind` decides who can reach the port at all:

- Browsers only: leave it at `127.0.0.1` and use `tailscale serve` (below).
- Other nodes call it: `--bind 100.64.0.7` (this machine's address from `tailscale ip -4`) so nodes reach `http://<node>:8787`,
  but then `tailscale serve` cannot reach a loopback listener. If the same machine also serves browsers, bind `0.0.0.0` instead:
  `serve` connects over the loopback address and nodes over the tailnet, and a request from anywhere else (the LAN) is refused
  by the whois check. Either way, the ACL decides who gets as far as the port (`docs/cluster.md`).

## `tailscale serve`

`tailscale serve` publishes a local port to the tailnet over HTTPS, with a certificate for the machine's name, and tells the
API who is calling. With the API running on this machine:

```bash
tailscale serve --bg 8787
tailscale serve status
```

The UI is then at `https://<machine>.<tailnet>.ts.net/` (the name is on the **Machines** page), from any browser on the tailnet.
`tailscale serve reset` takes it down. HTTPS must be enabled for the tailnet once (**DNS** page, **HTTPS Certificates**); the
first `serve` command offers the link.

Browsers use that `https://` URL. Nodes list each other by tailnet name, which reaches the API directly on port 8787 (`docs/cluster.md`).

### ACL

`tailscale serve` listens on port 443 of the tailnet name, so people reach a node on `tag:trainer:443`; other nodes call the API's
own port, `tag:trainer:8787`. The rules in `docs/cluster.md` have both. The user a browser request is attributed to is the
Tailscale login of the person who made it, so the ACL decides who can reach a node at all.

## As a service on macOS (launchd)

A launchd agent starts the API when you log in and restarts it if it stops. It runs in your session, which is what Metal
(MPS) needs. Save this as `~/Library/LaunchAgents/com.groundwork.api.plist`, replacing `/Users/you/TRAINER` with the checkout:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.groundwork.api</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/you/TRAINER/.venv/bin/python</string>
    <string>-m</string><string>studio_api</string>
    <string>--port</string><string>8787</string>
  </array>
  <key>WorkingDirectory</key><string>/Users/you/TRAINER</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>/Users/you/Library/Logs/groundwork-api.log</string>
  <key>StandardErrorPath</key><string>/Users/you/Library/Logs/groundwork-api.log</string>
</dict>
</plist>
```

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.groundwork.api.plist   # start now and at every login
launchctl kickstart -k gui/$(id -u)/com.groundwork.api                              # restart (after a git pull)
launchctl bootout gui/$(id -u)/com.groundwork.api                                   # stop and unload
tail -f ~/Library/Logs/groundwork-api.log
```

Then run `tailscale serve --bg 8787` once; Tailscale remembers it across restarts. The machine must not sleep while it is
a node: **System Settings**, **Battery** (or **Energy**), **Prevent automatic sleeping when the display is off**, or
`sudo pmset -a sleep 0` on a desktop.

## As a scheduled task on Windows

A scheduled task that runs at logon, with no time limit and restart on failure. In PowerShell, replacing `C:\TRAINER` with
the checkout (`pythonw.exe` is the windowless Python, so no console stays open):

```powershell
$action = New-ScheduledTaskAction -Execute "C:\TRAINER\.venv\Scripts\pythonw.exe" `
    -Argument "-m studio_api --port 8787" -WorkingDirectory "C:\TRAINER"
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName "Groundwork API" -Action $action -Trigger $trigger -Settings $settings
Start-ScheduledTask -TaskName "Groundwork API"
```

`-ExecutionTimeLimit` of zero matters: the default stops a task after 72 hours. `Stop-ScheduledTask -TaskName "Groundwork API"`
stops it, `Unregister-ScheduledTask -TaskName "Groundwork API" -Confirm:$false` removes it. `pythonw.exe` has no console, so
to see errors run the same command once in a terminal with `python.exe`. Run `tailscale serve --bg 8787` once (Tailscale
keeps it across restarts). Set **Power & sleep** to never sleep on this machine, and
allow `python.exe` through Windows Firewall only if you bind to a non-loopback address (nodes calling it directly); behind `tailscale serve` alone you do not.

## Checking it works

```bash
curl http://127.0.0.1:8787/jobs                 # [] : the API answers on the loopback
curl -N http://127.0.0.1:8787/events            # ": connected", then one "data: {...}" line per event
tailscale serve status                          # the HTTPS name that maps to it
```

From another machine on the tailnet, open `https://<machine>.<tailnet>.ts.net/`: the status strip at the top of the page
should say **Connected**.

## Frontend work without a workspace

`python -m studio_api --fake --port 8787` serves empty in-memory data, so the web UI can be built without a real workspace
(see `web/README.md` for running it beside the Next.js dev server).
