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
| `--fake` | Serve `FakeServices` (empty, in-memory data) instead of the workspace, for frontend work. Needs no workspace and is always open. |
| `--dump-openapi FILE` | Write the OpenAPI schema to `FILE` and exit. This is how `web/openapi.json` is made (see `web/README.md`). |

Install the requirements first (`python -m pip install -r requirements.txt`, as in the README); the API needs FastAPI,
uvicorn and the training engine. The web UI is served from `web/out`: build it once with `python tasks.py web:build` (Node and
pnpm needed; the API starts without it and then serves only the routes).

### Who is allowed in

The tailnet is the perimeter, and the API trusts the `Tailscale-User-Login` header that `tailscale serve` adds to each request.

- Bound to `127.0.0.1` (or `localhost`, `::1`) or run with `--fake`, the server is open: any process on this machine can use it,
  with or without the header. That is what you want behind `tailscale serve`, which connects from the loopback address.
- Bound to anything else (`--bind 100.64.0.7`, `--bind 0.0.0.0`), a request without `Tailscale-User-Login` gets `401`. Nothing
  on the tailnet sends that header unless `tailscale serve` (or another proxy you trust) is in front, so do not expose the
  port directly: leave `--bind` at its default and put `tailscale serve` in front.

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

Use that `https://` URL, with no port, when listing the node in another machine's `nodes` config (`docs/cluster.md`).

### ACL

Because `tailscale serve` listens on port 443 of the tailnet name, the ACL rules in `docs/cluster.md` must allow
`tag:trainer:443`, not `8787`: the API's own port is only open on the machine's loopback. Change the `dst` entries there to
`tag:trainer:443`. The user a request is attributed to is the Tailscale login of the person who made it, so the ACL decides
who can reach a node at all.

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
allow `python.exe` through Windows Firewall only if you ever bind to a non-loopback address (behind `tailscale serve` you do not).

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
