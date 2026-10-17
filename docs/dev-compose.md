# Development on the tailnet

The live dev loop (hot reload, one HTTPS URL on the tailnet) always trains on the machine's GPU. There is no CPU-only dev
variant: a Mac runs natively, because Docker cannot give a container Metal, and Docker is for NVIDIA machines.

| Machine                                    | Command                                                                  | `GET /node` reports |
| ------------------------------------------ | ------------------------------------------------------------------------ | ------------------- |
| Apple Silicon Mac                          | `.venv/bin/python tasks.py dev`, plus the host `tailscale serve` (below)  | `mps`               |
| Linux or Windows (WSL2) with an NVIDIA GPU | `docker compose up` (below)                                              | `cuda`              |
| Any machine, UI work without a GPU         | `python -m studio_api --fake` (docs/node.md) and `pnpm dev` in `web/`     | no training         |

A Mac does not run TRAINER in Docker. After starting, check the device: `/api/node` on the tailnet URL, or
`http://127.0.0.1:8787/node` on the machine. The API also prints `device: ...` at startup, and refuses to start when the
machine has a GPU (an Apple Silicon Mac, or `nvidia-smi` listing one) but torch reports `cpu`; `--allow-cpu` runs on the CPU
anyway. The Apple Neural Engine is not used: PyTorch runs on the GPU (MPS), and the ANE would need a Core ML export.

## Apple Silicon: `tasks.py dev`

The API runs natively from `.venv` (device `mps`) next to the Next dev server, both on `127.0.0.1`, and only the host's
`tailscale serve` (the Tailscale app on the Mac) puts them on the tailnet:

```bash
.venv/bin/python tasks.py dev      # API with --reload on 127.0.0.1:8787, `pnpm dev` on 127.0.0.1:3000
# once, in another terminal (the CLI is /Applications/Tailscale.app/Contents/MacOS/Tailscale if `tailscale` is not on PATH):
tailscale serve --bg http://127.0.0.1:3000
tailscale serve --bg --set-path=/api/ http://127.0.0.1:8787
tailscale serve status             # / -> :3000, /api/ -> :8787
curl https://<machine>.<tailnet>.ts.net/api/node    # "device": "mps"
```

`tasks.py dev` prints the two `serve` commands, installs `web/node_modules` when missing, and prints the device once the API
answers. Ctrl+C stops both processes. A saved `.py` restarts the API (open `/events` streams are cut after 2 seconds); a saved
file under `web/` hot-reloads. `tailscale serve` remembers its config: if the machine also serves the built app
(`tailscale serve --bg 8787`, docs/node.md), the commands above replace that `/` mapping; put it back with
`tailscale serve reset` and `tailscale serve --bg 8787`. Ports 8787 and 3000 must be free: stop a running API first (two APIs
on one `data/` and `runs/` is unsafe).

Why the host's `tailscale serve`, and not the compose `tailscale` and `web` containers proxying to `host.docker.internal:8787`:
the API trusts the `Tailscale-User-Login` header of a loopback peer (docs/node.md, "Who is allowed in"). A container reaches
the host only through `host.docker.internal`, so the API would have to listen beyond `127.0.0.1` (the Docker VM's address, or
`0.0.0.0` and with it the LAN), where it is no longer "open" by `open=args.bind in LOCAL` and a caller could type that header.
With the host's `tailscale serve` the API stays on `127.0.0.1` (`tasks.py dev` takes no `--bind`), nothing is published on the
LAN, and the only thing that adds the header is `tailscale serve`, which deletes any value the client sent. No bind, host or
auth logic changed. As for any native node, other local processes can call a loopback API.

## NVIDIA GPU: `docker compose up`

`docker compose up` runs the API and the web dev server from this checkout and puts them on your tailnet as one HTTPS URL,
so any device on the tailnet (a phone, another laptop) can open the app while you edit. The containers bind-mount the
source: a saved `.py` file restarts the API (uvicorn `--reload`), and a saved file under `web/` hot-reloads the page.

| Container   | Runs                                                        | Reached at (tailnet)                         |
| ----------- | ----------------------------------------------------------- | -------------------------------------------- |
| `tailscale` | `tailscaled` and `tailscale serve` (`docker/serve.json`)     | `https://<TS_HOSTNAME>.<tailnet>.ts.net`     |
| `web`       | `pnpm dev` on `web/`                                         | `/` on that URL                              |
| `api`       | `python -m studio_api --reload` on the repository root       | `/api/` on that URL, and `http://<TS_HOSTNAME>:8787` for other nodes |

`web` and `api` share the `tailscale` container's network and listen on `127.0.0.1` only, so nothing is published on the
host and every request arrives through `tailscale serve`, which adds the `Tailscale-User-Login` header. `/api/*` goes to
the API with the `/api` prefix stripped, so the page and the API are one origin (`NEXT_PUBLIC_API_URL=/api`).

## Once

1. In the Tailscale admin console, turn on **MagicDNS** and **HTTPS certificates** (**DNS** page).
2. Make an auth key (**Settings**, **Keys**, **Generate auth key**; reusable, so the node survives `docker compose down -v`).
3. `cp docker/.env.example .env` and set `TS_AUTHKEY`. `TS_HOSTNAME` names the machine (default `groundwork-dev`).

## Every time

```bash
docker compose up
```

Open `https://groundwork-dev.<tailnet>.ts.net`. The first start installs the web packages into a volume (Linux builds,
apart from the host's `web/node_modules`) and builds the API image, so it takes a few minutes; later starts take seconds.
After changing `requirements*.txt`, run `docker compose build api`. After changing `web/package.json`, restart `web`.

`API_ARGS=--fake` in `.env` serves `FakeServices` instead of this checkout's workspace (`data/`, `runs/`).

The API refuses a request whose `Host` is not one of its names (docs/node.md, "Host names"). Its container has no `tailscale`
command, so compose gives it `--allow-host` for `${TS_HOSTNAME}` and `${TS_HOSTNAME}.*.ts.net`, the name `tailscale serve` forwards. If you reach it under another name, add `--allow-host NAME` to `API_ARGS`.

### Requirements

`api` reserves every NVIDIA GPU (`deploy.resources.reservations.devices` in `compose.yaml`) and `docker/api.Dockerfile`
installs the CUDA 12.6 build of torch (`requirements-gpu.txt`). The host needs a recent NVIDIA driver and:

- **Linux:** the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
  then `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`. `docker run --rm --gpus all ubuntu nvidia-smi` should list the card.
- **Windows:** Docker Desktop with the WSL2 backend, and the NVIDIA driver for Windows (no driver inside WSL2). Run the
  compose command from a WSL2 shell.

Without the toolkit the container does not start (`could not select device driver "nvidia"`). Check the device after starting:

```bash
docker compose exec api python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8787/node').read().decode())"   # "device":"cuda"
```

CI has no GPU, so this is checked by hand.

## Limits

- Docker is for NVIDIA machines only. On a Mac use `tasks.py dev`: Docker Desktop there runs a Linux VM with no Metal.
- The container and a native API or Tk app on the same checkout share `data/` and `runs/`. Run one of them at a time.
- `api` and `web` live in the `tailscale` container's network. Compose restarts them when it restarts `tailscale`; if Docker
  restarts `tailscale` on its own (it crashed), run `docker compose restart api web`.
- The tailnet node is the container, not your machine: it shows up as its own machine in the admin console.
