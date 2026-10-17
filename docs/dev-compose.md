# Developing on the tailnet with Docker Compose

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

## Limits

- The API container is Linux, so torch is the CPU build: no Metal on a Mac. Train on the host (`python app.py`, or the API run
  natively as in `docs/node.md`) or on a CUDA node; use the container for UI and API work.
- The container and a native API or Tk app on the same checkout share `data/` and `runs/`. Run one of them at a time.
- `api` and `web` live in the `tailscale` container's network. Compose restarts them when it restarts `tailscale`; if Docker
  restarts `tailscale` on its own (it crashed), run `docker compose restart api web`.
- The tailnet node is the container, not your machine: it shows up as its own machine in the admin console.
