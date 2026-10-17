"""Shared pieces of the DiLoCo tests: a tiny corpus and model, a coordinator behind the real routes, and a peer client that calls them."""
import json
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np

import studio_diloco
from studio_api import app as api_app
from studio_api.app import create_app
from studio_api.hosts import ANY
from studio_ia.services import FakeServices
from studio_peer import PeerError

EXAMPLES = ([{"text": f"Document {n}: " + "the quick brown fox jumps over the lazy dog. " * (n % 5 + 2)} for n in range(40)]
            + [{"messages": [{"role": "user", "content": f"What is {n} plus {n}?"}, {"role": "assistant", "content": f"{n + n}."}]} for n in range(20)])
CONFIG = dict(device="cpu", width=16, layers=1, heads=2, context=32, expansion=2, dropout=.1, batch_size=2, learning_rate=.01, warmup=2,
              validation=.2, seed=7, sample_every=0, objective="all", name="diloco test")


def settings(**changes):
    return studio_diloco.Settings(**{"inner_steps": 3, "rounds": 4, "outer_lr": 1., "outer_momentum": 0., "shards": 4,
                                     "delta_dtype": "float32", **changes})


def build(directory, run="run1", clock=None, **changes):
    options = {"clock": clock} if clock else {}
    return studio_diloco.Coordinator.build(run, directory, CONFIG, EXAMPLES, settings(**changes), **options)


def app():
    with mock.patch.object(api_app, "WEB_OUT", Path(tempfile.gettempdir()) / "no-such-web-out"):
        return create_app(FakeServices(), open=True, hosts=ANY)


class Peer:
    """studio_peer.PeerClient's methods over a Starlette TestClient: an error status becomes a PeerError, as the real client raises."""

    def __init__(self, client):
        self.client = client

    def _check(self, response, path):
        if response.status_code >= 400:
            detail = response.json().get("detail", "") if response.headers.get("content-type", "").startswith("application/json") else ""
            raise PeerError(f"{response.status_code} {path}: {detail}", status=response.status_code, detail=str(detail))
        return response

    def get_json(self, path, params=None, timeout=None):
        return self._check(self.client.get(path, params=params), path).json()

    def post_json(self, path, body=None, params=None, timeout=None):
        response = self._check(self.client.post(path, json=body if body is not None else {}, params=params), path)
        return response.json() if response.content else None

    def get_bytes(self, path, params=None, timeout=None):
        return self._check(self.client.get(path, params=params), path).content

    def post_bytes(self, path, data, params=None, timeout=None):
        response = self._check(self.client.post(path, content=bytes(data), params=params, headers={"content-type": "application/octet-stream"}), path)
        return response.json() if response.content else None


def constant_delta(coordinator, value=.01):
    return {name: np.full(array.shape, value, dtype=np.float32) for name, array in coordinator.theta.items()}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
