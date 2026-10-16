"""Fakes for the scheduler tests: a clock you move by hand, and a network of fake peers behind the `client_for(address, **options)`
factory that studio_peer.PeerClient fills in production. No sockets, no sleeping."""
from studio_peer import PeerError


class Clock:
    def __init__(self, now=1000.):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class Net:
    """Addresses mapped to handlers `handler(method, path, body) -> json`. A handler that raises PeerError answers with that status;
    an address in `down` is unreachable (status None), like a laptop whose Wi-Fi dropped. Every call is logged."""

    def __init__(self):
        self.handlers, self.down, self.calls = {}, set(), []

    def client_for(self, address, **options):
        return Client(self, address, options)

    def call(self, address, method, path, body, options):
        self.calls.append((address, method, path.lstrip("/"), body, options))
        if address in self.down or address not in self.handlers:
            raise PeerError(f"{address} did not answer {method} /{path}: unreachable")
        return self.handlers[address](method, path.lstrip("/"), body)

    def to(self, address):
        return [(m, p, b) for a, m, p, b, _ in self.calls if a == address]


class Client:
    def __init__(self, net, address, options):
        self.net, self.address, self.options = net, address, options

    def get_json(self, path, params=None, timeout=None):
        return self.net.call(self.address, "GET", path, None, {**self.options, "timeout": timeout})

    def post_json(self, path, body=None, params=None, timeout=None):
        return self.net.call(self.address, "POST", path, body, {**self.options, "timeout": timeout})


def refuse(status, detail):
    return PeerError(f"answered {status}: {detail}", status=status, detail=detail)
