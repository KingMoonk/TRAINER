"""Opt-in timings: `with timed("name"):` or `@timed("name")` prints `name: 123.4 ms [thread]` to stderr when GROUNDWORK_TIMINGS=1.

Off by default, so the wrapped calls cost one environment lookup. `tasks.py profile` turns it on."""
import functools
import os
import sys
import threading
import time

ENV = "GROUNDWORK_TIMINGS"


class timed:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter() if os.environ.get(ENV) == "1" else None
        return self

    def __exit__(self, *_):
        if self.start is not None:
            ms = (time.perf_counter() - self.start) * 1000
            print(f"{self.name}: {ms:.1f} ms [{threading.current_thread().name}]", file=sys.stderr, flush=True)

    def __call__(self, fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with timed(self.name):
                return fn(*args, **kwargs)
        return wrapper
