"""Snapshots, commands and the render scheduler (#82), and the machine strip drawn from snapshots only (no display)."""
import gc
import os
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

import studio_ui_guard as guard
from studio_ia import threads
from studio_ia.services import FakeServices, Job, QueueState, Services
from studio_ia.shell import Studio, strip_lines
from studio_ia.snapshots import Render, Snapshots, Topic, Versions, versions_snapshot
from studio_ui_guard import OnUIThread


class Recorder(FakeServices):
    """FakeServices with no window: emit() delivers at once, on whichever thread emits, and is recorded."""

    def __init__(self):
        super().__init__()
        self.events, self.lock = [], threading.Lock()
        self.subscribe(self._record)

    def _record(self, kind, payload):
        with self.lock:
            self.events.append((kind, payload))

    def kinds(self, kind):
        with self.lock:
            return [payload for k, payload in self.events if k == kind]


class Bound(unittest.TestCase):
    """This test thread stands in for the Tk thread, with the tripwire raising."""

    def setUp(self):
        patch = mock.patch.dict(os.environ, {guard.STRICT: "1"})
        patch.start()
        self.addCleanup(patch.stop)
        guard.bind()
        self.addCleanup(guard.unbind)
        self.svc = Recorder()
        self.snaps = self.svc.snapshots
        self.addCleanup(self.snaps.stop)

    def settle(self):
        self.assertTrue(self.snaps.wait(10))


class Store(Bound):
    def test_nothing_is_there_before_the_first_publish(self):
        self.assertIsNone(self.snaps.get("queue"))
        self.assertEqual(self.snaps.get("queue", "none yet"), "none yet")
        self.assertIsNone(self.snaps.snapshot("queue"))

    def test_each_publish_is_the_next_version_of_its_topic(self):
        self.snaps.publish("queue", "a")
        self.snaps.publish("queue", "b")
        self.snaps.publish("versions", "c")
        self.assertEqual((self.snaps.snapshot("queue").version, self.snaps.get("queue")), (2, "b"))
        self.assertEqual(self.snaps.snapshot("versions").version, 1)
        self.assertEqual(self.svc.kinds("snapshot"), ["queue", "queue", "versions"])

    def test_computing_a_snapshot_on_the_tk_thread_raises(self):
        with self.assertRaisesRegex(OnUIThread, "computing the queue snapshot ran on the UI thread"):
            self.snaps.compute("queue")
        self.assertIsNone(self.snaps.get("queue"))

    def test_refresh_computes_on_a_worker_and_announces_it(self):
        self.snaps.refresh("queue", "versions")  # from the Tk thread: it only asks
        self.settle()
        self.assertIsInstance(self.snaps.get("queue"), QueueState)
        self.assertEqual(self.snaps.get("versions"), Versions())
        self.assertCountEqual(self.svc.kinds("snapshot"), ["queue", "versions"])

    def test_the_queue_snapshot_does_not_change_under_its_reader(self):
        self.snaps.refresh("queue")
        self.settle()
        self.svc.enqueue(Job("1", "pretrain"))  # FakeServices changes its QueueState in place
        self.assertEqual(self.snaps.get("queue").up_next, [])
        self.settle()  # the "queue" event asked for a new one only if start() subscribed: it did not
        self.assertEqual(self.snaps.snapshot("queue").version, 1)

    def test_versions_snapshot_follows_the_best_version_then_the_newest(self):
        self.svc.save_version({}, "first")
        snap = versions_snapshot(self.svc)
        self.assertEqual((len(snap.items), snap.current.number, snap.best, snap.chat.number), (1, 1, None, 1))
        self.svc._versions[0].best_checkpoint = "best.npz"
        self.svc.save_version({}, "second")
        snap = versions_snapshot(self.svc)
        self.assertEqual((snap.current.number, snap.best.number, snap.chat.number), (2, 1, 1))
        self.assertEqual(versions_snapshot(Services()), Versions())  # no versions behind it

    def test_a_failing_producer_keeps_the_last_snapshot(self):
        calls = []

        def make(_svc):
            calls.append(1)
            if len(calls) > 1:
                raise ValueError("disk went away")
            return "first"
        snaps = Snapshots(self.svc, {"t": Topic(make, ())})
        with self.assertLogs("studio_ia.snapshots", "ERROR"):
            snaps.refresh("t")
            snaps.wait()
            snaps.refresh("t")
            snaps.wait()
        self.assertEqual((snaps.get("t"), snaps.snapshot("t").version, len(calls)), ("first", 1, 2))


class Refresh(Bound):
    def test_events_ask_only_the_topics_that_follow_them(self):
        made = {"queue": 0, "versions": 0}

        def maker(name):
            def make(_svc):
                made[name] += 1
                return made[name]
            return make
        snaps = Snapshots(self.svc, {"queue": Topic(maker("queue"), ("queue", "train")), "versions": Topic(maker("versions"), ("versions",))})
        self.addCleanup(snaps.stop)
        snaps.start()
        snaps.wait()
        self.assertEqual(made, {"queue": 1, "versions": 1})  # start() computes every topic
        self.svc.emit("train", {"step": 3})
        snaps.wait()
        self.assertEqual(made, {"queue": 2, "versions": 1})
        self.svc.emit("snapshot", "queue")  # its own events ask nothing
        self.svc.emit("unrelated")
        snaps.wait()
        self.assertEqual(made, {"queue": 2, "versions": 1})

    def test_training_steps_ask_for_the_queue_at_most_once_a_second(self):
        now, made = [0.0], []
        snaps = Snapshots(self.svc, {"queue": Topic(lambda _s: made.append(1) or len(made), ("queue",), often=("train",))},
                          clock=lambda: now[0])
        self.addCleanup(snaps.stop)
        snaps.start()
        snaps.wait()
        for step in range(10):  # ten steps in half a second
            now[0] = step * .05
            snaps.on_event("train", {"step": step})
            snaps.tick()
            snaps.wait()
        self.assertEqual(len(made), 2)  # start(), then the first step; the rest wait for the second to pass
        now[0] = 1.0
        snaps.tick()
        snaps.wait()
        self.assertEqual(len(made), 3)  # the steps held back are not lost
        snaps.tick()
        snaps.wait()
        self.assertEqual(len(made), 3)
        snaps.on_event("queue", None)  # anything else is asked at once
        snaps.wait()
        self.assertEqual(len(made), 4)
        self.assertEqual(snaps.topics["queue"].every, 1.0)

    def test_requests_while_one_waits_are_merged(self):
        gate, made = threading.Event(), []

        def make(_svc):
            made.append(1)
            gate.wait(10)
            return len(made)
        snaps = Snapshots(self.svc, {"t": Topic(make, ())})
        self.addCleanup(snaps.stop)
        snaps.refresh("t")  # running, held at the gate
        for _ in range(2000):
            if made:
                break
            threading.Event().wait(.005)
        for _ in range(20):
            snaps.refresh("t")  # one more compute, for the state after the running one read it
        gate.set()
        snaps.wait()
        self.assertEqual(len(made), 2)
        self.assertEqual(snaps.get("t"), 2)

    def test_a_slow_topic_does_not_hold_up_another(self):
        gate = threading.Event()
        snaps = Snapshots(self.svc, {"slow": Topic(lambda _s: gate.wait(10), ()), "fast": Topic(lambda _s: "here", ())})
        self.addCleanup(snaps.stop)
        snaps.refresh("slow", "fast")
        for _ in range(1000):
            if snaps.get("fast"):
                break
            threading.Event().wait(.005)
        self.assertEqual((snaps.get("fast"), snaps.get("slow")), ("here", None))
        gate.set()
        snaps.wait()
        self.assertTrue(snaps.get("slow"))

    def test_after_stop_nothing_is_computed(self):
        self.snaps.start()
        self.settle()
        version = self.snaps.snapshot("queue").version
        self.snaps.stop()
        self.svc.emit("queue")
        self.snaps.refresh("queue")
        self.settle()
        self.assertEqual(self.snaps.snapshot("queue").version, version)
        self.snaps.start()  # the next app on the same services
        self.settle()
        self.assertEqual(self.snaps.snapshot("queue").version, version + 1)


class Commands(Bound):
    def test_a_command_runs_on_a_worker_then_its_topics_and_an_event_follow(self):
        ran_on = []

        def pause():
            ran_on.append(guard.on_ui_thread())
            self.svc.pause()
        self.snaps.send(pause, topics=("queue",), name="pause")  # returns at once
        self.settle()
        self.assertEqual(ran_on, [False])
        self.assertTrue(self.snaps.get("queue").paused)
        self.assertEqual(self.svc.kinds("command"), [{"name": "pause", "ok": True, "error": None}])
        kinds = [k for k, _ in self.svc.events if k in ("snapshot", "command")]
        self.assertEqual(kinds, ["snapshot", "command"])  # the snapshot is in by the time the UI hears the command is done

    def test_a_failed_command_says_so_and_its_topics_are_still_read_again(self):
        def broken(x):
            raise ValueError(x)
        with self.assertLogs("studio_ia.snapshots", "ERROR"):
            self.snaps.send(broken, "no such job", topics=("queue",), name="remove")
            self.settle()
        (done,) = self.svc.kinds("command")
        self.assertEqual((done["name"], done["ok"], str(done["error"])), ("remove", False, "no such job"))
        self.assertIsNotNone(self.snaps.get("queue"))


class Collector(unittest.TestCase):
    def test_a_worker_holds_the_collector_until_the_tk_thread_settles_it(self):
        threads.attach()
        self.addCleanup(threads.detach)
        svc = FakeServices()
        gate, running = threading.Event(), threading.Event()
        snaps = Snapshots(svc, {"t": Topic(lambda _s: (running.set(), gate.wait(10)), ())})
        snaps.refresh("t")
        self.assertTrue(running.wait(10))
        self.assertFalse(gc.isenabled())  # a collection on the worker could free Tk objects off the Tk thread
        gate.set()
        snaps.wait()
        for _ in range(1000):  # the thread is done only once its frames are gone
            threads.settle()
            if gc.isenabled():
                break
            threading.Event().wait(.005)
        self.assertTrue(gc.isenabled())


class RenderScheduler(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.render = Render(clock=lambda: self.now)
        self.calls = []
        self.render.subscribe(("queue", "versions"), lambda changed: self.calls.append(("strip", changed)))
        self.render.subscribe(("data",), lambda changed: self.calls.append(("data", changed)))

    def test_one_redraw_per_subscriber_per_frame(self):
        self.render.mark("queue")
        self.render.mark("queue", "versions")
        self.assertEqual(self.render.frame(), 1)
        self.assertEqual(self.calls, [("strip", {"queue", "versions"})])
        self.now += 1
        self.assertEqual(self.render.frame(), 0)  # nothing changed since

    def test_frames_are_at_most_about_30_a_second(self):
        self.render.mark("queue")
        self.render.frame()
        self.now += .01
        self.render.mark("data")
        self.assertEqual(self.render.frame(), 0)  # too soon: kept for the next frame
        self.now += .03
        self.assertEqual(self.render.frame(), 1)
        self.assertEqual(self.calls[-1], ("data", {"data"}))

    def test_a_failing_redraw_does_not_stop_the_others(self):
        self.render.subscribe(("queue",), lambda changed: 1 / 0)
        self.render.mark("queue", "data")
        with self.assertLogs("studio_ia.snapshots", "ERROR"):
            self.assertEqual(self.render.frame(), 3)
        self.assertEqual([who for who, _ in self.calls], ["strip", "data"])

    def test_unsubscribed_is_not_called(self):
        stop = self.render.subscribe(("queue",), lambda changed: self.calls.append(("gone", changed)))
        stop()
        self.render.mark("queue")
        self.render.frame()
        self.assertNotIn("gone", [who for who, _ in self.calls])


class Label:
    def __init__(self):
        self.options = {"text": ""}

    def configure(self, **options):
        self.options.update(options)

    def cget(self, name):
        return self.options[name]


class Strip(Bound):
    """The machine strip reads snapshots only: its services allow nothing else."""

    def shell(self):
        services = mock.NonCallableMock(spec=["snapshots"])  # any other service call is an AttributeError
        services.snapshots = self.snaps
        return SimpleNamespace(services=services, strip_now=Label(), strip_next=Label(), strip_chat=Label(),
                               strip_bar=mock.Mock(), _strip_resume=False)

    def test_before_the_first_queue_snapshot_it_waits(self):
        shell = self.shell()
        Studio.refresh_strip(shell)
        self.assertEqual(shell.strip_now.cget("text"), "")
        shell.strip_bar.set.assert_not_called()

    def test_it_draws_the_queue_and_versions_snapshots(self):
        shell = self.shell()
        self.snaps.publish("queue", QueueState(job=Job("1", "pretrain", "Version 5 catch-up"), step=50, total=200, phase="steady",
                                               up_next=[Job("2", "quicktest")]))
        Studio.refresh_strip(shell)
        self.assertEqual(shell.strip_now.cget("text"), "Version 5 catch-up · steady")
        self.assertEqual(shell.strip_next.cget("text"), "1 queued")
        self.assertEqual(shell.strip_chat.cget("text"), "Finding the version Chat uses…")
        shell.strip_bar.set.assert_called_with(.25)
        self.snaps.publish("versions", Versions(current=SimpleNamespace(number=3)))
        Studio.refresh_strip(shell)
        self.assertEqual(shell.strip_chat.cget("text"), "Chat uses version 3 · best so far")

    def test_resume_is_a_command(self):
        shell = self.shell()
        shell.services = self.svc
        self.svc.pause()
        self.snaps.refresh("queue")
        self.settle()
        Studio.refresh_strip(shell)
        self.assertTrue(shell._strip_resume)
        Studio._strip_chat_click(shell)  # the click returns at once; the queue resumes on the command worker
        self.settle()
        self.assertFalse(self.snaps.get("queue").paused)
        Studio.refresh_strip(shell)
        self.assertEqual(shell.strip_now.cget("text"), "Nothing is training")

    def test_strip_lines_while_the_chat_version_is_unknown(self):
        self.assertEqual(strip_lines(QueueState(), None, known=False)[2], "Finding the version Chat uses…")
        self.assertEqual(strip_lines(QueueState(paused=True), None, known=False)[2], "Resume ›")


class ShellEvents(unittest.TestCase):
    def test_snapshot_events_mark_the_render_and_never_reach_the_page(self):
        page = mock.Mock()
        shell = SimpleNamespace(render=Render(), route="queue", pages={"queue": page}, refresh_next=mock.Mock())
        Studio.on_event(shell, "snapshot", "queue")
        Studio.on_event(shell, "command", {"name": "resume", "ok": True, "error": None})
        page.on_event.assert_not_called()  # the Queue page rebuilds on any event that is not "train"
        self.assertEqual(shell.render._dirty, {"queue"})
        Studio.on_event(shell, "queue", None)
        page.on_event.assert_called_once_with("queue", None)
        shell.refresh_next.assert_called_once()


if __name__ == "__main__":
    unittest.main()
