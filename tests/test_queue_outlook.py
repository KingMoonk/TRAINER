"""The one place that says when a queued job starts: services.queue_outlook, the machine strip lines, the idle toggle."""
import unittest
from datetime import datetime

from studio_ia.services import FakeServices, Job, QueueState
from studio_queue import PASSIVE, Gate

NOON = datetime(2026, 1, 1, 12)


def outlook(asked=None, now=NOON, **state):
    """queue_outlook(asked) over a queue built from QueueState fields."""
    fake = FakeServices()
    fake.set_queue(QueueState(**state))
    return fake.queue_outlook(asked, now=now)


class Outlook(unittest.TestCase):
    def test_idle_queue_starts_now(self):
        self.assertEqual(outlook(), "Starts now")

    def test_paused_by_you_beats_everything_but_held(self):
        waiting = Job("w", "quicktest", "w")
        self.assertEqual(outlook(waiting, paused=True, up_next=[waiting], gate=Gate("on battery")), "Paused by you")
        held = Job("h", "quicktest", "h", status="held")
        self.assertEqual(outlook(held, paused=True, up_next=[held]), "Held; release it to run")

    def test_waiting_on_the_gate(self):
        self.assertEqual(outlook(gate=Gate("on battery")), "Waiting: on battery")
        self.assertEqual(outlook(gate=Gate("computer in use")), "Waiting: computer in use")
        self.assertEqual(outlook(gate=Gate(PASSIVE)), "Waiting: passive training is running")

    def test_outside_hours_names_the_next_start_and_the_plan(self):
        gate = Gate("outside hours", datetime(2026, 1, 1, 22))
        self.assertEqual(outlook(gate=gate, plan={"hours": "nights"}), "Starts tonight at 22:00 (plan: nights only)")
        gate = Gate("outside hours", datetime(2026, 1, 2, 9))
        self.assertEqual(outlook(gate=gate, now=datetime(2026, 1, 1, 18), plan={"hours": "set", "start_hour": 9, "end_hour": 17}),
                         "Starts tomorrow at 09:00 (plan: 09:00 to 17:00)")
        self.assertEqual(outlook(gate=Gate("outside hours")), "Waiting: outside hours")  # no resume time known

    def test_starts_after_the_running_job_with_the_time_left(self):
        running = Job("r", "pretrain", "Version 3")
        self.assertEqual(outlook(running, job=running, up_next=[]), "Running now")
        self.assertEqual(outlook(None, job=running, eta_s=1500.), "Starts after Version 3 (about 25 min)")
        self.assertEqual(outlook(None, job=running), "Starts after Version 3")  # no estimate yet
        queued = [Job("a", "quicktest", "First", eta_s=600.), Job("b", "quicktest", "Second", eta_s=120.)]
        self.assertEqual(outlook(queued[1], job=running, eta_s=1200., up_next=queued), "Starts after First (about 30 min)")
        self.assertEqual(outlook(queued[0], job=None, up_next=queued), "Starts now")

    def test_held_jobs_are_skipped_in_the_line_and_named_when_asked(self):
        held, nxt = Job("h", "quicktest", "Held", status="held"), Job("n", "quicktest", "Next", eta_s=60.)
        self.assertEqual(outlook(nxt, up_next=[held, nxt]), "Starts now")
        self.assertEqual(outlook(held, up_next=[held, nxt]), "Held; release it to run")

    def test_outside_hours_with_a_job_ahead(self):
        running = Job("r", "pretrain", "Version 3")
        gate = Gate("outside hours", datetime(2026, 1, 1, 22))
        self.assertEqual(outlook(None, job=running, gate=gate, plan={"hours": "nights"}),
                         "Starts after Version 3; the queue opens tonight at 22:00 (plan: nights only)")


if __name__ == "__main__":
    unittest.main()
