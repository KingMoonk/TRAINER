"""The one "what should I do next" answer. Pure: no Tk, no services; the sidebar's Next row and the pages' empty states all read it."""
from __future__ import annotations

from typing import NamedTuple


class NextStep(NamedTuple):
    route: str   # the page that does it
    title: str   # "Add some text"
    line: str    # one sentence of why


def _held(job):
    return getattr(job, "status", "waiting") == "held"


def next_step(has_text, versions, queue, outlook="") -> NextStep:
    """First state that applies, in this order:
    no text; no version; the newest version is an unsaved draft; a job running (or paused mid-run); then, while nothing has
    trained: nothing waiting, the queue paused, only held jobs, or jobs waiting to start (`outlook` is the queue's own
    sentence for that); finally a trained version to talk to.
    `versions` need .number, .status and .best_checkpoint; `queue` is a QueueState (job, paused, up_next, gate)."""
    if not has_text:
        return NextStep("your_data", "Add some text", "Training reads plain text first. Add some in Your data, or download the recommended collection.")
    if not versions:
        return NextStep("builder", "Design your first model", "Pick a shape in Build your model, then save it as version 1.")
    latest = versions[-1]
    trained = [v for v in versions if v.best_checkpoint]
    if latest.status == "draft":
        return NextStep("builder", "Save it", f"Version {latest.number} is a draft. Save it so it can be trained.")
    if queue.job is not None:
        name = queue.job.label or queue.job.kind
        if queue.paused:
            return NextStep("queue", "Training is paused", f"{name} is paused. Resume it in Queue.")
        return NextStep("queue", "Training now", f"{name} is running. Watch it, or pause it, in Queue.")
    waiting = list(queue.up_next)
    if not trained:
        if not waiting:
            return NextStep("builder", "Queue the training", f"Version {latest.number} has not trained yet. Queue it from Build your model.")
        if queue.paused:
            return NextStep("queue", "Resume the queue", "You paused the queue, so nothing will start. Resume it in Queue.")
        if all(_held(j) for j in waiting):
            return NextStep("queue", "Release a held job", "Everything in the queue is held. Release a job in Queue to run it.")
        return NextStep("queue", "Waiting to start", (outlook or "The queue will start as soon as the plan allows") + ".")
    best = trained[-1]
    return NextStep("chat", "Talk to it", f"Version {best.number} has trained. Ask it something in Chat.")


def empty_line(services) -> str:
    """The sentence a page with nothing to show puts under its "No trained version yet": what to do about it, from the same
    next_step() the sidebar uses. "" when it cannot be known without a long read, or when the answer is to use the page itself."""
    step = services.next_step_if_ready()
    return "" if step is None or step.route == "chat" else step.line
