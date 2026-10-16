"""studio_api.routers.chat over FakeServices: the reply streams on its own response, a closed response stops it, and what the user
teaches or rates reaches the services."""
import asyncio
import json
import threading
import unittest

from fastapi.testclient import TestClient

from studio_api.app import create_app
from studio_ia.services import FakeServices

ASK = {"version": 2, "messages": [{"role": "user", "content": "Hi"}]}


class ChattingServices(FakeServices):
    """chat() reports its reply a piece at a time, as a model does, and records what it was asked."""

    def __init__(self):
        super().__init__()
        self.asked, self.taught, self.rated = [], [], []
        self.fail = None

    def chat(self, version, messages, on_text, stop, ablate=None):
        self.asked.append((version, messages))
        if self.fail:
            raise self.fail
        for text in ("Hel", "Hello", "Hello there"):
            on_text(text)
        return "Hello there"

    def teach_correction(self, version, messages, correction, whole_answer=False):
        self.taught.append((version, messages, correction, whole_answer))

    def rate_pair(self, messages, chosen, rejected):
        self.rated.append((messages, chosen, rejected))
        return {"id": "pref1", "chosen": chosen["reply"], "rejected": rejected["reply"]}


def parse(body):
    """[(kind, payload)] of an SSE body, framed as on GET /events (data: {"kind", "payload"}); comments are skipped."""
    messages = [json.loads(block[6:]) for block in body.split("\n\n") if block.startswith("data: ")]
    return [(message["kind"], message["payload"]) for message in messages]


class ChatApiTest(unittest.TestCase):
    def setUp(self):
        self.services = ChattingServices()
        self.app = create_app(self.services, open=True)
        self.client = self.enterContext(TestClient(self.app))

    def test_the_reply_streams_as_events_and_ends_with_done(self):
        response = self.client.post("/chat/reply", json=ASK)
        self.assertEqual(response.headers["content-type"], "text/event-stream; charset=utf-8")
        self.assertEqual(parse(response.text), [("text", {"text": "Hel"}), ("text", {"text": "Hello"}), ("text", {"text": "Hello there"}),
                                                ("done", {"text": "Hello there"})])
        self.assertEqual(self.services.asked, [(2, [{"role": "user", "content": "Hi"}])])

    def test_a_reply_that_reports_nothing_is_sent_whole(self):
        self.app.state.services = FakeServices()
        self.assertEqual(parse(self.client.post("/chat/reply", json=ASK).text), [("text", {"text": "[reply]"}), ("done", {"text": "[reply]"})])

    def test_a_failure_is_a_sentence_in_the_stream(self):
        self.services.fail = ValueError("Version 2 has not trained yet.")
        self.assertEqual(parse(self.client.post("/chat/reply", json=ASK).text), [("error", {"message": "Version 2 has not trained yet."})])
        self.services.fail = RuntimeError("secret detail")
        events = parse(self.client.post("/chat/reply", json=ASK).text)
        self.assertEqual(events, [("error", {"message": "Something went wrong: RuntimeError. The log has the details."})])

    def test_a_reply_needs_a_message_with_a_known_role(self):
        self.assertEqual(self.client.post("/chat/reply", json={"messages": []}).status_code, 422)
        self.assertEqual(self.client.post("/chat/reply", json={"messages": [{"role": "bot", "content": "x"}]}).status_code, 422)
        self.assertEqual(self.services.asked, [])

    def test_closing_the_response_stops_the_model(self):
        stopped, started = threading.Event(), threading.Event()

        def slow(version, messages, on_text, stop, ablate=None):
            on_text("Hel")
            started.set()
            stopped.set() if stop.wait(5) else None
            return "Hel"
        self.services.chat = slow
        body = json.dumps(ASK).encode()
        scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST", "scheme": "http", "path": "/chat/reply",
                 "raw_path": b"/chat/reply", "query_string": b"", "client": ("127.0.0.1", 1), "server": ("testserver", 80),
                 "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]}

        async def drive():
            first_text, sent = asyncio.Event(), []

            async def receive():
                if not sent:
                    sent.append(1)
                    return {"type": "http.request", "body": body, "more_body": False}
                await first_text.wait()
                return {"type": "http.disconnect"}

            async def send(message):
                if message["type"] == "http.response.body" and b'"kind": "text"' in message.get("body", b""):
                    first_text.set()
            await self.app(scope, receive, send)
        asyncio.run(drive())
        self.assertTrue(started.is_set())
        self.assertTrue(stopped.wait(5), "the model was never told to stop")

    def test_a_correction_is_saved_for_the_teacher_or_as_a_whole_answer(self):
        messages = ASK["messages"]
        self.assertEqual(self.client.post("/chat/corrections", json={"version": 2, "messages": messages, "correction": "Hello!"}).status_code, 204)
        self.client.post("/chat/corrections", json={"messages": messages, "correction": "Hello!", "whole_answer": True})
        self.assertEqual(self.services.taught, [(2, [{"role": "user", "content": "Hi"}], "Hello!", False),
                                                (None, [{"role": "user", "content": "Hi"}], "Hello!", True)])
        self.assertEqual(self.client.post("/chat/corrections", json={"messages": messages, "correction": ""}).status_code, 422)

    def test_a_rated_pair_is_saved_and_the_record_comes_back(self):
        pair = {"messages": ASK["messages"], "chosen": {"version": 2, "reply": "Hello"}, "rejected": {"version": 1, "reply": "Go away"}}
        response = self.client.post("/chat/pairs", json=pair)
        self.assertEqual(response.json(), {"record": {"id": "pref1", "chosen": "Hello", "rejected": "Go away"}})
        self.assertEqual(self.services.rated[0][1:], ({"version": 2, "reply": "Hello"}, {"version": 1, "reply": "Go away"}))

    def test_a_pair_the_store_refuses_is_a_bad_request(self):
        def refuse(messages, chosen, rejected):
            raise ValueError("A preference needs a non-empty chosen answer that differs from the rejected one.")
        self.services.rate_pair = refuse
        pair = {"messages": ASK["messages"], "chosen": {"reply": "same"}, "rejected": {"reply": "same"}}
        self.assertEqual(self.client.post("/chat/pairs", json=pair).status_code, 400)


if __name__ == "__main__":
    unittest.main()
