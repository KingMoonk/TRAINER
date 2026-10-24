"""Receiver lifecycle invariants through the collective API, including allowed late updates."""
import hashlib
import time

import studio_diloco
from studio_peer import SenderEgressBudget
from studio_upload_admission import UploadAdmission

from studio_api.routers.collective import transfer_identity
from tests.test_api_collective import Collective


class UploadAdmissionAPI(Collective):
    def announce(self, coordinator, node="a", number=0):
        generation = coordinator.current.generation
        body = self.delta_body(coordinator)
        fields = dict(node=node, generation=generation, epoch=0, tokens=10, steps=2,
                      total=len(body), sha256=hashlib.sha256(body).hexdigest(), count=1, part_size=len(body))
        route = f"/collective/r1/rounds/{number}/delta/transfers"
        response = self.client.post(route, json=fields)
        self.assertEqual(response.status_code, 200, response.text)
        announcement = response.json()
        params = {**fields, "incarnation": announcement["incarnation"]}
        return route + "/" + announcement["transfer_id"], params, body

    def grant(self, route, params):
        response = self.client.get(route + "/grants", params=params)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["grants"][0]

    def send(self, route, params, grant, body):
        return self.client.post(route + "/parts", params={**params, "grant_id": grant["grant_id"], "part": grant["part"],
                               "size": len(body), "part_sha256": hashlib.sha256(body).hexdigest()}, content=body)

    def test_upload_that_crosses_a_round_deadline_can_still_be_carried_once(self):
        now = [0.]
        c = self.start(clock=lambda: now[0], min_members=2, inner_steps=2, rounds=5, missed_limit=3)
        c.join("a")
        c.join("b")
        route, params, body = self.announce(c, "b")
        c.submit(0, "a", 1, 0, 10, 2, studio_diloco.decode_arrays(body))
        now[0] = 2000.
        c.snapshot()
        self.assertEqual(c.current.number, 1)
        grant = self.grant(route, params)
        response = self.send(route, params, grant, body)
        self.assertEqual(response.status_code, 202, response.text)
        self.assertIn("b", c.late)
        self.assertTrue(self.client.get(route + "/grants", params=params).json()["complete"])
        self.assertEqual(self.send(route, params, grant, body).status_code, 202)
        self.assertEqual(list(c.late), ["b"])
        manager = self.client.app.state.upload_admission
        self.assertEqual((manager.outstanding_bytes, manager.assembly_bytes), (0, 0))

    def test_duplicate_inflight_request_cannot_release_the_original_requests_credit(self):
        c = self.start(inner_steps=2, rounds=3)
        c.join("a")
        route, params, body = self.announce(c)
        grant = self.grant(route, params)
        manager = self.client.app.state.upload_admission
        identity = transfer_identity("r1", 0, 1, "a", params["generation"], 0, 10, 2, 0., None, 0.)
        manager.begin_part(route.split('/')[-1], params["incarnation"], identity, grant["grant_id"], 0, len(body))
        response = self.send(route, params, grant, body)
        self.assertEqual(response.status_code, 409)
        self.assertEqual(manager.outstanding_bytes, len(body))
        self.assertEqual(manager.grants[grant["grant_id"]]["state"], "receiving")
        self.assertEqual(self.client.delete(route, params=params).status_code, 204)
        self.assertGreater(manager.assembly_bytes, 0)
        manager.end_part(grant["grant_id"])
        self.assertEqual((manager.outstanding_bytes, manager.assembly_bytes), (0, 0))

    def test_lifespan_sweep_releases_abandoned_upload_credit_without_another_upload_poll(self):
        now = [0.]
        budget = SenderEgressBudget(4)
        manager = UploadAdmission(clock=lambda: now[0], ttl=1., ingress_budget=budget)
        self.client.app.state.upload_admission = manager
        identity = ('run', 1, 0, 'sender', 1, 0, 1, 1, 0., None, 0.)
        transfer = manager.announce(identity, 4, hashlib.sha256(b'data').hexdigest(), 1, 4)
        manager.grant(transfer['transfer_id'], transfer['incarnation'], identity)
        self.assertEqual(budget.outstanding, 4)
        now[0] = 2.
        until, downloader = time.monotonic() + 3., object()
        while not budget.acquire(downloader, 4):
            self.assertLess(time.monotonic(), until, 'Lifespan did not sweep expired upload credit')
            time.sleep(.01)
        self.assertEqual(manager.outstanding_bytes, 0)
        budget.release(4)

    def test_status_recovers_assembly_when_the_last_request_disappears_before_apply(self):
        c = self.start(inner_steps=2, rounds=3)
        c.join("a")
        route, params, body = self.announce(c)
        grant = self.grant(route, params)
        manager = self.client.app.state.upload_admission
        identity = transfer_identity("r1", 0, 1, "a", params["generation"], 0, 10, 2, 0., None, 0.)
        ident, inc = route.split('/')[-1], params["incarnation"]
        manager.begin_part(ident, inc, identity, grant["grant_id"], 0, len(body))
        manager.finish_part(ident, inc, identity, grant["grant_id"], 0, body, hashlib.sha256(body).hexdigest())
        manager.end_part(grant["grant_id"])
        self.assertEqual(c.history, [])
        for _ in range(2):
            response = self.client.get(route + "/grants", params=params)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertTrue(response.json()["complete"])
        self.assertEqual(len(c.history), 1)
        self.assertEqual((manager.outstanding_bytes, manager.assembly_bytes), (0, 0))
