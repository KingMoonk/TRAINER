"""web/src/contract/fixtures/ are API responses recorded over FakeServices (tests/web_fixtures.py). They must match what the API answers."""

import unittest

from tests import web_fixtures


class WebFixturesFreshTest(unittest.TestCase):
    def test_recorded_fixtures_match_the_api(self):
        problems = web_fixtures.stale()
        self.assertEqual(
            problems, [],
            "The web contract fixtures are stale (a response the API gives has changed). Rewrite them with:\n"
            f"  {web_fixtures.REGENERATE}\nthen run `pnpm test` in web/ (the client must follow the new shape) and commit "
            "web/src/contract/fixtures/.")


if __name__ == "__main__":
    unittest.main()
