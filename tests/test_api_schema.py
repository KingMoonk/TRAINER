"""web/openapi.json is what the web client's types are generated from. It must match the app's schema."""

import unittest
from pathlib import Path

from studio_api.__main__ import schema_json

OPENAPI = Path(__file__).resolve().parent.parent / "web" / "openapi.json"
REGENERATE = "python -m studio_api --dump-openapi web/openapi.json && (cd web && pnpm gen:api)"


class OpenApiFreshTest(unittest.TestCase):
    def test_web_openapi_json_matches_the_app(self):
        self.assertTrue(OPENAPI.is_file(), f"web/openapi.json is missing. Regenerate it, and the TypeScript types, with:\n  {REGENERATE}")
        committed = OPENAPI.read_text(encoding="utf-8").replace("\r\n", "\n")
        self.assertEqual(
            committed, schema_json(),
            f"web/openapi.json is out of date with the API's routes. Regenerate it, and the TypeScript types, with:\n  {REGENERATE}\n"
            "and commit web/openapi.json and web/src/lib/api-schema.ts.")


if __name__ == "__main__":
    unittest.main()
