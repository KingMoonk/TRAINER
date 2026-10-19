"""studio_api.static: web/out is served with the segment payloads the Next.js client prefetches. No browser."""
import tempfile
import unittest
from pathlib import Path

from starlette.applications import Starlette
from starlette.testclient import TestClient

from studio_api.static import ExportFiles, nested


class Names(unittest.TestCase):
    def test_a_flat_segment_name_maps_to_the_nested_file(self):
        self.assertEqual(nested("builder/__next.builder.__PAGE__.txt"), "builder/__next.builder/__PAGE__.txt")
        self.assertEqual(nested("a/b/__next.a.b.__PAGE__.txt"), "a/b/__next.a/b/__PAGE__.txt")

    def test_other_names_are_left_alone(self):
        for path in ("builder/index.txt", "builder/__next._tree.txt", "builder/__next..txt", "_next/static/x.js"):
            self.assertIsNone(nested(path), path)


class Serving(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        root = Path(self.dir.name)
        (root / "builder" / "__next.builder").mkdir(parents=True)
        (root / "index.html").write_text("home", encoding="utf-8")
        (root / "builder" / "index.html").write_text("builder", encoding="utf-8")
        (root / "builder" / "__next._tree.txt").write_text("tree", encoding="utf-8")
        (root / "builder" / "__next.builder" / "__PAGE__.txt").write_text("page", encoding="utf-8")
        app = Starlette()
        app.mount("/", ExportFiles(directory=root, html=True))
        self.client = TestClient(app)

    def tearDown(self):
        self.dir.cleanup()

    def test_the_prefetched_payload_is_found(self):
        response = self.client.get("/builder/__next.builder.__PAGE__.txt")
        self.assertEqual((response.status_code, response.text), (200, "page"))

    def test_files_that_exist_and_pages_still_serve(self):
        self.assertEqual(self.client.get("/builder/__next._tree.txt").text, "tree")
        self.assertEqual(self.client.get("/builder/").text, "builder")

    def test_a_missing_payload_is_still_404(self):
        self.assertEqual(self.client.get("/chat/__next.chat.__PAGE__.txt").status_code, 404)

    def test_an_export_with_a_404_page_still_finds_the_payload(self):
        (Path(self.dir.name) / "404.html").write_text("not found page", encoding="utf-8")
        response = self.client.get("/builder/__next.builder.__PAGE__.txt")
        self.assertEqual((response.status_code, response.text), (200, "page"))
        self.assertEqual(self.client.get("/chat/__next.chat.__PAGE__.txt").status_code, 404)


if __name__ == "__main__":
    unittest.main()
