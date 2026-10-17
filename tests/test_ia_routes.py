"""Every web route has a page, and no web page still renders the placeholder.

The web routes are the {route, href} entries of web/src/lib/routes.ts. The page for a web route is
web/src/app/<href>/page.tsx. A page is a placeholder when its page.tsx renders PagePlaceholder
(web/src/components/page-placeholder.tsx): the test looks for that name, or its module path, in the page source.
"""

import re
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROUTES_TS = ROOT / "web" / "src" / "lib" / "routes.ts"
APP = ROOT / "web" / "src" / "app"
WEB_ROUTE = re.compile(r'\{\s*route:\s*"([^"]+)",\s*href:\s*"([^"]+)"')
PLACEHOLDER = re.compile(r"\bPagePlaceholder\b|page-placeholder")
# The run pages land in #344, #348 and #349; until they do these routes have no page.tsx yet.
PENDING_ROUTES = {"runs", "new_run", "run_view"}


def web_routes(source=ROUTES_TS):
    return WEB_ROUTE.findall(source.read_text(encoding="utf-8"))  # [(route, href)]


def page_file(href, app=APP):
    return app / href.strip("/") / "page.tsx"


def unbuilt(pairs, app=APP):
    return [route for route, href in pairs if not page_file(href, app).is_file()]


def placeholders(pairs, app=APP):
    return [route for route, href in pairs
            if page_file(href, app).is_file() and PLACEHOLDER.search(page_file(href, app).read_text(encoding="utf-8"))]


class RouteCheck(unittest.TestCase):
    def test_the_route_list_is_read(self):
        self.assertTrue(web_routes(), f"no routes found in {ROUTES_TS}")

    def test_every_web_route_has_a_page(self):
        missing = [route for route in unbuilt(web_routes()) if route not in PENDING_ROUTES]
        self.assertEqual(missing, [], f"web routes with no page.tsx under web/src/app: {missing}")

    def test_no_web_page_renders_the_placeholder(self):
        found = placeholders(web_routes())
        self.assertEqual(found, [], f"web pages that still render PagePlaceholder: {found}")

    def test_missing_and_placeholder_pages_are_named(self):
        pairs = [("done", "/done"), ("stub", "/stub"), ("absent", "/absent")]
        with tempfile.TemporaryDirectory() as tmp:
            app = Path(tmp)
            (app / "done").mkdir()
            (app / "done" / "page.tsx").write_text("export default function Page() { return null; }\n", encoding="utf-8")
            (app / "stub").mkdir()
            (app / "stub" / "page.tsx").write_text('import { PagePlaceholder } from "@/components/page-placeholder";\n', encoding="utf-8")
            self.assertEqual(unbuilt(pairs, app), ["absent"])
            self.assertEqual(placeholders(pairs, app), ["stub"])


if __name__ == "__main__":
    unittest.main()
