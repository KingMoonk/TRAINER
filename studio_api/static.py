"""Serve web/out, the Next.js static export, including the segment payloads its client router prefetches.

The export writes a page's segment payload as `<route>/__next.<segment>/__PAGE__.txt`, but the client asks for the flat name
`<route>/__next.<segment>.__PAGE__.txt`. A plain static server answers that with 404, so every in-app link falls back to a full
page load. ExportFiles answers the flat name from the nested file.
"""

import os
import posixpath

from starlette.exceptions import HTTPException
from starlette.staticfiles import StaticFiles

PREFIX, SUFFIX = "__next.", ".txt"


def nested(path):
    """The exported file for a flat segment-payload request, or None when `path` is not one:
    `builder/__next.builder.__PAGE__.txt` -> `builder/__next.builder/__PAGE__.txt`."""
    folder, name = posixpath.split(path)
    if not (name.startswith(PREFIX) and name.endswith(SUFFIX)):
        return None
    parts = name[len(PREFIX):-len(SUFFIX)].split(".")
    if len(parts) < 2 or not all(parts):
        return None
    return posixpath.join(folder, PREFIX + parts[0], *parts[1:-1], parts[-1] + SUFFIX)


class ExportFiles(StaticFiles):
    async def get_response(self, path, scope):
        """A miss is a raised 404, or (html=True with a 404.html in the export) that page with status 404: either way, try the
        nested file."""
        other = nested(path.replace(os.sep, "/"))
        try:
            response = await super().get_response(path, scope)
        except HTTPException as exc:
            if exc.status_code != 404 or other is None:
                raise
            return await super().get_response(other, scope)
        if response.status_code == 404 and other is not None:
            return await super().get_response(other, scope)
        return response
