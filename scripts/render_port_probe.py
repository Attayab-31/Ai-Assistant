"""Minimal HTTP server so Render free-tier web services detect an open port.

Celery worker and beat do not bind HTTP ports. On Render's free plan, background
workers are not available, so those processes are deployed as web services. This
probe satisfies Render's port scan while Celery runs in the foreground.
"""

from __future__ import annotations

import os
from http.server import BaseHTTPRequestHandler, HTTPServer


class _ProbeHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path in ("/", "/health"):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    server = HTTPServer(("0.0.0.0", port), _ProbeHandler)
    print(f"[render_port_probe] Listening on 0.0.0.0:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
