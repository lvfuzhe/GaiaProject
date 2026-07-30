from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from gaiazero.telemetry import read_events

WEB_ROOT = Path(__file__).with_name("web")
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        metrics_path: str | Path,
        *,
        quiet: bool = False,
    ) -> None:
        self.metrics_path = Path(metrics_path).resolve()
        self.quiet = quiet
        super().__init__(address, DashboardRequestHandler)


class DashboardRequestHandler(BaseHTTPRequestHandler):
    server: DashboardServer

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        request = urlparse(self.path)
        if request.path == "/api/events":
            self._serve_events(parse_qs(request.query))
            return
        if request.path == "/api/health":
            self._serve_health()
            return
        asset = ASSETS.get(request.path)
        if asset is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        filename, content_type = asset
        path = WEB_ROOT / filename
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._send(path.read_bytes(), content_type)

    def _serve_events(self, query: dict[str, list[str]]) -> None:
        try:
            after = max(0, int(query.get("after", ["0"])[0]))
            limit = min(5_000, max(1, int(query.get("limit", ["5000"])[0])))
        except ValueError:
            self._send_json({"error": "after and limit must be integers"}, HTTPStatus.BAD_REQUEST)
            return
        events = read_events(self.server.metrics_path, after=after, limit=limit)
        self._send_json(
            {
                "events": events,
                "last_sequence": events[-1]["sequence"] if events else after,
                "source": str(self.server.metrics_path),
                "exists": self.server.metrics_path.exists(),
            }
        )

    def _serve_health(self) -> None:
        path = self.server.metrics_path
        stat = path.stat() if path.exists() else None
        self._send_json(
            {
                "ok": True,
                "source": str(path),
                "exists": stat is not None,
                "size": stat.st_size if stat else 0,
                "modified": stat.st_mtime if stat else None,
            }
        )

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._send(body, "application/json; charset=utf-8", status)

    def _send(
        self,
        body: bytes,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; connect-src 'self'; img-src 'self' data:; "
            "style-src 'self'; script-src 'self'",
        )
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, message: str, *args: object) -> None:
        if not self.server.quiet:
            super().log_message(message, *args)


def create_dashboard_server(
    metrics_path: str | Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    quiet: bool = False,
) -> DashboardServer:
    return DashboardServer((host, port), metrics_path, quiet=quiet)


def serve_dashboard(
    metrics_path: str | Path,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    server = create_dashboard_server(metrics_path, host, port)
    print(f"GaiaZero dashboard: http://{host}:{server.server_port}")
    print(f"Metrics source: {server.metrics_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

