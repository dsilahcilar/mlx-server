"""GatewayHandler, ThreadedServer, and the ``serve()`` entry point."""
from __future__ import annotations

import http.server
import logging
import os
import signal
import threading
from urllib.parse import parse_qs, urlparse

from .backend import manager
from .config import DEFAULT_PORT, MLX_HOME
from .handlers.base import BaseHandlerMixin
from .handlers.ollama import OllamaHandlerMixin
from .handlers.openai import OpenAIHandlerMixin
from .metrics import metrics

log = logging.getLogger("mlx-gateway")


class GatewayHandler(
    OllamaHandlerMixin,
    OpenAIHandlerMixin,
    BaseHandlerMixin,
    http.server.BaseHTTPRequestHandler,
):
    """Routes OpenAI-compatible and Ollama-compatible requests to the right backend."""

    server_version = "mlx-server/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        log.debug(fmt, *args)

    def do_GET(self):
        if self.path == "/v1/models":
            self._handle_models()
        elif self.path == "/_/ps":
            self._json(200, {"models": manager.list_loaded()})
        elif self.path == "/_/health":
            self._json(200, {"status": "ok"})
        elif self.path == "/_/metrics":
            self._json(200, metrics.get_summary())
        elif self.path.startswith("/_/requests"):
            qs = parse_qs(urlparse(self.path).query)
            n = int(qs.get("n", ["50"])[0])
            self._json(200, {"requests": metrics.get_recent(n)})
        elif self.path == "/":
            self._json(200, {"status": "mlx-server is running"})
        elif self.path == "/api/tags":
            self._ollama_tags()
        elif self.path == "/api/ps":
            self._ollama_ps()
        elif self.path == "/api/version":
            self._json(200, {"version": "mlx-server-1.0"})
        else:
            self._json(404, self._error("Not found"))

    def do_POST(self):
        if self.path in (
            "/v1/chat/completions", "/v1/completions", "/v1/embeddings"
        ):
            self._proxy()
        elif self.path == "/_/load":
            self._handle_load()
        elif self.path == "/_/unload":
            self._handle_unload()
        elif self.path == "/api/chat":
            self._ollama_chat()
        elif self.path == "/api/generate":
            self._ollama_generate()
        elif self.path == "/api/show":
            self._ollama_show()
        else:
            self._json(404, self._error("Not found"))


class ThreadedServer(http.server.ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def serve(port: int = DEFAULT_PORT):
    MLX_HOME.mkdir(parents=True, exist_ok=True)

    pid_file = MLX_HOME / "gateway.pid"
    pid_file.write_text(str(os.getpid()))

    server = ThreadedServer(("0.0.0.0", port), GatewayHandler)

    def _shutdown(signum, _frame):
        log.info("Received signal %d, shutting down...", signum)
        manager.stop_all()
        pid_file.unlink(missing_ok=True)
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    log.info("Gateway listening on http://0.0.0.0:%d", port)
    print(f"mlx-server gateway listening on http://127.0.0.1:{port}")
    print(f"OpenAI API:  http://127.0.0.1:{port}/v1/")
    print("Models loaded on demand. Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        manager.stop_all()
        pid_file.unlink(missing_ok=True)
        server.server_close()
