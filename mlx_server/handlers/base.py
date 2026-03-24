"""Shared HTTP utilities mixin used by all handler mixins."""
from __future__ import annotations

import json


class BaseHandlerMixin:
    """JSON helpers shared by the OpenAI and Ollama handler mixins."""

    @staticmethod
    def _error(msg: str, etype: str = "invalid_request_error") -> dict:
        return {"error": {"message": msg, "type": etype}}

    def _read_body(self) -> bytes:
        cl = self.headers.get("Content-Length")
        te = self.headers.get("Transfer-Encoding", "")
        if cl is not None:
            return self.rfile.read(int(cl))
        if "chunked" in te.lower():
            buf = b""
            while True:
                line = self.rfile.readline().strip()
                if not line:
                    break
                size = int(line, 16)
                if size == 0:
                    self.rfile.readline()
                    break
                buf += self.rfile.read(size)
                self.rfile.read(2)
            return buf
        return b""

    def _json(self, status: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
