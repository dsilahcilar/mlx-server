"""OpenAI-compatible endpoint handlers (proxy, models list, load/unload)."""
from __future__ import annotations

import http.client
import json
import logging
import threading
import time

from ..aliases import resolve_model
from ..backend import manager
from ..metrics import metrics
from ..models import get_cached_models

log = logging.getLogger("mlx-gateway")


class OpenAIHandlerMixin:
    """Handles ``/v1/*`` endpoints."""

    def _handle_models(self):
        data = [
            {
                "id": m["id"],
                "object": "model",
                "created": int(m["modified"]),
                "owned_by": m["id"].split("/")[0] if "/" in m["id"] else "local",
            }
            for m in get_cached_models()
        ]
        self._json(200, {"object": "list", "data": data})

    def _proxy(self):
        """Forward a request to the appropriate backend, streaming if requested."""
        raw = self._read_body()
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            self._json(400, self._error("Invalid JSON body"))
            return

        model_id = body.get("model")
        if not model_id:
            self._json(400, self._error("'model' field is required"))
            return

        model_id = resolve_model(model_id)
        body["model"] = model_id

        backend, err = manager.get_or_start(model_id)
        if not backend:
            self._json(503, self._error(err or "Failed to load model", "model_error"))
            return

        is_stream = body.get("stream", False)
        if is_stream and self.path in ("/v1/chat/completions", "/v1/completions"):
            body.setdefault("stream_options", {})
            body["stream_options"]["include_usage"] = True

        t_start = time.monotonic()
        error_occurred = False
        prompt_tokens = completion_tokens = cached_tokens = 0

        try:
            forwarded = json.dumps(body).encode()
            conn = http.client.HTTPConnection(
                "127.0.0.1", backend.port, timeout=300
            )
            conn.request(
                "POST", self.path, body=forwarded,
                headers={
                    "Content-Type": "application/json",
                    "Content-Length": str(len(forwarded)),
                },
            )
            resp = conn.getresponse()
            self.send_response(resp.status)
            if resp.status >= 400:
                error_occurred = True

            if is_stream:
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                try:
                    while True:
                        line = resp.readline()
                        if not line:
                            break
                        self.wfile.write(line)
                        self.wfile.flush()
                        stripped = line.strip()
                        if (
                            stripped.startswith(b"data: ")
                            and stripped != b"data: [DONE]"
                        ):
                            try:
                                chunk = json.loads(stripped[6:])
                                usage = chunk.get("usage") or {}
                                if usage:
                                    prompt_tokens = usage.get("prompt_tokens", 0)
                                    completion_tokens = usage.get(
                                        "completion_tokens", 0
                                    )
                                    cached_tokens = (
                                        (usage.get("prompt_tokens_details") or {})
                                        .get("cached_tokens", 0)
                                    )
                            except (json.JSONDecodeError, KeyError):
                                pass
                        if stripped == b"data: [DONE]":
                            break
                except (BrokenPipeError, ConnectionResetError):
                    pass
                self.close_connection = True
            else:
                data = resp.read()
                for key, val in resp.getheaders():
                    if key.lower() == "content-type":
                        self.send_header(key, val)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                try:
                    resp_body = json.loads(data)
                    usage = resp_body.get("usage") or {}
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    cached_tokens = (
                        (usage.get("prompt_tokens_details") or {})
                        .get("cached_tokens", 0)
                    )
                except (json.JSONDecodeError, KeyError):
                    pass

            conn.close()

        except (BrokenPipeError, ConnectionResetError):
            error_occurred = True
        except Exception as exc:
            error_occurred = True
            log.exception("Proxy error for %s", model_id)
            try:
                self._json(
                    502, self._error(f"Backend error: {exc}", "proxy_error")
                )
            except Exception:
                pass
        finally:
            latency_ms = (time.monotonic() - t_start) * 1000
            metrics.record(
                model_id=model_id,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_tokens=cached_tokens,
                error=error_occurred,
                endpoint=self.path,
                stream=is_stream,
            )

    def _handle_load(self):
        raw = self._read_body()
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            self._json(400, self._error("Invalid JSON"))
            return

        model_id = body.get("model")
        if not model_id:
            self._json(400, self._error("'model' field required"))
            return
        model_id = resolve_model(model_id)

        def _load():
            _, err = manager.get_or_start(model_id)
            if err:
                log.error("Load %s failed: %s", model_id, err)

        threading.Thread(target=_load, daemon=True).start()
        self._json(200, {"status": "loading", "model": model_id})

    def _handle_unload(self):
        raw = self._read_body()
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            self._json(400, self._error("Invalid JSON"))
            return

        model_id = body.get("model")
        if not model_id:
            self._json(400, self._error("'model' field required"))
            return
        model_id = resolve_model(model_id)

        if manager.stop_model(model_id):
            self._json(200, {"status": "stopped", "model": model_id})
        else:
            self._json(404, self._error(f"Model '{model_id}' is not loaded"))
