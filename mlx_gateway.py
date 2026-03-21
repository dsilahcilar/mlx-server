#!/usr/bin/env python3
"""
mlx-gateway: Multi-model OpenAI-compatible API gateway for MLX.

Routes requests to per-model mlx_lm.server backends based on the 'model' field.
Single port, many models — like OpenAI's API, running locally on Apple Silicon.
"""
from __future__ import annotations

import http.client
import http.server
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_PORT = 11070
BACKEND_BASE_PORT = 18100
MAX_TOKENS_DEFAULT = 4096

MLX_HOME = Path.home() / ".mlx"
HF_CACHE = Path(os.environ.get(
    "HF_HOME", str(Path.home() / ".cache" / "huggingface")
)) / "hub"

log = logging.getLogger("mlx-gateway")


# ── Utilities ─────────────────────────────────────────────────────────────────

def find_mlx_server_bin() -> str | None:
    """Find the mlx_lm.server executable on PATH."""
    path = shutil.which("mlx_lm.server")
    if path:
        return path
    candidate = Path.home() / ".local" / "bin" / "mlx_lm.server"
    if candidate.exists():
        return str(candidate)
    return None


def is_model_cached(model_id: str) -> bool:
    cache_dir = HF_CACHE / f"models--{model_id.replace('/', '--')}"
    return cache_dir.is_dir()


def get_cached_models() -> list[dict]:
    models = []
    if not HF_CACHE.exists():
        return models
    for d in sorted(HF_CACHE.iterdir()):
        if d.is_dir() and d.name.startswith("models--"):
            model_id = d.name[len("models--"):].replace("--", "/", 1)
            try:
                size_bytes = sum(
                    f.stat().st_size for f in d.rglob("*") if f.is_file()
                )
            except OSError:
                size_bytes = 0
            models.append({
                "id": model_id,
                "size_bytes": size_bytes,
                "modified": d.stat().st_mtime,
            })
    return models


# ── Model Aliases ─────────────────────────────────────────────────────────────

def _generate_ollama_alias(model_id: str) -> str | None:
    """Generate an Ollama-style alias (e.g., qwen3:8b) from an HF model ID."""
    name = model_id.split("/")[-1] if "/" in model_id else model_id

    # Strip quantization/format suffixes
    name = re.sub(
        r'[-_](4bit|8bit|6bit|MLX|mlx|qat|MXFP4|Q8|bf16|fp16|fp32)$',
        '', name,
    )
    name = re.sub(
        r'[-_](4bit|8bit|6bit|MLX|mlx|qat|MXFP4|Q8|bf16|fp16|fp32)$',
        '', name,
    )
    # Strip instruction-tuning suffixes
    name = re.sub(
        r'[-_](Instruct|instruct|Chat|chat|it|IT|Base|base|Preview|preview)$',
        '', name,
    )

    # Find model size pattern: 8B, 32B, 1.7B, 0.6B, etc.
    size_match = re.search(r'[-_](\d+(?:\.\d+)?[Bb])(?:[-_]|$)', name)
    if not size_match:
        return None

    size = size_match.group(1).lower()
    family = name[:size_match.start()].rstrip('-_').lower()

    return f"{family}:{size}" if family else None


def build_alias_map() -> dict[str, str]:
    """Build a map from alias names to full HuggingFace model IDs."""
    aliases: dict[str, str] = {}

    # Load custom aliases from ~/.mlx/aliases.json
    alias_file = MLX_HOME / "aliases.json"
    if alias_file.exists():
        try:
            aliases.update(json.loads(alias_file.read_text()))
        except (json.JSONDecodeError, OSError):
            pass

    # Auto-generate aliases from cached models
    for model in get_cached_models():
        mid = model["id"]
        aliases[mid] = mid  # Full name always works

        # Short name: "Qwen3-8B-4bit"
        if "/" in mid:
            short = mid.split("/", 1)[1]
            aliases.setdefault(short, mid)
            aliases.setdefault(short.lower(), mid)

        # Ollama-style: "qwen3:8b"
        ollama = _generate_ollama_alias(mid)
        if ollama:
            aliases.setdefault(ollama, mid)

    return aliases


def resolve_model(name: str) -> str:
    """Resolve a model alias to its full HuggingFace ID."""
    aliases = build_alias_map()
    if name in aliases:
        return aliases[name]
    if name.lower() in aliases:
        return aliases[name.lower()]
    return name  # Return as-is; let backend fail with clear error


# ── Model Backend ─────────────────────────────────────────────────────────────

class ModelBackend:
    """Manages a single mlx_lm.server process serving one model."""

    def __init__(self, model_id: str, port: int,
                 max_tokens: int = MAX_TOKENS_DEFAULT):
        self.model_id = model_id
        self.port = port
        self.max_tokens = max_tokens
        self.process: subprocess.Popen | None = None
        self.ready = False
        self._loading = threading.Event()
        self._ready_event = threading.Event()
        self.last_used = time.time()
        self.error: str | None = None

    def start(self) -> bool:
        """Start the backend and block until ready (or failed)."""
        if self.ready:
            return True
        self._loading.set()

        mlx_bin = find_mlx_server_bin()
        if not mlx_bin:
            self.error = "mlx_lm.server not found. Install: pipx install mlx-lm"
            self._loading.clear()
            return False

        if not is_model_cached(self.model_id):
            self.error = (
                f"Model not found locally. Run: mlx-server pull {self.model_id}"
            )
            self._loading.clear()
            return False

        log_path = MLX_HOME / f"backend-{self.port}.log"
        cmd = [
            mlx_bin,
            "--model", self.model_id,
            "--port", str(self.port),
            "--max-tokens", str(self.max_tokens),
            "--chat-template-args", json.dumps({"enable_thinking": False}),
        ]

        log.info("Starting %s on internal port %d", self.model_id, self.port)

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=open(log_path, "w"),
                stderr=subprocess.STDOUT,
                env={**os.environ, "HF_HUB_OFFLINE": "1"},
            )
        except Exception as exc:
            self.error = str(exc)
            self._loading.clear()
            return False

        # Poll for readiness (up to 5 minutes)
        for _ in range(100):
            if self.process.poll() is not None:
                self.error = (
                    f"Process exited with code {self.process.returncode}"
                )
                self._loading.clear()
                log.error("Backend %s failed: %s", self.model_id, self.error)
                return False
            try:
                conn = http.client.HTTPConnection(
                    "127.0.0.1", self.port, timeout=2
                )
                conn.request("GET", "/v1/models")
                resp = conn.getresponse()
                resp.read()
                conn.close()
                if resp.status == 200:
                    self.ready = True
                    self._loading.clear()
                    self._ready_event.set()
                    self.last_used = time.time()
                    log.info(
                        "✅ %s ready on port %d (PID %d)",
                        self.model_id, self.port, self.process.pid,
                    )
                    return True
            except Exception:
                pass
            time.sleep(3)

        self.error = "Timeout (5 min) loading model"
        self._loading.clear()
        log.error("Backend %s timed out", self.model_id)
        return False

    def stop(self):
        if self.process and self.process.poll() is None:
            log.info("Stopping %s (PID %d)", self.model_id, self.process.pid)
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.ready = False
        self._ready_event.clear()

    @property
    def is_loading(self) -> bool:
        return self._loading.is_set()

    def wait_until_ready(self, timeout: float = 600) -> bool:
        return self._ready_event.wait(timeout=timeout)

    def touch(self):
        self.last_used = time.time()


# ── Backend Manager ───────────────────────────────────────────────────────────

class BackendManager:
    """Thread-safe registry of model backends."""

    def __init__(self):
        self.backends: dict[str, ModelBackend] = {}
        self._next_port = BACKEND_BASE_PORT
        self._lock = threading.Lock()

    def get_or_start(
        self, model_id: str
    ) -> tuple[ModelBackend | None, str | None]:
        """Return (backend, None) on success or (None, error_msg) on failure."""
        waiting_on = None

        with self._lock:
            if model_id in self.backends:
                b = self.backends[model_id]
                if b.ready:
                    b.touch()
                    return b, None
                if b.is_loading:
                    waiting_on = b
                else:
                    # Previous failure — remove and recreate
                    del self.backends[model_id]

            if model_id not in self.backends and waiting_on is None:
                if not is_model_cached(model_id):
                    return None, (
                        f"Model '{model_id}' not found locally. "
                        f"Run: mlx-server pull {model_id}"
                    )
                port = self._next_port
                self._next_port += 1
                self.backends[model_id] = ModelBackend(model_id, port)

        # If another thread is loading, wait for it
        if waiting_on is not None:
            if waiting_on.wait_until_ready(timeout=600):
                waiting_on.touch()
                return waiting_on, None
            return None, waiting_on.error or "Timeout waiting for model"

        # We own the start
        backend = self.backends.get(model_id)
        if backend is None:
            return None, "Internal error"

        if backend.start():
            return backend, None

        error = backend.error
        with self._lock:
            self.backends.pop(model_id, None)
        return None, error or "Failed to start model"

    def stop_model(self, model_id: str) -> bool:
        with self._lock:
            b = self.backends.pop(model_id, None)
        if b:
            b.stop()
            return True
        return False

    def stop_all(self):
        with self._lock:
            backends = list(self.backends.values())
            self.backends.clear()
        for b in backends:
            b.stop()

    def list_loaded(self) -> list[dict]:
        with self._lock:
            result = []
            for b in self.backends.values():
                info: dict = {
                    "model": b.model_id,
                    "port": b.port,
                    "ready": b.ready,
                    "loading": b.is_loading,
                }
                if b.process and b.process.poll() is None:
                    info["pid"] = b.process.pid
                if b.ready:
                    cache_dir = HF_CACHE / f"models--{b.model_id.replace('/', '--')}"
                    if cache_dir.exists():
                        try:
                            size = sum(
                                f.stat().st_size
                                for f in cache_dir.rglob("*") if f.is_file()
                            )
                            info["size_gb"] = round(size / (1024 ** 3), 1)
                        except OSError:
                            pass
                    idle = time.time() - b.last_used
                    if idle < 60:
                        info["idle"] = f"{int(idle)}s"
                    else:
                        info["idle"] = f"{int(idle / 60)}m"
                result.append(info)
            return result


# ── HTTP Gateway ──────────────────────────────────────────────────────────────

manager = BackendManager()


class GatewayHandler(http.server.BaseHTTPRequestHandler):
    """Routes OpenAI-compatible requests to the right model backend."""

    server_version = "mlx-server/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        log.debug(fmt, *args)

    # ── GET ───────────────────────────────────────────────────────────────

    def do_GET(self):
        if self.path == "/v1/models":
            self._handle_models()
        elif self.path == "/_/ps":
            self._json(200, {"models": manager.list_loaded()})
        elif self.path == "/_/health":
            self._json(200, {"status": "ok"})
        # ── Ollama API ────────────────────────────────────────────────────
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

    # ── POST ──────────────────────────────────────────────────────────────

    def do_POST(self):
        if self.path in (
            "/v1/chat/completions", "/v1/completions", "/v1/embeddings"
        ):
            self._proxy()
        elif self.path == "/_/load":
            self._handle_load()
        elif self.path == "/_/unload":
            self._handle_unload()
        # ── Ollama API ────────────────────────────────────────────────────
        elif self.path == "/api/chat":
            self._ollama_chat()
        elif self.path == "/api/generate":
            self._ollama_generate()
        elif self.path == "/api/show":
            self._ollama_show()
        else:
            self._json(404, self._error("Not found"))

    # ── Internals ─────────────────────────────────────────────────────────

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

    # ── /v1/models ────────────────────────────────────────────────────────

    def _handle_models(self):
        data = []
        for m in get_cached_models():
            data.append({
                "id": m["id"],
                "object": "model",
                "created": int(m["modified"]),
                "owned_by": (
                    m["id"].split("/")[0] if "/" in m["id"] else "local"
                ),
            })
        self._json(200, {"object": "list", "data": data})

    # ── Proxy ─────────────────────────────────────────────────────────────

    def _proxy(self):
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
        body["model"] = model_id  # Ensure backend gets the resolved name
        backend, err = manager.get_or_start(model_id)
        if not backend:
            self._json(
                503, self._error(err or "Failed to load model", "model_error")
            )
            return

        is_stream = body.get("stream", False)

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
                        if line.strip() == b"data: [DONE]":
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

            conn.close()

        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            log.exception("Proxy error for %s", model_id)
            try:
                self._json(
                    502,
                    self._error(f"Backend error: {exc}", "proxy_error"),
                )
            except Exception:
                pass

    # ── Management API ────────────────────────────────────────────────────

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
            self._json(
                404, self._error(f"Model '{model_id}' is not loaded")
            )

    # ── Ollama API ────────────────────────────────────────────────────────

    def _ollama_model_entry(self, model: dict) -> dict:
        """Build an Ollama-style model entry for /api/tags."""
        mid = model["id"]
        alias = _generate_ollama_alias(mid) or mid
        ts = datetime.fromtimestamp(
            model["modified"], tz=timezone.utc
        ).isoformat()

        # Parse config for details
        details = {"format": "mlx", "family": "", "parameter_size": "",
                    "quantization_level": ""}
        cache_dir = HF_CACHE / f"models--{mid.replace('/', '--')}"
        config_files = list(cache_dir.glob("snapshots/*/config.json"))
        if config_files:
            try:
                cfg = json.loads(config_files[0].read_text())
                details["family"] = cfg.get("model_type", "")
                quant = cfg.get("quantization", {})
                if isinstance(quant, dict):
                    details["quantization_level"] = (
                        f"Q{quant.get('bits', '?')}"
                    )
                size_match = re.search(
                    r'(\d+(?:\.\d+)?[Bb])', mid.split("/")[-1]
                )
                if size_match:
                    details["parameter_size"] = size_match.group(1)
            except (json.JSONDecodeError, OSError):
                pass

        return {
            "name": alias,
            "model": alias,
            "modified_at": ts,
            "size": model["size_bytes"],
            "digest": f"sha256:{mid}",
            "details": details,
        }

    def _ollama_tags(self):
        """GET /api/tags — list models in Ollama format."""
        models = [
            self._ollama_model_entry(m) for m in get_cached_models()
        ]
        self._json(200, {"models": models})

    def _ollama_ps(self):
        """GET /api/ps — list running models in Ollama format."""
        loaded = manager.list_loaded()
        models = []
        for b in loaded:
            if not b.get("ready"):
                continue
            alias = _generate_ollama_alias(b["model"]) or b["model"]
            size_bytes = int(b.get("size_gb", 0) * 1024 ** 3)
            models.append({
                "name": alias,
                "model": alias,
                "size": size_bytes,
                "digest": f"sha256:{b['model']}",
                "details": {"format": "mlx"},
                "expires_at": (
                    datetime.now(timezone.utc).isoformat()
                ),
                "size_vram": size_bytes,
            })
        self._json(200, {"models": models})

    def _ollama_chat(self):
        """POST /api/chat — Ollama chat endpoint, proxied via OpenAI."""
        raw = self._read_body()
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            self._json(400, self._error("Invalid JSON"))
            return

        ollama_model = body.get("model", "")
        model_id = resolve_model(ollama_model)
        is_stream = body.get("stream", True)  # Ollama defaults stream=true

        # Convert Ollama request → OpenAI request
        openai_body: dict = {
            "model": model_id,
            "messages": body.get("messages", []),
            "stream": is_stream,
        }
        opts = body.get("options", {})
        if "temperature" in opts:
            openai_body["temperature"] = opts["temperature"]
        if "top_p" in opts:
            openai_body["top_p"] = opts["top_p"]
        if "num_predict" in opts:
            openai_body["max_tokens"] = opts["num_predict"]
        if "stop" in opts:
            openai_body["stop"] = opts["stop"]

        # Get or start backend
        backend, err = manager.get_or_start(model_id)
        if not backend:
            self._json(404, {"error": err or "model not found"})
            return

        openai_raw = json.dumps(openai_body).encode()

        try:
            conn = http.client.HTTPConnection(
                "127.0.0.1", backend.port, timeout=300
            )
            conn.request(
                "POST", "/v1/chat/completions", body=openai_raw,
                headers={
                    "Content-Type": "application/json",
                    "Content-Length": str(len(openai_raw)),
                },
            )
            resp = conn.getresponse()

            if resp.status != 200:
                err_body = resp.read()
                conn.close()
                self.send_response(resp.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err_body)))
                self.end_headers()
                self.wfile.write(err_body)
                return

            if is_stream:
                self._ollama_stream_chat(resp, ollama_model)
            else:
                self._ollama_nonstream_chat(resp, ollama_model)

            conn.close()

        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            log.exception("Ollama chat proxy error")
            try:
                self._json(502, {"error": str(exc)})
            except Exception:
                pass

    def _ollama_stream_chat(self, resp, model_name: str):
        """Convert OpenAI SSE stream to Ollama NDJSON stream."""
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()

        buf = b""
        t_start = time.monotonic()
        eval_count = 0

        try:
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                buf += chunk

                while b"\n" in buf:
                    line_bytes, buf = buf.split(b"\n", 1)
                    line = line_bytes.decode().strip()
                    if not line:
                        continue
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]

                    if data_str == "[DONE]":
                        duration_ns = int(
                            (time.monotonic() - t_start) * 1e9
                        )
                        final = json.dumps({
                            "model": model_name,
                            "created_at": datetime.now(
                                timezone.utc
                            ).isoformat(),
                            "message": {
                                "role": "assistant", "content": ""
                            },
                            "done": True,
                            "total_duration": duration_ns,
                            "eval_count": eval_count,
                        })
                        self.wfile.write(final.encode() + b"\n")
                        self.wfile.flush()
                        return

                    try:
                        oai = json.loads(data_str)
                        content = (
                            oai.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content", "")
                        )
                        if content:
                            eval_count += 1
                            out = json.dumps({
                                "model": model_name,
                                "created_at": datetime.now(
                                    timezone.utc
                                ).isoformat(),
                                "message": {
                                    "role": "assistant",
                                    "content": content,
                                },
                                "done": False,
                            })
                            self.wfile.write(out.encode() + b"\n")
                            self.wfile.flush()
                    except json.JSONDecodeError:
                        pass
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _ollama_nonstream_chat(self, resp, model_name: str):
        """Convert OpenAI non-streaming response to Ollama format."""
        data = json.loads(resp.read())

        content = ""
        if data.get("choices"):
            content = data["choices"][0].get("message", {}).get(
                "content", ""
            )

        usage = data.get("usage", {})
        ollama_resp = {
            "model": model_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "message": {"role": "assistant", "content": content},
            "done": True,
            "total_duration": 0,
            "prompt_eval_count": usage.get("prompt_tokens", 0),
            "eval_count": usage.get("completion_tokens", 0),
        }
        self._json(200, ollama_resp)

    def _ollama_generate(self):
        """POST /api/generate — Ollama generate endpoint."""
        raw = self._read_body()
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            self._json(400, self._error("Invalid JSON"))
            return

        ollama_model = body.get("model", "")
        model_id = resolve_model(ollama_model)
        prompt = body.get("prompt", "")
        is_stream = body.get("stream", True)

        # Convert to chat format
        openai_body: dict = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "stream": is_stream,
        }
        opts = body.get("options", {})
        if "temperature" in opts:
            openai_body["temperature"] = opts["temperature"]
        if "num_predict" in opts:
            openai_body["max_tokens"] = opts["num_predict"]

        backend, err = manager.get_or_start(model_id)
        if not backend:
            self._json(404, {"error": err or "model not found"})
            return

        openai_raw = json.dumps(openai_body).encode()

        try:
            conn = http.client.HTTPConnection(
                "127.0.0.1", backend.port, timeout=300
            )
            conn.request(
                "POST", "/v1/chat/completions", body=openai_raw,
                headers={
                    "Content-Type": "application/json",
                    "Content-Length": str(len(openai_raw)),
                },
            )
            resp = conn.getresponse()

            if resp.status != 200:
                err_body = resp.read()
                conn.close()
                self.send_response(resp.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err_body)))
                self.end_headers()
                self.wfile.write(err_body)
                return

            if is_stream:
                self._ollama_stream_generate(resp, ollama_model)
            else:
                self._ollama_nonstream_generate(resp, ollama_model)

            conn.close()

        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            log.exception("Ollama generate proxy error")
            try:
                self._json(502, {"error": str(exc)})
            except Exception:
                pass

    def _ollama_stream_generate(self, resp, model_name: str):
        """Convert OpenAI SSE stream to Ollama generate NDJSON."""
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()

        buf = b""
        t_start = time.monotonic()

        try:
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                buf += chunk

                while b"\n" in buf:
                    line_bytes, buf = buf.split(b"\n", 1)
                    line = line_bytes.decode().strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]

                    if data_str == "[DONE]":
                        duration_ns = int(
                            (time.monotonic() - t_start) * 1e9
                        )
                        final = json.dumps({
                            "model": model_name,
                            "created_at": datetime.now(
                                timezone.utc
                            ).isoformat(),
                            "response": "",
                            "done": True,
                            "total_duration": duration_ns,
                        })
                        self.wfile.write(final.encode() + b"\n")
                        self.wfile.flush()
                        return

                    try:
                        oai = json.loads(data_str)
                        content = (
                            oai.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content", "")
                        )
                        if content:
                            out = json.dumps({
                                "model": model_name,
                                "created_at": datetime.now(
                                    timezone.utc
                                ).isoformat(),
                                "response": content,
                                "done": False,
                            })
                            self.wfile.write(out.encode() + b"\n")
                            self.wfile.flush()
                    except json.JSONDecodeError:
                        pass
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _ollama_nonstream_generate(self, resp, model_name: str):
        """Convert OpenAI response to Ollama generate format."""
        data = json.loads(resp.read())
        content = ""
        if data.get("choices"):
            content = data["choices"][0].get("message", {}).get(
                "content", ""
            )
        ollama_resp = {
            "model": model_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "response": content,
            "done": True,
        }
        self._json(200, ollama_resp)

    def _ollama_show(self):
        """POST /api/show — model info in Ollama format."""
        raw = self._read_body()
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            self._json(400, self._error("Invalid JSON"))
            return

        model_name = body.get("model") or body.get("name", "")
        model_id = resolve_model(model_name)

        if not is_model_cached(model_id):
            self._json(404, {"error": f"model '{model_name}' not found"})
            return

        cache_dir = HF_CACHE / f"models--{model_id.replace('/', '--')}"
        details = {"format": "mlx", "family": "", "parameter_size": "",
                    "quantization_level": ""}
        model_info = {}

        config_files = list(cache_dir.glob("snapshots/*/config.json"))
        if config_files:
            try:
                cfg = json.loads(config_files[0].read_text())
                details["family"] = cfg.get("model_type", "")
                quant = cfg.get("quantization", {})
                if isinstance(quant, dict):
                    details["quantization_level"] = (
                        f"Q{quant.get('bits', '?')}"
                    )
                model_info = {
                    "hidden_size": cfg.get("hidden_size"),
                    "num_hidden_layers": cfg.get("num_hidden_layers"),
                    "num_attention_heads": cfg.get("num_attention_heads"),
                    "vocab_size": cfg.get("vocab_size"),
                    "context_length": cfg.get("max_position_embeddings"),
                }
            except (json.JSONDecodeError, OSError):
                pass

        self._json(200, {
            "model_info": model_info,
            "details": details,
            "modified_at": datetime.fromtimestamp(
                cache_dir.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
        })


# ── Server ────────────────────────────────────────────────────────────────────

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
        # Shutdown in a thread to avoid deadlock in signal handler
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


# ── Interactive Chat REPL ─────────────────────────────────────────────────────

def chat_repl(base_url: str, model: str):
    """Interactive multi-turn chat with streaming output."""
    import urllib.error
    import urllib.request

    messages: list[dict] = []

    print(f"Running {model}. Send a message or type /bye to exit.\n")

    while True:
        try:
            prompt = input("\033[1;32m>>> \033[0m")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not prompt.strip():
            continue
        if prompt.strip() in ("/bye", "/exit", "/quit"):
            print("Bye!")
            break

        messages.append({"role": "user", "content": prompt})

        payload = json.dumps({
            "model": model,
            "messages": messages,
            "stream": True,
        }).encode()

        req = urllib.request.Request(
            f"{base_url}/v1/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(payload)),
            },
        )

        full_response = ""
        try:
            resp = urllib.request.urlopen(req, timeout=300)
            for raw_line in resp:
                line = raw_line.decode().strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        chunk = json.loads(line[6:])
                        content = (
                            chunk.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content", "")
                        )
                        sys.stdout.write(content)
                        sys.stdout.flush()
                        full_response += content
                    except json.JSONDecodeError:
                        pass
            print("\n")
        except KeyboardInterrupt:
            print("\n")
            messages.pop()  # Remove interrupted user message
            continue
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            print(f"\nError ({e.code}): {body}\n")
            messages.pop()  # Remove failed user message
            continue
        except Exception as e:
            print(f"\nError: {e}\n")
            messages.pop()
            continue

        messages.append({"role": "assistant", "content": full_response})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    if len(sys.argv) >= 2 and sys.argv[1] == "chat":
        # Usage: mlx_gateway.py chat <base_url> <model>
        if len(sys.argv) < 4:
            print("Usage: mlx_gateway.py chat <base_url> <model>")
            sys.exit(1)
        chat_repl(sys.argv[2], sys.argv[3])
    else:
        port = DEFAULT_PORT
        for i, arg in enumerate(sys.argv[1:], 1):
            if arg == "--port" and i < len(sys.argv):
                port = int(sys.argv[i + 1])
        serve(port)
