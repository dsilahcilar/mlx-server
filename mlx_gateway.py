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
import shutil
import signal
import subprocess
import sys
import threading
import time
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

        backend, err = manager.get_or_start(model_id)
        if not backend:
            self._json(
                503, self._error(err or "Failed to load model", "model_error")
            )
            return

        is_stream = body.get("stream", False)

        try:
            conn = http.client.HTTPConnection(
                "127.0.0.1", backend.port, timeout=300
            )
            conn.request(
                "POST", self.path, body=raw,
                headers={
                    "Content-Type": "application/json",
                    "Content-Length": str(len(raw)),
                },
            )
            resp = conn.getresponse()

            self.send_response(resp.status)

            if is_stream:
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                try:
                    while True:
                        chunk = resp.read(4096)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
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

        if manager.stop_model(model_id):
            self._json(200, {"status": "stopped", "model": model_id})
        else:
            self._json(
                404, self._error(f"Model '{model_id}' is not loaded")
            )


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
        server.shutdown()

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
