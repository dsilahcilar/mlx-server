#!/usr/bin/env python3
"""
Embedding model server — OpenAI-compatible /v1/embeddings endpoint.
Uses sentence-transformers with Apple Silicon MPS backend.

Usage: mlx_embedding_server.py <model_id> <port>
"""
from __future__ import annotations

import http.server
import json
import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] mlx-embedding-server — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mlx-embedding-server")


def main():
    if len(sys.argv) < 3:
        print("Usage: mlx_embedding_server.py <model_id> <port>", file=sys.stderr)
        sys.exit(1)

    model_id = sys.argv[1]
    port = int(sys.argv[2])

    log.info("Loading %s ...", model_id)
    try:
        from sentence_transformers import SentenceTransformer
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = SentenceTransformer(model_id, device=device)
        dim = model.get_sentence_embedding_dimension()
        log.info("✅ %s ready (dim=%d, device=%s)", model_id, dim, device)
    except Exception as exc:
        log.error("Failed to load %s: %s", model_id, exc)
        sys.exit(1)

    class EmbeddingHandler(http.server.BaseHTTPRequestHandler):
        server_version = "mlx-embedding-server/1.0"

        def log_message(self, fmt, *args):
            log.debug(fmt, *args)

        def _json(self, code: int, obj: dict):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, msg: str):
            return {"error": {"message": msg, "type": "invalid_request_error"}}

        def do_GET(self):
            if self.path == "/v1/models":
                self._json(200, {
                    "object": "list",
                    "data": [{
                        "id": model_id,
                        "object": "model",
                        "owned_by": "mlx-server",
                        "type": "embedding",
                    }],
                })
            else:
                self._json(404, self._error("Not found"))

        def do_POST(self):
            if self.path != "/v1/embeddings":
                self._json(404, self._error("Not found"))
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
            except Exception:
                self._json(400, self._error("Invalid JSON body"))
                return

            raw_input = body.get("input")
            if raw_input is None:
                self._json(400, self._error("'input' field is required"))
                return

            texts: list[str] = [raw_input] if isinstance(raw_input, str) else raw_input

            try:
                t0 = time.time()
                normalize = body.get("normalize_embeddings", True)
                embeddings = model.encode(
                    texts,
                    normalize_embeddings=normalize,
                    show_progress_bar=False,
                )
                elapsed = time.time() - t0
                log.debug("Encoded %d text(s) in %.2fs", len(texts), elapsed)
            except Exception as exc:
                log.exception("Encode error")
                self._json(500, self._error(str(exc)))
                return

            # Count tokens (approximate via word count; proper token count
            # would require running the tokenizer separately)
            try:
                encoded = model.tokenize(texts)
                token_count = int(encoded["input_ids"].shape[0] * encoded["input_ids"].shape[1])
            except Exception:
                token_count = sum(len(t.split()) for t in texts)

            data = [
                {
                    "object": "embedding",
                    "index": i,
                    "embedding": emb.tolist(),
                }
                for i, emb in enumerate(embeddings)
            ]

            self._json(200, {
                "object": "list",
                "data": data,
                "model": model_id,
                "usage": {
                    "prompt_tokens": token_count,
                    "total_tokens": token_count,
                },
            })

    server = http.server.HTTPServer(("127.0.0.1", port), EmbeddingHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
