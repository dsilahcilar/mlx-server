#!/usr/bin/env python3
"""
mlx-gateway entry point — delegates to the mlx_server package.

This file is kept at the repository root so the mlx-server bash CLI can
reference it via a stable path (GATEWAY_SCRIPT).
"""
from __future__ import annotations

import logging
import sys


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


if __name__ == "__main__":
    _setup_logging()

    from mlx_server.config import DEFAULT_PORT
    from mlx_server.chat import chat_repl
    from mlx_server.gateway import serve

    if len(sys.argv) >= 2 and sys.argv[1] == "chat":
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
