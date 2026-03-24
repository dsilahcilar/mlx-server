#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Installing mlx-server..."
echo ""

# Create symlink (so edits in the repo take effect immediately)
sudo ln -sf "${SCRIPT_DIR}/mlx-server" /usr/local/bin/mlx-server 2>/dev/null \
  || ln -sf "${SCRIPT_DIR}/mlx-server" /usr/local/bin/mlx-server
chmod +x "${SCRIPT_DIR}/mlx-server"
chmod +x "${SCRIPT_DIR}/mlx_gateway.py"
chmod +x "${SCRIPT_DIR}/mlx_server/embedding_server.py"
echo "✅ Installed: /usr/local/bin/mlx-server → ${SCRIPT_DIR}/mlx-server"

# Check dependencies
echo ""
echo "Dependencies:"

if command -v python3 &>/dev/null; then
    echo "  ✅ python3 $(python3 --version 2>&1 | awk '{print $2}')"
else
    echo "  ❌ python3 not found"
fi

if command -v mlx_lm.server &>/dev/null; then
    echo "  ✅ mlx_lm.server found"
else
    echo "  ⚠️  mlx_lm.server not found — install: pipx install mlx-lm"
fi

echo ""
echo "Get started:"
echo "  mlx-server serve"
echo "  mlx-server run mlx-community/Qwen3-32B-4bit"
