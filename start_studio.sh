#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "=========================================================="
echo " BIOS V2 — PS08: Own Inference Engine | Inference Studio"
echo "=========================================================="

# 1. Verify or build C++ inference engine binary
if [ ! -f "build/inference-engine" ]; then
    echo "[1/4] Building C++ inference engine binary..."
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DINFERENCE_ENGINE_BUILD_TESTS=ON
    cmake --build build --parallel 4
else
    echo "[1/4] C++ inference engine binary verified at build/inference-engine."
fi

# 2. Check model assets
if [ ! -f "models/tinyllama-1.1b-q4_0.ie" ]; then
    echo "[WARNING] models/tinyllama-1.1b-q4_0.ie not found."
    if [ -f "models/tinyllama-1.1b-fp32.ie" ]; then
        echo "Found FP32 model: models/tinyllama-1.1b-fp32.ie"
    fi
else
    echo "[2/4] Model verified: models/tinyllama-1.1b-q4_0.ie ($(du -h models/tinyllama-1.1b-q4_0.ie | cut -f1))"
fi

# 3. Clean up any stale process occupying port 3000
PORT="${PORT:-3000}"
EXISTING_PID=$(lsof -ti :"$PORT" 2>/dev/null || true)
if [ -n "$EXISTING_PID" ]; then
    echo "[3/4] Cleaning up stale process ($EXISTING_PID) on port $PORT..."
    kill -9 $EXISTING_PID 2>/dev/null || true
    sleep 1
else
    echo "[3/4] Port $PORT is available."
fi

# 4. Verify or build frontend static bundle
if [ ! -f "studio/dist/index.html" ]; then
    echo "[4/4] Building production frontend bundle (Vite + Tailwind)..."
    npm --prefix studio run build
else
    echo "[4/4] Production frontend bundle verified at studio/dist/."
fi

echo ""
echo "----------------------------------------------------------"
echo " Starting Inference Studio on http://localhost:$PORT"
echo " Press Ctrl+C to terminate."
echo "----------------------------------------------------------"
echo ""

exec npm --prefix studio start

