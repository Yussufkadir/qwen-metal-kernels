#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "=== Compiling Metal kernels ==="
xcrun -sdk macosx metal -c kernels.metal -o kernels.air
xcrun -sdk macosx metallib kernels.air -o default.metallib
echo "Created default.metallib"

echo "=== Compiling C bridge ==="
clang++ -std=c++17 -O2 \
    -framework Metal \
    -framework Foundation \
    -dynamiclib \
    bridge.mm \
    -o ../../libmetal_kernels.dylib
echo "Created libmetal_kernels.dylib"

echo "=== Done ==="
echo ""
echo "Files created:"
echo "  default.metallib          (in this folder)"
echo "  libmetal_kernels.dylib    (in parent folder: qwen_benchmark/)"
echo ""
echo "Now run: cd ../.. && python benchmark_qwen.py"
