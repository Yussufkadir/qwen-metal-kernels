#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "=== Compiling Metal kernels ==="
metal_bin="$(xcrun --find metal)"
metallib_bin="$(xcrun --find metallib)"
"${metal_bin}" -c kernels.metal -o kernels.air
"${metal_bin}" -c optimized_kernels.metal -o optimized_kernels.air
"${metallib_bin}" kernels.air optimized_kernels.air -o default.metallib
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
