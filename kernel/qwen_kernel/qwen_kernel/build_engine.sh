#!/bin/zsh
set -euo pipefail

cd "${0:A:h}"

metal_bin="$(xcrun --find metal)"
metallib_bin="$(xcrun --find metallib)"
"${metal_bin}" -c kernels.metal -o kernels.air
"${metal_bin}" -c optimized_kernels.metal -o optimized_kernels.air
"${metallib_bin}" kernels.air optimized_kernels.air -o default.metallib

xcrun clang++ -std=c++20 -O3 -fobjc-arc -Wall -Wextra \
  main_cli.mm qwen_engine.mm \
  -framework Foundation \
  -framework Metal \
  -framework MetalPerformanceShaders \
  -o inference_engine

echo "Built ${PWD}/inference_engine"
