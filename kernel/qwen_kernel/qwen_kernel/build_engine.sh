#!/bin/zsh
set -euo pipefail

cd "${0:A:h}"
xcrun clang++ -std=c++20 -O3 -fobjc-arc -Wall -Wextra \
  main_cli.mm qwen_engine.mm \
  -framework Foundation \
  -framework Metal \
  -framework MetalPerformanceShaders \
  -o inference_engine

echo "Built ${PWD}/inference_engine"
