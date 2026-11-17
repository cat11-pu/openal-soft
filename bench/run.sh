#!/bin/sh
# Build and run the HRTF mix benchmark (baseline vs optimized kernels).
# Usage: bench/run.sh
set -e
cd "$(dirname "$0")/.."
g++ -std=c++20 -O3 -DNDEBUG -Wall \
    -I bench/gen -I . -I common -I gsl/include \
    bench/hrtf_mix_bench.cpp \
    core/mixer/mixer_c.cpp \
    core/mixer/mixer_sse.cpp \
    core/mixer/mixer_avx2.cpp \
    core/mixer/mixer_avx512.cpp \
    core/filters/splitter.cpp \
    -o bench/hrtf_mix_bench
exec ./bench/hrtf_mix_bench "$@"
