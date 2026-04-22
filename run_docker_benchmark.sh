#!/usr/bin/env bash
set -euo pipefail

BENCHMARK_NUM="${1:?Usage: $0 <benchmark-number> <timeout-seconds> [output-json]}"
TIMEOUT_SECONDS="${2:?Usage: $0 <benchmark-number> <timeout-seconds> [output-json]}"
OUTPUT_JSON="${3:-out.json}"

BENCHMARK_JSON="benchmarks/mlsys-2026-${BENCHMARK_NUM}.json"

if [[ ! -f "$BENCHMARK_JSON" ]]; then
    echo "Benchmark not found: $BENCHMARK_JSON" >&2
    exit 1
fi

docker run --rm --platform=linux/amd64 \
    --cpus=8 --memory=32g \
    -v "$PWD":/work -w /work \
    ubuntu:22.04 bash -lc "
set -e
timeout ${TIMEOUT_SECONDS}s ./mlsys ${BENCHMARK_JSON} ${OUTPUT_JSON}
test -s ${OUTPUT_JSON}
"
