#!/usr/bin/env bash
set -euo pipefail

BENCHMARK_NUM="${1:?Usage: $0 <benchmark-number> <timeout-seconds> [output-json]}"
TIMEOUT_SECONDS="${2:?Usage: $0 <benchmark-number> <timeout-seconds> [output-json]}"
OUTPUT_JSON="${3:-out.json}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BENCHMARK_JSON="benchmarks/mlsys-2026-${BENCHMARK_NUM}.json"

if [[ ! -f "${REPO_ROOT}/${BENCHMARK_JSON}" ]]; then
    echo "Benchmark not found: $BENCHMARK_JSON" >&2
    exit 1
fi

if [[ ! -x "${REPO_ROOT}/mlsys" ]]; then
    echo "Executable not found: ${REPO_ROOT}/mlsys" >&2
    echo "Run scripts/run_docker.sh first." >&2
    exit 1
fi

docker run --rm --platform=linux/amd64 \
    --cpus=8 --memory=32g \
    -v "${REPO_ROOT}":/work -w /work \
    ubuntu:22.04 bash -lc "
set -e
timeout ${TIMEOUT_SECONDS}s ./mlsys ${BENCHMARK_JSON} ${OUTPUT_JSON}
test -s ${OUTPUT_JSON}
"
