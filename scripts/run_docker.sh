#!/usr/bin/env bash
set -euo pipefail

# Build inside Ubuntu 22.04 amd64.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

docker run --rm --platform=linux/amd64 \
    -v "${REPO_ROOT}":/work -w /work \
    ubuntu:22.04 bash -lc '
set -e
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  ca-certificates git build-essential g++-11 cmake ninja-build file

rm -rf build-ubuntu22-v3

cmake -S . -B build-ubuntu22-v3 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=g++-11 \
  -DCMAKE_CXX_FLAGS="-march=x86-64-v3 -mtune=generic" \
  -DCMAKE_EXE_LINKER_FLAGS="-static-libstdc++ -static-libgcc"

cmake --build build-ubuntu22-v3 --target mlsys -j8

cp build-ubuntu22-v3/mlsys ./mlsys
strip ./mlsys
chmod +x ./mlsys

file ./mlsys
ldd ./mlsys || true
'
