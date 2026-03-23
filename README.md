# MLSys 2026 Google Graph Scheduling Competition
![Build Status](https://github.com/jerryyiransun/MLSys2026-Google-Graph-Scheduling-Competition/actions/workflows/ci.yaml/badge.svg)

This repository contains our C++ scheduling algorithm for Track A of the MLSys 2026 competition.

## Prerequisites

- **CMake** (Version 3.14 or higher)
- **C++ Compiler** supporting C++17
- **clang-format** (for pre-commit formatting)

## How to Build

Open your terminal at the project root and run:

```bash
# Generate the build files
cmake -B build

# build both debug and release mode
cmake -B build-debug -DCMAKE_BUILD_TYPE=Debug

# Compile the project
cmake --build build
```

## How to Run

```bash
./build/mlsys path/to/input.json path/to/output.json

# example
./build/mlsys benchmarks/mlsys-2026-1.json out.json
```

## Git Hooks (clang-format pre-commit)

This repository includes a versioned pre-commit hook at `.githooks/pre-commit`
that auto-formats staged `*.cpp`, `*.h`, and `*.hpp` files using `.clang-format`.

Run this once per clone from the project root:

```bash
git config core.hooksPath .githooks
```

Optional verification:

```bash
ls .githooks
```

Then make a small commit. The hook will format staged C/C++ files and re-stage
them automatically before commit.
