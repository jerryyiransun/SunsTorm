# MLSys 2026 Google Graph Scheduling Competition
![Build Status](https://github.com/jerryyiransun/MLSys2026-Google-Graph-Scheduling-Competition/actions/workflows/ci.yaml/badge.svg)

This repository contains our C++ scheduling algorithm for Track A of the MLSys 2026 competition.

## Prerequisites

- **CMake** (Version 3.14 or higher)
- **C++ Compiler** supporting C++17

## How to Build

Open your terminal at the project root and run:

```bash
# Generate the build files
cmake -B build

# or Build in DEBUG mode
cmake -B build -DCMAKE_BUILD_TYPE=Debug

# Compile the project
cmake --build build
```

## How to Run

```bash
./build/mlsys path/to/input.json path/to/output.json

# example
./build/mlsys benchmarks/mlsys-2026-1.json out.json
```
