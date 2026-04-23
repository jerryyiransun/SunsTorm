# Profiling

## Build

```bash
# Profiling build
cmake -B build-relwithdebinfo -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-relwithdebinfo
```

## Profiling with `perf`

Install `perf` on Ubuntu:

```bash
sudo apt update
sudo apt install linux-tools-common linux-tools-generic
```

Record a profile for the optimized, symbolized binary:

```bash
perf record -g -- ./build-relwithdebinfo/mlsys benchmarks/mlsys-2026-1.json out.json
```

Inspect the resulting profile:

```bash
perf report
```

### `perf` on WSL2

On WSL2, the usual `linux-tools-<kernel>` package often does not exist for the
Microsoft kernel, even when the `perf` launcher suggests it. In that case, build
`perf` from the WSL kernel source and use that binary directly.

Install the build dependencies:

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  flex \
  bison \
  pkg-config \
  python3-dev \
  libssl-dev \
  libelf-dev \
  libtraceevent-dev \
  libdw-dev \
  libdebuginfod-dev \
  libunwind-dev \
  libslang2-dev \
  libperl-dev \
  liblzma-dev \
  libzstd-dev \
  libcap-dev \
  libnuma-dev \
  libbabeltrace-dev \
  libpfm4-dev \
  systemtap-sdt-dev
```

Clone the WSL kernel source and build `perf`:

```bash
mkdir -p ~/tools
git clone --depth=1 https://github.com/microsoft/WSL2-Linux-Kernel.git ~/tools/WSL2-Linux-Kernel
cd ~/tools/WSL2-Linux-Kernel/tools/perf
make
```

Confirm the binary exists:

```bash
~/tools/WSL2-Linux-Kernel/tools/perf/perf --version
```

Add it to your shell `PATH`:

```bash
echo 'export PATH="$HOME/tools/WSL2-Linux-Kernel/tools/perf:$PATH"' >> ~/.bashrc
source ~/.bashrc
which perf
perf --version
```

Then profile `mlsys` as usual:

```bash
perf record -g -- ./build-relwithdebinfo/mlsys benchmarks/mlsys-2026-1.json out.json
perf report
```

WSL2 notes:

- Some hardware counters may still be unavailable under WSL2.
- `perf record -g` stack sampling is still the recommended first step for
  identifying hotspots in `mlsys`.
- If `make` fails because a library is missing, install the package named in the
  error and rerun `make`.
