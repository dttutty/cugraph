# Multi-GPU sampling communication benchmark

This directory contains a two-rank synthetic benchmark for the communication
paths used by homogeneous uniform neighbor sampling. It was added alongside the
`sampling-comm-optimization` branch so the optimization and its validation tool
live in the same repository.

The executable uses the public libcugraph C API and can therefore be built
against either a baseline or candidate cuGraph CMake export without modifying
the source. Graph construction is outside the measured interval. Each measured
iteration covers the sampling API call through `cudaDeviceSynchronize()`.

## Build

First build the cuGraph checkout that should be measured. Its build directory
must contain `cugraph-config.cmake`. Then build the benchmark:

```bash
CUGRAPH_DIR=/path/to/cugraph/cpp/build \
BENCH_BUILD_DIR=/tmp/mg-sampling-candidate \
./benchmarks/cugraph/cpp/mg_sampling/build.sh
```

To compare two revisions, use separate Git worktrees and build directories:

```bash
CUGRAPH_DIR=/path/to/baseline/cpp/build \
BENCH_BUILD_DIR=/tmp/mg-sampling-baseline \
./benchmarks/cugraph/cpp/mg_sampling/build.sh

CUGRAPH_DIR=/path/to/candidate/cpp/build \
BENCH_BUILD_DIR=/tmp/mg-sampling-candidate \
./benchmarks/cugraph/cpp/mg_sampling/build.sh
```

`CMAKE_PREFIX_PATH` should name the RAPIDS environment used to build cuGraph.
The CMake targets exported by cuGraph supply the matching RAFT, RMM, CCCL, and
other transitive dependencies; the benchmark does not hard-code their include
or library paths. Use the same C++ compiler and CUDA toolkit as the cuGraph
build; mixing a system compiler with a Conda-built cuGraph can produce C++ ABI
or CCCL errors.

## Local two-GPU run

`run_local_pair.sh` starts one process on each of two local GPUs. The default
timeout is 1,200 seconds.

```bash
BENCH_BINARY=/tmp/mg-sampling-candidate/mg_sampling_bench \
RUN_LABEL=candidate-edge-ids \
./benchmarks/cugraph/cpp/mg_sampling/run_local_pair.sh \
  --vertices=1000000 \
  --degree=32 \
  --seeds-per-rank=4096 \
  --fanout=30,30 \
  --warmup=2 \
  --iters=5 \
  --edge-ids=true
```

Set `RANK0_CUDA_VISIBLE_DEVICES` and `RANK1_CUDA_VISIBLE_DEVICES` to change the
GPU assignment. `PORT`, `TIMEOUT_SECONDS`, `LOG_ROOT`, `NCCL_DEBUG`, and
`NCCL_SOCKET_IFNAME` are also configurable. If `BENCH_BINARY` was built from a
different worktree, set `CUGRAPH_GIT_SHA` to that worktree's exact commit; the
launcher also records the binary SHA-256 digest.

## Dual-host run

Build and deploy the matching benchmark binary and runtime libraries on both
hosts. The launcher deliberately requires the host-specific network settings
instead of embedding machine names or addresses:

```bash
REMOTE_HOST=user@rank1-host \
MASTER_ADDR=192.0.2.10 \
NCCL_SOCKET_IFNAME=ens6f0 \
BENCH_BINARY=/path/on/rank0/mg_sampling_bench \
REMOTE_BINARY=/path/on/rank1/mg_sampling_bench \
RUN_LABEL=candidate-edge-ids \
./benchmarks/cugraph/cpp/mg_sampling/run_dual.sh \
  --vertices=1000000 \
  --degree=32 \
  --seeds-per-rank=4096 \
  --fanout=30,30 \
  --warmup=2 \
  --iters=5 \
  --edge-ids=true
```

Optional `LOCAL_LD_LIBRARY_PATH` and `REMOTE_LD_LIBRARY_PATH` values are passed
only to the benchmark processes. Set `NCCL_IB_DISABLE` explicitly when a TCP
comparison is intended; leaving it unset allows NCCL to select the configured
transport. Set `CUGRAPH_GIT_SHA` to the exact revision used for both deployed
binaries.

The launcher uses SSH only to start rank 1. It applies a 20-minute timeout by
default and terminates its local SSH child when rank 0 fails.

## Validate and compare logs

Both launchers write `meta.txt`, `rank0.log`, and `rank1.log`. The parser rejects
incomplete runs, duplicate iterations, mismatched rank iteration sets, and
changing output sizes.

```bash
uv run --no-project python \
  benchmarks/cugraph/cpp/mg_sampling/parse_logs.py \
  --compare \
  /path/to/baseline-logdir \
  /path/to/candidate-logdir
```

The reported iteration time is `max(rank0, rank1)`, which represents the
two-rank critical path. A positive `time_change_percent` means the candidate is
slower; a speedup greater than one means it is faster.

## Benchmark arguments

Run `mg_sampling_bench --help` for the compact usage text. Important arguments
include:

- `--vertices`, `--degree`, and `--seeds-per-rank` for the synthetic workload;
- `--fanout=15,15` for per-hop fanout;
- `--warmup` and `--iters` for timing;
- `--edge-ids=true` to exercise the edge-property return path;
- `--renumber`, `--return-hops`, `--label-offsets`, and
  `--with-replacement` for sampling options.

See [RESULTS.md](RESULTS.md) for the historical measurements that motivated
moving this benchmark into the branch. Those measurements are evidence for the
optimization direction, not a reproducible validation of the current commit,
because their exact Git SHAs were not recorded.
