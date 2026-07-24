# Historical two-host results

These measurements were captured on 2026-07-01 using the predecessor of this
benchmark. They compare the then-current `main` build with the legacy
`sampling-comm-reduction-stage2` candidate from which the
`sampling-comm-optimization` changes were migrated.

The preserved per-rank, per-iteration measurements are available in
[`historical_results_2026-07-01.json`](historical_results_2026-07-01.json).

For every measured iteration, the value below is the maximum elapsed time of
rank 0 and rank 1. The mean is then taken across iterations. Negative time
change means the candidate was faster.

| Vertices | Degree | Seeds/rank | Fanout | Edge IDs | Warmup/iters | Main mean (s) | Candidate mean (s) | Speedup | Time change |
|---:|---:|---:|:---:|:---:|:---:|---:|---:|---:|---:|
| 200,000 | 32 | 1,024 | 15,15 | no | 2/5 | 0.075743620 | 0.073149120 | 1.035469x | -3.425371% |
| 1,000,000 | 32 | 4,096 | 30,30 | no | 2/5 | 0.848810400 | 0.847386200 | 1.001681x | -0.167788% |
| 1,000 | 4 | 16 | 2,2 | yes | 0/1 | 0.236223000 | 0.214049000 | 1.103593x | -9.386893% |
| 1,000,000 | 32 | 4,096 | 30,30 | yes | 2/5 | 1.737094000 | 0.662484800 | 2.622089x | -61.862467% |
| 200,000 | 32 | 1,024 | 15,15 | yes | 0/1 | 0.337478000 | 0.354117000 | 0.953013x | +4.930395% |

All five comparison pairs had complete rank-0 and rank-1 iterations and `done`
events. Baseline and candidate output sizes matched for every pair. The 1M
vertex cases produced 7,618,560 sampled edges per iteration; the 200K cases
produced 491,520; and the 1K smoke case produced 192. Only array sizes were
checked—the sampled endpoints and edge-ID values were not compared.

The strongest signal is the 1M-vertex edge-ID case: five steady-state
iterations show a 2.62x speedup. The equivalent workload without edge IDs is
effectively unchanged. The two one-iteration, zero-warmup cases include
first-call overhead and must be treated as smoke diagnostics, not performance
conclusions. The 200K edge-ID diagnostic also used verbose NCCL logging, which
may have distorted its timing.

## Environment and reproducibility limits

- The run used two hosts with one GPU process per host and world size 2. Rank 0
  was `markov`; rank 1 was `sun`; the master address was `10.102.160.93`.
- `renumber`, `return_hops`, and label offsets were enabled;
  `with_replacement` was disabled.
- The launcher set `NCCL_SOCKET_IFNAME=enp` and `NCCL_IB_DISABLE=1`. These are
  socket/TCP measurements, not IB or RoCE results.
- Logs report NCCL 2.30.7 with CUDA 12.9 and Git version `fe96036-dirty`.
- The build used g++ 11 and a local `cugraph-gnn-2606-profile` environment.
  GPU model, driver, CPU, exact NIC, OS, and power/clock state were not recorded.
- Exact baseline and candidate Git SHAs were not recorded. Directory labels
  such as `main` and `refactor` are not sufficient to reproduce a revision.
- Each configuration was run in a single process-level trial, with baseline and
  candidate run sequentially rather than interleaved or randomized.
- The graph was generated synthetically and deterministically. Timing excludes
  graph construction and cannot be extrapolated directly to end-to-end training.

Future results should record both Git SHAs, the complete hardware/software
inventory, network transport, raw per-iteration values, and multiple
process-level repetitions.
