#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, Lei Zhao
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../../.." && pwd)"
binary=${BENCH_BINARY:-${script_dir}/build/mg_sampling_bench}
port=${PORT:-39001}
timeout_seconds=${TIMEOUT_SECONDS:-1200}
run_label=${RUN_LABEL:-local}
log_root=${LOG_ROOT:-${script_dir}/logs}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
log_dir=${log_root}/${run_label}_${stamp}

if [[ ! -x "${binary}" ]]; then
  echo "Benchmark binary is not executable: ${binary}" >&2
  exit 2
fi

mkdir -p "${log_dir}"

benchmark_args=(
  --world-size=2
  --master-addr=127.0.0.1
  --port="${port}"
  --vertices=200000
  --degree=16
  --seeds-per-rank=512
  --fanout=15,15
  --warmup=1
  --iters=3
  "$@"
)

rank0_pid=
cleanup() {
  if [[ -n "${rank0_pid}" ]] && kill -0 "${rank0_pid}" 2>/dev/null; then
    kill "${rank0_pid}" 2>/dev/null || true
    wait "${rank0_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

git_sha=${CUGRAPH_GIT_SHA:-$(git -C "${repo_root}" rev-parse HEAD)}
binary_sha256=$(sha256sum "${binary}" | awk '{print $1}')
printf 'cugraph_git_sha=%s\nbinary=%s\nbinary_sha256=%s\nargs=' \
  "${git_sha}" "${binary}" "${binary_sha256}" >"${log_dir}/meta.txt"
printf '%q ' "${benchmark_args[@]}" >>"${log_dir}/meta.txt"
printf '\n' >>"${log_dir}/meta.txt"

CUDA_VISIBLE_DEVICES="${RANK0_CUDA_VISIBLE_DEVICES:-0}" \
NCCL_DEBUG="${NCCL_DEBUG:-WARN}" \
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}" \
timeout "${timeout_seconds}s" "${binary}" --rank=0 --device=0 "${benchmark_args[@]}" \
  >"${log_dir}/rank0.log" 2>&1 &
rank0_pid=$!

CUDA_VISIBLE_DEVICES="${RANK1_CUDA_VISIBLE_DEVICES:-1}" \
NCCL_DEBUG="${NCCL_DEBUG:-WARN}" \
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}" \
timeout "${timeout_seconds}s" "${binary}" --rank=1 --device=0 "${benchmark_args[@]}" \
  >"${log_dir}/rank1.log" 2>&1

wait "${rank0_pid}"
rank0_pid=
trap - EXIT INT TERM
printf '%s\n' "${log_dir}"
