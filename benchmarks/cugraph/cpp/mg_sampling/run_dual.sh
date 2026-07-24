#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, Lei Zhao
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../../.." && pwd)"
remote_host=${REMOTE_HOST:?Set REMOTE_HOST to the rank-1 SSH destination}
master_addr=${MASTER_ADDR:?Set MASTER_ADDR to the rank-0 address reachable by rank 1}
local_binary=${BENCH_BINARY:-${script_dir}/build/mg_sampling_bench}
remote_binary=${REMOTE_BINARY:?Set REMOTE_BINARY to the rank-1 benchmark path}
port=${PORT:-39001}
timeout_seconds=${TIMEOUT_SECONDS:-1200}
socket_ifname=${NCCL_SOCKET_IFNAME:-}
run_label=${RUN_LABEL:-dual}
log_root=${LOG_ROOT:-${script_dir}/logs}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
log_dir=${log_root}/${run_label}_${stamp}

if [[ ! -x "${local_binary}" ]]; then
  echo "Local benchmark binary is not executable: ${local_binary}" >&2
  exit 2
fi
if [[ -z "${socket_ifname}" ]]; then
  echo "Set NCCL_SOCKET_IFNAME explicitly for a dual-host run." >&2
  exit 2
fi

mkdir -p "${log_dir}"
benchmark_args=(
  --world-size=2
  --master-addr="${master_addr}"
  --port="${port}"
  --vertices=200000
  --degree=16
  --seeds-per-rank=512
  --fanout=15,15
  --warmup=1
  --iters=3
  "$@"
)

common_env=(
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
  NCCL_SOCKET_IFNAME="${socket_ifname}"
)
if [[ -n "${NCCL_IB_DISABLE+x}" ]]; then
  common_env+=(NCCL_IB_DISABLE="${NCCL_IB_DISABLE}")
fi

remote_env=("${common_env[@]}")
local_env=("${common_env[@]}")
if [[ -n "${REMOTE_LD_LIBRARY_PATH:-}" ]]; then
  remote_env+=(LD_LIBRARY_PATH="${REMOTE_LD_LIBRARY_PATH}")
fi
if [[ -n "${LOCAL_LD_LIBRARY_PATH:-}" ]]; then
  local_env+=(LD_LIBRARY_PATH="${LOCAL_LD_LIBRARY_PATH}")
fi

remote_command=(
  timeout "${timeout_seconds}s"
  env "${remote_env[@]}"
  "${remote_binary}"
  --rank=1
  --device=0
  "${benchmark_args[@]}"
)
printf -v remote_command_quoted '%q ' "${remote_command[@]}"

remote_pid=
cleanup() {
  if [[ -n "${remote_pid}" ]] && kill -0 "${remote_pid}" 2>/dev/null; then
    kill "${remote_pid}" 2>/dev/null || true
    wait "${remote_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

{
  git_sha=${CUGRAPH_GIT_SHA:-$(git -C "${repo_root}" rev-parse HEAD)}
  local_binary_sha256=$(sha256sum "${local_binary}" | awk '{print $1}')
  printf 'cugraph_git_sha=%s\nlocal_binary=%s\nlocal_binary_sha256=%s\n' \
    "${git_sha}" "${local_binary}" "${local_binary_sha256}"
  printf 'remote_host=%s\nremote_binary=%s\n' "${remote_host}" "${remote_binary}"
  printf 'master_addr=%s\nport=%s\nargs=' "${master_addr}" "${port}"
  printf '%q ' "${benchmark_args[@]}"
  printf '\n'
} >"${log_dir}/meta.txt"

ssh "${remote_host}" "${remote_command_quoted}" >"${log_dir}/rank1.log" 2>&1 &
remote_pid=$!

timeout "${timeout_seconds}s" env "${local_env[@]}" \
  "${local_binary}" --rank=0 --device=0 "${benchmark_args[@]}" \
  >"${log_dir}/rank0.log" 2>&1

wait "${remote_pid}"
remote_pid=
trap - EXIT INT TERM
printf '%s\n' "${log_dir}"
