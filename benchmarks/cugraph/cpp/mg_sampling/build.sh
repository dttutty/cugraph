#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, Lei Zhao
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../../.." && pwd)"
build_dir=${BENCH_BUILD_DIR:-${script_dir}/build}
cugraph_dir=${CUGRAPH_DIR:-${repo_root}/cpp/build}

if [[ ! -f "${cugraph_dir}/cugraph-config.cmake" ]]; then
  echo "cuGraph CMake export not found: ${cugraph_dir}/cugraph-config.cmake" >&2
  echo "Build cuGraph first, or set CUGRAPH_DIR to a build/install CMake package directory." >&2
  exit 2
fi

cmake -S "${script_dir}" -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}" \
  -Dcugraph_DIR="${cugraph_dir}" \
  "$@"
cmake --build "${build_dir}" --parallel "${BUILD_JOBS:-4}"

printf '%s\n' "${build_dir}/mg_sampling_bench"
