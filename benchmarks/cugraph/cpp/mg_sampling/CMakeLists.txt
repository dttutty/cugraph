# =============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2026, Lei Zhao
# SPDX-License-Identifier: Apache-2.0
# =============================================================================

cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

project(cugraph_mg_sampling_benchmark LANGUAGES CXX)

find_package(CUDAToolkit REQUIRED)
find_package(cugraph CONFIG REQUIRED)

find_path(NCCL_INCLUDE_DIR nccl.h REQUIRED)
find_library(NCCL_LIBRARY NAMES nccl REQUIRED)
find_library(RAFT_LIBRARY NAMES raft REQUIRED)
find_library(RAPIDS_LOGGER_LIBRARY NAMES rapids_logger REQUIRED)
find_library(CUVS_LIBRARY NAMES cuvs REQUIRED)
find_library(UCXX_LIBRARY NAMES ucxx REQUIRED)
find_library(UCP_LIBRARY NAMES ucp REQUIRED)
find_library(UCS_LIBRARY NAMES ucs REQUIRED)
find_library(UCT_LIBRARY NAMES uct REQUIRED)

add_executable(mg_sampling_bench mg_sampling_bench.cpp)
target_compile_features(mg_sampling_bench PRIVATE cxx_std_20)
target_include_directories(mg_sampling_bench PRIVATE "${NCCL_INCLUDE_DIR}")
target_link_libraries(
    mg_sampling_bench
    PRIVATE
        cugraph::cugraph
        cugraph::cugraph_c
        CUDA::cudart
        CUDA::curand
        "${NCCL_LIBRARY}"
        "${RAFT_LIBRARY}"
        "${RAPIDS_LOGGER_LIBRARY}"
        "${CUVS_LIBRARY}"
        "${UCXX_LIBRARY}"
        "${UCP_LIBRARY}"
        "${UCS_LIBRARY}"
        "${UCT_LIBRARY}"
)

set_target_properties(
    mg_sampling_bench
    PROPERTIES
        BUILD_RPATH_USE_ORIGIN TRUE
        INSTALL_RPATH_USE_LINK_PATH TRUE
)
