# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

include("${CMAKE_CURRENT_LIST_DIR}/_helpers.cmake")

set(CUGRAPH_C_API_SAMPLING_SOURCE_PATTERNS
    "^src/c_api/negative_sampling\\.cpp$"
    "^src/c_api/neighbor_sampling\\.cpp$"
    "^src/c_api/random_walks\\.cpp$"
    "^src/c_api/sampling_result\\.cpp$"
    "^src/c_api/temporal_neighbor_sampling\\.cpp$"
)

cugraph_filter_sources(CUGRAPH_C_API_SAMPLING_SOURCES CUGRAPH_C_API_SOURCES
                       ${CUGRAPH_C_API_SAMPLING_SOURCE_PATTERNS})
cugraph_source_group("component_c_api_sampling" CUGRAPH_C_API_SAMPLING_SOURCES)
