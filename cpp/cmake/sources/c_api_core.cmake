# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

include("${CMAKE_CURRENT_LIST_DIR}/_helpers.cmake")

set(CUGRAPH_C_API_CORE_SOURCE_PATTERNS
    "^src/c_api/allgather\\.cpp$"
    "^src/c_api/array\\.cpp$"
    "^src/c_api/capi_helper\\.cu$"
    "^src/c_api/decompress_to_edgelist\\.cpp$"
    "^src/c_api/edgelist\\.cpp$"
    "^src/c_api/error\\.cpp$"
    "^src/c_api/extract_vertex_list\\.cpp$"
    "^src/c_api/graph_functions\\.cpp$"
    "^src/c_api/graph_helper_mg\\.cu$"
    "^src/c_api/graph_helper_sg\\.cu$"
    "^src/c_api/graph_mg\\.cpp$"
    "^src/c_api/graph_sg\\.cpp$"
    "^src/c_api/random\\.cpp$"
    "^src/c_api/renumber_arbitrary_edgelist\\.cu$"
    "^src/c_api/resource_handle\\.cpp$"
)

cugraph_filter_sources(CUGRAPH_C_API_CORE_SOURCES CUGRAPH_C_API_SOURCES
                       ${CUGRAPH_C_API_CORE_SOURCE_PATTERNS})
cugraph_source_group("component_c_api_core" CUGRAPH_C_API_CORE_SOURCES)
