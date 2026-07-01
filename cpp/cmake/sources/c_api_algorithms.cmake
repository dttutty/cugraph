# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

include("${CMAKE_CURRENT_LIST_DIR}/_helpers.cmake")

set(CUGRAPH_C_API_ALGORITHM_SOURCE_PATTERNS
    "^src/c_api/betweenness_centrality\\.cpp$"
    "^src/c_api/bfs\\.cpp$"
    "^src/c_api/centrality_result\\.cpp$"
    "^src/c_api/core_number\\.cpp$"
    "^src/c_api/core_result\\.cpp$"
    "^src/c_api/degrees\\.cu$"
    "^src/c_api/degrees_result\\.cpp$"
    "^src/c_api/ecg\\.cpp$"
    "^src/c_api/eigenvector_centrality\\.cpp$"
    "^src/c_api/extract_ego\\.cpp$"
    "^src/c_api/extract_paths\\.cpp$"
    "^src/c_api/graph_generators\\.cpp$"
    "^src/c_api/hierarchical_clustering_result\\.cpp$"
    "^src/c_api/hits\\.cpp$"
    "^src/c_api/induced_subgraph\\.cpp$"
    "^src/c_api/induced_subgraph_result\\.cpp$"
    "^src/c_api/k_core\\.cpp$"
    "^src/c_api/k_truss\\.cpp$"
    "^src/c_api/katz\\.cpp$"
    "^src/c_api/labeling_result\\.cpp$"
    "^src/c_api/legacy_fa2\\.cpp$"
    "^src/c_api/legacy_mst\\.cpp$"
    "^src/c_api/legacy_spectral\\.cpp$"
    "^src/c_api/leiden\\.cpp$"
    "^src/c_api/lookup_src_dst\\.cpp$"
    "^src/c_api/louvain\\.cpp$"
    "^src/c_api/pagerank\\.cpp$"
    "^src/c_api/similarity\\.cpp$"
    "^src/c_api/sssp\\.cpp$"
    "^src/c_api/strongly_connected_components\\.cpp$"
    "^src/c_api/triangle_count\\.cpp$"
    "^src/c_api/weakly_connected_components\\.cpp$"
)

cugraph_filter_sources(CUGRAPH_C_API_ALGORITHM_SOURCES CUGRAPH_C_API_SOURCES
                       ${CUGRAPH_C_API_ALGORITHM_SOURCE_PATTERNS})
cugraph_source_group("component_c_api_algorithms" CUGRAPH_C_API_ALGORITHM_SOURCES)
