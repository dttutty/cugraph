# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

include("${CMAKE_CURRENT_LIST_DIR}/_helpers.cmake")

set(CUGRAPH_ALGORITHM_SOURCE_PATTERNS
    "^src/centrality/"
    "^src/community/"
    "^src/components/"
    "^src/cores/"
    "^src/dag/"
    "^src/generators/"
    "^src/layout/"
    "^src/linear_assignment/"
    "^src/link_analysis/"
    "^src/link_prediction/"
    "^src/traversal/"
    "^src/tree/"
)

cugraph_filter_sources(CUGRAPH_ALGORITHM_SOURCES CUGRAPH_NATIVE_SOURCES
                       ${CUGRAPH_ALGORITHM_SOURCE_PATTERNS})
cugraph_source_group("component_algorithms" CUGRAPH_ALGORITHM_SOURCES)
