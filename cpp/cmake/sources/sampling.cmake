# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

include("${CMAKE_CURRENT_LIST_DIR}/_helpers.cmake")

set(CUGRAPH_SAMPLING_SOURCE_PATTERNS
    "^src/sampling/"
)

cugraph_filter_sources(CUGRAPH_SAMPLING_SOURCES CUGRAPH_NATIVE_SOURCES
                       ${CUGRAPH_SAMPLING_SOURCE_PATTERNS})
cugraph_source_group("component_sampling" CUGRAPH_SAMPLING_SOURCES)
