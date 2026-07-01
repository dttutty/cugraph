# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

include("${CMAKE_CURRENT_LIST_DIR}/_helpers.cmake")

set(CUGRAPH_CORE_SOURCE_PATTERNS
    "^src/converters/"
    "^src/detail/"
    "^src/lookup/"
    "^src/mtmg/"
    "^src/structure/"
    "^src/utilities/"
)

cugraph_filter_sources(CUGRAPH_CORE_SOURCES CUGRAPH_NATIVE_SOURCES
                       ${CUGRAPH_CORE_SOURCE_PATTERNS})
cugraph_source_group("component_core" CUGRAPH_CORE_SOURCES)
