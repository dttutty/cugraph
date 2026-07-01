# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

function(cugraph_filter_sources out_var all_sources_var)
  set(filtered)
  foreach(source IN LISTS ${all_sources_var})
    foreach(pattern IN LISTS ARGN)
      if(source MATCHES "${pattern}")
        list(APPEND filtered "${source}")
        break()
      endif()
    endforeach()
  endforeach()
  set(${out_var} "${filtered}" PARENT_SCOPE)
endfunction()

function(cugraph_source_group group_name source_list_var)
  set(sources ${${source_list_var}})
  if(sources)
    source_group("${group_name}" FILES ${sources})
  endif()
endfunction()
