# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

function(cugraph_append_python_package_cmake_prefixes)
  set(_cugraph_python_package_prefixes "${CMAKE_PREFIX_PATH}")
  foreach(_cugraph_python_package_prefix IN LISTS _cugraph_python_package_prefixes)
    foreach(_cugraph_python_cmake_subdir lib64/cmake lib/cmake cmake)
      if(IS_DIRECTORY "${_cugraph_python_package_prefix}/${_cugraph_python_cmake_subdir}")
        list(
          APPEND
          CMAKE_PREFIX_PATH
          "${_cugraph_python_package_prefix}/${_cugraph_python_cmake_subdir}"
        )
      endif()
    endforeach()
  endforeach()
  list(REMOVE_DUPLICATES CMAKE_PREFIX_PATH)
  set(CMAKE_PREFIX_PATH
      "${CMAKE_PREFIX_PATH}"
      PARENT_SCOPE
  )
endfunction()
