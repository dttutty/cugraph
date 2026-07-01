/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/export.hpp>

#include <thrust/scatter.h>

namespace CUGRAPH_EXPORT cugraph {

/**
 * @brief Scatter [input_first, input_last) to @p output_first using @p map_first.
 *
 * Compatibility wrapper for the historical split thrust-wrapper header path.
 */
struct scatter_t {
  template <typename ExecutionPolicy,
            typename InputIterator,
            typename MapIterator,
            typename OutputIterator>
  void operator()(ExecutionPolicy const& policy,
                  InputIterator input_first,
                  InputIterator input_last,
                  MapIterator map_first,
                  OutputIterator output_first) const
  {
    thrust::scatter(policy, input_first, input_last, map_first, output_first);
  }
};

inline constexpr scatter_t scatter{};

}  // namespace CUGRAPH_EXPORT cugraph
