/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/export.hpp>

#include <thrust/gather.h>

namespace CUGRAPH_EXPORT cugraph {

/**
 * @brief Gather from @p input_first into @p output_first using @p map_first.
 *
 * Compatibility wrapper for the historical split thrust-wrapper header path.
 */
struct gather_t {
  template <typename ExecutionPolicy,
            typename MapIterator,
            typename InputIterator,
            typename OutputIterator>
  auto operator()(ExecutionPolicy const& policy,
                  MapIterator map_first,
                  MapIterator map_last,
                  InputIterator input_first,
                  OutputIterator output_first) const
  {
    return thrust::gather(policy, map_first, map_last, input_first, output_first);
  }
};

inline constexpr gather_t gather{};

}  // namespace CUGRAPH_EXPORT cugraph
