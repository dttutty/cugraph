/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/edge_property.hpp>
#include <cugraph/export.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/src_dst_lookup_container.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <tuple>

namespace CUGRAPH_EXPORT cugraph {

template <typename vertex_t, typename edge_t, typename edge_type_t, bool multi_gpu>
lookup_container_t<edge_t, edge_type_t, vertex_t> build_edge_id_and_type_to_src_dst_lookup_map(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  edge_property_view_t<edge_t, edge_t const*> edge_id_view,
  edge_property_view_t<edge_t, edge_type_t const*> edge_type_view);

template <typename vertex_t, typename edge_t, typename edge_type_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>
lookup_endpoints_from_edge_ids_and_single_type(
  raft::handle_t const& handle,
  lookup_container_t<edge_t, edge_type_t, vertex_t> const& lookup_container,
  raft::device_span<edge_t const> edge_ids_to_lookup,
  edge_type_t edge_type_to_lookup);

template <typename vertex_t, typename edge_t, typename edge_type_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>
lookup_endpoints_from_edge_ids_and_types(
  raft::handle_t const& handle,
  lookup_container_t<edge_t, edge_type_t, vertex_t> const& lookup_container,
  raft::device_span<edge_t const> edge_ids_to_lookup,
  raft::device_span<edge_type_t const> edge_types_to_lookup);

}  // namespace CUGRAPH_EXPORT cugraph
