/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "detail/shuffle_wrappers.hpp"

#include <cugraph/arithmetic_variant_types.hpp>
#include <cugraph/export.hpp>
#include <cugraph/large_buffer_manager.hpp>
#include <cugraph/utilities/shuffle_comm.cuh>
#include <cugraph/utilities/thrust_wrappers.hpp>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/adjacent_difference.h>
#include <thrust/binary_search.h>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <optional>
#include <vector>

namespace cugraph {

namespace {

template <typename T0, typename T1, typename T2>
struct packed_record_3_t {
  T0 v0;
  T1 v1;
  T2 v2;
};

template <typename T0, typename T1, typename T2, typename T3>
struct packed_record_4_t {
  T0 v0;
  T1 v1;
  T2 v2;
  T3 v3;
};

template <typename T>
auto gather_property_by_position(raft::handle_t const& handle,
                                 rmm::device_uvector<T> const& property,
                                 rmm::device_uvector<size_t> const& property_positions,
                                 std::optional<large_buffer_type_t> large_buffer_type)
{
  auto tmp = large_buffer_type ? large_buffer_manager::allocate_memory_buffer<T>(
                                   property.size(), handle.get_stream())
                               : rmm::device_uvector<T>(property.size(), handle.get_stream());
  thrust::gather(handle.get_thrust_policy(),
                 property_positions.begin(),
                 property_positions.end(),
                 property.begin(),
                 tmp.begin());
  return tmp;
}

template <typename record_t>
rmm::device_uvector<record_t> shuffle_property_records(
  raft::handle_t const& handle,
  rmm::device_uvector<record_t> const& tx_records,
  rmm::device_uvector<size_t> const& d_tx_counts)
{
  auto [tx_counts, tx_displs, tx_dst_ranks, rx_counts, rx_displs, rx_src_ranks] =
    cugraph::detail::compute_tx_rx_counts_displs_ranks(
      handle.get_comms(),
      raft::device_span<size_t const>(d_tx_counts.data(), d_tx_counts.size()),
      false,
      handle.get_stream());

  auto rx_buffer_size = rx_displs.size() > 0 ? rx_displs.back() + rx_counts.back() : size_t{0};
  rmm::device_uvector<record_t> rx_records(rx_buffer_size, handle.get_stream());

  handle.get_comms().device_multicast_sendrecv(tx_records.data(),
                                               tx_counts,
                                               tx_displs,
                                               tx_dst_ranks,
                                               rx_records.data(),
                                               rx_counts,
                                               rx_displs,
                                               rx_src_ranks,
                                               handle.get_stream().value());

  return rx_records;
}

template <typename vertex_t>
bool try_shuffle_sampled_edges_without_property(
  raft::handle_t const& handle,
  std::vector<arithmetic_device_uvector_t>& properties,
  rmm::device_uvector<size_t> const& property_positions,
  rmm::device_uvector<size_t> const& tx_counts,
  std::optional<large_buffer_type_t> large_buffer_type)
{
  if (large_buffer_type) { return false; }
  if ((properties.size() != 3) ||
      !std::holds_alternative<rmm::device_uvector<vertex_t>>(properties[0]) ||
      !std::holds_alternative<rmm::device_uvector<vertex_t>>(properties[1]) ||
      !std::holds_alternative<rmm::device_uvector<size_t>>(properties[2])) {
    return false;
  }

  auto& majors             = std::get<rmm::device_uvector<vertex_t>>(properties[0]);
  auto& minors             = std::get<rmm::device_uvector<vertex_t>>(properties[1]);
  auto& original_positions = std::get<rmm::device_uvector<size_t>>(properties[2]);

  auto tmp_majors             = gather_property_by_position(handle, majors, property_positions, std::nullopt);
  auto tmp_minors             = gather_property_by_position(handle, minors, property_positions, std::nullopt);
  auto tmp_original_positions = gather_property_by_position(
    handle, original_positions, property_positions, std::nullopt);

  using record_t = packed_record_3_t<vertex_t, vertex_t, size_t>;
  rmm::device_uvector<record_t> tx_records(majors.size(), handle.get_stream());
  thrust::for_each(
    handle.get_thrust_policy(),
    thrust::make_counting_iterator(size_t{0}),
    thrust::make_counting_iterator(majors.size()),
    [major_first = tmp_majors.data(),
     minor_first = tmp_minors.data(),
     original_position_first = tmp_original_positions.data(),
     tx_record_first = tx_records.data()] __device__(auto i) {
      tx_record_first[i] =
        record_t{major_first[i], minor_first[i], original_position_first[i]};
    });

  auto rx_records = shuffle_property_records(handle, tx_records, tx_counts);

  rmm::device_uvector<vertex_t> rx_majors(rx_records.size(), handle.get_stream());
  rmm::device_uvector<vertex_t> rx_minors(rx_records.size(), handle.get_stream());
  rmm::device_uvector<size_t> rx_original_positions(rx_records.size(), handle.get_stream());
  thrust::for_each(
    handle.get_thrust_policy(),
    thrust::make_counting_iterator(size_t{0}),
    thrust::make_counting_iterator(rx_records.size()),
    [rx_record_first = rx_records.data(),
     major_first = rx_majors.data(),
     minor_first = rx_minors.data(),
     original_position_first = rx_original_positions.data()] __device__(auto i) {
      auto record                 = rx_record_first[i];
      major_first[i]              = record.v0;
      minor_first[i]              = record.v1;
      original_position_first[i] = record.v2;
    });

  majors             = std::move(rx_majors);
  minors             = std::move(rx_minors);
  original_positions = std::move(rx_original_positions);

  return true;
}

template <typename vertex_t, typename property_t>
void shuffle_sampled_edges_with_one_property(
  raft::handle_t const& handle,
  rmm::device_uvector<property_t>& property,
  rmm::device_uvector<vertex_t>& majors,
  rmm::device_uvector<vertex_t>& minors,
  rmm::device_uvector<size_t>& original_positions,
  rmm::device_uvector<size_t> const& property_positions,
  rmm::device_uvector<size_t> const& tx_counts)
{
  auto tmp_property = gather_property_by_position(handle, property, property_positions, std::nullopt);
  auto tmp_majors   = gather_property_by_position(handle, majors, property_positions, std::nullopt);
  auto tmp_minors   = gather_property_by_position(handle, minors, property_positions, std::nullopt);
  auto tmp_original_positions = gather_property_by_position(
    handle, original_positions, property_positions, std::nullopt);

  using record_t = packed_record_4_t<property_t, vertex_t, vertex_t, size_t>;
  rmm::device_uvector<record_t> tx_records(property.size(), handle.get_stream());
  thrust::for_each(
    handle.get_thrust_policy(),
    thrust::make_counting_iterator(size_t{0}),
    thrust::make_counting_iterator(property.size()),
    [property_first = tmp_property.data(),
     major_first = tmp_majors.data(),
     minor_first = tmp_minors.data(),
     original_position_first = tmp_original_positions.data(),
     tx_record_first = tx_records.data()] __device__(auto i) {
      tx_record_first[i] =
        record_t{property_first[i], major_first[i], minor_first[i], original_position_first[i]};
    });

  auto rx_records = shuffle_property_records(handle, tx_records, tx_counts);

  rmm::device_uvector<property_t> rx_property(rx_records.size(), handle.get_stream());
  rmm::device_uvector<vertex_t> rx_majors(rx_records.size(), handle.get_stream());
  rmm::device_uvector<vertex_t> rx_minors(rx_records.size(), handle.get_stream());
  rmm::device_uvector<size_t> rx_original_positions(rx_records.size(), handle.get_stream());
  thrust::for_each(
    handle.get_thrust_policy(),
    thrust::make_counting_iterator(size_t{0}),
    thrust::make_counting_iterator(rx_records.size()),
    [rx_record_first = rx_records.data(),
     property_first = rx_property.data(),
     major_first = rx_majors.data(),
     minor_first = rx_minors.data(),
     original_position_first = rx_original_positions.data()] __device__(auto i) {
      auto record                 = rx_record_first[i];
      property_first[i]           = record.v0;
      major_first[i]              = record.v1;
      minor_first[i]              = record.v2;
      original_position_first[i] = record.v3;
    });

  property           = std::move(rx_property);
  majors             = std::move(rx_majors);
  minors             = std::move(rx_minors);
  original_positions = std::move(rx_original_positions);
}

template <typename vertex_t>
bool try_shuffle_sampled_edges_with_one_property(
  raft::handle_t const& handle,
  std::vector<arithmetic_device_uvector_t>& properties,
  rmm::device_uvector<size_t> const& property_positions,
  rmm::device_uvector<size_t> const& tx_counts,
  std::optional<large_buffer_type_t> large_buffer_type)
{
  if (large_buffer_type) { return false; }
  if ((properties.size() != 4) ||
      !std::holds_alternative<rmm::device_uvector<vertex_t>>(properties[1]) ||
      !std::holds_alternative<rmm::device_uvector<vertex_t>>(properties[2]) ||
      !std::holds_alternative<rmm::device_uvector<size_t>>(properties[3])) {
    return false;
  }

  cugraph::variant_type_dispatch(properties[0], [&](auto& property) {
    auto& majors             = std::get<rmm::device_uvector<vertex_t>>(properties[1]);
    auto& minors             = std::get<rmm::device_uvector<vertex_t>>(properties[2]);
    auto& original_positions = std::get<rmm::device_uvector<size_t>>(properties[3]);
    shuffle_sampled_edges_with_one_property(
      handle, property, majors, minors, original_positions, property_positions, tx_counts);
  });

  return true;
}

bool try_shuffle_sampled_edges(
  raft::handle_t const& handle,
  std::vector<arithmetic_device_uvector_t>& properties,
  rmm::device_uvector<size_t> const& property_positions,
  rmm::device_uvector<size_t> const& tx_counts,
  std::optional<large_buffer_type_t> large_buffer_type)
{
  return try_shuffle_sampled_edges_without_property<int32_t>(
           handle, properties, property_positions, tx_counts, large_buffer_type) ||
         try_shuffle_sampled_edges_without_property<int64_t>(
           handle, properties, property_positions, tx_counts, large_buffer_type) ||
         try_shuffle_sampled_edges_with_one_property<int32_t>(
           handle, properties, property_positions, tx_counts, large_buffer_type) ||
         try_shuffle_sampled_edges_with_one_property<int64_t>(
           handle, properties, property_positions, tx_counts, large_buffer_type);
}

}  // namespace

CUGRAPH_EXPORT std::vector<arithmetic_device_uvector_t> shuffle_properties(
  raft::handle_t const& handle,
  rmm::device_uvector<int>&& gpus,
  std::vector<arithmetic_device_uvector_t>&& properties,
  std::optional<large_buffer_type_t> large_buffer_type)
{
  auto const comm_size = handle.get_comms().get_size();

  if (properties.size() == 0) {
    gpus.resize(0, handle.get_stream());
    gpus.shrink_to_fit(handle.get_stream());
  } else if (properties.size() == 1) {
    cugraph::variant_type_dispatch(properties[0], [&handle, &gpus](auto& prop) {
      thrust::sort_by_key(handle.get_thrust_policy(), gpus.begin(), gpus.end(), prop.begin());
    });

    rmm::device_uvector<size_t> tx_counts(comm_size, handle.get_stream());
    {
      rmm::device_uvector<size_t> lasts(comm_size, handle.get_stream());
      thrust::upper_bound(handle.get_thrust_policy(),
                          gpus.begin(),
                          gpus.end(),
                          thrust::make_counting_iterator(int{0}),
                          thrust::make_counting_iterator(comm_size),
                          lasts.begin());
      gpus.resize(0, handle.get_stream());
      gpus.shrink_to_fit(handle.get_stream());
      thrust::adjacent_difference(
        handle.get_thrust_policy(), lasts.begin(), lasts.end(), tx_counts.begin());
    }

    cugraph::variant_type_dispatch(
      properties[0], [&handle, &tx_counts, large_buffer_type](auto& prop) {
        std::tie(prop, std::ignore) =
          shuffle_values(handle.get_comms(),
                         prop.begin(),
                         raft::device_span<size_t const>(tx_counts.data(), tx_counts.size()),
                         handle.get_stream(),
                         large_buffer_type);
      });
  } else {
    rmm::device_uvector<size_t> property_positions =
      large_buffer_type
        ? large_buffer_manager::allocate_memory_buffer<size_t>(gpus.size(), handle.get_stream())
        : rmm::device_uvector<size_t>(gpus.size(), handle.get_stream());
    cugraph::sequence(
      handle.get_thrust_policy(), property_positions.begin(), property_positions.end(), size_t{0});
    thrust::sort_by_key(
      handle.get_thrust_policy(), gpus.begin(), gpus.end(), property_positions.begin());

    rmm::device_uvector<size_t> tx_counts(comm_size, handle.get_stream());
    {
      rmm::device_uvector<size_t> lasts(comm_size, handle.get_stream());
      thrust::upper_bound(handle.get_thrust_policy(),
                          gpus.begin(),
                          gpus.end(),
                          thrust::make_counting_iterator(int{0}),
                          thrust::make_counting_iterator(comm_size),
                          lasts.begin());
      gpus.resize(0, handle.get_stream());
      gpus.shrink_to_fit(handle.get_stream());
      thrust::adjacent_difference(
        handle.get_thrust_policy(), lasts.begin(), lasts.end(), tx_counts.begin());
    }

    if (!try_shuffle_sampled_edges(
          handle, properties, property_positions, tx_counts, large_buffer_type)) {
      std::for_each(
        properties.begin(),
        properties.end(),
        [&handle, &property_positions, &tx_counts, large_buffer_type](auto& property) {
          cugraph::variant_type_dispatch(
            property, [&handle, &property_positions, &tx_counts, large_buffer_type](auto& prop) {
              using T = typename std::remove_reference<decltype(prop)>::type::value_type;
              auto tmp =
                gather_property_by_position(handle, prop, property_positions, large_buffer_type);
              std::tie(prop, std::ignore) =
                shuffle_values(handle.get_comms(),
                               tmp.begin(),
                               raft::device_span<size_t const>(tx_counts.data(), tx_counts.size()),
                               handle.get_stream(),
                               large_buffer_type);
            });
        });
    }
  }

  return std::move(properties);
}

}  // namespace cugraph
