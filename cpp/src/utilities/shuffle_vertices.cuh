/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "detail/shuffle_wrappers.hpp"

#include <cugraph/arithmetic_variant_types.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/large_buffer_manager.hpp>
#include <cugraph/utilities/graph_partition_utils.cuh>
#include <cugraph/utilities/shuffle_comm.cuh>
#include <cugraph/utilities/thrust_wrappers.hpp>

#include <cuda/std/tuple>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>

#include <tuple>

namespace cugraph {

namespace {

enum class vertex_shuffle_mode_t { global, two_level_major, two_level_minor };

template <typename T0, typename T1>
struct packed_record_2_t {
  T0 v0;
  T1 v1;
};

template <typename T0, typename T1, typename T2>
struct packed_record_3_t {
  T0 v0;
  T1 v1;
  T2 v2;
};

template <typename record_t>
rmm::device_uvector<record_t> shuffle_vertex_records(
  raft::comms::comms_t const& comm,
  rmm::device_uvector<record_t> const& tx_records,
  raft::device_span<size_t const> d_tx_value_counts,
  rmm::cuda_stream_view stream_view)
{
  auto [tx_counts, tx_displs, tx_dst_ranks, rx_counts, rx_displs, rx_src_ranks] =
    cugraph::detail::compute_tx_rx_counts_displs_ranks(
      comm, d_tx_value_counts, false, stream_view);

  auto rx_buffer_size = rx_displs.size() > 0 ? rx_displs.back() + rx_counts.back() : size_t{0};
  rmm::device_uvector<record_t> rx_records(rx_buffer_size, stream_view);

  comm.device_multicast_sendrecv(tx_records.data(),
                                 tx_counts,
                                 tx_displs,
                                 tx_dst_ranks,
                                 rx_records.data(),
                                 rx_counts,
                                 rx_displs,
                                 rx_src_ranks,
                                 stream_view.value());

  return rx_records;
}

template <typename vertex_t, typename T>
void shuffle_vertex_and_property(raft::handle_t const& handle,
                                 raft::comms::comms_t const& comm,
                                 rmm::device_uvector<vertex_t>& vertices,
                                 rmm::device_uvector<T>& property,
                                 raft::device_span<size_t const> d_tx_value_counts)
{
  using record_t = packed_record_2_t<vertex_t, T>;

  rmm::device_uvector<record_t> tx_records(vertices.size(), handle.get_stream());
  thrust::for_each(handle.get_thrust_policy(),
                   thrust::make_counting_iterator(size_t{0}),
                   thrust::make_counting_iterator(vertices.size()),
                   [vertex_first    = vertices.data(),
                    property_first  = property.data(),
                    tx_record_first = tx_records.data()] __device__(auto i) {
                     tx_record_first[i] = record_t{vertex_first[i], property_first[i]};
                   });

  auto rx_records = shuffle_vertex_records(comm, tx_records, d_tx_value_counts, handle.get_stream());

  rmm::device_uvector<vertex_t> rx_vertices(rx_records.size(), handle.get_stream());
  rmm::device_uvector<T> rx_property(rx_records.size(), handle.get_stream());
  thrust::for_each(handle.get_thrust_policy(),
                   thrust::make_counting_iterator(size_t{0}),
                   thrust::make_counting_iterator(rx_records.size()),
                   [rx_record_first = rx_records.data(),
                    vertex_first    = rx_vertices.data(),
                    property_first  = rx_property.data()] __device__(auto i) {
                     auto record       = rx_record_first[i];
                     vertex_first[i]   = record.v0;
                     property_first[i] = record.v1;
                   });

  vertices = std::move(rx_vertices);
  property = std::move(rx_property);
}

template <typename vertex_t, typename T0, typename T1>
void shuffle_vertex_and_two_properties(raft::handle_t const& handle,
                                       raft::comms::comms_t const& comm,
                                       rmm::device_uvector<vertex_t>& vertices,
                                       rmm::device_uvector<T0>& property0,
                                       rmm::device_uvector<T1>& property1,
                                       rmm::device_uvector<size_t> const& property_positions,
                                       raft::device_span<size_t const> d_tx_value_counts)
{
  using record_t = packed_record_3_t<vertex_t, T0, T1>;

  rmm::device_uvector<T0> tmp0(property0.size(), handle.get_stream());
  rmm::device_uvector<T1> tmp1(property1.size(), handle.get_stream());
  thrust::gather(handle.get_thrust_policy(),
                 property_positions.begin(),
                 property_positions.end(),
                 property0.begin(),
                 tmp0.begin());
  thrust::gather(handle.get_thrust_policy(),
                 property_positions.begin(),
                 property_positions.end(),
                 property1.begin(),
                 tmp1.begin());

  rmm::device_uvector<record_t> tx_records(vertices.size(), handle.get_stream());
  thrust::for_each(handle.get_thrust_policy(),
                   thrust::make_counting_iterator(size_t{0}),
                   thrust::make_counting_iterator(vertices.size()),
                   [vertex_first    = vertices.data(),
                    property0_first = tmp0.data(),
                    property1_first = tmp1.data(),
                    tx_record_first = tx_records.data()] __device__(auto i) {
                     tx_record_first[i] =
                       record_t{vertex_first[i], property0_first[i], property1_first[i]};
                   });

  auto rx_records = shuffle_vertex_records(comm, tx_records, d_tx_value_counts, handle.get_stream());

  rmm::device_uvector<vertex_t> rx_vertices(rx_records.size(), handle.get_stream());
  rmm::device_uvector<T0> rx_property0(rx_records.size(), handle.get_stream());
  rmm::device_uvector<T1> rx_property1(rx_records.size(), handle.get_stream());
  thrust::for_each(handle.get_thrust_policy(),
                   thrust::make_counting_iterator(size_t{0}),
                   thrust::make_counting_iterator(rx_records.size()),
                   [rx_record_first = rx_records.data(),
                    vertex_first    = rx_vertices.data(),
                    property0_first = rx_property0.data(),
                    property1_first = rx_property1.data()] __device__(auto i) {
                     auto record         = rx_record_first[i];
                     vertex_first[i]     = record.v0;
                     property0_first[i] = record.v1;
                     property1_first[i] = record.v2;
                   });

  vertices  = std::move(rx_vertices);
  property0 = std::move(rx_property0);
  property1 = std::move(rx_property1);
}

template <typename vertex_t, typename func_t>
struct vertex_groupby_functor_t {
  func_t func_;
  int major_comm_size{};
  int minor_comm_size{};
  vertex_shuffle_mode_t shuffle_mode{};

  auto __device__ operator()(vertex_t v) const
  {
    auto comm_rank = func_(v);
    if (shuffle_mode == vertex_shuffle_mode_t::global) {
      return comm_rank;
    } else if (shuffle_mode == vertex_shuffle_mode_t::two_level_major) {
      return partition_manager::compute_major_comm_rank_from_global_comm_rank(
        major_comm_size, minor_comm_size, comm_rank);
    } else {
      assert(shuffle_mode == vertex_shuffle_mode_t::two_level_minor);
      return partition_manager::compute_minor_comm_rank_from_global_comm_rank(
        major_comm_size, minor_comm_size, comm_rank);
    }
  }
};

template <typename vertex_t, typename v_to_gpu_id_op_t>
std::tuple<rmm::device_uvector<vertex_t>, std::vector<arithmetic_device_uvector_t>>
shuffle_vertices(raft::handle_t const& handle,
                 rmm::device_uvector<vertex_t>&& vertices,
                 std::vector<arithmetic_device_uvector_t>&& vertex_properties,
                 v_to_gpu_id_op_t v_to_gpu_id_op,
                 std::optional<large_buffer_type_t> large_buffer_type)
{
  constexpr int max_global_comm_all2all_size =
    512;  // NCCL currently has limitations in supportingg All-to-All between a large number of
          // ranks, as a workaround, use a two step approach using major_comm and minor_comm

  auto& comm                 = handle.get_comms();
  auto const comm_size       = comm.get_size();
  auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
  auto const major_comm_size = major_comm.get_size();
  auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
  auto const minor_comm_size = minor_comm.get_size();

  bool two_level_shuffle = false;
  if (comm_size >
      std::max(max_global_comm_all2all_size, std::max(major_comm_size, minor_comm_size))) {
    two_level_shuffle = true;
  }

  auto element_size = sizeof(vertex_t);
  if (vertex_properties.size() == 1) {
    element_size +=
      cugraph::variant_type_dispatch(vertex_properties[0], cugraph::sizeof_arithmetic_element{});
  } else if (vertex_properties.size() > 1) {
    element_size += sizeof(size_t);
  }

  auto mem_frugal_threshold = std::numeric_limits<size_t>::max();
  if (!large_buffer_type) {
    auto total_global_mem = handle.get_device_properties().totalGlobalMem;
    auto constexpr mem_frugal_ratio =
      0.1;  // if the expected temporary buffer size exceeds the mem_frugal_ratio of the
            // total_global_mem, switch to the memory frugal approach (thrust::sort is used to
            // group-by by default, and thrust::sort requires temporary buffer comparable to the
            // input data size)
    mem_frugal_threshold =
      static_cast<size_t>(static_cast<double>(total_global_mem / element_size) * mem_frugal_ratio);
  }

  for (size_t step = 0; step < (two_level_shuffle ? size_t{2} : size_t{1}); ++step) {
    vertex_shuffle_mode_t shuffle_mode = two_level_shuffle
                                           ? (step == 0 ? vertex_shuffle_mode_t::two_level_major
                                                        : vertex_shuffle_mode_t::two_level_minor)
                                           : vertex_shuffle_mode_t::global;
    auto& this_step_comm    = two_level_shuffle ? (step == 0 ? major_comm : minor_comm) : comm;
    int this_step_comm_size = this_step_comm.get_size();

    rmm::device_uvector<size_t> d_tx_value_counts(0, handle.get_stream());
    vertex_groupby_functor_t<vertex_t, v_to_gpu_id_op_t> groupby_functor{
      v_to_gpu_id_op, major_comm_size, minor_comm_size, shuffle_mode};

    std::optional<rmm::device_uvector<size_t>> property_positions{std::nullopt};
    if (vertex_properties.size() == 0) {
      d_tx_value_counts = cugraph::groupby_and_count(vertices.begin(),
                                                     vertices.end(),
                                                     groupby_functor,
                                                     this_step_comm_size,
                                                     mem_frugal_threshold,
                                                     handle.get_stream(),
                                                     large_buffer_type);
    } else if (vertex_properties.size() == 1) {
      cugraph::variant_type_dispatch(vertex_properties[0],
                                     [&handle,
                                      &vertices,
                                      &d_tx_value_counts,
                                      groupby_functor,
                                      large_buffer_type,
                                      this_step_comm_size,
                                      mem_frugal_threshold](auto& prop) {
                                       d_tx_value_counts =
                                         cugraph::groupby_and_count(vertices.begin(),
                                                                    vertices.end(),
                                                                    prop.begin(),
                                                                    groupby_functor,
                                                                    this_step_comm_size,
                                                                    mem_frugal_threshold,
                                                                    handle.get_stream(),
                                                                    large_buffer_type);
                                     });
    } else {
      property_positions = large_buffer_type
                             ? large_buffer_manager::allocate_memory_buffer<size_t>(
                                 vertices.size(), handle.get_stream())
                             : rmm::device_uvector<size_t>(vertices.size(), handle.get_stream());
      cugraph::sequence(handle.get_thrust_policy(),
                        property_positions->begin(),
                        property_positions->end(),
                        size_t{0});

      d_tx_value_counts = cugraph::groupby_and_count(vertices.begin(),
                                                     vertices.end(),
                                                     property_positions->begin(),
                                                     groupby_functor,
                                                     this_step_comm_size,
                                                     mem_frugal_threshold,
                                                     handle.get_stream(),
                                                     large_buffer_type);
    }

    if (vertex_properties.size() == 0) {
      std::tie(vertices, std::ignore) = shuffle_values(
        this_step_comm,
        vertices.begin(),
        raft::device_span<size_t const>(d_tx_value_counts.data(), d_tx_value_counts.size()),
        handle.get_stream(),
        large_buffer_type);
    } else if ((vertex_properties.size() == 1) && !large_buffer_type) {
      cugraph::variant_type_dispatch(
        vertex_properties[0],
        [&handle, &vertices, &this_step_comm, &d_tx_value_counts](auto& prop) {
          shuffle_vertex_and_property(
            handle,
            this_step_comm,
            vertices,
            prop,
            raft::device_span<size_t const>(d_tx_value_counts.data(), d_tx_value_counts.size()));
        });
    } else if (vertex_properties.size() == 1) {
      std::tie(vertices, std::ignore) = shuffle_values(
        this_step_comm,
        vertices.begin(),
        raft::device_span<size_t const>(d_tx_value_counts.data(), d_tx_value_counts.size()),
        handle.get_stream(),
        large_buffer_type);
      cugraph::variant_type_dispatch(
        vertex_properties[0],
        [&handle, &this_step_comm, &d_tx_value_counts, large_buffer_type](auto& prop) {
          std::tie(prop, std::ignore) = shuffle_values(
            this_step_comm,
            prop.begin(),
            raft::device_span<size_t const>(d_tx_value_counts.data(), d_tx_value_counts.size()),
            handle.get_stream(),
            large_buffer_type);
        });
    } else if ((vertex_properties.size() == 2) && !large_buffer_type) {
      raft::device_span<size_t const> d_tx_value_counts_span(d_tx_value_counts.data(),
                                                             d_tx_value_counts.size());

      cugraph::variant_type_dispatch(
        vertex_properties[0],
        [&handle,
         &vertices,
         &this_step_comm,
         &property_positions,
         d_tx_value_counts_span,
         &vertex_properties](auto& prop0) {
          cugraph::variant_type_dispatch(
            vertex_properties[1],
            [&handle,
             &vertices,
             &this_step_comm,
             &property_positions,
             d_tx_value_counts_span,
             &prop0](auto& prop1) {
              shuffle_vertex_and_two_properties(handle,
                                                this_step_comm,
                                                vertices,
                                                prop0,
                                                prop1,
                                                *property_positions,
                                                d_tx_value_counts_span);
            });
        });
    } else {
      raft::device_span<size_t const> d_tx_value_counts_span(d_tx_value_counts.data(),
                                                             d_tx_value_counts.size());

      std::tie(vertices, std::ignore) = shuffle_values(this_step_comm,
                                                       vertices.begin(),
                                                       d_tx_value_counts_span,
                                                       handle.get_stream(),
                                                       large_buffer_type);

      std::for_each(
        vertex_properties.begin(),
        vertex_properties.end(),
        [&handle, &this_step_comm, &property_positions, d_tx_value_counts_span, large_buffer_type](
          auto& property) {
          cugraph::variant_type_dispatch(
            property,
            [&handle,
             &this_step_comm,
             &property_positions,
             d_tx_value_counts_span,
             large_buffer_type](auto& prop) {
              using T  = typename std::remove_reference<decltype(prop)>::type::value_type;
              auto tmp = large_buffer_type
                           ? large_buffer_manager::allocate_memory_buffer<T>(prop.size(),
                                                                             handle.get_stream())
                           : rmm::device_uvector<T>(prop.size(), handle.get_stream());
              thrust::gather(handle.get_thrust_policy(),
                             property_positions->begin(),
                             property_positions->end(),
                             prop.begin(),
                             tmp.begin());
              std::tie(prop, std::ignore) = shuffle_values(this_step_comm,
                                                           tmp.begin(),
                                                           d_tx_value_counts_span,
                                                           handle.get_stream(),
                                                           large_buffer_type);
            });
        });
    }
  }

  return std::make_tuple(std::move(vertices), std::move(vertex_properties));
}

}  // namespace

template <typename vertex_t>
std::tuple<rmm::device_uvector<vertex_t>, std::vector<arithmetic_device_uvector_t>>
shuffle_ext_vertices(raft::handle_t const& handle,
                     rmm::device_uvector<vertex_t>&& vertices,
                     std::vector<arithmetic_device_uvector_t>&& vertex_properties,
                     std::optional<large_buffer_type_t> large_buffer_type)
{
  auto const comm_size       = handle.get_comms().get_size();
  auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
  auto const major_comm_size = major_comm.get_size();
  auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
  auto const minor_comm_size = minor_comm.get_size();

  return shuffle_vertices(handle,
                          std::move(vertices),
                          std::move(vertex_properties),
                          cugraph::detail::compute_gpu_id_from_ext_vertex_t<vertex_t>{
                            comm_size, major_comm_size, minor_comm_size},
                          large_buffer_type);
}

template <typename vertex_t>
std::tuple<rmm::device_uvector<vertex_t>, std::vector<arithmetic_device_uvector_t>>
shuffle_int_vertices(raft::handle_t const& handle,
                     rmm::device_uvector<vertex_t>&& vertices,
                     std::vector<arithmetic_device_uvector_t>&& vertex_properties,
                     raft::host_span<vertex_t const> vertex_partition_range_lasts,
                     std::optional<large_buffer_type_t> large_buffer_type)
{
  auto const comm_size       = handle.get_comms().get_size();
  auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
  auto const major_comm_size = major_comm.get_size();
  auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
  auto const minor_comm_size = minor_comm.get_size();

  rmm::device_uvector<vertex_t> d_vertex_partition_range_lasts(vertex_partition_range_lasts.size(),
                                                               handle.get_stream());
  raft::update_device(d_vertex_partition_range_lasts.data(),
                      vertex_partition_range_lasts.data(),
                      vertex_partition_range_lasts.size(),
                      handle.get_stream());

  return shuffle_vertices(
    handle,
    std::move(vertices),
    std::move(vertex_properties),
    cugraph::detail::compute_gpu_id_from_int_vertex_t<vertex_t>{
      raft::device_span<vertex_t const>(d_vertex_partition_range_lasts.data(),
                                        d_vertex_partition_range_lasts.size()),
      major_comm_size,
      minor_comm_size},
    large_buffer_type);
}

}  // namespace cugraph
