/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cugraph/arithmetic_variant_types.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/export.hpp>
#include <cugraph/shuffle_functions.hpp>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/graph_partition_utils.cuh>
#include <cugraph/utilities/shuffle_comm.cuh>
#include <cugraph/utilities/thrust_wrappers.hpp>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/iterator>
#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <optional>

namespace cugraph {
namespace detail {

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

template <typename TxValueIterator>
rmm::device_uvector<typename thrust::iterator_traits<TxValueIterator>::value_type>
shuffle_values_with_precomputed_metadata(
  raft::comms::comms_t const& comm,
  TxValueIterator tx_value_first,
  std::vector<size_t> const& tx_counts,
  std::vector<size_t> const& tx_displs,
  std::vector<int> const& tx_dst_ranks,
  std::vector<size_t> const& rx_counts,
  std::vector<size_t> const& rx_displs,
  std::vector<int> const& rx_src_ranks,
  rmm::cuda_stream_view stream_view)
{
  using value_t = typename thrust::iterator_traits<TxValueIterator>::value_type;

  auto rx_buffer_size = rx_displs.size() > 0 ? rx_displs.back() + rx_counts.back() : size_t{0};
  rmm::device_uvector<value_t> rx_value_buffer(rx_buffer_size, stream_view);

  cugraph::device_multicast_sendrecv(
    comm,
    tx_value_first,
    raft::host_span<size_t const>(tx_counts.data(), tx_counts.size()),
    raft::host_span<size_t const>(tx_displs.data(), tx_displs.size()),
    raft::host_span<int const>(tx_dst_ranks.data(), tx_dst_ranks.size()),
    rx_value_buffer.begin(),
    raft::host_span<size_t const>(rx_counts.data(), rx_counts.size()),
    raft::host_span<size_t const>(rx_displs.data(), rx_displs.size()),
    raft::host_span<int const>(rx_src_ranks.data(), rx_src_ranks.size()),
    stream_view);

  return rx_value_buffer;
}

template <typename record_t>
rmm::device_uvector<record_t> shuffle_records_with_precomputed_metadata(
  raft::comms::comms_t const& comm,
  rmm::device_uvector<record_t> const& tx_records,
  std::vector<size_t> const& tx_counts,
  std::vector<size_t> const& tx_displs,
  std::vector<int> const& tx_dst_ranks,
  std::vector<size_t> const& rx_counts,
  std::vector<size_t> const& rx_displs,
  std::vector<int> const& rx_src_ranks,
  rmm::cuda_stream_view stream_view)
{
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

template <typename T>
rmm::device_uvector<T> gather_by_position(raft::handle_t const& handle,
                                          rmm::device_uvector<T> const& values,
                                          rmm::device_uvector<size_t> const& positions)
{
  rmm::device_uvector<T> gathered(values.size(), handle.get_stream());
  thrust::gather(handle.get_thrust_policy(),
                 positions.begin(),
                 positions.end(),
                 values.begin(),
                 gathered.begin());
  return gathered;
}

template <typename T0, typename T1>
void shuffle_labels_and_two_properties(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>& labels,
  rmm::device_uvector<T0>& property0,
  rmm::device_uvector<T1>& property1,
  rmm::device_uvector<size_t> const& property_position,
  std::vector<size_t> const& tx_counts,
  std::vector<size_t> const& tx_displs,
  std::vector<int> const& tx_dst_ranks,
  std::vector<size_t> const& rx_counts,
  std::vector<size_t> const& rx_displs,
  std::vector<int> const& rx_src_ranks)
{
  auto tmp0 = gather_by_position(handle, property0, property_position);
  auto tmp1 = gather_by_position(handle, property1, property_position);

  using record_t = packed_record_3_t<int32_t, T0, T1>;
  rmm::device_uvector<record_t> tx_records(labels.size(), handle.get_stream());
  thrust::for_each(
    handle.get_thrust_policy(),
    thrust::make_counting_iterator(size_t{0}),
    thrust::make_counting_iterator(labels.size()),
    [label_first = labels.data(),
     property0_first = tmp0.data(),
     property1_first = tmp1.data(),
     tx_record_first = tx_records.data()] __device__(auto i) {
      tx_record_first[i] = record_t{label_first[i], property0_first[i], property1_first[i]};
    });

  auto rx_records = shuffle_records_with_precomputed_metadata(
    handle.get_comms(),
    tx_records,
    tx_counts,
    tx_displs,
    tx_dst_ranks,
    rx_counts,
    rx_displs,
    rx_src_ranks,
    handle.get_stream());

  rmm::device_uvector<int32_t> rx_labels(rx_records.size(), handle.get_stream());
  rmm::device_uvector<T0> rx_property0(rx_records.size(), handle.get_stream());
  rmm::device_uvector<T1> rx_property1(rx_records.size(), handle.get_stream());
  thrust::for_each(
    handle.get_thrust_policy(),
    thrust::make_counting_iterator(size_t{0}),
    thrust::make_counting_iterator(rx_records.size()),
    [rx_record_first = rx_records.data(),
     label_first = rx_labels.data(),
     property0_first = rx_property0.data(),
     property1_first = rx_property1.data()] __device__(auto i) {
      auto record         = rx_record_first[i];
      label_first[i]     = record.v0;
      property0_first[i] = record.v1;
      property1_first[i] = record.v2;
    });

  labels    = std::move(rx_labels);
  property0 = std::move(rx_property0);
  property1 = std::move(rx_property1);
}

template <typename T0, typename T1, typename T2>
void shuffle_labels_and_three_properties(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>& labels,
  rmm::device_uvector<T0>& property0,
  rmm::device_uvector<T1>& property1,
  rmm::device_uvector<T2>& property2,
  rmm::device_uvector<size_t> const& property_position,
  std::vector<size_t> const& tx_counts,
  std::vector<size_t> const& tx_displs,
  std::vector<int> const& tx_dst_ranks,
  std::vector<size_t> const& rx_counts,
  std::vector<size_t> const& rx_displs,
  std::vector<int> const& rx_src_ranks)
{
  auto tmp0 = gather_by_position(handle, property0, property_position);
  auto tmp1 = gather_by_position(handle, property1, property_position);
  auto tmp2 = gather_by_position(handle, property2, property_position);

  using record_t = packed_record_4_t<int32_t, T0, T1, T2>;
  rmm::device_uvector<record_t> tx_records(labels.size(), handle.get_stream());
  thrust::for_each(
    handle.get_thrust_policy(),
    thrust::make_counting_iterator(size_t{0}),
    thrust::make_counting_iterator(labels.size()),
    [label_first = labels.data(),
     property0_first = tmp0.data(),
     property1_first = tmp1.data(),
     property2_first = tmp2.data(),
     tx_record_first = tx_records.data()] __device__(auto i) {
      tx_record_first[i] = record_t{
        label_first[i], property0_first[i], property1_first[i], property2_first[i]};
    });

  auto rx_records = shuffle_records_with_precomputed_metadata(
    handle.get_comms(),
    tx_records,
    tx_counts,
    tx_displs,
    tx_dst_ranks,
    rx_counts,
    rx_displs,
    rx_src_ranks,
    handle.get_stream());

  rmm::device_uvector<int32_t> rx_labels(rx_records.size(), handle.get_stream());
  rmm::device_uvector<T0> rx_property0(rx_records.size(), handle.get_stream());
  rmm::device_uvector<T1> rx_property1(rx_records.size(), handle.get_stream());
  rmm::device_uvector<T2> rx_property2(rx_records.size(), handle.get_stream());
  thrust::for_each(
    handle.get_thrust_policy(),
    thrust::make_counting_iterator(size_t{0}),
    thrust::make_counting_iterator(rx_records.size()),
    [rx_record_first = rx_records.data(),
     label_first = rx_labels.data(),
     property0_first = rx_property0.data(),
     property1_first = rx_property1.data(),
     property2_first = rx_property2.data()] __device__(auto i) {
      auto record         = rx_record_first[i];
      label_first[i]     = record.v0;
      property0_first[i] = record.v1;
      property1_first[i] = record.v2;
      property2_first[i] = record.v3;
    });

  labels    = std::move(rx_labels);
  property0 = std::move(rx_property0);
  property1 = std::move(rx_property1);
  property2 = std::move(rx_property2);
}

}  // namespace

CUGRAPH_EXPORT std::tuple<std::vector<cugraph::arithmetic_device_uvector_t>,
                          std::optional<rmm::device_uvector<int32_t>>,
                          std::optional<rmm::device_uvector<int32_t>>,
                          std::optional<rmm::device_uvector<size_t>>>
shuffle_and_organize_output(
  raft::handle_t const& handle,
  std::vector<cugraph::arithmetic_device_uvector_t>&& property_edges,
  std::optional<rmm::device_uvector<int32_t>>&& labels,
  std::optional<rmm::device_uvector<int32_t>>&& hops,
  std::optional<int32_t> input_hops,
  std::optional<raft::device_span<int32_t const>> label_to_output_comm_rank)
{
  std::optional<rmm::device_uvector<size_t>> offsets{std::nullopt};

  if (labels) {
    if (label_to_output_comm_rank) {
      indirection_t<int32_t, int32_t const*> key_to_gpu_op{label_to_output_comm_rank->begin()};

      auto comm_size = handle.get_comms().get_size();
      size_t element_size{sizeof(int32_t) + sizeof(size_t)};
      auto total_global_mem = handle.get_device_properties().totalGlobalMem;
      auto constexpr mem_frugal_ratio =
        0.1;  // if the expected temporary buffer size exceeds the mem_frugal_ratio of the
              // total_global_mem, switch to the memory frugal approach (thrust::sort is used to
              // group-by by default, and thrust::sort requires temporary buffer comparable to the
              // input data size)
      auto mem_frugal_threshold = static_cast<size_t>(
        static_cast<double>(total_global_mem / element_size) * mem_frugal_ratio);

      rmm::device_uvector<size_t> property_position(labels->size(), handle.get_stream());
      cugraph::sequence(rmm::exec_policy(handle.get_stream()),
                        property_position.data(),
                        property_position.data() + property_position.size(),
                        size_t{0});

      auto d_tx_value_counts = cugraph::groupby_and_count(labels->begin(),
                                                          labels->end(),
                                                          property_position.begin(),
                                                          key_to_gpu_op,
                                                          comm_size,
                                                          mem_frugal_threshold,
                                                          handle.get_stream());

      raft::device_span<size_t const> d_tx_value_counts_span{d_tx_value_counts.data(),
                                                             d_tx_value_counts.size()};

      auto [tx_counts, tx_displs, tx_dst_ranks, rx_counts, rx_displs, rx_src_ranks] =
        compute_tx_rx_counts_displs_ranks(
          handle.get_comms(), d_tx_value_counts_span, false, handle.get_stream());

      size_t num_tuple_shuffled_properties{0};
      if (property_edges.size() >= 3) {
        cugraph::variant_type_dispatch(property_edges[0], [&](auto& prop0) {
          cugraph::variant_type_dispatch(property_edges[1], [&](auto& prop1) {
            cugraph::variant_type_dispatch(property_edges[2], [&](auto& prop2) {
              shuffle_labels_and_three_properties(handle,
                                                  *labels,
                                                  prop0,
                                                  prop1,
                                                  prop2,
                                                  property_position,
                                                  tx_counts,
                                                  tx_displs,
                                                  tx_dst_ranks,
                                                  rx_counts,
                                                  rx_displs,
                                                  rx_src_ranks);
            });
          });
        });
        num_tuple_shuffled_properties = 3;
      } else {
        cugraph::variant_type_dispatch(property_edges[0], [&](auto& prop0) {
          cugraph::variant_type_dispatch(property_edges[1], [&](auto& prop1) {
            shuffle_labels_and_two_properties(handle,
                                              *labels,
                                              prop0,
                                              prop1,
                                              property_position,
                                              tx_counts,
                                              tx_displs,
                                              tx_dst_ranks,
                                              rx_counts,
                                              rx_displs,
                                              rx_src_ranks);
          });
        });
        num_tuple_shuffled_properties = 2;
      }

      std::for_each(
        property_edges.begin() + num_tuple_shuffled_properties,
        property_edges.end(),
        [&handle,
         &property_position,
         &tx_counts,
         &tx_displs,
         &tx_dst_ranks,
         &rx_counts,
         &rx_displs,
         &rx_src_ranks](auto& property) {
          cugraph::variant_type_dispatch(
            property,
            [&handle,
             &property_position,
             &tx_counts,
             &tx_displs,
             &tx_dst_ranks,
             &rx_counts,
             &rx_displs,
             &rx_src_ranks](auto& prop) {
              using T = typename std::remove_reference<decltype(prop)>::type::value_type;
              rmm::device_uvector<T> tmp(prop.size(), handle.get_stream());

              thrust::gather(handle.get_thrust_policy(),
                             property_position.begin(),
                             property_position.end(),
                             prop.begin(),
                             tmp.begin());

              prop = shuffle_values_with_precomputed_metadata(handle.get_comms(),
                                                              tmp.begin(),
                                                              tx_counts,
                                                              tx_displs,
                                                              tx_dst_ranks,
                                                              rx_counts,
                                                              rx_displs,
                                                              rx_src_ranks,
                                                              handle.get_stream());
            });
        });

      if (hops) {
        rmm::device_uvector<int32_t> tmp(hops->size(), handle.get_stream());
        thrust::gather(handle.get_thrust_policy(),
                       property_position.begin(),
                       property_position.end(),
                       hops->begin(),
                       tmp.begin());

        *hops = shuffle_values_with_precomputed_metadata(handle.get_comms(),
                                                         tmp.begin(),
                                                         tx_counts,
                                                         tx_displs,
                                                         tx_dst_ranks,
                                                         rx_counts,
                                                         rx_displs,
                                                         rx_src_ranks,
                                                         handle.get_stream());
      }
    }

    // Sort the tuples by hop/label
    rmm::device_uvector<size_t> indices(labels->size(), handle.get_stream());
    cugraph::sequence(handle.get_thrust_policy(), indices.begin(), indices.end(), size_t{0});
    if (hops) {
      thrust::sort_by_key(handle.get_thrust_policy(),
                          thrust::make_zip_iterator(labels->begin(), hops->begin()),
                          thrust::make_zip_iterator(labels->end(), hops->end()),
                          indices.begin());
    } else {
      thrust::sort_by_key(
        handle.get_thrust_policy(), labels->begin(), labels->end(), indices.begin());
    }

    std::for_each(
      property_edges.begin(), property_edges.end(), [&handle, &indices](auto& property) {
        cugraph::variant_type_dispatch(property, [&handle, &indices](auto& edge_vector) {
          using T = typename std::remove_reference<decltype(edge_vector)>::type::value_type;
          rmm::device_uvector<T> tmp(indices.size(), handle.get_stream());
          thrust::gather(handle.get_thrust_policy(),
                         indices.begin(),
                         indices.end(),
                         edge_vector.begin(),
                         tmp.begin());

          edge_vector = std::move(tmp);
        });
      });

    // Need to generate offsets for each unique label (not each seed) on each GPU
    rmm::device_uvector<int32_t> unique_labels(labels->size(), handle.get_stream());
    raft::copy(unique_labels.data(), labels->data(), labels->size(), handle.get_stream());
    cugraph::sort(handle.get_thrust_policy(), unique_labels.begin(), unique_labels.end());
    auto unique_end =
      cugraph::unique(handle.get_thrust_policy(), unique_labels.begin(), unique_labels.end());
    size_t num_unique_labels =
      static_cast<size_t>(cuda::std::distance(unique_labels.begin(), unique_end));

    unique_labels.resize(num_unique_labels, handle.get_stream());

    offsets = rmm::device_uvector<size_t>(unique_labels.size() + 1, handle.get_stream());

    thrust::lower_bound(handle.get_thrust_policy(),
                        labels->begin(),
                        labels->end(),
                        unique_labels.begin(),
                        unique_labels.end(),
                        offsets->begin());

    size_t last_offset = labels->size();
    offsets->set_element_async(unique_labels.size(), last_offset, handle.get_stream());
    handle.sync_stream();
  }

  return std::make_tuple(
    std::move(property_edges), std::move(labels), std::move(hops), std::move(offsets));
}

}  // namespace detail
}  // namespace cugraph
