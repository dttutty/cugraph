# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libc.stddef cimport size_t
from libc.stdint cimport uintptr_t
from libcpp cimport bool
from libcpp.memory cimport unique_ptr

from cuda.bindings.cyruntime cimport (
    cudaError_t,
    cudaGetErrorString,
    cudaMemcpyAsync,
    cudaMemcpyDeviceToDevice,
    cudaStream_t,
    cudaStreamSynchronize,
)
from pylibraft.common.handle cimport handle_t

import cupy as cp
import numpy as np


cdef extern from "cugraph/legacy/graph.hpp" namespace "cugraph::legacy":
    cdef cppclass GraphCOOView[VT, ET, WT]:
        VT* src_indices
        VT* dst_indices
        WT* edge_data
        VT number_of_vertices
        ET number_of_edges

        GraphCOOView()
        GraphCOOView(const VT*, const ET*, const WT*, size_t, size_t)

    cdef cppclass GraphCSRView[VT, ET, WT]:
        ET* offsets
        VT* indices
        WT* edge_data
        VT number_of_vertices
        ET number_of_edges

        GraphCSRView()
        GraphCSRView(const VT*, const ET*, const WT*, size_t, size_t)
        void get_source_indices(VT*) const

    cdef cppclass GraphCOO[VT, ET, WT]:
        GraphCOOView[VT, ET, WT] view()

    cdef cppclass GraphCSR[VT, ET, WT]:
        GraphCSRView[VT, ET, WT] view()


cdef extern from "cugraph/legacy/functions.hpp" namespace "cugraph":
    cdef unique_ptr[GraphCSR[VT, ET, WT]] c_coo_to_csr "cugraph::coo_to_csr" [
        VT,
        ET,
        WT,
    ](
        const GraphCOOView[VT, ET, WT]& graph
    ) except +

    cdef void c_comms_bcast "cugraph::comms_bcast" [value_t](
        const handle_t& handle,
        value_t* dst,
        size_t size,
    ) except +


cdef extern from "cugraph/algorithms.hpp" namespace "cugraph":
    cdef weight_t c_hungarian "cugraph::hungarian" [vertex_t, edge_t, weight_t](
        const handle_t& handle,
        const GraphCOOView[vertex_t, edge_t, weight_t]& graph,
        vertex_t num_workers,
        const vertex_t* workers,
        vertex_t* assignments,
        weight_t epsilon,
    ) except +

    cdef unique_ptr[GraphCOO[VT, ET, WT]] c_minimum_spanning_tree "cugraph::minimum_spanning_tree" [
        VT,
        ET,
        WT,
    ](
        const handle_t& handle,
        const GraphCSRView[VT, ET, WT]& graph,
    ) except +


cdef extern from "cugraph/algorithms.hpp":
    cdef weight_t c_dense_hungarian_eps "cugraph::dense::hungarian" [
        vertex_t,
        weight_t,
    ](
        const handle_t& handle,
        const weight_t* costs,
        vertex_t num_rows,
        vertex_t num_columns,
        vertex_t* assignments,
        weight_t epsilon,
    ) except +

    cdef weight_t c_dense_hungarian_no_eps "cugraph::dense::hungarian" [
        vertex_t,
        weight_t,
    ](
        const handle_t& handle,
        const weight_t* costs,
        vertex_t num_rows,
        vertex_t num_columns,
        vertex_t* assignments,
    ) except +


cdef extern from "cugraph/utilities/path_retrieval.hpp" namespace "cugraph":
    cdef void c_get_traversed_cost "cugraph::get_traversed_cost" [vertex_t, weight_t](
        const handle_t& handle,
        const vertex_t* vertices,
        const vertex_t* preds,
        const weight_t* info_weights,
        weight_t* out,
        vertex_t stop_vertex,
        vertex_t num_vertices,
    ) except +


cdef void _cuda_check(cudaError_t status, str api_name):
    cdef const char* error_string
    if status == 0:
        return
    error_string = cudaGetErrorString(status)
    raise RuntimeError(
        f"{api_name} failed with CUDA error {status}: "
        f"{error_string.decode('utf-8')}"
    )


cdef uintptr_t _device_pointer(object obj, str name):
    try:
        cai = obj.__cuda_array_interface__
    except AttributeError as exc:
        raise TypeError(f"{name} does not support __cuda_array_interface__") from exc

    cdef uintptr_t pointer = <uintptr_t>cai["data"][0]
    if pointer == 0:
        raise ValueError(f"{name} has null device pointer")
    return pointer


cdef void _require_dtype(object obj, object dtype, str name):
    if obj.dtype != np.dtype(dtype):
        raise TypeError(f"{name} must have dtype {np.dtype(dtype)}")


cdef void _require_same_length(object lhs, str lhs_name, object rhs, str rhs_name):
    if len(lhs) != len(rhs):
        raise ValueError(f"{lhs_name} and {rhs_name} must have the same length")


cdef object _empty_device_array(size_t size, object dtype):
    return cp.empty(<Py_ssize_t>size, dtype=dtype)


cdef void _copy_to_device_array(
    object destination,
    const void* source,
    size_t element_count,
    size_t item_size,
    cudaStream_t stream,
):
    cdef uintptr_t destination_pointer
    if element_count == 0:
        return
    destination_pointer = _device_pointer(destination, "destination")
    _cuda_check(
        cudaMemcpyAsync(
            <void*>destination_pointer,
            source,
            element_count * item_size,
            cudaMemcpyDeviceToDevice,
            stream,
        ),
        "cudaMemcpyAsync",
    )
    _cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize")


cdef object _copy_device_array(
    const void* source,
    size_t element_count,
    object dtype,
    size_t item_size,
    cudaStream_t stream,
):
    output = _empty_device_array(element_count, dtype)
    _copy_to_device_array(output, source, element_count, item_size, stream)
    return output


cdef handle_t* _handle_pointer(object handle):
    cdef uintptr_t pointer
    if handle is None:
        raise TypeError("handle must not be None")
    if hasattr(handle, "getHandle"):
        pointer = <uintptr_t>handle.getHandle()
    elif isinstance(handle, int):
        pointer = <uintptr_t>handle
    else:
        raise TypeError("handle must be a RAFT handle object or integer pointer")
    if pointer == 0:
        raise ValueError("handle pointer is null")
    return <handle_t*>pointer


cdef tuple _coo_to_csr_float(object sources, object destinations, object weights):
    cdef size_t num_edges = len(sources)
    cdef int* source_ptr = <int*>_device_pointer(sources, "sources")
    cdef int* dest_ptr = <int*>_device_pointer(destinations, "destinations")
    cdef float* weight_ptr = NULL
    cdef GraphCOOView[int, int, float] graph
    cdef unique_ptr[GraphCSR[int, int, float]] csr
    cdef GraphCSRView[int, int, float] csr_view
    cdef cudaStream_t stream = NULL

    if weights is not None:
        _require_dtype(weights, np.float32, "weights")
        weight_ptr = <float*>_device_pointer(weights, "weights")

    graph = GraphCOOView[int, int, float](
        source_ptr,
        dest_ptr,
        weight_ptr,
        <size_t>0,
        num_edges,
    )
    csr = c_coo_to_csr[int, int, float](graph)
    csr_view = csr.get().view()

    offsets = _copy_device_array(
        <const void*>csr_view.offsets,
        <size_t>csr_view.number_of_vertices + 1,
        np.int32,
        sizeof(int),
        stream,
    )
    indices = _copy_device_array(
        <const void*>csr_view.indices,
        <size_t>csr_view.number_of_edges,
        np.int32,
        sizeof(int),
        stream,
    )
    copied_weights = None
    if weights is not None:
        copied_weights = _copy_device_array(
            <const void*>csr_view.edge_data,
            <size_t>csr_view.number_of_edges,
            np.float32,
            sizeof(float),
            stream,
        )
    return offsets, indices, copied_weights


cdef tuple _coo_to_csr_double(object sources, object destinations, object weights):
    cdef size_t num_edges = len(sources)
    cdef int* source_ptr = <int*>_device_pointer(sources, "sources")
    cdef int* dest_ptr = <int*>_device_pointer(destinations, "destinations")
    cdef double* weight_ptr = NULL
    cdef GraphCOOView[int, int, double] graph
    cdef unique_ptr[GraphCSR[int, int, double]] csr
    cdef GraphCSRView[int, int, double] csr_view
    cdef cudaStream_t stream = NULL

    _require_dtype(weights, np.float64, "weights")
    weight_ptr = <double*>_device_pointer(weights, "weights")

    graph = GraphCOOView[int, int, double](
        source_ptr,
        dest_ptr,
        weight_ptr,
        <size_t>0,
        num_edges,
    )
    csr = c_coo_to_csr[int, int, double](graph)
    csr_view = csr.get().view()

    offsets = _copy_device_array(
        <const void*>csr_view.offsets,
        <size_t>csr_view.number_of_vertices + 1,
        np.int32,
        sizeof(int),
        stream,
    )
    indices = _copy_device_array(
        <const void*>csr_view.indices,
        <size_t>csr_view.number_of_edges,
        np.int32,
        sizeof(int),
        stream,
    )
    copied_weights = _copy_device_array(
        <const void*>csr_view.edge_data,
        <size_t>csr_view.number_of_edges,
        np.float64,
        sizeof(double),
        stream,
    )
    return offsets, indices, copied_weights


def coo_to_csr(sources, destinations, weights=None):
    _require_same_length(sources, "sources", destinations, "destinations")
    _require_dtype(sources, np.int32, "sources")
    _require_dtype(destinations, np.int32, "destinations")

    if len(sources) == 0:
        return cp.zeros(1, dtype=np.int32), cp.zeros(1, dtype=np.int32), weights

    if weights is None or weights.dtype == np.dtype(np.float32):
        return _coo_to_csr_float(sources, destinations, weights)
    if weights.dtype == np.dtype(np.float64):
        return _coo_to_csr_double(sources, destinations, weights)
    raise TypeError("weights must be None, float32, or float64")


def csr_to_edgelist(offsets, indices, weights=None):
    cdef size_t offset_count = len(offsets)
    cdef size_t edge_count = len(indices)
    cdef int* offsets_ptr
    cdef int* indices_ptr
    cdef int* sources_ptr
    cdef GraphCSRView[int, int, float] graph
    cdef cudaStream_t stream = NULL

    if offset_count == 0:
        raise ValueError("offsets must not be empty")
    _require_dtype(offsets, np.int32, "offsets")
    _require_dtype(indices, np.int32, "indices")

    offsets_ptr = <int*>_device_pointer(offsets, "offsets")
    indices_ptr = <int*>_device_pointer(indices, "indices")
    graph = GraphCSRView[int, int, float](
        offsets_ptr,
        indices_ptr,
        <float*>NULL,
        offset_count - 1,
        edge_count,
    )

    sources = _empty_device_array(edge_count, np.int32)
    sources_ptr = <int*>_device_pointer(sources, "sources")
    graph.get_source_indices(sources_ptr)
    _cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize")

    return sources, indices, weights


cdef tuple _sparse_hungarian_float(
    object sources,
    object destinations,
    object weights,
    object workers,
    int num_vertices,
    object epsilon,
):
    cdef int* source_ptr = <int*>_device_pointer(sources, "sources")
    cdef int* dest_ptr = <int*>_device_pointer(destinations, "destinations")
    cdef float* weight_ptr = <float*>_device_pointer(weights, "weights")
    cdef int* worker_ptr = <int*>_device_pointer(workers, "workers")
    cdef size_t worker_count = len(workers)
    cdef GraphCOOView[int, int, float] graph
    cdef handle_t handle
    cdef float eps = 1e-6 if epsilon is None else <float>epsilon
    cdef float cost
    cdef int* assignment_ptr

    assignments = _empty_device_array(worker_count, np.int32)
    assignment_ptr = <int*>_device_pointer(assignments, "assignments")
    graph = GraphCOOView[int, int, float](
        source_ptr,
        dest_ptr,
        weight_ptr,
        <size_t>num_vertices,
        <size_t>len(sources),
    )
    cost = c_hungarian[int, int, float](
        handle,
        graph,
        <int>worker_count,
        worker_ptr,
        assignment_ptr,
        eps,
    )
    handle.sync_stream()
    return cost, assignments


cdef tuple _sparse_hungarian_double(
    object sources,
    object destinations,
    object weights,
    object workers,
    int num_vertices,
    object epsilon,
):
    cdef int* source_ptr = <int*>_device_pointer(sources, "sources")
    cdef int* dest_ptr = <int*>_device_pointer(destinations, "destinations")
    cdef double* weight_ptr = <double*>_device_pointer(weights, "weights")
    cdef int* worker_ptr = <int*>_device_pointer(workers, "workers")
    cdef size_t worker_count = len(workers)
    cdef GraphCOOView[int, int, double] graph
    cdef handle_t handle
    cdef double eps = 1e-6 if epsilon is None else <double>epsilon
    cdef double cost
    cdef int* assignment_ptr

    assignments = _empty_device_array(worker_count, np.int32)
    assignment_ptr = <int*>_device_pointer(assignments, "assignments")
    graph = GraphCOOView[int, int, double](
        source_ptr,
        dest_ptr,
        weight_ptr,
        <size_t>num_vertices,
        <size_t>len(sources),
    )
    cost = c_hungarian[int, int, double](
        handle,
        graph,
        <int>worker_count,
        worker_ptr,
        assignment_ptr,
        eps,
    )
    handle.sync_stream()
    return cost, assignments


def sparse_hungarian(
    sources,
    destinations,
    weights,
    workers,
    int num_vertices,
    epsilon=None,
):
    _require_same_length(sources, "sources", destinations, "destinations")
    _require_same_length(sources, "sources", weights, "weights")
    _require_dtype(sources, np.int32, "sources")
    _require_dtype(destinations, np.int32, "destinations")
    _require_dtype(workers, np.int32, "workers")

    if weights.dtype == np.dtype(np.float32):
        return _sparse_hungarian_float(
            sources,
            destinations,
            weights,
            workers,
            num_vertices,
            epsilon,
        )
    if weights.dtype == np.dtype(np.float64):
        return _sparse_hungarian_double(
            sources,
            destinations,
            weights,
            workers,
            num_vertices,
            epsilon,
        )
    raise TypeError("weights must be float32 or float64")


cdef tuple _dense_hungarian_float(
    object costs,
    int num_rows,
    int num_columns,
    object epsilon,
):
    cdef float* cost_ptr = <float*>_device_pointer(costs, "costs")
    cdef handle_t handle
    cdef float eps = 1e-6 if epsilon is None else <float>epsilon
    cdef int* assignment_ptr
    cdef float cost

    assignments = _empty_device_array(num_rows, np.int32)
    assignment_ptr = <int*>_device_pointer(assignments, "assignments")
    cost = c_dense_hungarian_eps[int, float](
        handle,
        cost_ptr,
        num_rows,
        num_columns,
        assignment_ptr,
        eps,
    )
    handle.sync_stream()
    return cost, assignments


cdef tuple _dense_hungarian_double(
    object costs,
    int num_rows,
    int num_columns,
    object epsilon,
):
    cdef double* cost_ptr = <double*>_device_pointer(costs, "costs")
    cdef handle_t handle
    cdef double eps = 1e-6 if epsilon is None else <double>epsilon
    cdef int* assignment_ptr
    cdef double cost

    assignments = _empty_device_array(num_rows, np.int32)
    assignment_ptr = <int*>_device_pointer(assignments, "assignments")
    cost = c_dense_hungarian_eps[int, double](
        handle,
        cost_ptr,
        num_rows,
        num_columns,
        assignment_ptr,
        eps,
    )
    handle.sync_stream()
    return cost, assignments


cdef tuple _dense_hungarian_int(
    object costs,
    int num_rows,
    int num_columns,
):
    cdef int* cost_ptr = <int*>_device_pointer(costs, "costs")
    cdef handle_t handle
    cdef int* assignment_ptr
    cdef int cost

    assignments = _empty_device_array(num_rows, np.int32)
    assignment_ptr = <int*>_device_pointer(assignments, "assignments")
    cost = c_dense_hungarian_no_eps[int, int](
        handle,
        cost_ptr,
        num_rows,
        num_columns,
        assignment_ptr,
    )
    handle.sync_stream()
    return cost, assignments


def dense_hungarian(costs, int num_rows, int num_columns, epsilon=None):
    if costs.dtype == np.dtype(np.float32):
        return _dense_hungarian_float(costs, num_rows, num_columns, epsilon)
    if costs.dtype == np.dtype(np.float64):
        return _dense_hungarian_double(costs, num_rows, num_columns, epsilon)
    if costs.dtype == np.dtype(np.int32):
        return _dense_hungarian_int(costs, num_rows, num_columns)
    raise TypeError("costs must be int32, float32, or float64")


cdef tuple _minimum_spanning_tree_from_csr_float(
    object offsets,
    object indices,
    object weights,
    int num_vertices,
    int num_edges,
):
    cdef int* offsets_ptr = <int*>_device_pointer(offsets, "offsets")
    cdef int* indices_ptr = <int*>_device_pointer(indices, "indices")
    cdef float* weights_ptr = <float*>_device_pointer(weights, "weights")
    cdef GraphCSRView[int, int, float] graph
    cdef unique_ptr[GraphCOO[int, int, float]] result
    cdef GraphCOOView[int, int, float] result_view
    cdef handle_t handle

    graph = GraphCSRView[int, int, float](
        offsets_ptr,
        indices_ptr,
        weights_ptr,
        <size_t>num_vertices,
        <size_t>num_edges,
    )
    result = c_minimum_spanning_tree[int, int, float](handle, graph)
    result_view = result.get().view()
    sources = _copy_device_array(
        <const void*>result_view.src_indices,
        <size_t>result_view.number_of_edges,
        np.int32,
        sizeof(int),
        handle.get_stream(),
    )
    destinations = _copy_device_array(
        <const void*>result_view.dst_indices,
        <size_t>result_view.number_of_edges,
        np.int32,
        sizeof(int),
        handle.get_stream(),
    )
    result_weights = _copy_device_array(
        <const void*>result_view.edge_data,
        <size_t>result_view.number_of_edges,
        np.float32,
        sizeof(float),
        handle.get_stream(),
    )
    return sources, destinations, result_weights


cdef tuple _minimum_spanning_tree_from_csr_double(
    object offsets,
    object indices,
    object weights,
    int num_vertices,
    int num_edges,
):
    cdef int* offsets_ptr = <int*>_device_pointer(offsets, "offsets")
    cdef int* indices_ptr = <int*>_device_pointer(indices, "indices")
    cdef double* weights_ptr = <double*>_device_pointer(weights, "weights")
    cdef GraphCSRView[int, int, double] graph
    cdef unique_ptr[GraphCOO[int, int, double]] result
    cdef GraphCOOView[int, int, double] result_view
    cdef handle_t handle

    graph = GraphCSRView[int, int, double](
        offsets_ptr,
        indices_ptr,
        weights_ptr,
        <size_t>num_vertices,
        <size_t>num_edges,
    )
    result = c_minimum_spanning_tree[int, int, double](handle, graph)
    result_view = result.get().view()
    sources = _copy_device_array(
        <const void*>result_view.src_indices,
        <size_t>result_view.number_of_edges,
        np.int32,
        sizeof(int),
        handle.get_stream(),
    )
    destinations = _copy_device_array(
        <const void*>result_view.dst_indices,
        <size_t>result_view.number_of_edges,
        np.int32,
        sizeof(int),
        handle.get_stream(),
    )
    result_weights = _copy_device_array(
        <const void*>result_view.edge_data,
        <size_t>result_view.number_of_edges,
        np.float64,
        sizeof(double),
        handle.get_stream(),
    )
    return sources, destinations, result_weights


def minimum_spanning_tree_from_csr(
    offsets,
    indices,
    weights,
    int num_vertices,
    int num_edges,
):
    _require_dtype(offsets, np.int32, "offsets")
    _require_dtype(indices, np.int32, "indices")
    if weights.dtype == np.dtype(np.float32):
        return _minimum_spanning_tree_from_csr_float(
            offsets,
            indices,
            weights,
            num_vertices,
            num_edges,
        )
    if weights.dtype == np.dtype(np.float64):
        return _minimum_spanning_tree_from_csr_double(
            offsets,
            indices,
            weights,
            num_vertices,
            num_edges,
        )
    raise TypeError("weights must be float32 or float64")


cdef object _get_traversed_cost_float(
    object vertices,
    object predecessors,
    object info_weights,
    int stop_vertex,
):
    cdef int* vertex_ptr = <int*>_device_pointer(vertices, "vertices")
    cdef int* pred_ptr = <int*>_device_pointer(predecessors, "predecessors")
    cdef float* weight_ptr = <float*>_device_pointer(info_weights, "info_weights")
    cdef float* output_ptr
    cdef size_t vertex_count = len(vertices)
    cdef handle_t handle

    output = _empty_device_array(vertex_count, np.float32)
    output_ptr = <float*>_device_pointer(output, "output")
    c_get_traversed_cost[int, float](
        handle,
        vertex_ptr,
        pred_ptr,
        weight_ptr,
        output_ptr,
        stop_vertex,
        <int>vertex_count,
    )
    handle.sync_stream()
    return output


cdef object _get_traversed_cost_double(
    object vertices,
    object predecessors,
    object info_weights,
    int stop_vertex,
):
    cdef int* vertex_ptr = <int*>_device_pointer(vertices, "vertices")
    cdef int* pred_ptr = <int*>_device_pointer(predecessors, "predecessors")
    cdef double* weight_ptr = <double*>_device_pointer(info_weights, "info_weights")
    cdef double* output_ptr
    cdef size_t vertex_count = len(vertices)
    cdef handle_t handle

    output = _empty_device_array(vertex_count, np.float64)
    output_ptr = <double*>_device_pointer(output, "output")
    c_get_traversed_cost[int, double](
        handle,
        vertex_ptr,
        pred_ptr,
        weight_ptr,
        output_ptr,
        stop_vertex,
        <int>vertex_count,
    )
    handle.sync_stream()
    return output


def get_traversed_cost(vertices, predecessors, info_weights, int stop_vertex):
    _require_same_length(vertices, "vertices", predecessors, "predecessors")
    _require_same_length(vertices, "vertices", info_weights, "info_weights")
    _require_dtype(vertices, np.int32, "vertices")
    _require_dtype(predecessors, np.int32, "predecessors")
    if info_weights.dtype == np.dtype(np.float32):
        return _get_traversed_cost_float(
            vertices,
            predecessors,
            info_weights,
            stop_vertex,
        )
    if info_weights.dtype == np.dtype(np.float64):
        return _get_traversed_cost_double(
            vertices,
            predecessors,
            info_weights,
            stop_vertex,
        )
    raise TypeError("info_weights must be float32 or float64")


def comms_bcast(handle, values):
    cdef handle_t* handle_ptr = _handle_pointer(handle)
    cdef size_t count = len(values)

    if values.dtype == np.dtype(np.int32):
        c_comms_bcast[int](
            handle_ptr[0],
            <int*>_device_pointer(values, "values"),
            count,
        )
    elif values.dtype == np.dtype(np.float32):
        c_comms_bcast[float](
            handle_ptr[0],
            <float*>_device_pointer(values, "values"),
            count,
        )
    elif values.dtype == np.dtype(np.float64):
        c_comms_bcast[double](
            handle_ptr[0],
            <double*>_device_pointer(values, "values"),
            count,
        )
    else:
        raise TypeError("values must be int32, float32, or float64")
    handle_ptr[0].sync_stream()
