# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import enum

import cudf
import numpy as np
from pylibcugraph import legacy as plc_legacy


def datatype_cast(cols, dtypes):
    cols_out = []
    for col in cols:
        if col is None or col.dtype.type in dtypes:
            cols_out.append(col)
        else:
            cols_out.append(col.astype(dtypes[0]))
    return cols_out


class Direction(enum.Enum):
    ALL = 0
    IN = 1
    OUT = 2


def coo2csr(source_col, dest_col, weights=None):
    if len(source_col) != len(dest_col):
        raise ValueError("source_col and dest_col should have the same number of elements")
    if source_col.dtype != dest_col.dtype:
        raise TypeError("source_col and dest_col should be the same type")
    if source_col.dtype != np.int32:
        raise TypeError("source_col and dest_col must be type np.int32")
    if len(source_col) == 0:
        return (
            cudf.Series(np.zeros(1, dtype=np.int32)),
            cudf.Series(np.zeros(1, dtype=np.int32)),
            weights,
        )

    offsets, indices, csr_weights = plc_legacy.coo_to_csr(source_col, dest_col, weights)
    return cudf.Series(offsets), cudf.Series(indices), (
        None if csr_weights is None else cudf.Series(csr_weights)
    )


def view_adj_list(input_graph):
    if input_graph.adjlist is not None:
        return None
    if input_graph.edgelist is None:
        raise ValueError("Graph is Empty")

    src = input_graph.edgelist.edgelist_df["src"]
    dst = input_graph.edgelist.edgelist_df["dst"]
    src, dst = datatype_cast([src, dst], [np.int32])
    weights = None
    if input_graph.edgelist.weights:
        (weights,) = datatype_cast(
            [input_graph.edgelist.edgelist_df["weights"]],
            [np.float32, np.float64],
        )
    return coo2csr(src, dst, weights)


def view_transposed_adj_list(input_graph):
    if input_graph.transposedadjlist is not None:
        return None
    if input_graph.edgelist is None:
        if input_graph.adjlist is None:
            raise ValueError("Graph is Empty")
        input_graph.view_edge_list()

    src = input_graph.edgelist.edgelist_df["src"]
    dst = input_graph.edgelist.edgelist_df["dst"]
    src, dst = datatype_cast([src, dst], [np.int32])
    weights = None
    if input_graph.edgelist.weights:
        (weights,) = datatype_cast(
            [input_graph.edgelist.edgelist_df["weights"]],
            [np.float32, np.float64],
        )
    return coo2csr(dst, src, weights)


def view_edge_list(input_graph):
    if input_graph.adjlist is None:
        raise RuntimeError("Graph is Empty")

    offsets, indices = datatype_cast(
        [input_graph.adjlist.offsets, input_graph.adjlist.indices],
        [np.int32],
    )
    weights = input_graph.adjlist.weights
    if weights is not None:
        (weights,) = datatype_cast([weights], [np.float32, np.float64])

    src, dst, weights = plc_legacy.csr_to_edgelist(offsets, indices, weights)
    return cudf.Series(src), dst, weights


def _degree(input_graph, direction=Direction.ALL):
    if direction == Direction.IN:
        df = input_graph.in_degree()
        return df["vertex"], df["degree"]
    if direction == Direction.OUT:
        df = input_graph.out_degree()
        return df["vertex"], df["degree"]
    df = input_graph.degree()
    return df["vertex"], df["degree"]


def _degrees(input_graph):
    df = input_graph.degrees()
    return df["vertex"], df["in_degree"], df["out_degree"]


def weight_type(input_graph):
    if input_graph.edgelist is not None and input_graph.edgelist.weights:
        return input_graph.edgelist.edgelist_df["weights"].dtype
    if input_graph.adjlist is not None and input_graph.adjlist.weights is not None:
        return input_graph.adjlist.weights.dtype
    return None
