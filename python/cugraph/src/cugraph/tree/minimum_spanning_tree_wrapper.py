# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import cudf
import cupy as cp
import numpy as np
from pylibcugraph import legacy as plc_legacy

from cugraph.structure import graph_primtypes_wrapper


def minimum_spanning_tree(input_graph):
    if not input_graph.adjlist:
        input_graph.view_adj_list()

    offsets, indices = graph_primtypes_wrapper.datatype_cast(
        [input_graph.adjlist.offsets, input_graph.adjlist.indices],
        [np.int32],
    )
    num_verts = input_graph.number_of_vertices()
    num_edges = input_graph.number_of_edges(directed_edges=True)

    if input_graph.adjlist.weights is not None:
        (weights,) = graph_primtypes_wrapper.datatype_cast(
            [input_graph.adjlist.weights],
            [np.float32, np.float64],
        )
    else:
        weights = cudf.Series(cp.full(num_edges, 1.0, dtype=np.float32))

    src, dst, edge_weights = plc_legacy.minimum_spanning_tree_from_csr(
        offsets,
        indices,
        weights,
        num_verts,
        num_edges,
    )

    df = cudf.DataFrame()
    df["src"] = cudf.Series(src)
    df["dst"] = cudf.Series(dst)
    df["weight"] = cudf.Series(edge_weights)
    return df


def maximum_spanning_tree(input_graph):
    return minimum_spanning_tree(input_graph)
