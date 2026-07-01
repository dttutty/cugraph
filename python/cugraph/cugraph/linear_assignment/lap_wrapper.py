# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import cudf
import numpy as np
from pylibcugraph import legacy as plc_legacy

from cugraph.structure import graph_primtypes_wrapper


def sparse_hungarian(input_graph, workers, epsilon):
    if not input_graph.edgelist:
        input_graph.view_edge_list()
    if input_graph.edgelist.weights is None:
        raise ValueError("hungarian algorithm requires weighted graph")

    src = input_graph.edgelist.edgelist_df["src"]
    dst = input_graph.edgelist.edgelist_df["dst"]
    weights = input_graph.edgelist.edgelist_df["weights"]

    src, dst = graph_primtypes_wrapper.datatype_cast([src, dst], [np.int32])
    (weights,) = graph_primtypes_wrapper.datatype_cast([weights], [np.float32, np.float64])
    (local_workers,) = graph_primtypes_wrapper.datatype_cast([workers], [np.int32])

    cost, assignments = plc_legacy.sparse_hungarian(
        src,
        dst,
        weights,
        local_workers,
        input_graph.number_of_vertices(),
        epsilon,
    )

    df = cudf.DataFrame()
    df["vertex"] = workers
    df["assignment"] = cudf.Series(assignments)
    return cost, df


def dense_hungarian(costs, num_rows, num_columns, epsilon):
    if type(costs) is not cudf.Series:
        raise TypeError("costs must be a cudf.Series")

    cost, assignment = plc_legacy.dense_hungarian(
        costs,
        num_rows,
        num_columns,
        epsilon,
    )
    return cost, cudf.Series(assignment)
