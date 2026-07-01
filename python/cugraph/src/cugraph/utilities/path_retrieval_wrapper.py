# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import cudf
import numpy as np
from pylibcugraph import legacy as plc_legacy


def get_traversed_cost(input_df, stop_vertex):
    vertices = input_df["vertex"]
    predecessors = input_df["predecessor"]
    if vertices.dtype != np.int32:
        vertices = vertices.astype(np.int32)
    if predecessors.dtype != np.int32:
        predecessors = predecessors.astype(np.int32)

    info = plc_legacy.get_traversed_cost(
        vertices,
        predecessors,
        input_df["weights"],
        int(stop_vertex),
    )

    df = cudf.DataFrame()
    df["vertex"] = input_df["vertex"]
    df["info"] = cudf.Series(info)
    return df
