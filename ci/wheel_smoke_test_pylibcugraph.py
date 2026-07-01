# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import cupy
from pylibcugraph import GraphProperties, ResourceHandle, SGGraph


if __name__ == "__main__":
    src_array = cupy.asarray([0, 1, 2, 2], dtype="int32")
    dst_array = cupy.asarray([1, 2, 3, 0], dtype="int32")
    wgt_array = cupy.asarray([1.0, 1.0, 1.0, 1.0], dtype="float32")

    resource_handle = ResourceHandle()

    G = SGGraph(
        resource_handle,
        GraphProperties(is_symmetric=False, is_multigraph=False),
        src_array,
        dst_array,
        wgt_array,
        store_transposed=False,
        renumber=True,
        do_expensive_check=True,
    )

    assert G is not None
