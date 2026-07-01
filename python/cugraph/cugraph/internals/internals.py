# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from numba.cuda.api import from_cuda_array_interface


class PyCallback:
    def get_numba_matrix(self, positions, shape, typestr):
        itemsize = 4 if typestr == "float32" else 8
        desc = {
            "shape": shape,
            "strides": (itemsize, shape[0] * itemsize),
            "typestr": typestr,
            "data": [positions],
            "order": "C",
            "version": 1,
        }
        return from_cuda_array_interface(desc)


class GraphBasedDimRedCallback(PyCallback):
    def get_native_callback(self):
        raise RuntimeError(
            "GraphBasedDimRedCallback no longer exposes a native callback. "
            "force_atlas2 callback support was removed in version 25.10."
        )
