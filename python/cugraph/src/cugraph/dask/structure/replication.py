# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import cudf
import dask.distributed as dd
import dask_cudf
import numpy as np
from pylibcugraph import legacy as plc_legacy

import cugraph.dask.comms.comms as Comms
import cugraph.dask.common.mg_utils as mg_utils
from cugraph.dask.common.input_utils import get_mg_batch_data


def replicate_cudf_dataframe(cudf_dataframe, client=None, comms=None):
    if type(cudf_dataframe) is not cudf.DataFrame:
        raise TypeError("Expected a cudf.DataFrame to replicate")
    client = mg_utils.get_client() if client is None else client
    comms = Comms.get_comms() if comms is None else comms
    dask_cudf_df = dask_cudf.from_cudf(cudf_dataframe, npartitions=1)
    df_length = len(dask_cudf_df)

    _df_data = get_mg_batch_data(dask_cudf_df, batch_enabled=True)
    df_data = mg_utils.prepare_worker_to_parts(_df_data, client)

    workers_to_futures = {
        worker: client.submit(
            _replicate_cudf_dataframe,
            (data, cudf_dataframe.columns.values, cudf_dataframe.dtypes, df_length),
            comms.sessionId,
            workers=[worker],
        )
        for (worker, data) in df_data.worker_to_parts.items()
    }
    dd.wait(workers_to_futures)
    return workers_to_futures


def _replicate_cudf_dataframe(input_data, session_id):
    handle = Comms.get_handle(session_id)
    _data, columns, dtypes, df_length = input_data
    data = _data[0]
    has_data = type(data) is cudf.DataFrame

    df_data = {}
    for idx, column in enumerate(columns):
        if has_data:
            series = data[column]
        else:
            series = cudf.Series(np.zeros(df_length), dtype=dtypes[idx])
            df_data[column] = series
        plc_legacy.comms_bcast(handle, series)

    if has_data:
        return data
    return cudf.DataFrame(data=df_data)


def replicate_cudf_series(cudf_series, client=None, comms=None):
    if type(cudf_series) is not cudf.Series:
        raise TypeError("Expected a cudf.Series to replicate")
    client = mg_utils.get_client() if client is None else client
    comms = Comms.get_comms() if comms is None else comms
    dask_cudf_series = dask_cudf.from_cudf(cudf_series, npartitions=1)
    series_length = len(dask_cudf_series)
    _series_data = get_mg_batch_data(dask_cudf_series, batch_enabled=True)
    series_data = mg_utils.prepare_worker_to_parts(_series_data)

    dtype = cudf_series.dtype
    workers_to_futures = {
        worker: client.submit(
            _replicate_cudf_series,
            (data, series_length, dtype),
            comms.sessionId,
            workers=[worker],
        )
        for (worker, data) in series_data.worker_to_parts.items()
    }
    dd.wait(workers_to_futures)
    return workers_to_futures


def _replicate_cudf_series(input_data, session_id):
    handle = Comms.get_handle(session_id)
    _data, size, dtype = input_data

    data = _data[0]
    if type(data) is cudf.Series:
        result = data
    else:
        result = cudf.Series(np.zeros(size), dtype=dtype)

    plc_legacy.comms_bcast(handle, result)
    return result
