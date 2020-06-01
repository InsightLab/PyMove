import os

from dask.dataframe import DataFrame as DaskDataFrame
from dask.dataframe import from_pandas
from pandas import DataFrame, Timestamp

from pymove import DaskMoveDataFrame, MoveDataFrame, PandasMoveDataFrame, read_csv
from pymove.utils.constants import (
    DATETIME,
    LATITUDE,
    LONGITUDE,
    TRAJ_ID,
    TYPE_DASK,
    TYPE_PANDAS,
)

list_data = [
    [39.984094, 116.319236, '2008-10-23 05:53:05', 1],
    [39.984198, 116.319322, '2008-10-23 05:53:06', 1],
    [39.984224, 116.319402, '2008-10-23 05:53:11', 2],
    [39.984224, 116.319402, '2008-10-23 05:53:11', 2],
]

str_data_default = """
lat,lon,datetime,id
39.984093,116.319236,2008-10-23 05:53:05,4
39.9842,116.319322,2008-10-23 05:53:06,1
39.984222,116.319402,2008-10-23 05:53:11,2
39.984222,116.319402,2008-10-23 05:53:11,2
"""

str_data_different = """
latitude,longitude,time,traj_id
39.984093,116.319236,2008-10-23 05:53:05,4
39.9842,116.319322,2008-10-23 05:53:06,1
39.984222,116.319402,2008-10-23 05:53:11,2
39.984222,116.319402,2008-10-23 05:53:11,2
"""

str_data_missing = """
39.984093,116.319236,2008-10-23 05:53:05,4
39.9842,116.319322,2008-10-23 05:53:06,1
39.984222,116.319402,2008-10-23 05:53:11,2
39.984222,116.319402,2008-10-23 05:53:11,2
"""


def _default_move_df():
    return MoveDataFrame(
        data=list_data,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        datetime=DATETIME,
        traj_id=TRAJ_ID,
        type_=TYPE_DASK,
    )


def _default_dask_df():
    df = DataFrame(
        data=[
            [39.984094, 116.319236, Timestamp('2008-10-23 05:53:05'), 1],
            [39.984198, 116.319322, Timestamp('2008-10-23 05:53:06'), 1],
            [39.984224, 116.319402, Timestamp('2008-10-23 05:53:11'), 2],
            [39.984224, 116.319402, Timestamp('2008-10-23 05:53:11'), 2],
        ],
        columns=['lat', 'lon', 'datetime', 'id'],
        index=[0, 1, 2, 3],
    )
    return from_pandas(df, npartitions=1)


def _has_columns(data):
    cols = data.columns
    if LATITUDE in cols and LONGITUDE in cols and DATETIME in cols:
        return True
    return False


def _validate_move_data_frame_data(data):
    try:
        if data.dtypes.lat != 'float64':
            return False
        if data.dtypes.lon != 'float64':
            return False
        if data.dtypes.datetime != 'datetime64[ns]':
            return False
        return True
    except AttributeError:
        print(AttributeError)


def test_move_data_frame_from_list():
    move_df = _default_move_df()
    assert _has_columns(move_df)
    assert _validate_move_data_frame_data(move_df)
    assert isinstance(move_df, DaskMoveDataFrame)


def test_move_data_frame_from_file(tmpdir):
    d = tmpdir.mkdir('prepossessing')

    file_default_columns = d.join('test_read_default.csv')
    file_default_columns.write(str_data_default)
    filename_default = os.path.join(
        file_default_columns.dirname, file_default_columns.basename
    )

    move_df = read_csv(filename_default, type_=TYPE_DASK)
    assert _has_columns(move_df)
    assert _validate_move_data_frame_data(move_df)
    assert isinstance(move_df, DaskMoveDataFrame)

    file_different_columns = d.join('test_read_different.csv')
    file_different_columns.write(str_data_different)
    filename_diferent = os.path.join(
        file_different_columns.dirname, file_different_columns.basename
    )

    move_df = read_csv(
        filename_diferent,
        latitude='latitude',
        longitude='longitude',
        datetime='time',
        traj_id='traj_id',
        type_=TYPE_DASK,
    )
    assert _has_columns(move_df)
    assert _validate_move_data_frame_data(move_df)
    assert isinstance(move_df, DaskMoveDataFrame)

    file_missing_columns = d.join('test_read_missing.csv')
    file_missing_columns.write(str_data_missing)
    filename_missing = os.path.join(
        file_missing_columns.dirname, file_missing_columns.basename
    )

    move_df = read_csv(
        filename_missing,
        names=[LATITUDE, LONGITUDE, DATETIME, TRAJ_ID],
        type_=TYPE_DASK,
    )
    assert _has_columns(move_df)
    assert _validate_move_data_frame_data(move_df)
    assert isinstance(move_df, DaskMoveDataFrame)


def test_move_data_frame_from_dict():
    dict_data = {
        LATITUDE: [39.984198, 39.984224, 39.984094],
        LONGITUDE: [116.319402, 116.319322, 116.319402],
        DATETIME: [
            '2008-10-23 05:53:11',
            '2008-10-23 05:53:06',
            '2008-10-23 05:53:06',
        ],
    }
    move_df = MoveDataFrame(
        data=dict_data,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        datetime=DATETIME,
        traj_id=TRAJ_ID,
        type_=TYPE_DASK,
    )
    assert _has_columns(move_df)
    assert _validate_move_data_frame_data(move_df)
    assert isinstance(move_df, DaskMoveDataFrame)


def test_move_data_frame_from_data_frame():
    df = DataFrame(
        data=[
            [39.984094, 116.319236, Timestamp('2008-10-23 05:53:05'), 1],
            [39.984198, 116.319322, Timestamp('2008-10-23 05:53:06'), 1],
            [39.984224, 116.319402, Timestamp('2008-10-23 05:53:11'), 2],
            [39.984224, 116.319402, Timestamp('2008-10-23 05:53:11'), 2],
        ],
        columns=['lat', 'lon', 'datetime', 'id'],
        index=[0, 1, 2, 3],
    )
    move_df = MoveDataFrame(
        data=df,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        datetime=DATETIME,
        type_=TYPE_DASK,
    )
    assert _has_columns(move_df)
    assert _validate_move_data_frame_data(move_df)
    assert isinstance(move_df, DaskMoveDataFrame)


def test_attribute_error_from_data_frame():
    df = DataFrame(
        data=[
            [39.984094, 116.319236, Timestamp('2008-10-23 05:53:05'), 1],
            [39.984198, 116.319322, Timestamp('2008-10-23 05:53:06'), 1],
            [39.984224, 116.319402, Timestamp('2008-10-23 05:53:11'), 2],
            [39.984224, 116.319402, Timestamp('2008-10-23 05:53:11'), 2],
        ],
        columns=['laterr', 'lon', 'datetime', 'id'],
        index=[0, 1, 2, 3],
    )
    try:
        MoveDataFrame(
            data=df,
            latitude=LATITUDE,
            longitude=LONGITUDE,
            datetime=DATETIME,
            type_=TYPE_DASK,
        )
        raise AssertionError(
            'AttributeError error not raised by MoveDataFrame'
        )
    except AttributeError:
        pass

    df = DataFrame(
        data=[
            [39.984094, 116.319236, Timestamp('2008-10-23 05:53:05'), 1],
            [39.984198, 116.319322, Timestamp('2008-10-23 05:53:06'), 1],
            [39.984224, 116.319402, Timestamp('2008-10-23 05:53:11'), 2],
            [39.984224, 116.319402, Timestamp('2008-10-23 05:53:11'), 2],
        ],
        columns=['lat', 'lonerr', 'datetime', 'id'],
        index=[0, 1, 2, 3],
    )
    try:
        MoveDataFrame(
            data=df,
            latitude=LATITUDE,
            longitude=LONGITUDE,
            datetime=DATETIME,
            type_=TYPE_DASK,
        )
        raise AssertionError(
            'AttributeError error not raised by MoveDataFrame'
        )
    except AttributeError:
        pass

    df = DataFrame(
        data=[
            [39.984094, 116.319236, Timestamp('2008-10-23 05:53:05'), 1],
            [39.984198, 116.319322, Timestamp('2008-10-23 05:53:06'), 1],
            [39.984224, 116.319402, Timestamp('2008-10-23 05:53:11'), 2],
            [39.984224, 116.319402, Timestamp('2008-10-23 05:53:11'), 2],
        ],
        columns=['lat', 'lon', 'datetimerr', 'id'],
        index=[0, 1, 2, 3],
    )
    try:
        MoveDataFrame(
            data=df,
            latitude=LATITUDE,
            longitude=LONGITUDE,
            datetime=DATETIME,
            type_=TYPE_DASK,
        )
        raise AssertionError(
            'AttributeError error not raised by MoveDataFrame'
        )
    except AttributeError:
        pass


def test_convert_to():
    move_df = _default_move_df()

    assert move_df._type == TYPE_DASK
    assert isinstance(move_df, DaskMoveDataFrame)
    assert isinstance(move_df._data, DaskDataFrame)

    move_df_pandas = move_df.convert_to('pandas')
    assert move_df_pandas._type == TYPE_PANDAS
    assert isinstance(move_df_pandas, PandasMoveDataFrame)

    assert move_df._type == TYPE_DASK
    assert isinstance(move_df, DaskMoveDataFrame)
    assert isinstance(move_df._data, DaskDataFrame)


def test_get_type():
    move_df = _default_move_df()

    assert move_df.get_type() == TYPE_DASK
