import os
from datetime import date

from dask.dataframe import DataFrame as DaskDataFrame
from matplotlib.testing.compare import compare_images
from numpy import nan, ndarray
from numpy.testing import assert_allclose, assert_array_equal
from pandas import DataFrame, Series, Timedelta, Timestamp
from pandas.testing import assert_frame_equal, assert_series_equal

import pymove
from pymove import DaskMoveDataFrame, MoveDataFrame, PandasMoveDataFrame, read_csv
from pymove.utils.constants import (
    DATE,
    DATETIME,
    DAY,
    DIST_PREV_TO_NEXT,
    DIST_TO_PREV,
    HOUR,
    HOUR_SIN,
    LATITUDE,
    LONGITUDE,
    PERIOD,
    SITUATION,
    SPEED_PREV_TO_NEXT,
    TID,
    TIME_PREV_TO_NEXT,
    TRAJ_ID,
    TYPE_DASK,
    TYPE_PANDAS,
    UID,
    WEEK_END,
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
    )


def _default_pandas_df():
    return DataFrame(
        data=[
            [39.984094, 116.319236, Timestamp('2008-10-23 05:53:05'), 1],
            [39.984198, 116.319322, Timestamp('2008-10-23 05:53:06'), 1],
            [39.984224, 116.319402, Timestamp('2008-10-23 05:53:11'), 2],
            [39.984224, 116.319402, Timestamp('2008-10-23 05:53:11'), 2],
        ],
        columns=['lat', 'lon', 'datetime', 'id'],
        index=[0, 1, 2, 3],
    )


def _has_columns(data):
    if LATITUDE in data and LONGITUDE in data and DATETIME in data:
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
    assert isinstance(move_df, PandasMoveDataFrame)


def test_move_data_frame_from_file(tmpdir):
    d = tmpdir.mkdir('prepossessing')

    file_default_columns = d.join('test_read_default.csv')
    file_default_columns.write(str_data_default)
    filename_default = os.path.join(
        file_default_columns.dirname, file_default_columns.basename
    )

    move_df = read_csv(filename_default)
    assert _has_columns(move_df)
    assert _validate_move_data_frame_data(move_df)
    assert isinstance(move_df, PandasMoveDataFrame)

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
    )
    assert _has_columns(move_df)
    assert _validate_move_data_frame_data(move_df)
    assert isinstance(move_df, PandasMoveDataFrame)

    file_missing_columns = d.join('test_read_missing.csv')
    file_missing_columns.write(str_data_missing)
    filename_missing = os.path.join(
        file_missing_columns.dirname, file_missing_columns.basename
    )

    move_df = read_csv(
        filename_missing, names=[LATITUDE, LONGITUDE, DATETIME, TRAJ_ID]
    )
    assert _has_columns(move_df)
    assert _validate_move_data_frame_data(move_df)
    assert isinstance(move_df, PandasMoveDataFrame)


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
    )
    assert _has_columns(move_df)
    assert _validate_move_data_frame_data(move_df)
    assert isinstance(move_df, PandasMoveDataFrame)


def test_move_data_frame_from_data_frame():
    df = _default_pandas_df()
    move_df = MoveDataFrame(
        data=df, latitude=LATITUDE, longitude=LONGITUDE, datetime=DATETIME
    )
    assert _has_columns(move_df)
    assert _validate_move_data_frame_data(move_df)
    assert isinstance(move_df, PandasMoveDataFrame)


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
            data=df, latitude=LATITUDE, longitude=LONGITUDE, datetime=DATETIME
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
            data=df, latitude=LATITUDE, longitude=LONGITUDE, datetime=DATETIME
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
            data=df, latitude=LATITUDE, longitude=LONGITUDE, datetime=DATETIME
        )
        raise AssertionError(
            'AttributeError error not raised by MoveDataFrame'
        )
    except AttributeError:
        pass


def test_lat():
    move_df = _default_move_df()

    lat = move_df.lat
    srs = Series(
        data=[39.984094, 39.984198, 39.984224, 39.984224],
        index=[0, 1, 2, 3],
        dtype='float64',
        name='lat',
    )
    assert_series_equal(lat, srs)


def test_lon():
    move_df = _default_move_df()

    lon = move_df.lon
    srs = Series(
        data=[116.319236, 116.319322, 116.319402, 116.319402],
        index=[0, 1, 2, 3],
        dtype='float64',
        name='lon',
    )
    assert_series_equal(lon, srs)


def test_datetime():
    move_df = _default_move_df()

    datetime = move_df.datetime
    srs = Series(
        data=[
            '2008-10-23 05:53:05',
            '2008-10-23 05:53:06',
            '2008-10-23 05:53:11',
            '2008-10-23 05:53:11',
        ],
        index=[0, 1, 2, 3],
        dtype='datetime64[ns]',
        name='datetime',
    )
    assert_series_equal(datetime, srs)


def test_loc():
    move_df = _default_move_df()

    assert move_df.loc[0, TRAJ_ID] == 1

    loc_ = move_df.loc[move_df[LONGITUDE] > 116.319321]
    expected = DataFrame(
        data=[
            [39.984198, 116.319322, Timestamp('2008-10-23 05:53:06'), 1],
            [39.984224, 116.319402, Timestamp('2008-10-23 05:53:11'), 2],
            [39.984224, 116.319402, Timestamp('2008-10-23 05:53:11'), 2],
        ],
        columns=['lat', 'lon', 'datetime', 'id'],
        index=[1, 2, 3],
    )
    assert_frame_equal(loc_, expected)


def test_iloc():
    move_df = _default_move_df()

    expected = Series(
        data=[39.984094, 116.319236, Timestamp('2008-10-23 05:53:05'), 1],
        index=['lat', 'lon', 'datetime', 'id'],
        dtype='object',
        name=0,
    )

    assert_series_equal(move_df.iloc[0], expected)


def test_at():
    move_df = _default_move_df()

    assert move_df.at[0, TRAJ_ID] == 1


def test_values():
    move_df = _default_move_df()

    expected = [
        [39.984094, 116.319236, Timestamp('2008-10-23 05:53:05'), 1],
        [39.984198, 116.319322, Timestamp('2008-10-23 05:53:06'), 1],
        [39.984224, 116.319402, Timestamp('2008-10-23 05:53:11'), 2],
        [39.984224, 116.319402, Timestamp('2008-10-23 05:53:11'), 2],
    ]
    assert_array_equal(move_df.values, expected)


def test_columns():
    move_df = _default_move_df()

    assert_array_equal(
        move_df.columns, [LATITUDE, LONGITUDE, DATETIME, TRAJ_ID]
    )


def test_index():
    move_df = _default_move_df()

    assert_array_equal(move_df.index, [0, 1, 2, 3])


def test_dtypes():
    move_df = _default_move_df()

    expected = Series(
        data=['float64', 'float64', '<M8[ns]', 'int64'],
        index=['lat', 'lon', 'datetime', 'id'],
        dtype='object',
        name=None,
    )
    assert_series_equal(move_df.dtypes, expected)


def test_shape():
    move_df = _default_move_df()

    assert move_df.shape == (4, 4)


def test_len():
    move_df = _default_move_df()

    assert move_df.len() == 4


def test_unique():
    move_df = _default_move_df()
    assert_array_equal(move_df['id'].unique(), [1, 2])


def test_head():
    move_df = _default_move_df()

    expected = DataFrame(
        data=[
            [39.984094, 116.319236, Timestamp('2008-10-23 05:53:05'), 1],
            [39.984198, 116.319322, Timestamp('2008-10-23 05:53:06'), 1],
        ],
        columns=['lat', 'lon', 'datetime', 'id'],
        index=[0, 1],
    )
    assert_frame_equal(move_df.head(2), expected)


def test_tail():
    move_df = _default_move_df()

    expected = DataFrame(
        data=[
            [39.984224, 116.319402, Timestamp('2008-10-23 05:53:11'), 2],
            [39.984224, 116.319402, Timestamp('2008-10-23 05:53:11'), 2],
        ],
        columns=['lat', 'lon', 'datetime', 'id'],
        index=[2, 3],
    )
    assert_frame_equal(move_df.tail(2), expected)


def test_number_users():
    move_df = MoveDataFrame(
        data=list_data,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        datetime=DATETIME,
        traj_id=TRAJ_ID,
    )

    assert move_df.get_users_number() == 1

    move_df[UID] = [1, 1, 2, 3]
    assert move_df.get_users_number() == 3


def test_to_numpy():
    move_df = MoveDataFrame(
        data=list_data,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        datetime=DATETIME,
        traj_id=TRAJ_ID,
    )

    assert isinstance(move_df.to_numpy(), ndarray)


def test_to_dict():
    move_df = MoveDataFrame(
        data=list_data,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        datetime=DATETIME,
        traj_id=TRAJ_ID,
    )

    assert isinstance(move_df.to_dict(), dict)


def test_to_grid():
    move_df = MoveDataFrame(
        data=list_data,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        datetime=DATETIME,
        traj_id=TRAJ_ID,
    )

    assert isinstance(move_df.to_grid(8), pymove.core.grid.Grid)


def test_to_data_frame():
    move_df = MoveDataFrame(
        data=list_data,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        datetime=DATETIME,
        traj_id=TRAJ_ID,
    )

    assert isinstance(move_df.to_data_frame(), DataFrame)


def test_describe():
    move_df = _default_move_df()

    expected = DataFrame(
        data=[
            [4.0, 4.0, 4.0],
            [39.984185, 116.31934049999998, 1.5],
            [6.189237971348586e-05, 7.921910543639078e-05, 0.5773502691896257],
            [39.984094, 116.319236, 1.0],
            [39.984172, 116.3193005, 1.0],
            [39.984211, 116.319362, 1.5],
            [39.984224, 116.319402, 2.0],
            [39.984224, 116.319402, 2.0],
        ],
        columns=['lat', 'lon', 'id'],
        index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'],
    )
    assert_frame_equal(move_df.describe(), expected)


def test_memory_usage():
    move_df = _default_move_df()

    expected = Series(
        data=[128, 32, 32, 32, 32],
        index=['Index', 'lat', 'lon', 'datetime', 'id'],
        dtype='int64',
        name=None,
    )
    assert_series_equal(move_df.memory_usage(), expected)


def test_copy():
    move_df = _default_move_df()

    cp = move_df.copy()
    assert_frame_equal(move_df, cp)
    cp.at[0, TRAJ_ID] = 0
    assert move_df.loc[0, TRAJ_ID] == 1
    assert move_df.loc[0, TRAJ_ID] != cp.loc[0, TRAJ_ID]


def test_generate_tid_based_on_id_datetime():
    move_df = _default_move_df()

    new_move_df = move_df.generate_tid_based_on_id_datetime(inplace=False)
    expected = DataFrame(
        data=[
            [
                39.984094,
                116.319236,
                Timestamp('2008-10-23 05:53:05'),
                1,
                '12008102305',
            ],
            [
                39.984198,
                116.319322,
                Timestamp('2008-10-23 05:53:06'),
                1,
                '12008102305',
            ],
            [
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                2,
                '22008102305',
            ],
            [
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                2,
                '22008102305',
            ],
        ],
        columns=['lat', 'lon', 'datetime', 'id', 'tid'],
        index=[0, 1, 2, 3],
    )
    assert_frame_equal(new_move_df, expected)
    assert isinstance(new_move_df, PandasMoveDataFrame)
    assert TID not in move_df

    move_df.generate_tid_based_on_id_datetime()
    assert_frame_equal(move_df, expected)


def test_generate_date_features():
    move_df = _default_move_df()

    new_move_df = move_df.generate_date_features(inplace=False)
    expected = DataFrame(
        data=[
            [
                39.984094,
                116.319236,
                Timestamp('2008-10-23 05:53:05'),
                1,
                date(2008, 10, 23),
            ],
            [
                39.984198,
                116.319322,
                Timestamp('2008-10-23 05:53:06'),
                1,
                date(2008, 10, 23),
            ],
            [
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                2,
                date(2008, 10, 23),
            ],
            [
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                2,
                date(2008, 10, 23),
            ],
        ],
        columns=['lat', 'lon', 'datetime', 'id', 'date'],
        index=[0, 1, 2, 3],
    )
    assert_frame_equal(new_move_df, expected)
    assert isinstance(new_move_df, PandasMoveDataFrame)
    assert DATE not in move_df

    move_df.generate_date_features()
    assert_frame_equal(move_df, expected)


def test_generate_hour_features():
    move_df = _default_move_df()

    new_move_df = move_df.generate_hour_features(inplace=False)
    expected = DataFrame(
        data=[
            [39.984094, 116.319236, Timestamp('2008-10-23 05:53:05'), 1, 5],
            [39.984198, 116.319322, Timestamp('2008-10-23 05:53:06'), 1, 5],
            [39.984224, 116.319402, Timestamp('2008-10-23 05:53:11'), 2, 5],
            [39.984224, 116.319402, Timestamp('2008-10-23 05:53:11'), 2, 5],
        ],
        columns=['lat', 'lon', 'datetime', 'id', 'hour'],
        index=[0, 1, 2, 3],
    )
    assert_frame_equal(new_move_df, expected)
    assert isinstance(new_move_df, PandasMoveDataFrame)
    assert HOUR not in move_df

    move_df.generate_hour_features()
    assert_frame_equal(move_df, expected)


def test_generate_day_of_the_week_features():
    move_df = _default_move_df()

    new_move_df = move_df.generate_day_of_the_week_features(inplace=False)
    expected = DataFrame(
        data=[
            [
                39.984094,
                116.319236,
                Timestamp('2008-10-23 05:53:05'),
                1,
                'Thursday',
            ],
            [
                39.984198,
                116.319322,
                Timestamp('2008-10-23 05:53:06'),
                1,
                'Thursday',
            ],
            [
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                2,
                'Thursday',
            ],
            [
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                2,
                'Thursday',
            ],
        ],
        columns=['lat', 'lon', 'datetime', 'id', 'day'],
        index=[0, 1, 2, 3],
    )
    assert_frame_equal(new_move_df, expected)
    assert isinstance(new_move_df, PandasMoveDataFrame)
    assert DAY not in move_df

    move_df.generate_day_of_the_week_features()
    assert_frame_equal(move_df, expected)


def test_generate_weekend_features():
    move_df = _default_move_df()

    new_move_df = move_df.generate_weekend_features(inplace=False)
    expected = DataFrame(
        data=[
            [39.984094, 116.319236, Timestamp('2008-10-23 05:53:05'), 1, 0],
            [39.984198, 116.319322, Timestamp('2008-10-23 05:53:06'), 1, 0],
            [39.984224, 116.319402, Timestamp('2008-10-23 05:53:11'), 2, 0],
            [39.984224, 116.319402, Timestamp('2008-10-23 05:53:11'), 2, 0],
        ],
        columns=['lat', 'lon', 'datetime', 'id', 'weekend'],
        index=[0, 1, 2, 3],
    )
    assert_frame_equal(new_move_df, expected)
    assert isinstance(new_move_df, PandasMoveDataFrame)
    assert WEEK_END not in move_df

    move_df.generate_weekend_features()
    assert_frame_equal(move_df, expected)


def test_generate_time_of_day_features():
    move_df = _default_move_df()

    new_move_df = move_df.generate_time_of_day_features(inplace=False)
    expected = DataFrame(
        data=[
            [
                39.984094,
                116.319236,
                Timestamp('2008-10-23 05:53:05'),
                1,
                'Early morning',
            ],
            [
                39.984198,
                116.319322,
                Timestamp('2008-10-23 05:53:06'),
                1,
                'Early morning',
            ],
            [
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                2,
                'Early morning',
            ],
            [
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                2,
                'Early morning',
            ],
        ],
        columns=['lat', 'lon', 'datetime', 'id', 'period'],
        index=[0, 1, 2, 3],
    )
    assert_frame_equal(new_move_df, expected)
    assert isinstance(new_move_df, PandasMoveDataFrame)
    assert PERIOD not in move_df

    move_df.generate_time_of_day_features()
    assert_frame_equal(move_df, expected)


def test_generate_datetime_in_format_cyclical():
    move_df = _default_move_df()

    new_move_df = move_df.generate_datetime_in_format_cyclical(inplace=False)
    expected = DataFrame(
        data=[
            [
                39.984094,
                116.319236,
                Timestamp('2008-10-23 05:53:05'),
                1,
                0.9790840876823229,
                0.20345601305263375,
            ],
            [
                39.984198,
                116.319322,
                Timestamp('2008-10-23 05:53:06'),
                1,
                0.9790840876823229,
                0.20345601305263375,
            ],
            [
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                2,
                0.9790840876823229,
                0.20345601305263375,
            ],
            [
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                2,
                0.9790840876823229,
                0.20345601305263375,
            ],
        ],
        columns=['lat', 'lon', 'datetime', 'id', 'hour_sin', 'hour_cos'],
        index=[0, 1, 2, 3],
    )
    assert_frame_equal(new_move_df, expected)
    assert isinstance(new_move_df, PandasMoveDataFrame)
    assert HOUR_SIN not in move_df

    move_df.generate_datetime_in_format_cyclical()
    assert_frame_equal(move_df, expected)


def test_generate_dist_time_speed_features():
    move_df = _default_move_df()

    new_move_df = move_df.generate_dist_time_speed_features(inplace=False)
    expected = DataFrame(
        data=[
            [
                1,
                39.984094,
                116.319236,
                Timestamp('2008-10-23 05:53:05'),
                nan,
                nan,
                nan,
            ],
            [
                1,
                39.984198,
                116.319322,
                Timestamp('2008-10-23 05:53:06'),
                13.690153134343689,
                1.0,
                13.690153134343689,
            ],
            [
                2,
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                nan,
                nan,
                nan,
            ],
            [
                2,
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                0.0,
                0.0,
                nan,
            ],
        ],
        columns=[
            'id',
            'lat',
            'lon',
            'datetime',
            'dist_to_prev',
            'time_to_prev',
            'speed_to_prev',
        ],
        index=[0, 1, 2, 3],
    )
    assert_frame_equal(new_move_df, expected)
    assert isinstance(new_move_df, PandasMoveDataFrame)
    assert DIST_TO_PREV not in move_df

    move_df.generate_dist_time_speed_features()
    print(move_df)
    assert_frame_equal(move_df, expected)


def test_generate_dist_features():
    move_df = _default_move_df()

    new_move_df = move_df.generate_dist_features(inplace=False)
    expected = DataFrame(
        data=[
            [
                1,
                39.984094,
                116.319236,
                Timestamp('2008-10-23 05:53:05'),
                nan,
                13.690153134343689,
                nan,
            ],
            [
                1,
                39.984198,
                116.319322,
                Timestamp('2008-10-23 05:53:06'),
                13.690153134343689,
                nan,
                nan,
            ],
            [
                2,
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                nan,
                0.0,
                nan,
            ],
            [
                2,
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                0.0,
                nan,
                nan,
            ],
        ],
        columns=[
            'id',
            'lat',
            'lon',
            'datetime',
            'dist_to_prev',
            'dist_to_next',
            'dist_prev_to_next',
        ],
        index=[0, 1, 2, 3],
    )
    assert_frame_equal(new_move_df, expected)
    assert isinstance(new_move_df, PandasMoveDataFrame)
    assert DIST_PREV_TO_NEXT not in move_df

    move_df.generate_dist_features()
    print(move_df)
    assert_frame_equal(move_df, expected)


def test_generate_time_features():
    move_df = _default_move_df()

    new_move_df = move_df.generate_time_features(inplace=False)
    expected = DataFrame(
        data=[
            [
                1,
                39.984094,
                116.319236,
                Timestamp('2008-10-23 05:53:05'),
                nan,
                1.0,
                nan,
            ],
            [
                1,
                39.984198,
                116.319322,
                Timestamp('2008-10-23 05:53:06'),
                1.0,
                nan,
                nan,
            ],
            [
                2,
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                nan,
                0.0,
                nan,
            ],
            [
                2,
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                0.0,
                nan,
                nan,
            ],
        ],
        columns=[
            'id',
            'lat',
            'lon',
            'datetime',
            'time_to_prev',
            'time_to_next',
            'time_prev_to_next',
        ],
        index=[0, 1, 2, 3],
    )
    assert_frame_equal(new_move_df, expected)
    assert isinstance(new_move_df, PandasMoveDataFrame)
    assert TIME_PREV_TO_NEXT not in move_df

    move_df.generate_time_features()
    print(move_df)
    assert_frame_equal(move_df, expected)


def test_generate_speed_features():
    move_df = _default_move_df()

    new_move_df = move_df.generate_speed_features(inplace=False)
    expected = DataFrame(
        data=[
            [
                0,
                39.984094,
                116.319236,
                Timestamp('2008-10-23 05:53:05'),
                1,
                nan,
                13.690153134343689,
                nan,
            ],
            [
                1,
                39.984198,
                116.319322,
                Timestamp('2008-10-23 05:53:06'),
                1,
                13.690153134343689,
                nan,
                nan,
            ],
            [
                2,
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                2,
                nan,
                nan,
                nan,
            ],
            [
                3,
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                2,
                nan,
                nan,
                nan,
            ],
        ],
        columns=[
            'index',
            'lat',
            'lon',
            'datetime',
            'id',
            'speed_to_prev',
            'speed_to_next',
            'speed_prev_to_next',
        ],
        index=[0, 1, 2, 3],
    )
    assert_frame_equal(new_move_df, expected)
    assert isinstance(new_move_df, PandasMoveDataFrame)
    assert SPEED_PREV_TO_NEXT not in move_df

    move_df.generate_speed_features()
    print(move_df)
    assert_frame_equal(move_df, expected)


def test_generate_move_and_stop_by_radius():
    move_df = _default_move_df()

    new_move_df = move_df.generate_move_and_stop_by_radius(inplace=False)
    expected = DataFrame(
        data=[
            [
                1,
                39.984094,
                116.319236,
                Timestamp('2008-10-23 05:53:05'),
                nan,
                13.690153134343689,
                nan,
                'nan',
            ],
            [
                1,
                39.984198,
                116.319322,
                Timestamp('2008-10-23 05:53:06'),
                13.690153134343689,
                nan,
                nan,
                'move',
            ],
            [
                2,
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                nan,
                0.0,
                nan,
                'nan',
            ],
            [
                2,
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                0.0,
                nan,
                nan,
                'stop',
            ],
        ],
        columns=[
            'id',
            'lat',
            'lon',
            'datetime',
            'dist_to_prev',
            'dist_to_next',
            'dist_prev_to_next',
            'situation',
        ],
        index=[0, 1, 2, 3],
    )
    assert_frame_equal(new_move_df, expected)
    assert isinstance(new_move_df, PandasMoveDataFrame)
    assert SITUATION not in move_df

    move_df.generate_move_and_stop_by_radius()
    print(move_df)
    assert_frame_equal(move_df, expected)


def test_time_interval():
    move_df = _default_move_df()
    assert move_df.time_interval() == Timedelta('0 days 00:00:06')


def test_get_bbox():
    move_df = _default_move_df()

    assert_allclose(
        move_df.get_bbox(), (39.984093, 116.31924, 39.984222, 116.319405)
    )


def test_plot_traj_id():
    move_df = _default_move_df()
    move_df[TID] = ['1', '2', '3', '4']

    df, img = move_df.plot_traj_id('1')
    expected = DataFrame(
        data=[
            [39.984094, 116.319236, Timestamp('2008-10-23 05:53:05'), 1, '1'],
        ],
        columns=['lat', 'lon', 'datetime', 'id', 'tid'],
        index=[0],
    )
    assert_frame_equal(df, expected)
    assert isinstance(df, PandasMoveDataFrame)


def test_min():
    move_df = _default_move_df()

    expected = Series(
        data=[39.984094, 116.319236, Timestamp('2008-10-23 05:53:05'), 1],
        index=['lat', 'lon', 'datetime', 'id'],
        dtype='object',
        name=None,
    )
    assert_series_equal(move_df.min(), expected)


def test_max():
    move_df = _default_move_df()

    expected = Series(
        data=[39.984224, 116.319402, Timestamp('2008-10-23 05:53:11'), 2],
        index=['lat', 'lon', 'datetime', 'id'],
        dtype='object',
        name=None,
    )
    assert_series_equal(move_df.max(), expected)


def test_count():
    move_df = _default_move_df()

    expected = Series(
        data=[4, 4, 4, 4],
        index=['lat', 'lon', 'datetime', 'id'],
        dtype='int64',
        name=None,
    )
    assert_series_equal(move_df.count(), expected)


def test_group_by():
    move_df = _default_move_df()

    expected = _default_pandas_df()
    expected = expected.groupby('id').mean()
    assert_frame_equal(move_df.groupby(TRAJ_ID).mean(), expected)


def test_select_dtypes():
    move_df = _default_move_df()

    expected = DataFrame(
        data=[
            [39.984094, 116.319236],
            [39.984198, 116.319322],
            [39.984224, 116.319402],
            [39.984224, 116.319402],
        ],
        columns=['lat', 'lon'],
        index=[0, 1, 2, 3],
    )
    assert_frame_equal(move_df.select_dtypes(include='float64'), expected)


def test_astype():
    move_df = _default_move_df()

    expected = DataFrame(
        data=[
            [39, 116, 1224741185000000000, 1],
            [39, 116, 1224741186000000000, 1],
            [39, 116, 1224741191000000000, 2],
            [39, 116, 1224741191000000000, 2],
        ],
        columns=['lat', 'lon', 'datetime', 'id'],
        index=[0, 1, 2, 3],
    )
    assert_frame_equal(move_df.astype('int64'), expected)


def test_sort_values():
    move_df = _default_move_df()

    expected = DataFrame(
        data=[
            [39.984224, 116.319402, Timestamp('2008-10-23 05:53:11'), 2],
            [39.984224, 116.319402, Timestamp('2008-10-23 05:53:11'), 2],
            [39.984094, 116.319236, Timestamp('2008-10-23 05:53:05'), 1],
            [39.984198, 116.319322, Timestamp('2008-10-23 05:53:06'), 1],
        ],
        columns=['lat', 'lon', 'datetime', 'id'],
        index=[2, 3, 0, 1],
    )

    assert_frame_equal(
        move_df.sort_values(by=TRAJ_ID, ascending=False), expected
    )


def test_reset_index():
    move_df = _default_move_df()

    move_df = move_df.loc[1:]
    assert_array_equal(move_df.index, [1, 2, 3])
    move_df.reset_index(inplace=True)
    assert_array_equal(move_df.index, [0, 1, 2])


def test_set_index():
    move_df = _default_move_df()

    expected = DataFrame(
        data=[
            [39.984094, 116.319236, Timestamp('2008-10-23 05:53:05')],
            [39.984198, 116.319322, Timestamp('2008-10-23 05:53:06')],
            [39.984224, 116.319402, Timestamp('2008-10-23 05:53:11')],
            [39.984224, 116.319402, Timestamp('2008-10-23 05:53:11')],
        ],
        columns=['lat', 'lon', 'datetime'],
        index=[1, 1, 2, 2],
    )
    expected.index.name = 'id'
    assert_frame_equal(move_df.set_index('id'), expected)

    try:
        move_df.set_index('datetime', inplace=True)
        assert False
    except AttributeError:
        assert True


def test_drop():
    move_df = _default_move_df()
    move_df[UID] = [1, 1, 2, 3]

    move_test = move_df.drop(columns=[UID])
    assert UID not in move_test
    assert UID in move_df
    assert isinstance(move_test, PandasMoveDataFrame)

    move_test = move_df.drop(index=[0, 1])
    assert move_test.len() == 2
    assert isinstance(move_test, PandasMoveDataFrame)

    move_df.drop(columns=[UID], inplace=True)
    assert UID not in move_df
    assert isinstance(move_df, PandasMoveDataFrame)

    try:
        move_df.drop(columns=[LATITUDE], inplace=True)
        raise AssertionError(
            'AttributeError error not raised by MoveDataFrame'
        )
    except AttributeError:
        pass

    try:
        move_df.drop(columns=[LONGITUDE], inplace=True)
        raise AssertionError(
            'AttributeError error not raised by MoveDataFrame'
        )
    except AttributeError:
        pass

    try:
        move_df.drop(columns=[DATETIME], inplace=True)
        raise AssertionError(
            'AttributeError error not raised by MoveDataFrame'
        )
    except AttributeError:
        pass


def test_duplicated():
    move_df = _default_move_df()

    expected = [False, True, False, True]
    assert_array_equal(move_df.duplicated(TRAJ_ID), expected)
    expected = [False, False, True, False]
    assert_array_equal(
        move_df.duplicated(subset=DATETIME, keep='last'), expected
    )


def test_drop_duplicates():
    move_df = _default_move_df()

    move_test = move_df.drop_duplicates()
    expected = DataFrame(
        data=[
            [39.984094, 116.319236, Timestamp('2008-10-23 05:53:05'), 1],
            [39.984198, 116.319322, Timestamp('2008-10-23 05:53:06'), 1],
            [39.984224, 116.319402, Timestamp('2008-10-23 05:53:11'), 2],
        ],
        columns=['lat', 'lon', 'datetime', 'id'],
        index=[0, 1, 2],
    )
    assert_frame_equal(move_test, expected)
    assert isinstance(move_test, PandasMoveDataFrame)
    assert move_df.len() == 4

    move_df.drop_duplicates(inplace=True)
    assert_frame_equal(move_test, expected)
    assert isinstance(move_df, PandasMoveDataFrame)
    assert move_df.len() == 3


def test_all():
    move_df = _default_move_df()
    move_df['teste'] = [False, False, True, True]

    assert_array_equal(move_df.all(), [True, True, True, True, False])
    assert_array_equal(move_df.all(axis=1), [False, False, True, True])


def test_any():
    move_df = _default_move_df()
    move_df['teste'] = [False, False, False, False]

    assert_array_equal(move_df.any(), [True, True, True, True, False])
    assert_array_equal(move_df.any(axis=1), [True, True, True, True])


def test_isna():
    move_df = _default_move_df()
    move_df.at[0, DATETIME] = nan

    expected = DataFrame(
        data=[
            [False, False, True, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
        ],
        columns=['lat', 'lon', 'datetime', 'id'],
        index=[0, 1, 2, 3],
    )
    assert_frame_equal(move_df.isna(), expected)


def test_fillna():
    move_df = _default_move_df()
    move_df.at[0, LATITUDE] = nan

    move_test = move_df.fillna(0)
    expected = DataFrame(
        data=[
            [0, 116.319236, Timestamp('2008-10-23 05:53:05'), 1],
            [39.984198, 116.319322, Timestamp('2008-10-23 05:53:06'), 1],
            [39.984224, 116.319402, Timestamp('2008-10-23 05:53:11'), 2],
            [39.984224, 116.319402, Timestamp('2008-10-23 05:53:11'), 2],
        ],
        columns=['lat', 'lon', 'datetime', 'id'],
        index=[0, 1, 2, 3],
    )
    assert_frame_equal(move_test, expected)
    assert isinstance(move_test, PandasMoveDataFrame)
    assert move_df.isna().any(axis=None)

    move_df.fillna(0, inplace=True)
    assert_frame_equal(move_df, expected)


def test_dropna():
    move_df = _default_move_df()
    move_df.at[0, LATITUDE] = nan

    move_test = move_df.dropna(axis=1)
    expected = DataFrame(
        data=[
            [116.319236, Timestamp('2008-10-23 05:53:05'), 1],
            [116.319322, Timestamp('2008-10-23 05:53:06'), 1],
            [116.319402, Timestamp('2008-10-23 05:53:11'), 2],
            [116.319402, Timestamp('2008-10-23 05:53:11'), 2],
        ],
        columns=['lon', 'datetime', 'id'],
        index=[0, 1, 2, 3],
    )
    assert_frame_equal(move_test, expected)
    assert move_test.shape == (4, 3)
    assert isinstance(move_test, DataFrame)
    assert move_df.shape == (4, 4)

    move_test = move_df.dropna(axis=0)
    expected = DataFrame(
        data=[
            [39.984198, 116.319322, Timestamp('2008-10-23 05:53:06'), 1],
            [39.984224, 116.319402, Timestamp('2008-10-23 05:53:11'), 2],
            [39.984224, 116.319402, Timestamp('2008-10-23 05:53:11'), 2],
        ],
        columns=['lat', 'lon', 'datetime', 'id'],
        index=[1, 2, 3],
    )
    assert_frame_equal(move_test, expected)
    assert move_test.shape == (3, 4)
    assert isinstance(move_test, PandasMoveDataFrame)
    assert move_df.shape == (4, 4)

    try:
        move_df.dropna(axis=1, inplace=True)
        assert False
    except AttributeError:
        assert True

    move_df.dropna(axis=0, inplace=True)
    assert_frame_equal(move_df, expected)


def test_sample():
    move_df = _default_move_df()

    sample_test = move_df.sample(n=2, random_state=42)
    expected = DataFrame(
        data=[
            [39.984198, 116.319322, Timestamp('2008-10-23 05:53:06'), 1],
            [39.984224, 116.319402, Timestamp('2008-10-23 05:53:11'), 2],
        ],
        columns=['lat', 'lon', 'datetime', 'id'],
        index=[1, 3],
    )
    assert_frame_equal(sample_test, expected)
    assert isinstance(sample_test, PandasMoveDataFrame)


def test_isin():
    move_df = _default_move_df()
    move_df_copy = _default_move_df()

    assert move_df.isin(move_df_copy).all(axis=None)

    move_df_copy.at[0, LATITUDE] = 0
    assert not move_df.isin(move_df_copy).all(axis=None)


def test_append():
    move_df = _default_move_df()
    df = DataFrame(
        data=[[39.984094, 116.319236, Timestamp('2008-10-23 05:53:05'), 1]],
        columns=['lat', 'lon', 'datetime', 'id'],
        index=[0],
    )
    mdf = MoveDataFrame(df)

    expected = DataFrame(
        data=[
            [39.984094, 116.319236, Timestamp('2008-10-23 05:53:05'), 1],
            [39.984198, 116.319322, Timestamp('2008-10-23 05:53:06'), 1],
            [39.984224, 116.319402, Timestamp('2008-10-23 05:53:11'), 2],
            [39.984224, 116.319402, Timestamp('2008-10-23 05:53:11'), 2],
            [39.984094, 116.319236, Timestamp('2008-10-23 05:53:05'), 1],
        ],
        columns=['lat', 'lon', 'datetime', 'id'],
        index=[0, 1, 2, 3, 0],
    )
    assert_frame_equal(move_df.append(df), expected)
    assert_frame_equal(move_df.append(mdf), expected)


def test_join():
    move_df = _default_move_df()
    other = DataFrame({'id': [1, 1, 2, 2], 'key': ['a', 'b', 'c', 'd']})

    result = move_df.join(other, on='id', lsuffix='_l', rsuffix='_r')
    expected = DataFrame(
        data=[
            [
                39.984094,
                116.319236,
                Timestamp('2008-10-23 05:53:05'),
                1,
                1,
                'b',
            ],
            [
                39.984198,
                116.319322,
                Timestamp('2008-10-23 05:53:06'),
                1,
                1,
                'b',
            ],
            [
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                2,
                2,
                'c',
            ],
            [
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                2,
                2,
                'c',
            ],
        ],
        columns=['lat', 'lon', 'datetime', 'id_l', 'id_r', 'key'],
        index=[0, 1, 2, 3],
    )
    assert_frame_equal(result, expected)


def test_merge():
    move_df = _default_move_df()
    other = DataFrame({'key': [1], 'value': ['a']})

    result = move_df.merge(other, left_on='id', right_on='key')
    expected = DataFrame(
        data=[
            [
                39.984094,
                116.319236,
                Timestamp('2008-10-23 05:53:05'),
                1,
                1,
                'a',
            ],
            [
                39.984198,
                116.319322,
                Timestamp('2008-10-23 05:53:06'),
                1,
                1,
                'a',
            ],
        ],
        columns=['lat', 'lon', 'datetime', 'id', 'key', 'value'],
        index=[0, 1],
    )
    assert_frame_equal(result, expected)


def test_nunique():
    move_df = _default_move_df()
    assert_array_equal(move_df.nunique(), [3, 3, 3, 2])


def test_write_file(tmpdir):
    d = tmpdir.mkdir('prepossessing')

    file_write_default = d.join('test_write_default.csv')
    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )
    move_df = _default_move_df()
    move_df.write_file(filename_write_default)
    move_df_test = read_csv(filename_write_default)
    assert_frame_equal(move_df, move_df_test)

    file_write_semi = d.join('test_write_semi.csv')
    filename_write_semi = os.path.join(
        file_write_semi.dirname, file_write_semi.basename
    )
    move_df = _default_move_df()
    move_df.write_file(filename_write_semi, separator=';')
    move_df_test = read_csv(filename_write_semi, sep=';')
    assert_frame_equal(move_df, move_df_test)


def test_to_csv(tmpdir):
    d = tmpdir.mkdir('prepossessing')

    file_csv = d.join('test_csv.csv')
    filename_csv = os.path.join(file_csv.dirname, file_csv.basename)
    move_df = _default_move_df()
    move_df.to_csv(filename_csv, index=False)
    move_df_test = read_csv(filename_csv)
    assert_frame_equal(move_df, move_df_test)


def test_convert_to():
    move_df = _default_move_df()

    assert move_df._type == TYPE_PANDAS
    assert isinstance(move_df, PandasMoveDataFrame)
    assert isinstance(move_df._data, DataFrame)

    move_df_dask = move_df.convert_to('dask')
    assert move_df_dask._type == TYPE_DASK
    assert isinstance(move_df_dask, DaskMoveDataFrame)
    assert isinstance(move_df_dask, DaskDataFrame)

    assert move_df._type == TYPE_PANDAS
    assert isinstance(move_df, PandasMoveDataFrame)
    assert isinstance(move_df._data, DataFrame)


def test_get_type():
    move_df = _default_move_df()

    assert move_df.get_type() == TYPE_PANDAS


def test_rename():

    expected = _default_move_df()

    expected_columns = DataFrame(
        data=[
            [
                39.984094,
                116.319236,
                Timestamp('2008-10-23 05:53:05'),
                1
            ],
            [
                39.984198,
                116.319322,
                Timestamp('2008-10-23 05:53:06'),
                1
            ],
            [
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                2
            ],
            [
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                2
            ],
        ],
        columns=['lat', 'lon', 'datetime', 'ID'],
        index=[0, 1, 2, 3],
    )

    expected_index = DataFrame(
        data=[
            [
                39.984094,
                116.319236,
                Timestamp('2008-10-23 05:53:05'),
                1
            ],
            [
                39.984198,
                116.319322,
                Timestamp('2008-10-23 05:53:06'),
                1
            ],
            [
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                2
            ],
            [
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                2
            ],
        ],
        columns=['lat', 'lon', 'datetime', 'id'],
        index=['a', 'b', 2, 3],
    )

    expected_lat = DataFrame(
        data=[
            [
                39.984094,
                116.319236,
                Timestamp('2008-10-23 05:53:05'),
                1
            ],
            [
                39.984198,
                116.319322,
                Timestamp('2008-10-23 05:53:06'),
                1
            ],
            [
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                2
            ],
            [
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                2
            ],
        ],
        columns=['LAT', 'lon', 'datetime', 'id'],
        index=[0, 1, 2, 3],
    )

    move_df = _default_move_df()

    new_move_df = move_df.rename(columns={'id': 'ID'}, inplace=False)

    assert_frame_equal(new_move_df, expected_columns)

    assert isinstance(new_move_df, PandasMoveDataFrame)

    assert_frame_equal(move_df, expected)

    new_move_df = move_df.rename(index={0: 'a', 1: 'b'}, inplace=False)

    assert_frame_equal(new_move_df, expected_index)

    assert isinstance(new_move_df, PandasMoveDataFrame)

    new_move = move_df.rename({0: 'a', 1: 'b'}, axis='index', inplace=False)

    assert_frame_equal(new_move_df, expected_index)

    assert isinstance(new_move_df, PandasMoveDataFrame)

    new_move_df = move_df.rename(columns={'lat': 'LAT'}, inplace=False)

    assert_frame_equal(new_move_df, expected_lat)

    assert isinstance(new_move_df, DataFrame)

    move_df.rename(columns={'id': 'ID'}, inplace=True)

    assert_frame_equal(move_df, expected_columns)

    assert isinstance(move_df, PandasMoveDataFrame)

    try:
        move_df.rename(columns={'lat': 'LAT'}, inplace=True)
        raise AssertionError(
            'AttributeError error not raised by MoveDataFrame'
        )
    except AttributeError:
        pass

    try:
        move_df.rename(columns={'lon': 'LON'}, inplace=True)
        raise AssertionError(
            'AttributeError error not raised by MoveDataFrame'
        )
    except AttributeError:
        pass

    try:
        move_df.rename(columns={'datetime': 'DATETIME'}, inplace=True)
        raise AssertionError(
            'AttributeError error not raised by MoveDataFrame'
        )
    except AttributeError:
        pass


def test_plot_all_features(tmpdir):

    move_df = _default_move_df()

    d = tmpdir.mkdir('prepossessing')

    file_write_default = d.join('features.png')
    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    move_df.plot_all_features(save_fig=True, name=filename_write_default)

    test_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(test_dir, 'baseline/features.png')

    compare_images(data_dir,
                   filename_write_default,
                   0.0001,
                   in_decorator=False)

    move_df['lat'] = move_df['lat'].astype('float32')
    move_df['lon'] = move_df['lon'].astype('float32')

    try:
        move_df.plot_all_features(name=filename_write_default)
        raise AssertionError(
            'AttributeError error not raised by MoveDataFrame'
        )
    except AttributeError:
        pass


def test_plot_trajs(tmpdir):

    move_df = _default_move_df()

    d = tmpdir.mkdir('prepossessing')

    file_write_default = d.join('trajectories.png')
    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    move_df.plot_trajs(save_fig=True, name=filename_write_default)

    test_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(test_dir, 'baseline/trajectories.png')

    compare_images(data_dir,
                   filename_write_default,
                   0.0001,
                   in_decorator=False)
