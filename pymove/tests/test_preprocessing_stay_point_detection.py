from numpy import nan
from pandas import DataFrame, Timestamp
from pandas.testing import assert_frame_equal

from pymove import MoveDataFrame, stay_point_detection
from pymove.utils.constants import DATETIME, LATITUDE, LONGITUDE, TRAJ_ID

list_data = [
    [39.984094, 116.319236, '2008-10-23 05:53:05', 1],
    [39.984198, 116.319322, '2008-10-23 05:53:06', 1],
    [39.984224, 116.319402, '2008-10-23 05:53:11', 2],
    [39.984224, 116.319402, '2008-10-23 05:53:15', 2],
]

list_data_test = [
    [39.984093, 116.319237, '2008-10-23 05:53:05', 1],
    [39.984200, 116.319321, '2008-10-23 05:53:06', 1],
    [39.984222, 116.319405, '2008-10-23 05:53:11', 1],
    [39.984211, 116.319389, '2008-10-23 05:53:16', 1],
    [39.984219, 116.319420, '2008-10-23 05:53:21', 1],
]


def _prepare_default_df():
    move_df = MoveDataFrame(
        data=list_data,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        datetime=DATETIME,
        traj_id=TRAJ_ID,
    )
    cols = ['lat', 'lon', 'datetime', 'id']
    return move_df, cols


def test_create_or_update_datetime_in_format_cyclical():
    move_df, cols = _prepare_default_df()

    stay_point_detection.create_or_update_datetime_in_format_cyclical(move_df)

    expected = DataFrame(
        data=[
            [
                39.984094,
                116.319236,
                Timestamp('2008-10-23 05:53:05'),
                1,
                0.979084,
                0.203456,
            ],
            [
                39.984198,
                116.319322,
                Timestamp('2008-10-23 05:53:06'),
                1,
                0.979084,
                0.203456,
            ],
            [
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                2,
                0.979084,
                0.203456,
            ],
            [
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:15'),
                2,
                0.979084,
                0.203456,
            ],
        ],
        columns=cols + ['hour_sin', 'hour_cos'],
        index=[0, 1, 2, 3],
    )
    assert_frame_equal(move_df, expected)


def test_create_or_update_move_stop_by_dist_time():
    move_df = MoveDataFrame(
        data=list_data,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        datetime=DATETIME,
        traj_id=TRAJ_ID,
    )
    cols = [
        'segment_stop',
        'id',
        'lat',
        'lon',
        'datetime',
        'dist_to_prev',
        'time_to_prev',
        'speed_to_prev',
        'stop',
    ]

    stay_point_detection.create_or_update_move_stop_by_dist_time(
        move_df, dist_radius=3.5, time_radius=0.5, inplace=True
    )
    expected = DataFrame(
        data=[
            [
                1,
                1,
                39.984094,
                116.319236,
                Timestamp('2008-10-23 05:53:05'),
                nan,
                nan,
                nan,
                False,
            ],
            [
                2,
                1,
                39.984198,
                116.319322,
                Timestamp('2008-10-23 05:53:06'),
                nan,
                nan,
                nan,
                False,
            ],
            [
                3,
                2,
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                nan,
                nan,
                nan,
                True,
            ],
            [
                3,
                2,
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:15'),
                0.0,
                4.0,
                0.0,
                True,
            ],
        ],
        columns=cols,
        index=[0, 1, 2, 3],
    )

    assert_frame_equal(move_df, expected)


def test_create_or_update_move_and_stop_by_radius():
    move_df = MoveDataFrame(
        data=list_data,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        datetime=DATETIME,
        traj_id=TRAJ_ID,
    )
    cols = [
        'id',
        'lat',
        'lon',
        'datetime',
        'dist_to_prev',
        'dist_to_next',
        'dist_prev_to_next',
        'situation',
    ]

    stay_point_detection.create_or_update_move_and_stop_by_radius(
        move_df, radius=4.0
    )
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
                Timestamp('2008-10-23 05:53:15'),
                0.0,
                nan,
                nan,
                'stop',
            ],
        ],
        columns=cols,
        index=[0, 1, 2, 3],
    )
    assert_frame_equal(move_df, expected)
