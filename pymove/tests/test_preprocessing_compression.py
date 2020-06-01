from numpy import nan
from pandas import DataFrame, Timestamp
from pandas.testing import assert_frame_equal

from pymove import MoveDataFrame, compression
from pymove.utils.constants import (
    DATETIME,
    DIST_TO_PREV,
    LATITUDE,
    LONGITUDE,
    SEGMENT_STOP,
    SPEED_TO_PREV,
    STOP,
    TIME_TO_PREV,
    TRAJ_ID,
)

list_data = [
    [39.984093, 116.319237, '2008-10-23 05:53:05', 1],
    [39.984200, 116.319321, '2008-10-23 05:53:06', 1],
    [39.984222, 116.319405, '2008-10-23 05:53:11', 1],
    [39.984211, 116.319389, '2008-10-23 05:53:16', 1],
    [39.984219, 116.319420, '2008-10-23 05:53:21', 1],
]


def test_compress_segment_to_stop_point():
    move_df = MoveDataFrame(
        data=list_data,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        datetime=DATETIME,
        traj_id=TRAJ_ID,
    )
    move_df[SEGMENT_STOP] = [1, 2, 3, 3, 3]
    move_df[DIST_TO_PREV] = [
        nan,
        nan,
        nan,
        1.8315003203265166,
        2.786977605326357,
    ]
    move_df[TIME_TO_PREV] = [nan, nan, nan, 5.0, 5.0]
    move_df[SPEED_TO_PREV] = [
        nan,
        nan,
        nan,
        0.3663000640653033,
        0.5573955210652713,
    ]
    move_df[STOP] = [False, False, True, True, True]
    cols = [
        'lat',
        'lon',
        'datetime',
        'id',
        'segment_stop',
        'dist_to_prev',
        'time_to_prev',
        'speed_to_prev',
        'stop',
        'lat_mean',
        'lon_mean',
    ]

    compressed_trajs_mean = compression.compress_segment_stop_to_point(
        move_df, dist_radius=3.5, time_radius=0.5
    )
    expected = DataFrame(
        data=[
            [
                39.984093,
                116.319237,
                Timestamp('2008-10-23 05:53:05'),
                1,
                1,
                nan,
                nan,
                nan,
                False,
                nan,
                nan,
            ],
            [
                39.9842,
                116.319321,
                Timestamp('2008-10-23 05:53:06'),
                1,
                2,
                nan,
                nan,
                nan,
                False,
                nan,
                nan,
            ],
            [
                39.984222,
                116.319405,
                Timestamp('2008-10-23 05:53:11'),
                1,
                3,
                nan,
                nan,
                nan,
                True,
                39.984222,
                116.319405,
            ],
            [
                39.984219,
                116.31942,
                Timestamp('2008-10-23 05:53:21'),
                1,
                3,
                2.786977605326357,
                5.0,
                0.5573955210652713,
                True,
                39.984222,
                116.319405,
            ],
        ],
        columns=cols,
        index=[0, 1, 2, 4],
    )
    assert_frame_equal(compressed_trajs_mean, expected)
    assert move_df.len() == 5

    compressed_trajs_centroid = compression.compress_segment_stop_to_point(
        move_df, dist_radius=3.5, time_radius=0.5, point_mean='centroid'
    )
    expected = DataFrame(
        data=[
            [
                39.984093,
                116.319237,
                Timestamp('2008-10-23 05:53:05'),
                1,
                1,
                nan,
                nan,
                nan,
                False,
                nan,
                nan,
            ],
            [
                39.9842,
                116.319321,
                Timestamp('2008-10-23 05:53:06'),
                1,
                2,
                nan,
                nan,
                nan,
                False,
                nan,
                nan,
            ],
            [
                39.984222,
                116.319405,
                Timestamp('2008-10-23 05:53:11'),
                1,
                3,
                nan,
                nan,
                nan,
                True,
                39.984217,
                116.319405,
            ],
            [
                39.984219,
                116.31942,
                Timestamp('2008-10-23 05:53:21'),
                1,
                3,
                2.786977605326357,
                5.0,
                0.5573955210652713,
                True,
                39.984217,
                116.319405,
            ],
        ],
        columns=cols,
        index=[0, 1, 2, 4],
    )
    assert_frame_equal(compressed_trajs_centroid, expected)
    assert move_df.len() == 5

    compression.compress_segment_stop_to_point(
        move_df,
        dist_radius=3.5,
        time_radius=0.5,
        point_mean='centroid',
        drop_moves=True,
        inplace=True,
    )
    expected = DataFrame(
        data=[
            [
                39.984222,
                116.319405,
                Timestamp('2008-10-23 05:53:11'),
                1,
                3,
                nan,
                nan,
                nan,
                True,
                39.984217,
                116.319405,
            ],
            [
                39.984219,
                116.31942,
                Timestamp('2008-10-23 05:53:21'),
                1,
                3,
                2.786977605326357,
                5.0,
                0.5573955210652713,
                True,
                39.984217,
                116.319405,
            ],
        ],
        columns=cols,
        index=[2, 4],
    )
    assert_frame_equal(move_df, expected)
    assert move_df.len() == 2
