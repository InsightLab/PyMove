from numpy import nan
from numpy.testing import assert_allclose
from pandas import DataFrame, Timestamp
from pandas.testing import assert_frame_equal

from pymove import MoveDataFrame, segmentation
from pymove.utils.constants import (
    DATETIME,
    LATITUDE,
    LONGITUDE,
    TID_DIST,
    TID_PART,
    TID_SPEED,
    TID_TIME,
    TRAJ_ID,
)

list_data = [
    [39.984094, 116.319236, '2008-10-23 05:53:05', 1],
    [39.984198, 116.319322, '2008-10-23 05:53:06', 1],
    [39.984224, 116.319402, '2008-10-23 05:53:11', 2],
    [39.984224, 116.319402, '2008-10-23 05:53:15', 2],
]


def _prepare_df_tid(tid):
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
        'time_to_prev',
        'speed_to_prev',
        tid,
    ]
    return move_df, cols


def test_bbox_split():
    bbox = (39.984093, 116.31924, 39.984222, 116.319405)
    segmented_bbox = segmentation.bbox_split(bbox, 3)
    expected = [
        [39.984093, 116.31924, 39.984222, 116.319295],
        [39.984093, 116.319295, 39.984222, 116.31935],
        [39.984093, 116.31935, 39.984222, 116.319405],
    ]
    assert_allclose(segmented_bbox, expected)


def test_by_dist_time_speed():
    move_df, cols = _prepare_df_tid(TID_PART)

    segmented_dts = segmentation.by_dist_time_speed(move_df, inplace=False)
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
                1,
            ],
            [
                1,
                39.984198,
                116.319322,
                Timestamp('2008-10-23 05:53:06'),
                13.690153134343689,
                1.0,
                13.690153134343689,
                1,
            ],
            [
                2,
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                nan,
                nan,
                nan,
                2,
            ],
            [
                2,
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:15'),
                0.0,
                4.0,
                0.000000,
                2,
            ],
        ],
        columns=cols,
        index=[0, 1, 2, 3],
    )
    assert_frame_equal(segmented_dts, expected)
    assert move_df.shape[1] == 4

    segmentation.by_dist_time_speed(move_df, inplace=True)
    assert_frame_equal(move_df, expected)
    assert move_df.shape[1] == 8


def test_by_max_dist():
    move_df, cols = _prepare_df_tid(TID_DIST)

    segmented_dist = segmentation.by_max_dist(
        move_df, max_dist_between_adj_points=0.5, inplace=False
    )
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
                1,
            ],
            [
                1,
                39.984198,
                116.319322,
                Timestamp('2008-10-23 05:53:06'),
                13.690153134343689,
                1.0,
                13.690153134343689,
                2,
            ],
            [
                2,
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                nan,
                nan,
                nan,
                3,
            ],
            [
                2,
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:15'),
                0.0,
                4.0,
                0.000000,
                3,
            ],
        ],
        columns=cols,
        index=[0, 1, 2, 3],
    )
    assert_frame_equal(segmented_dist, expected)
    assert move_df.shape[1] == 4

    segmentation.by_max_dist(
        move_df, max_dist_between_adj_points=0.5, inplace=True
    )
    assert_frame_equal(move_df, expected)
    assert move_df.shape[1] == 8


def test_by_max_time():
    move_df, cols = _prepare_df_tid(TID_TIME)

    segmented_time = segmentation.by_max_time(
        move_df, max_time_between_adj_points=0, inplace=False
    )

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
                1,
            ],
            [
                1,
                39.984198,
                116.319322,
                Timestamp('2008-10-23 05:53:06'),
                13.690153134343689,
                1.0,
                13.690153134343689,
                2,
            ],
            [
                2,
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                nan,
                nan,
                nan,
                3,
            ],
            [
                2,
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:15'),
                0.0,
                4.0,
                0.000000,
                4,
            ],
        ],
        columns=cols,
        index=[0, 1, 2, 3],
    )
    assert_frame_equal(segmented_time, expected)
    assert move_df.shape[1] == 4

    segmentation.by_max_time(
        move_df, max_time_between_adj_points=0, inplace=True
    )
    assert_frame_equal(move_df, expected)
    assert move_df.shape[1] == 8


def test_by_max_speed():
    move_df, cols = _prepare_df_tid(TID_SPEED)

    segmented_speed = segmentation.by_max_speed(
        move_df, max_speed_between_adj_points=10, inplace=False
    )
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
                1,
            ],
            [
                1,
                39.984198,
                116.319322,
                Timestamp('2008-10-23 05:53:06'),
                13.690153134343689,
                1.0,
                13.690153134343689,
                2,
            ],
            [
                2,
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                nan,
                nan,
                nan,
                3,
            ],
            [
                2,
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:15'),
                0.0,
                4.0,
                0.000000,
                3,
            ],
        ],
        columns=cols,
        index=[0, 1, 2, 3],
    )
    assert_frame_equal(segmented_speed, expected)
    assert move_df.shape[1] == 4

    segmentation.by_max_speed(
        move_df, max_speed_between_adj_points=0, inplace=True
    )
    assert_frame_equal(move_df, expected)
    assert move_df.shape[1] == 8
