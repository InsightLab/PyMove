from numpy import nan
from pandas import DataFrame, Timestamp
from pandas.testing import assert_frame_equal

from pymove import MoveDataFrame, filters
from pymove.utils.constants import (
    DATETIME,
    DIST_PREV_TO_NEXT,
    DIST_TO_NEXT,
    DIST_TO_PREV,
    LATITUDE,
    LONGITUDE,
    SPEED_TO_PREV,
    TIME_TO_PREV,
    TRAJ_ID,
)

list_data_1 = [
    [39.984092712402344, 116.31923675537101, '2008-10-23 05:53:05', 1],
    [39.984199952392578, 116.31932067871094, '2008-10-23 05:53:06', 1],
    [39.984222412109375, 116.31940460205078, '2008-10-23 05:53:11', 2],
    [39.984222412109375, 116.31940460205078, '2008-10-23 05:53:11', 2],
]

list_data_2 = [
    [39.984093, 116.319237, '2008-10-23 05:53:05', 1],
    [39.984200, 116.319321, '2008-10-23 05:53:06', 1],
    [38.984211, 115.319389, '2008-10-23 05:53:11', 1],
    [39.984222, 116.319405, '2008-10-23 05:53:16', 1],
    [39.984219, 116.319420, '2008-10-23 05:53:21', 1],
]

list_data_3 = [
    [39.984093, 116.319237, '2008-10-23 05:53:05', 1],
    [39.984200, 116.319321, '2008-10-23 05:54:06', 1],
    [39.984222, 116.319405, '2008-10-23 05:55:16', 1],
    [39.984219, 116.319420, '2008-10-23 05:56:21', 1],
    [39.984199, 116.319320, '2008-10-23 05:53:06', 2],
    [39.974222, 116.339404, '2008-10-23 05:53:11', 2],
]


def _prepare_df_default(list_data):
    move_df = MoveDataFrame(
        data=list_data,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        datetime=DATETIME,
        traj_id=TRAJ_ID,
    )
    cols = ['lat', 'lon', 'datetime', 'id']
    return move_df, cols


def _prepare_df_with_distances():
    move_df = MoveDataFrame(
        data=list_data_2,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        datetime=DATETIME,
    )
    move_df[DIST_TO_PREV] = [
        nan,
        13.884481484192932,
        140454.64510608764,
        140460.97747220137,
        1.3208180727070074,
    ]
    move_df[DIST_TO_NEXT] = [
        13.884481484192932,
        140454.64510608764,
        140460.97747220137,
        1.3208180727070074,
        nan,
    ]
    move_df[DIST_PREV_TO_NEXT] = [
        nan,
        140440.86267282354,
        7.563335999941849,
        140461.50100279532,
        nan,
    ]
    cols = [
        'id',
        'lat',
        'lon',
        'datetime',
        'dist_to_prev',
        'dist_to_next',
        'dist_prev_to_next',
    ]
    return move_df[cols], cols


def _prepare_df_with_dist_time_speed():
    move_df = MoveDataFrame(
        data=list_data_2,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        datetime=DATETIME,
    )
    move_df[DIST_TO_PREV] = [
        nan,
        13.884481484192932,
        140454.64510608764,
        140460.97747220137,
        1.3208180727070074,
    ]
    move_df[TIME_TO_PREV] = [nan, 1.0, 5.0, 5.0, 5.0]
    move_df[SPEED_TO_PREV] = [
        nan,
        13.884481484192932,
        28090.92902121753,
        28092.195494440275,
        0.26416361454140147,
    ]
    cols = [
        'id',
        'lat',
        'lon',
        'datetime',
        'dist_to_prev',
        'time_to_prev',
        'speed_to_prev',
    ]
    return move_df[cols], cols


def _prepare_df_with_long_trajectory():
    move_df = MoveDataFrame(
        data=list_data_3,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        datetime=DATETIME,
    )
    move_df[DIST_TO_PREV] = [
        nan,
        13.884481484192932,
        7.563335999941849,
        1.3208180727070074,
        nan,
        2039.4197149375298,
    ]
    move_df[TIME_TO_PREV] = [nan, 61.00000000000001, 70.0, 65.0, nan, 5.0]
    move_df[SPEED_TO_PREV] = [
        nan,
        0.22761445056053983,
        0.10804765714202641,
        0.020320278041646267,
        nan,
        407.883942987506,
    ]
    cols = [
        'id',
        'lat',
        'lon',
        'datetime',
        'dist_to_prev',
        'time_to_prev',
        'speed_to_prev',
    ]
    return move_df[cols], cols


def test_by_bbox():
    move_df, cols = _prepare_df_default(list_data_1)
    bbox = (39.984193, 116.31924, 39.984222, 116.319405)

    filter_values = filters.by_bbox(move_df, bbox)
    expected = DataFrame(
        data=[
            [
                39.984199952392578,
                116.31932067871094,
                Timestamp('2008-10-23 05:53:06'),
                1,
            ]
        ],
        columns=cols,
        index=[1],
    )
    assert_frame_equal(filter_values, expected)
    assert move_df.len() == 4

    filter_out_values = filters.by_bbox(move_df, bbox, filter_out=True)
    expected = DataFrame(
        data=[
            [
                39.984092712402344,
                116.31923675537101,
                Timestamp('2008-10-23 05:53:05'),
                1,
            ],
            [
                39.984222412109375,
                116.31940460205078,
                Timestamp('2008-10-23 05:53:11'),
                2,
            ],
            [
                39.984222412109375,
                116.31940460205078,
                Timestamp('2008-10-23 05:53:11'),
                2,
            ],
        ],
        columns=cols,
        index=[0, 2, 3],
    )
    assert_frame_equal(filter_out_values, expected)
    assert move_df.len() == 4

    filters.by_bbox(move_df, bbox, inplace=True)
    expected = DataFrame(
        data=[
            [
                39.984199952392578,
                116.31932067871094,
                Timestamp('2008-10-23 05:53:06'),
                1,
            ]
        ],
        columns=cols,
        index=[1],
    )
    assert_frame_equal(move_df, expected)
    assert move_df.len() == 1


def test_by_datetime():
    move_df, cols = _prepare_df_default(list_data_1)

    filter_values_start = filters.by_datetime(
        move_df, start_datetime='2008-10-23 05:53:06'
    )
    expected = DataFrame(
        data=[
            [
                39.984199952392578,
                116.31932067871094,
                Timestamp('2008-10-23 05:53:06'),
                1,
            ],
            [
                39.984222412109375,
                116.31940460205078,
                Timestamp('2008-10-23 05:53:11'),
                2,
            ],
            [
                39.984222412109375,
                116.31940460205078,
                Timestamp('2008-10-23 05:53:11'),
                2,
            ],
        ],
        columns=cols,
        index=[1, 2, 3],
    )
    assert_frame_equal(filter_values_start, expected)
    assert move_df.len() == 4

    filter_values_end = filters.by_datetime(
        move_df, end_datetime='2008-10-23 05:53:05'
    )
    expected = DataFrame(
        data=[
            [
                39.984092712402344,
                116.31923675537101,
                Timestamp('2008-10-23 05:53:05'),
                1,
            ]
        ],
        columns=cols,
        index=[0],
    )
    assert_frame_equal(filter_values_end, expected)
    assert move_df.len() == 4

    filter_values_start_end = filters.by_datetime(
        move_df,
        start_datetime='2008-10-23 05:53:04',
        end_datetime='2008-10-23 05:53:06',
    )
    expected = DataFrame(
        data=[
            [
                39.984092712402344,
                116.31923675537101,
                Timestamp('2008-10-23 05:53:05'),
                1,
            ],
            [
                39.984199952392578,
                116.31932067871094,
                Timestamp('2008-10-23 05:53:06'),
                1,
            ],
        ],
        columns=cols,
        index=[0, 1],
    )
    assert_frame_equal(filter_values_start_end, expected)
    assert move_df.len() == 4

    filter_out_values = filters.by_datetime(
        move_df,
        start_datetime='2008-10-23 05:53:04',
        end_datetime='2008-10-23 05:53:06',
        filter_out=True,
    )
    expected = DataFrame(
        [
            [
                39.984222412109375,
                116.31940460205078,
                Timestamp('2008-10-23 05:53:11'),
                2,
            ],
            [
                39.984222412109375,
                116.31940460205078,
                Timestamp('2008-10-23 05:53:11'),
                2,
            ],
        ],
        columns=cols,
        index=[2, 3],
    )
    assert_frame_equal(filter_out_values, expected)
    assert move_df.len() == 4

    filters.by_datetime(
        move_df,
        start_datetime='2008-10-23 05:53:04',
        end_datetime='2008-10-23 05:53:06',
        filter_out=True,
        inplace=True,
    )
    expected = DataFrame(
        [
            [
                39.984222412109375,
                116.31940460205078,
                Timestamp('2008-10-23 05:53:11'),
                2,
            ],
            [
                39.984222412109375,
                116.31940460205078,
                Timestamp('2008-10-23 05:53:11'),
                2,
            ],
        ],
        columns=cols,
        index=[2, 3],
    )
    assert_frame_equal(move_df, expected)
    assert move_df.len() == 2


def test_by_label():
    move_df, cols = _prepare_df_default(list_data_1)

    filter_values = filters.by_label(move_df, value=1, label_name=TRAJ_ID)
    expected = DataFrame(
        [
            [
                39.984092712402344,
                116.31923675537101,
                Timestamp('2008-10-23 05:53:05'),
                1,
            ],
            [
                39.984199952392578,
                116.31932067871094,
                Timestamp('2008-10-23 05:53:06'),
                1,
            ],
        ],
        columns=cols,
        index=[0, 1],
    )
    assert_frame_equal(filter_values, expected)
    assert move_df.len() == 4

    filter_out_values = filters.by_label(
        move_df, value=1, label_name=TRAJ_ID, filter_out=True
    )
    expected = DataFrame(
        [
            [
                39.984222412109375,
                116.31940460205078,
                Timestamp('2008-10-23 05:53:11'),
                2,
            ],
            [
                39.984222412109375,
                116.31940460205078,
                Timestamp('2008-10-23 05:53:11'),
                2,
            ],
        ],
        columns=cols,
        index=[2, 3],
    )
    assert_frame_equal(filter_out_values, expected)
    assert move_df.len() == 4

    filters.by_label(
        move_df, value=1, label_name=TRAJ_ID, filter_out=True, inplace=True
    )
    expected = DataFrame(
        [
            [
                39.984222412109375,
                116.31940460205078,
                Timestamp('2008-10-23 05:53:11'),
                2,
            ],
            [
                39.984222412109375,
                116.31940460205078,
                Timestamp('2008-10-23 05:53:11'),
                2,
            ],
        ],
        columns=cols,
        index=[2, 3],
    )
    assert_frame_equal(filter_out_values, expected)
    assert move_df.len() == 2


def test_by_id():
    move_df, cols = _prepare_df_default(list_data_1)

    filter_values = filters.by_id(move_df, id_=1)
    expected = DataFrame(
        [
            [
                39.984092712402344,
                116.31923675537101,
                Timestamp('2008-10-23 05:53:05'),
                1,
            ],
            [
                39.984199952392578,
                116.31932067871094,
                Timestamp('2008-10-23 05:53:06'),
                1,
            ],
        ],
        columns=cols,
        index=[0, 1],
    )
    assert_frame_equal(filter_values, expected)
    assert move_df.len() == 4

    filter_out_values = filters.by_id(move_df, id_=1, filter_out=True)
    expected = DataFrame(
        [
            [
                39.984222412109375,
                116.31940460205078,
                Timestamp('2008-10-23 05:53:11'),
                2,
            ],
            [
                39.984222412109375,
                116.31940460205078,
                Timestamp('2008-10-23 05:53:11'),
                2,
            ],
        ],
        columns=cols,
        index=[2, 3],
    )
    assert_frame_equal(filter_out_values, expected)
    assert move_df.len() == 4

    filters.by_id(move_df, id_=1, filter_out=True, inplace=True)
    expected = DataFrame(
        [
            [
                39.984222412109375,
                116.31940460205078,
                Timestamp('2008-10-23 05:53:11'),
                2,
            ],
            [
                39.984222412109375,
                116.31940460205078,
                Timestamp('2008-10-23 05:53:11'),
                2,
            ],
        ],
        columns=cols,
        index=[2, 3],
    )
    assert_frame_equal(move_df, expected)
    assert move_df.len() == 2


def test_by_tid():
    move_df = MoveDataFrame(
        data=list_data_1,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        datetime=DATETIME,
        traj_id=TRAJ_ID,
    )
    move_df['tid'] = ['123', '456', '789', '789']
    cols = ['lat', 'lon', 'datetime', 'id', 'tid']

    filter_values = filters.by_tid(move_df, tid_='456')
    expected = DataFrame(
        [
            [
                39.984199952392578,
                116.31932067871094,
                Timestamp('2008-10-23 05:53:06'),
                1,
                '456',
            ],
        ],
        columns=cols,
        index=[1],
    )
    assert_frame_equal(filter_values, expected)
    assert move_df.len() == 4

    filter_out_values = filters.by_tid(move_df, tid_='456', filter_out=True)
    expected = DataFrame(
        [
            [
                39.984092712402344,
                116.31923675537101,
                Timestamp('2008-10-23 05:53:05'),
                1,
                '123',
            ],
            [
                39.984222412109375,
                116.31940460205078,
                Timestamp('2008-10-23 05:53:11'),
                2,
                '789',
            ],
            [
                39.984222412109375,
                116.31940460205078,
                Timestamp('2008-10-23 05:53:11'),
                2,
                '789',
            ],
        ],
        columns=cols,
        index=[0, 2, 3],
    )
    assert_frame_equal(filter_out_values, expected)
    assert move_df.len() == 4

    filters.by_tid(move_df, tid_='789', filter_out=False, inplace=True)
    expected = DataFrame(
        [
            [
                39.984222412109375,
                116.31940460205078,
                Timestamp('2008-10-23 05:53:11'),
                2,
                '789',
            ],
            [
                39.984222412109375,
                116.31940460205078,
                Timestamp('2008-10-23 05:53:11'),
                2,
                '789',
            ],
        ],
        columns=cols,
        index=[2, 3],
    )

    assert_frame_equal(move_df, expected)
    assert move_df.len() == 2


def test_outliers():
    move_df, cols = _prepare_df_with_distances()

    outliers = filters.outliers(move_data=move_df, jump_coefficient=1)
    expected = DataFrame(
        data=[
            [
                1,
                38.984211,
                115.319389,
                Timestamp('2008-10-23 05:53:11'),
                140454.64510608764,
                140460.97747220137,
                7.563335999941849,
            ],
        ],
        columns=cols,
        index=[2],
    )
    assert_frame_equal(outliers, expected)
    assert move_df.len() == 5

    not_outliers = filters.outliers(
        move_data=move_df, jump_coefficient=1, filter_out=True
    )
    expected = DataFrame(
        data=[
            [
                1,
                39.984093,
                116.319237,
                Timestamp('2008-10-23 05:53:05'),
                nan,
                13.884481484192932,
                nan,
            ],
            [
                1,
                39.9842,
                116.319321,
                Timestamp('2008-10-23 05:53:06'),
                13.884481484192932,
                140454.64510608764,
                140440.86267282354,
            ],
            [
                1,
                39.984222,
                116.319405,
                Timestamp('2008-10-23 05:53:16'),
                140460.97747220137,
                1.3208180727070074,
                140461.50100279532,
            ],
            [
                1,
                39.984219,
                116.31942,
                Timestamp('2008-10-23 05:53:21'),
                1.3208180727070074,
                nan,
                nan,
            ],
        ],
        columns=cols,
        index=[0, 1, 3, 4],
    )
    assert_frame_equal(not_outliers, expected)
    assert move_df.len() == 5

    filters.outliers(move_data=move_df, jump_coefficient=1, inplace=True)
    expected = DataFrame(
        data=[
            [
                1,
                38.984211,
                115.319389,
                Timestamp('2008-10-23 05:53:11'),
                140454.64510608764,
                140460.97747220137,
                7.563335999941849,
            ]
        ],
        columns=cols,
        index=[2],
    )
    assert_frame_equal(move_df, expected)
    assert move_df.len() == 1


def test_clean_duplicates():
    move_df, cols = _prepare_df_default(list_data_1)

    duplicates_all = filters.clean_duplicates(move_df)
    expected = DataFrame(
        [
            [
                39.984092712402344,
                116.31923675537101,
                Timestamp('2008-10-23 05:53:05'),
                1,
            ],
            [
                39.984199952392578,
                116.31932067871094,
                Timestamp('2008-10-23 05:53:06'),
                1,
            ],
            [
                39.984222412109375,
                116.31940460205078,
                Timestamp('2008-10-23 05:53:11'),
                2,
            ],
        ],
        columns=cols,
        index=[0, 1, 2],
    )
    assert_frame_equal(duplicates_all, expected)
    assert move_df.len() == 4

    duplicates_subset = filters.clean_duplicates(move_df, subset=TRAJ_ID)
    expected = DataFrame(
        [
            [
                39.984092712402344,
                116.31923675537101,
                Timestamp('2008-10-23 05:53:05'),
                1,
            ],
            [
                39.984222412109375,
                116.31940460205078,
                Timestamp('2008-10-23 05:53:11'),
                2,
            ],
        ],
        columns=cols,
        index=[0, 2],
    )
    assert_frame_equal(duplicates_subset, expected)
    assert move_df.len() == 4

    duplicates_last = filters.clean_duplicates(
        move_df, subset=TRAJ_ID, keep='last'
    )
    expected = DataFrame(
        [
            [
                39.984199952392578,
                116.31932067871094,
                Timestamp('2008-10-23 05:53:06'),
                1,
            ],
            [
                39.984222412109375,
                116.31940460205078,
                Timestamp('2008-10-23 05:53:11'),
                2,
            ],
        ],
        columns=cols,
        index=[1, 3],
    )
    assert_frame_equal(duplicates_last, expected)
    assert move_df.len() == 4

    filters.clean_duplicates(move_df, subset=TRAJ_ID, inplace=True)
    expected = DataFrame(
        [
            [
                39.984092712402344,
                116.31923675537101,
                Timestamp('2008-10-23 05:53:05'),
                1,
            ],
            [
                39.984222412109375,
                116.31940460205078,
                Timestamp('2008-10-23 05:53:11'),
                2,
            ],
        ],
        columns=cols,
        index=[0, 2],
    )
    assert_frame_equal(move_df, expected)
    assert move_df.len() == 2


def test_clean_nan_values():
    move_df, cols = _prepare_df_default(list_data_1)
    move_df.loc[3, TRAJ_ID] = nan

    drop_nan = filters.clean_nan_values(move_df)
    expected = DataFrame(
        [
            [
                39.984092712402344,
                116.31923675537101,
                Timestamp('2008-10-23 05:53:05'),
                1.0,
            ],
            [
                39.984199952392578,
                116.31932067871094,
                Timestamp('2008-10-23 05:53:06'),
                1.0,
            ],
            [
                39.984222412109375,
                116.31940460205078,
                Timestamp('2008-10-23 05:53:11'),
                2.0,
            ],
        ],
        columns=cols,
        index=[0, 1, 2],
    )
    assert_frame_equal(drop_nan, expected)
    assert move_df.len() == 4

    filters.clean_nan_values(move_df, inplace=True)
    expected = DataFrame(
        [
            [
                39.984092712402344,
                116.31923675537101,
                Timestamp('2008-10-23 05:53:05'),
                1.0,
            ],
            [
                39.984199952392578,
                116.31932067871094,
                Timestamp('2008-10-23 05:53:06'),
                1.0,
            ],
            [
                39.984222412109375,
                116.31940460205078,
                Timestamp('2008-10-23 05:53:11'),
                2.0,
            ],
        ],
        columns=cols,
        index=[0, 1, 2],
    )
    assert_frame_equal(drop_nan, expected)
    assert move_df.len() == 3


def test_clean_gps_jumps_by_distance():
    move_df, cols = _prepare_df_with_distances()

    not_jumps = filters.clean_gps_jumps_by_distance(
        move_data=move_df, jump_coefficient=1
    )
    expected = DataFrame(
        data=[
            [
                1,
                39.984093,
                116.319237,
                Timestamp('2008-10-23 05:53:05'),
                nan,
                13.884481484192932,
                nan,
            ],
            [
                1,
                39.9842,
                116.319321,
                Timestamp('2008-10-23 05:53:06'),
                13.884481484192932,
                140454.64510608764,
                140440.86267282354,
            ],
            [
                1,
                39.984222,
                116.319405,
                Timestamp('2008-10-23 05:53:16'),
                140460.97747220137,
                1.3208180727070074,
                140461.50100279532,
            ],
            [
                1,
                39.984219,
                116.31942,
                Timestamp('2008-10-23 05:53:21'),
                1.3208180727070074,
                nan,
                nan,
            ],
        ],
        columns=cols,
        index=[0, 1, 3, 4],
    )
    assert_frame_equal(not_jumps, expected)
    assert move_df.len() == 5

    filters.clean_gps_jumps_by_distance(
        move_data=move_df, jump_coefficient=1, inplace=True
    )
    assert_frame_equal(move_df, expected)
    assert move_df.len() == 4


def test_clean_gps_nearby_points_by_distances():
    move_df, cols = _prepare_df_with_distances()

    not_nearby = filters.clean_gps_nearby_points_by_distances(
        move_df, radius_area=20.0
    )
    expected = DataFrame(
        data=[
            [
                1,
                39.984093,
                116.319237,
                Timestamp('2008-10-23 05:53:05'),
                nan,
                13.884481484192932,
                nan,
            ],
            [
                1,
                38.984211,
                115.319389,
                Timestamp('2008-10-23 05:53:11'),
                140454.64510608764,
                140460.97747220137,
                7.563335999941849,
            ],
            [
                1,
                39.984222,
                116.319405,
                Timestamp('2008-10-23 05:53:16'),
                140460.97747220137,
                1.3208180727070074,
                140461.50100279532,
            ],
        ],
        columns=cols,
        index=[0, 2, 3],
    )
    assert_frame_equal(not_nearby, expected)
    assert move_df.len() == 5

    filters.clean_gps_nearby_points_by_distances(
        move_df, radius_area=20.0, inplace=True
    )
    assert_frame_equal(move_df, expected)
    assert move_df.len() == 3


def test_clean_gps_nearby_points_by_speed():
    move_df, cols = _prepare_df_with_dist_time_speed()

    speedy = filters.clean_gps_nearby_points_by_speed(
        move_df, speed_radius=5.0
    )

    expected = DataFrame(
        data=[
            [
                1,
                39.984093,
                116.319237,
                Timestamp('2008-10-23 05:53:05'),
                nan,
                nan,
                nan,
            ],
            [
                1,
                39.9842,
                116.319321,
                Timestamp('2008-10-23 05:53:06'),
                13.884481484192932,
                1.0,
                13.884481484192932,
            ],
            [
                1,
                38.984211,
                115.319389,
                Timestamp('2008-10-23 05:53:11'),
                140454.64510608764,
                5.0,
                28090.92902121753,
            ],
            [
                1,
                39.984222,
                116.319405,
                Timestamp('2008-10-23 05:53:16'),
                140460.97747220137,
                5.0,
                28092.195494440275,
            ],
        ],
        columns=cols,
        index=[0, 1, 2, 3],
    )
    assert_frame_equal(speedy, expected)
    assert move_df.len() == 5

    filters.clean_gps_nearby_points_by_speed(
        move_df, speed_radius=5.0, inplace=True
    )
    assert_frame_equal(move_df, expected)
    assert move_df.len() == 4


def test_clean_gps_speed_max_radius():
    move_df, cols = _prepare_df_with_dist_time_speed()

    not_speedy = filters.clean_gps_speed_max_radius(move_df, speed_max=15)
    expected = DataFrame(
        data=[
            [
                1,
                39.984093,
                116.319237,
                Timestamp('2008-10-23 05:53:05'),
                nan,
                nan,
                nan,
            ],
            [
                1,
                39.9842,
                116.319321,
                Timestamp('2008-10-23 05:53:06'),
                13.884481484192932,
                1.0,
                13.884481484192932,
            ],
        ],
        columns=cols,
        index=[0, 1],
    )
    assert_frame_equal(not_speedy, expected)
    assert move_df.len() == 5

    filters.clean_gps_speed_max_radius(move_df, speed_max=15, inplace=True)
    assert_frame_equal(move_df, expected)
    assert move_df.len() == 2


def test_clean_trajectories_with_few_points():
    move_df, cols = _prepare_df_default(list_data_3)

    not_few_points = filters.clean_trajectories_with_few_points(
        move_df, min_points_per_trajectory=2, label_tid=TRAJ_ID
    )

    expected = DataFrame(
        data=[
            [39.984093, 116.319237, Timestamp('2008-10-23 05:53:05'), 1],
            [39.984200, 116.319321, Timestamp('2008-10-23 05:54:06'), 1],
            [39.984222, 116.319405, Timestamp('2008-10-23 05:55:16'), 1],
            [39.984219, 116.319420, Timestamp('2008-10-23 05:56:21'), 1],
            [39.984199, 116.319320, Timestamp('2008-10-23 05:53:06'), 2],
            [39.974222, 116.339404, Timestamp('2008-10-23 05:53:11'), 2],
        ],
        columns=cols,
        index=[0, 1, 2, 3, 4, 5],
    )
    assert_frame_equal(not_few_points, expected)
    assert move_df.len() == 6

    filters.clean_trajectories_with_few_points(
        move_df, min_points_per_trajectory=3, label_tid=TRAJ_ID, inplace=True
    )
    expected = DataFrame(
        data=[
            [39.984093, 116.319237, Timestamp('2008-10-23 05:53:05'), 1],
            [39.984200, 116.319321, Timestamp('2008-10-23 05:54:06'), 1],
            [39.984222, 116.319405, Timestamp('2008-10-23 05:55:16'), 1],
            [39.984219, 116.319420, Timestamp('2008-10-23 05:56:21'), 1],
        ],
        columns=cols,
        index=[0, 1, 2, 3],
    )
    assert_frame_equal(move_df, expected)
    assert move_df.len() == 4


def test_clean_trajectories_short_and_few_points():
    move_df, cols = _prepare_df_with_long_trajectory()

    not_short_and_few = filters.clean_trajectories_short_and_few_points(
        move_df,
        label_id=TRAJ_ID,
        min_trajectory_distance=100,
        min_points_per_trajectory=2,
    )
    expected = DataFrame(
        data=[
            [
                2,
                39.984199,
                116.31932,
                Timestamp('2008-10-23 05:53:06'),
                nan,
                nan,
                nan,
            ],
            [
                2,
                39.974222,
                116.339404,
                Timestamp('2008-10-23 05:53:11'),
                2039.4197149375298,
                5.0,
                407.883942987506,
            ],
        ],
        columns=cols,
        index=[4, 5],
    )
    assert_frame_equal(not_short_and_few, expected)
    assert move_df.len() == 6

    filters.clean_trajectories_short_and_few_points(
        move_df,
        label_id=TRAJ_ID,
        min_trajectory_distance=100,
        min_points_per_trajectory=3,
        inplace=True,
    )
    expected = DataFrame(
        data=[
            [
                1,
                39.984093,
                116.319237,
                Timestamp('2008-10-23 05:53:05'),
                nan,
                nan,
                nan,
            ],
        ],
        columns=cols,
        index=[0],
    )
    expected.drop(index=0, inplace=True)
    assert_frame_equal(move_df, expected)
    assert move_df.len() == 0


def test_clean_id_by_time_max():
    move_df, cols = _prepare_df_with_long_trajectory()

    long_time = filters.clean_id_by_time_max(
        move_df, label_id=TRAJ_ID, time_max=120
    )
    expected = DataFrame(
        data=[
            [
                1,
                39.984093,
                116.319237,
                Timestamp('2008-10-23 05:53:05'),
                nan,
                nan,
                nan,
            ],
            [
                1,
                39.9842,
                116.319321,
                Timestamp('2008-10-23 05:54:06'),
                13.884481484192932,
                61.00000000000001,
                0.22761445056053983,
            ],
            [
                1,
                39.984222,
                116.319405,
                Timestamp('2008-10-23 05:55:16'),
                7.563335999941849,
                70.0,
                0.10804765714202641,
            ],
            [
                1,
                39.984219,
                116.31942,
                Timestamp('2008-10-23 05:56:21'),
                1.3208180727070074,
                65.0,
                0.020320278041646267,
            ],
        ],
        columns=cols,
        index=[0, 1, 2, 3],
    )
    assert_frame_equal(long_time, expected)
    assert move_df.len() == 6

    filters.clean_id_by_time_max(
        move_df, label_id=TRAJ_ID, time_max=3600, inplace=True
    )
    expected = DataFrame(
        data=[
            [
                1,
                39.984093,
                116.319237,
                Timestamp('2008-10-23 05:53:05'),
                nan,
                nan,
                nan,
            ],
        ],
        columns=cols,
        index=[0],
    )
    expected.drop(index=0, inplace=True)
    assert_frame_equal(move_df, expected)
    assert move_df.len() == 0
