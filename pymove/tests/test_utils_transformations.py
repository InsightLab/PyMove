
import numpy as np
from pandas import DataFrame, Timestamp
from pandas.testing import assert_frame_equal

from pymove import MoveDataFrame, transformations
from pymove.utils.constants import DATETIME, LATITUDE, LONGITUDE, TRAJ_ID

list_data = [
    [39.984094, 116.319236, '2008-10-23 05:53:05', 1],
    [39.984198, 116.319322, '2008-10-23 05:53:06', 1],
    [39.984224, 116.319402, '2008-10-23 05:53:11', 1],
    [39.984224, 116.319402, '2008-10-23 05:53:11', 2],
]


def _default_move_df():
    return MoveDataFrame(
        data=list_data,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        datetime=DATETIME,
        traj_id=TRAJ_ID,
    )


def test_feature_values_using_filter():

    expected = DataFrame(
        data=[
            [
                5.0,
                116.319236,
                Timestamp('2008-10-23 05:53:05'),
                1
            ],
            [
                5.0,
                116.319322,
                Timestamp('2008-10-23 05:53:06'),
                1
            ],
            [
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                1
            ],
            [
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                2
            ],
        ],
        columns=[LATITUDE, LONGITUDE, DATETIME, TRAJ_ID],
        index=[0, 1, 2, 3],
    )

    expected_one_filter = DataFrame(
        data=[
            [
                5.0,
                116.319236,
                Timestamp('2008-10-23 05:53:05'),
                1
            ],
            [
                5.0,
                116.319322,
                Timestamp('2008-10-23 05:53:06'),
                1
            ],
            [
                5.0,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                1
            ],
            [
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                2
            ],
        ],
        columns=[LATITUDE, LONGITUDE, DATETIME, TRAJ_ID],
        index=[0, 1, 2, 3],
    )

    expected.set_index(TRAJ_ID, inplace=True)
    expected_one_filter.set_index(TRAJ_ID, inplace=True)

    move_df = _default_move_df()
    move_df.set_index(TRAJ_ID, inplace=True)

    new_move_df = transformations.feature_values_using_filter(move_df,
                                                              1,
                                                              LATITUDE,
                                                              np.array([0, 1]),
                                                              5,
                                                              inplace=False)

    assert_frame_equal(new_move_df, expected)

    new_move_df = transformations.feature_values_using_filter(move_df,
                                                              1,
                                                              LATITUDE,
                                                              np.array(1),
                                                              5,
                                                              inplace=False)

    assert_frame_equal(new_move_df, expected_one_filter)

    transformations.feature_values_using_filter(move_df,
                                                1,
                                                LATITUDE,
                                                np.array([0, 1]),
                                                5,
                                                inplace=True)

    assert_frame_equal(move_df, expected)


def test_feature_values_using_filter_and_indexes():

    expected = DataFrame(
        data=[
            [
                5.0,
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
                5.0,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                1
            ],
            [
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                2
            ],
        ],
        columns=[LATITUDE, LONGITUDE, DATETIME, TRAJ_ID],
        index=[0, 1, 2, 3],
    )

    expected_one_filter = DataFrame(
        data=[
            [
                5.0,
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
                1
            ],
            [
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                2
            ],
        ],
        columns=[LATITUDE, LONGITUDE, DATETIME, TRAJ_ID],
        index=[0, 1, 2, 3],
    )

    expected.set_index(TRAJ_ID, inplace=True)
    expected_one_filter.set_index(TRAJ_ID, inplace=True)

    move_df = _default_move_df()
    move_df.set_index(TRAJ_ID, inplace=True)

    new_move_df = transformations.feature_values_using_filter_and_indexes(move_df,
                                                                          1,
                                                                          LATITUDE,
                                                                          [0 , 1, 2],
                                                                          [0, 2],
                                                                          5.0,
                                                                          inplace=False)

    assert_frame_equal(new_move_df, expected)

    new_move_df = transformations.feature_values_using_filter_and_indexes(move_df,
                                                                          1,
                                                                          LATITUDE,
                                                                          [0],
                                                                          [0],
                                                                          5.0,
                                                                          inplace=False)

    assert_frame_equal(new_move_df, expected_one_filter)

    transformations.feature_values_using_filter_and_indexes(move_df,
                                                            1,
                                                            LATITUDE,
                                                            [0 , 1, 2],
                                                            [0, 2],
                                                            5.0,
                                                            inplace=True)

    assert_frame_equal(move_df, expected)
