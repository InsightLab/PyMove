from pymove import stay_point_detection
from pymove import MoveDataFrame
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd

list_data = [[39.984094, 116.319236, '2008-10-23 05:53:05', 1],
             [39.984198, 116.319322, '2008-10-23 05:53:06', 1],
             [39.984224, 116.319402, '2008-10-23 05:53:11', 2],
             [39.984224, 116.319402, '2008-10-23 05:53:11', 2]]


list_data_test =[[39.984093, 116.319237, '2008-10-23 05:53:05', 1],
                 [39.984200, 116.319321, '2008-10-23 05:53:06', 1],
                 [39.984222, 116.319405, '2008-10-23 05:53:11', 1],
                 [39.984211, 116.319389,'2008-10-23 05:53:16', 1],
                 [39.984219, 116.319420, '2008-10-23 05:53:21', 1]]
move_df = MoveDataFrame(data=list_data_test, latitude="lat", longitude="lon", datetime="datetime", traj_id="id")


def test_create_update_datetime_in_format_cyclical():
    stay_point_detection.create_update_datetime_in_format_cyclical(move_df)
    assert_array_equal(move_df['hour_sin'], [0.9790840876823229, 0.9790840876823229, 0.9790840876823229,
                                            0.9790840876823229,0.9790840876823229])
    assert_array_equal(move_df['hour_cos'], [0.20345601305263375,0.20345601305263375,0.20345601305263375,
                                             0.20345601305263375,0.20345601305263375])
    move_df.drop(['hour_sin', 'hour_cos'], axis=1, inplace=True)


def test_create_or_update_move_stop_by_dist_time():
    df_move = MoveDataFrame(data=list_data_test, latitude="lat", longitude="lon", datetime="datetime", traj_id="id")
    stay_point_detection.create_or_update_move_stop_by_dist_time(df_move, dist_radius=3.5, time_radius=0.5)
    assert_array_equal(df_move['stop'], [False, False, True, True, True])


def test_create_update_move_and_stop_by_radius():
    stay_point_detection.create_update_move_and_stop_by_radius(move_df, radius=4.0)
    assert_array_equal(move_df['situation'].astype(str), ['nan', 'move', 'move', 'stop', 'stop'])