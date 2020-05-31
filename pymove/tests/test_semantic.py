import os
import time

import pandas as pd
from numpy import array, nan
from shapely.geometry import Polygon
from pandas import DataFrame, Timestamp
from pymove import MoveDataFrame, semantic
from pandas.testing import assert_frame_equal
from numpy.testing import assert_array_equal

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
    [39.984224, 116.319402, '2008-10-23 05:53:11', 1],
    [39.984211, 116.319389, '2008-10-23 05:53:16', 1],
    [39.984217, 116.319422, '2008-10-23 05:53:21', 1],
    [39.984710, 116.319865, '2008-10-23 05:53:23', 1],
    [39.984674, 116.319810, '2008-10-23 05:53:28', 1],
    [39.984623, 116.319773, '2008-10-23 05:53:33', 1],
    [39.984606, 116.319732, '2008-10-23 05:53:38', 1],
    [39.984555, 116.319728, '2008-10-23 05:53:43', 1]
]


def _default_move_df(list_data=list_data):
    return MoveDataFrame(
        data=list_data,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        datetime=DATETIME,
        traj_id=TRAJ_ID,
    )


def test_end_create_operation():
    move_df = _default_move_df()
    expected = DataFrame(
        data=[
            [39.984094, 116.319236, Timestamp('2008-10-23 05:53:05'), 1],
            [39.984198, 116.319322, Timestamp('2008-10-23 05:53:06'), 1],
            [39.984224, 116.319402, Timestamp('2008-10-23 05:53:11'), 1],
            [39.984211, 116.319389, Timestamp('2008-10-23 05:53:16'), 1],
            [39.984217, 116.319422, Timestamp('2008-10-23 05:53:21'), 1],
            [39.984710, 116.319865, Timestamp('2008-10-23 05:53:23'), 1],
            [39.984674, 116.319810, Timestamp('2008-10-23 05:53:28'), 1],
            [39.984623, 116.319773, Timestamp('2008-10-23 05:53:33'), 1],
            [39.984606, 116.319732, Timestamp('2008-10-23 05:53:38'), 1],
            [39.984555, 116.319728, Timestamp('2008-10-23 05:53:43'), 1]
        ],
        columns=['lat', 'lon', 'datetime', 'id'],
        index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    )

    new_move_df = semantic._end_create_operation(move_df, 'lat', time.time(), False)

    assert_frame_equal(new_move_df, expected)

    semantic._end_create_operation(move_df, 'lat', time.time(), True)

    assert_frame_equal(move_df, expected)


def test_process_simple_filter():

    move_df = _default_move_df()

    expected = DataFrame(
        data=[
            [39.984094, 116.319236, Timestamp('2008-10-23 05:53:05'), 1, False],
            [39.984198, 116.319322, Timestamp('2008-10-23 05:53:06'), 1, True],
            [39.984224, 116.319402, Timestamp('2008-10-23 05:53:11'), 1, True],
            [39.984211, 116.319389, Timestamp('2008-10-23 05:53:16'), 1, True],
            [39.984217, 116.319422, Timestamp('2008-10-23 05:53:21'), 1, True],
            [39.984710, 116.319865, Timestamp('2008-10-23 05:53:23'), 1, True],
            [39.984674, 116.319810, Timestamp('2008-10-23 05:53:28'), 1, True],
            [39.984623, 116.319773, Timestamp('2008-10-23 05:53:33'), 1, True],
            [39.984606, 116.319732, Timestamp('2008-10-23 05:53:38'), 1, True],
            [39.984555, 116.319728, Timestamp('2008-10-23 05:53:43'), 1, True]
        ],
        columns=['lat', 'lon', 'datetime', 'id', 'new_label'],
        index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    )

    new_move_df = semantic._process_simple_filter(move_df,
                                                  'new_label',
                                                  'lat',
                                                  39.984217,
                                                  time.time(),
                                                  False
                                                  )

    assert_frame_equal(new_move_df, expected)

    semantic._process_simple_filter(move_df,
                                    'new_label',
                                    'lat',
                                    39.984217,
                                    time.time(),
                                    True)

    assert_frame_equal(move_df, expected)


def test_create_or_update_out_of_the_bbox():
    bbox = (39.984217, 116.319236, 39.98471, 116.319865)
    move_df = _default_move_df()

    expected = DataFrame(
        data=[
            [39.984094, 116.319236, Timestamp('2008-10-23 05:53:05'), 1, True],
            [39.984198, 116.319322, Timestamp('2008-10-23 05:53:06'), 1, True],
            [39.984224, 116.319402, Timestamp('2008-10-23 05:53:11'), 1, False],
            [39.984211, 116.319389, Timestamp('2008-10-23 05:53:16'), 1, True],
            [39.984217, 116.319422, Timestamp('2008-10-23 05:53:21'), 1, False],
            [39.984710, 116.319865, Timestamp('2008-10-23 05:53:23'), 1, False],
            [39.984674, 116.319810, Timestamp('2008-10-23 05:53:28'), 1, False],
            [39.984623, 116.319773, Timestamp('2008-10-23 05:53:33'), 1, False],
            [39.984606, 116.319732, Timestamp('2008-10-23 05:53:38'), 1, False],
            [39.984555, 116.319728, Timestamp('2008-10-23 05:53:43'), 1, False]
        ],
        columns=['lat', 'lon', 'datetime', 'id', 'out_bbox'],
        index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    )

    semantic.create_or_update_out_of_the_bbox(move_df, bbox)

    assert_frame_equal(move_df, expected)


def test_create_or_update_gps_deactivated_signal():
    move_df = _default_move_df()

    expected = DataFrame(
        data=[
            [1, 39.984094, 116.319236, Timestamp('2008-10-23 05:53:05'),
             nan, 1.0, nan, False],
            [1, 39.984198, 116.319322, Timestamp('2008-10-23 05:53:06'),
             1.0, 5.0, 6.0, True],
            [1, 39.984224, 116.319402, Timestamp('2008-10-23 05:53:11'),
             5.0, 5.0, 10.0, True],
            [1, 39.984211, 116.319389, Timestamp('2008-10-23 05:53:16'),
             5.0, 5.0, 10.0, True],
            [1, 39.984217, 116.319422, Timestamp('2008-10-23 05:53:21'),
             5.0, 2.0, 7.0, True],
            [1, 39.984710, 116.319865, Timestamp('2008-10-23 05:53:23'),
             2.0, 5.0, 7.0, True],
            [1, 39.984674, 116.319810, Timestamp('2008-10-23 05:53:28'),
             5.0, 5.0, 10.0, True],
            [1, 39.984623, 116.319773, Timestamp('2008-10-23 05:53:33'),
             5.0, 5.0, 10.0, True],
            [1, 39.984606, 116.319732, Timestamp('2008-10-23 05:53:38'),
             5.0, 5.0, 10.0, True],
            [1, 39.984555, 116.319728, Timestamp('2008-10-23 05:53:43'),
             5.0, nan, nan, True]
        ],
        columns=['id',
                 'lat',
                 'lon',
                 'datetime',
                 'time_to_prev',
                 'time_to_next',
                 'time_prev_to_next',
                 'deactivated_signal'],
        index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    )

    new_move_df = semantic.create_or_update_gps_deactivated_signal(move_df,
                                                                   max_time_between_adj_points=5.0,
                                                                   inplace=False)

    assert_frame_equal(new_move_df, expected)

    semantic.create_or_update_gps_deactivated_signal(move_df,
                                                     max_time_between_adj_points=5.0)

    assert_frame_equal(move_df, expected)


def test_create_or_update_gps_jump():
    move_df = _default_move_df()

    expected = DataFrame(
        data=[
            [1, 39.984094, 116.319236, Timestamp('2008-10-23 05:53:05'),
             nan, 13.690153, nan, True],
            [1, 39.984198, 116.319322, Timestamp('2008-10-23 05:53:06'),
             13.690153, 7.403788, 20.223428, True],
            [1, 39.984224, 116.319402, Timestamp('2008-10-23 05:53:11'),
             7.403788, 1.821083, 5.888579, True],
            [1, 39.984211, 116.319389, Timestamp('2008-10-23 05:53:16'),
             1.821083, 2.889671, 1.873356, False],
            [1, 39.984217, 116.319422, Timestamp('2008-10-23 05:53:21'),
             2.889671, 66.555997, 68.727260, True],
            [1, 39.984710, 116.319865, Timestamp('2008-10-23 05:53:23'),
             66.555997, 6.162987, 60.622358, True],
            [1, 39.984674, 116.319810, Timestamp('2008-10-23 05:53:28'),
             6.162987, 6.488225, 12.450907, True],
            [1, 39.984623, 116.319773, Timestamp('2008-10-23 05:53:33'),
             6.488225, 3.971848, 10.066577, True],
            [1, 39.984606, 116.319732, Timestamp('2008-10-23 05:53:38'),
             3.971848, 5.681172, 8.477733, True],
            [1, 39.984555, 116.319728, Timestamp('2008-10-23 05:53:43'),
             5.681172, nan, nan, True]
        ],
        columns=['id',
                 'lat',
                 'lon',
                 'datetime',
                 'dist_to_prev',
                 'dist_to_next',
                 'dist_prev_to_next',
                 'gps_jump'],
        index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    )

    new_move_df = semantic.create_or_update_gps_jump(move_df,
                                                     max_dist_between_adj_points=5.0,
                                                     inplace=False)

    assert_frame_equal(new_move_df, expected)

    semantic.create_or_update_gps_jump(move_df, max_dist_between_adj_points=5.0)

    assert_frame_equal(move_df, expected)


def test_create_or_update_short_trajectory():
    move_df = _default_move_df()
    move_df.at[[6, 7, 8, 9], 'id'] = 2

    expected = DataFrame(
        data=[
            [1, 39.984094, 116.319236, Timestamp('2008-10-23 05:53:05'),
             nan, nan, nan, 1, False],
            [1, 39.984198, 116.319322, Timestamp('2008-10-23 05:53:06'),
             13.690153, 1.0, 13.690153, 1, False],
            [1, 39.984224, 116.319402, Timestamp('2008-10-23 05:53:11'),
             7.403788, 5.0, 1.480758, 1, False],
            [1, 39.984211, 116.319389, Timestamp('2008-10-23 05:53:16'),
             1.821083, 5.0, 0.364217, 1, False],
            [1, 39.984217, 116.319422, Timestamp('2008-10-23 05:53:21'),
             2.889671, 5.0, 0.577934, 1, False],
            [1, 39.984710, 116.319865, Timestamp('2008-10-23 05:53:23'),
             66.555997, 2.0, 33.277998, 1, False],
            [2, 39.984674, 116.319810, Timestamp('2008-10-23 05:53:28'),
             nan, nan, nan, 2, True],
            [2, 39.984623, 116.319773, Timestamp('2008-10-23 05:53:33'),
             6.488225, 5.0, 1.297645, 2, True],
            [2, 39.984606, 116.319732, Timestamp('2008-10-23 05:53:38'),
             3.971848, 5.0, 0.794370, 2, True],
            [2, 39.984555, 116.319728, Timestamp('2008-10-23 05:53:43'),
             5.681172, 5.0, 1.136234, 2, True]
        ],
        columns=['id',
                 'lat',
                 'lon',
                 'datetime',
                 'dist_to_prev',
                 'time_to_prev',
                 'speed_to_prev',
                 'tid_part',
                 'short_traj'],
        index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    )

    new_move_df = semantic.create_or_update_short_trajectory(move_df,
                                                             k_segment_max=4,
                                                             inplace=False)

    assert_frame_equal(new_move_df, expected)

    assert('short_traj' not in move_df)

    semantic.create_or_update_short_trajectory(move_df, k_segment_max=4)

    assert_frame_equal(move_df, expected)


def test_create_or_update_gps_block_signal():
    move_df = _default_move_df()

    expected = DataFrame(
        data=[
            [1, 39.984094, 116.319236, Timestamp('2008-10-23 05:53:05'),
             nan, nan, nan, False],
            [2, 39.984198, 116.319322, Timestamp('2008-10-23 05:53:06'),
             nan, nan, nan, False],
            [3, 39.984224, 116.319402, Timestamp('2008-10-23 05:53:11'),
             nan, nan, nan, False],
            [4, 39.984211, 116.319389, Timestamp('2008-10-23 05:53:16'),
             nan, nan, nan, False],
            [5, 39.984217, 116.319422, Timestamp('2008-10-23 05:53:21'),
             nan, nan, nan, False],
            [6, 39.984710, 116.319865, Timestamp('2008-10-23 05:53:23'),
             nan, nan, nan, False],
            [7, 39.984674, 116.319810, Timestamp('2008-10-23 05:53:28'),
             nan, nan, nan, False],
            [8, 39.984623, 116.319773, Timestamp('2008-10-23 05:53:33'),
             nan, nan, nan, False],
            [9, 39.984606, 116.319732, Timestamp('2008-10-23 05:53:38'),
             nan, nan, nan, False],
            [10, 39.984555, 116.319728, Timestamp('2008-10-23 05:53:43'),
             nan, nan, nan, False]
        ],
        columns=['id',
                 'lat',
                 'lon',
                 'datetime',
                 'dist_to_prev',
                 'time_to_prev',
                 'speed_to_prev',
                 'block_signal'],
        index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    )

    new_move_df = semantic.create_or_update_gps_block_signal(move_df,
                                                             inplace=False)

    assert_frame_equal(new_move_df, expected)

    assert('block_signal' not in move_df)

    semantic.create_or_update_gps_block_signal(move_df)

    assert_frame_equal(move_df, expected)


def test_filter_block_signal_by_repeated_amount_of_points():
    move_df = _default_move_df()

    expected = ['id',
                'lat',
                'lon',
                'datetime',
                'dist_to_prev',
                'time_to_prev',
                'speed_to_prev',
                'block_signal']

    new_move_df = semantic.filter_block_signal_by_repeated_amount_of_points(move_df, 
                                                       inplace=False)
    
    assert(new_move_df.empty)
    
    assert_array_equal(new_move_df.columns, expected)

    new_move_df = semantic.filter_block_signal_by_repeated_amount_of_points(move_df, 
                                                                            filter_out=True,
                                                                            inplace=False)
    
    assert(new_move_df.empty)
    
    assert_array_equal(new_move_df.columns, expected)

    semantic.filter_block_signal_by_repeated_amount_of_points(move_df, 
                                                              inplace=True)
    
    assert(move_df.empty)


def test_filter_block_signal_by_time():
    move_df = _default_move_df()

    expected = ['id',
                'lat',
                'lon',
                'datetime',
                'dist_to_prev',
                'time_to_prev',
                'speed_to_prev',
                'block_signal']  

    new_move_df = semantic.filter_block_signal_by_time(move_df, 
                                                       inplace=False)
    
    assert(new_move_df.empty)
    
    assert_array_equal(new_move_df.columns, expected)

    new_move_df = semantic.filter_block_signal_by_time(move_df, 
                                                       filter_out=True,
                                                       inplace=False)
    
    assert(new_move_df.empty)
    
    assert_array_equal(new_move_df.columns, expected)

    semantic.filter_block_signal_by_time(move_df, inplace=True)
    
    assert(move_df.empty)
    
    assert_array_equal(move_df.columns, expected)


def test_filter_longer_time_to_stop_segment_by_id():

    move_df = _default_move_df()   
    
    expected = ['segment_stop', 
                'id', 
                'lat', 
                'lon', 
                'datetime', 
                'dist_to_prev',
                'time_to_prev', 
                'speed_to_prev', 
                'stop']           
    
    new_move_df = semantic.filter_longer_time_to_stop_segment_by_id(move_df, 
                                                                    inplace=False)
    
    assert(new_move_df.empty)
    
    assert_array_equal(new_move_df.columns, expected)

    new_move_df = semantic.filter_longer_time_to_stop_segment_by_id(move_df, 
                                                                    filter_out=True,
                                                                    inplace=False)
    
    assert(new_move_df.empty)
    
    assert_array_equal(new_move_df.columns, expected)

    semantic.filter_longer_time_to_stop_segment_by_id(move_df, inplace=True)
    
    assert(move_df.empty)
    
    assert_array_equal(move_df.columns, expected)

