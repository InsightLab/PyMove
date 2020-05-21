from pymove import segmentation
from pymove import MoveDataFrame
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd


list_data = [[39.984094, 116.319236, '2008-10-23 05:53:05', 1],
             [39.984198, 116.319322, '2008-10-23 05:53:06', 1],
             [39.984224, 116.319402, '2008-10-23 05:53:11', 2],
             [39.984224, 116.319402, '2008-10-23 05:53:11', 2]]
#move_df = MoveDataFrame(data=list_data, latitude="lat", longitude="lon", datetime="datetime", traj_id="id")


list_data_test =[[39.984093, 116.319237, '2008-10-23 05:53:05', 1],
                 [39.984200, 116.319321, '2008-10-23 05:53:06', 1],
                 [39.984222, 116.319405, '2008-10-23 05:53:11', 1],
                 [39.984211, 116.319389,'2008-10-23 05:53:16', 1],
                 [39.984219, 116.319420, '2008-10-23 05:53:21', 1]]
#df_move = MoveDataFrame(data=list_data_test, latitude="lat", longitude="lon", datetime="datetime", traj_id="id")


data = [[39.984094, 116.319236, '2008-10-23 05:53:05', 1],
     [39.984198, 116.319322, '2008-10-23 05:53:06', 1],
     [39.984224, 116.319402, '2008-10-23 05:53:11', 1],
     [39.984224, 116.319402, '2008-10-23 05:53:11', 2]]


def test_bbox_split():
    bbox = (39.984093, 116.31924, 39.984222, 116.319405)
    segmented_bbox = segmentation.bbox_split(bbox, 3)
    assert_array_equal(segmented_bbox, [[ 39.984093, 116.31924 ,  39.984222, 116.319295],
                                       [ 39.984093, 116.319295,  39.984222, 116.31935 ],
                                       [ 39.984093, 116.31935 ,  39.984222, 116.319405]])

def test_by_dist_time_speed():
    move = MoveDataFrame(data=data, latitude="lat", longitude="lon", datetime="datetime", traj_id="id")
    segmented_move = segmentation.by_dist_time_speed(move, inplace=False)
    assert_array_equal(segmented_move['tid_part'], [1, 1, 1])
    assert (segmented_move.len() == 3)
    assert (segmented_move.index.name is None)

    df_move = MoveDataFrame(data=list_data, latitude="lat", longitude="lon", datetime="datetime", traj_id="id")
    segmentation.by_dist_time_speed(df_move)
    assert_array_equal(df_move['tid_part'], [1, 1, 2, 2])
    assert (df_move.index.name is None)


def test_by_max_dist():
    move = MoveDataFrame(data=list_data_test, latitude="lat", longitude="lon", datetime="datetime", traj_id="id")
    segmentation.by_max_dist(move, max_dist_between_adj_points=5.0)
    assert_array_equal(move['tid_dist'], [1,2,3,3,3])
    assert(move.index.name is None)


def test_by_max_time():
    move = MoveDataFrame(data=list_data_test, latitude="lat", longitude="lon", datetime="datetime", traj_id="id")
    segmentation.by_max_time(move, max_time_between_adj_points=1.0)
    assert_array_equal(move['tid_time'], [1,1,2,3,4])
    assert(move.index.name is None)


def test_by_max_speed():
    move = MoveDataFrame(data=list_data_test, latitude="lat", longitude="lon", datetime="datetime", traj_id="id")
    segmentation.by_max_speed(move, max_speed_between_adj_points=1.0)
    assert_array_equal(move['tid_speed'], [1,2,3,3,3])
    assert(move.index.name is None)


