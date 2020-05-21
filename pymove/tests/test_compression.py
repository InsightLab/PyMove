from pymove import compression
from pymove import MoveDataFrame
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd

list_data_test =[[39.984093, 116.319237, '2008-10-23 05:53:05', 1],
                 [39.984200, 116.319321, '2008-10-23 05:53:06', 1],
                 [39.984222, 116.319405, '2008-10-23 05:53:11', 1],
                 [39.984211, 116.319389,'2008-10-23 05:53:16', 1],
                 [39.984219, 116.319420, '2008-10-23 05:53:21', 1]]


def test_compress_segment_stop_to_point():
    # Testing inplace = False and lat_mean and lon_mean defined based on point that repeats most within the segment
    move_df = MoveDataFrame(data=list_data_test, latitude="lat", longitude="lon", datetime="datetime", traj_id="id")
    compressed_trajs = compression.compress_segment_stop_to_point(move_df, dist_radius=3.5, time_radius=0.5)
    assert_array_equal(compressed_trajs[['lat_mean', 'lon_mean']].astype(str).values, [['nan', 'nan'],
                                                                                       ['nan', 'nan'],
                                                                                       ['39.984222412109375',
                                                                                        '116.31940460205078'],
                                                                                       ['39.984222412109375',
                                                                                        '116.31940460205078']])

    # Testing inplace = False and lat_mean and lon_mean defined using centroids
    compressed_trajs = compression.compress_segment_stop_to_point(move_df, dist_radius=3.5, time_radius=0.5,
                                                                  point_mean='centroid')
    assert_array_equal(compressed_trajs[['lat_mean', 'lon_mean']].astype(str).values, [['nan', 'nan'],
                                                                                       ['nan', 'nan'],
                                                                                       ['39.98421859741211',
                                                                                        '116.31940460205078'],
                                                                                       ['39.98421859741211',
                                                                                        '116.31940460205078']])

    # Testing droping move
    compressed_trajs = compression.compress_segment_stop_to_point(move_df, dist_radius=3.5, time_radius=0.5,
                                                                  point_mean='centroid', drop_moves=True)
    assert_array_equal(compressed_trajs[['lat_mean', 'lon_mean']].astype(str).values,
                       [['39.98421859741211', '116.31940460205078'],
                        ['39.98421859741211', '116.31940460205078']])

    # testing inplace =True
    compression.compress_segment_stop_to_point(move_df, dist_radius=3.5, time_radius=0.5, point_mean='centroid',
                                               drop_moves=True, inplace=True)
    assert_array_equal(move_df[['lat_mean', 'lon_mean']].astype(str).values,
                       [['39.98421859741211', '116.31940460205078'],
                        ['39.98421859741211', '116.31940460205078']])


def test_compress_segment_stop_to_point_optimizer():
    # Testing inplace = False and lat_mean and lon_mean defined based on point that repeats most within the segment
    move_df = MoveDataFrame(data=list_data_test, latitude="lat", longitude="lon", datetime="datetime", traj_id="id")
    compressed_trajs = compression.compress_segment_stop_to_point_optimizer(move_df, dist_radius=3.5, time_radius=0.5)
    assert_array_equal(compressed_trajs[['lat_mean', 'lon_mean']].astype(str).values, [['nan', 'nan'],
                                                                                       ['nan', 'nan'],
                                                                                       ['39.984222', '116.319405'],
                                                                                       ['39.984222', '116.319405']])

    # Testing inplace = False and lat_mean and lon_mean defined using centroids
    compressed_trajs = compression.compress_segment_stop_to_point_optimizer(move_df, dist_radius=3.5, time_radius=0.5,
                                                                            point_mean='centroid')
    assert_array_equal(compressed_trajs[['lat_mean', 'lon_mean']].astype(str).values, [['nan', 'nan'],
                                                                                       ['nan', 'nan'],
                                                                                       ['39.98422', '116.319405'],
                                                                                       ['39.98422', '116.319405']])
    # Testing droping move
    compressed_trajs = compression.compress_segment_stop_to_point_optimizer(move_df, dist_radius=3.5, time_radius=0.5,
                                                                            point_mean='centroid', drop_moves=True)
    assert_array_equal(compressed_trajs[['lat_mean', 'lon_mean']].astype(str).values, [['39.98422', '116.319405'],
                                                                                       ['39.98422', '116.319405']])

    # testing inplace =True
    compression.compress_segment_stop_to_point_optimizer(move_df, dist_radius=3.5, time_radius=0.5,
                                                         point_mean='centroid', drop_moves=True, inplace=True)
    assert_array_equal(move_df[['lat_mean', 'lon_mean']].astype(str).values, [['39.98422', '116.319405'],
                                                                              ['39.98422', '116.319405']])