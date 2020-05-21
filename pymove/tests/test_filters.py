from pymove import filters
from pymove import MoveDataFrame
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd

list_data = [[39.984094, 116.319236, '2008-10-23 05:53:05', 1],
             [39.984198, 116.319322, '2008-10-23 05:53:06', 1],
             [39.984224, 116.319402, '2008-10-23 05:53:11', 2],
             [39.984224, 116.319402, '2008-10-23 05:53:11', 2]]
move_df = MoveDataFrame(data=list_data, latitude="lat", longitude="lon", datetime="datetime", traj_id="id")

list_data_test =[[39.984093, 116.319237, '2008-10-23 05:53:05', 1],
                 [39.984200, 116.319321, '2008-10-23 05:53:06', 1],
                 [39.984222, 116.319405, '2008-10-23 05:53:11', 1],
                 [39.984211, 116.319389,'2008-10-23 05:53:16', 1],
                 [39.984219, 116.319420, '2008-10-23 05:53:21', 1]]


def test_by_bbox():
    move_df_bbox = MoveDataFrame(data=list_data, latitude="lat", longitude="lon", datetime="datetime", traj_id="id")
    bbox = (39.984193, 116.31924, 39.984222, 116.319405)

    filter_values = filters.by_bbox(move_df_bbox, bbox)
    assert_array_equal(filter_values, [[39.98419952392578, 116.31932067871094,
                                        pd.Timestamp('2008-10-23 05:53:06'), 1],
                                       [39.984222412109375, 116.31940460205078,
                                        pd.Timestamp('2008-10-23 05:53:11'), 2],
                                       [39.984222412109375, 116.31940460205078,
                                        pd.Timestamp('2008-10-23 05:53:11'), 2]])
    assert move_df_bbox.len() == 4
    filter_values = filters.by_bbox(move_df_bbox, bbox, filter_out=True)
    assert_array_equal(filter_values, [[39.984092712402344, 116.3192367553711,
                                        pd.Timestamp('2008-10-23 05:53:05'), 1]])
    assert move_df_bbox.len() == 4
    filters.by_bbox(move_df_bbox, bbox, inplace=True)
    assert_array_equal(move_df_bbox, [[39.98419952392578, 116.31932067871094,
                                       pd.Timestamp('2008-10-23 05:53:06'), 1],
                                      [39.984222412109375, 116.31940460205078,
                                       pd.Timestamp('2008-10-23 05:53:11'), 2],
                                      [39.984222412109375, 116.31940460205078,
                                       pd.Timestamp('2008-10-23 05:53:11'), 2]])
    assert move_df_bbox.len() == 3


def test_by_datetime():
    filter_values_start = filters.by_datetime(move_df, start_datetime='2008-10-23 05:53:06')
    assert_array_equal(filter_values_start, [[39.98419952392578, 116.31932067871094,
                                              pd.Timestamp('2008-10-23 05:53:06'), 1],
                                             [39.984222412109375, 116.31940460205078,
                                              pd.Timestamp('2008-10-23 05:53:11'), 2],
                                             [39.984222412109375, 116.31940460205078,
                                              pd.Timestamp('2008-10-23 05:53:11'), 2]])

    filter_values_end = filters.by_datetime(move_df, end_datetime='2008-10-23 05:53:05')
    assert_array_equal(filter_values_end, [[39.984092712402344, 116.3192367553711,
                                            pd.Timestamp('2008-10-23 05:53:05'), 1]])

    filter_values = filters.by_datetime(move_df, start_datetime='2008-10-23 05:53:04',
                                        end_datetime='2008-10-23 05:53:06')
    assert_array_equal(filter_values, [[39.984092712402344, 116.3192367553711,
                                        pd.Timestamp('2008-10-23 05:53:05'), 1],
                                       [39.98419952392578, 116.31932067871094,
                                        pd.Timestamp('2008-10-23 05:53:06'), 1]])

    filter_out_values = filters.by_datetime(move_df, start_datetime='2008-10-23 05:53:06',
                                            end_datetime='2008-10-23 05:53:11', filter_out=True)
    assert_array_equal(filter_out_values, [[39.984092712402344, 116.3192367553711,
                                            pd.Timestamp('2008-10-23 05:53:05'), 1]])


def test_by_label():
    filter_values = filters.by_label(move_df, value=1, label_name='id')
    assert_array_equal(filter_values, [[39.984092712402344, 116.3192367553711,
                                        pd.Timestamp('2008-10-23 05:53:05'), 1],
                                       [39.98419952392578, 116.31932067871094,
                                        pd.Timestamp('2008-10-23 05:53:06'), 1]])

    filter_out_values = filters.by_label(move_df, value=1, label_name='id', filter_out=True)
    assert_array_equal(filter_out_values, [[39.984222412109375, 116.31940460205078,
                                            pd.Timestamp('2008-10-23 05:53:11'), 2],
                                           [39.984222412109375, 116.31940460205078,
                                            pd.Timestamp('2008-10-23 05:53:11'), 2]])


def test_by_id():
    filter_values = filters.by_id(move_df, 1)
    assert_array_equal(filter_values, [[39.984092712402344, 116.3192367553711,
                                        pd.Timestamp('2008-10-23 05:53:05'), 1],
                                       [39.98419952392578, 116.31932067871094,
                                        pd.Timestamp('2008-10-23 05:53:06'), 1]])

def test_by_tid():
    filter_values = filters.by_tid(move_df, '12008102305')
    assert_array_equal(filter_values, [[39.984092712402344, 116.3192367553711,
                                        pd.Timestamp('2008-10-23 05:53:05'), 1, '12008102305'],
                                       [39.98419952392578, 116.31932067871094,
                                        pd.Timestamp('2008-10-23 05:53:06'), 1, '12008102305']])
    move_df.drop('tid', axis=1, inplace=True)


def test_outliers():
    df_move = MoveDataFrame(data=list_data_test, latitude="latp", longitude="lon", datetime="datetime")
    outliers = filters.outliers(move_data=df_move, jump_coefficient=1)
    assert_array_equal(outliers, [[1, 39.98421096801758, 116.31938934326172,
                                   pd.Timestamp('2008-10-23 05:53:16'), 1.6286216204832726,
                                   2.4484945931533275, 1.2242472060393084]])
    not_outliers = filters.outliers(move_data=df_move, jump_coefficient=1, filter_out=True)
    assert_array_equal(not_outliers.values.astype(str), [['1', '39.984092712402344', '116.3192367553711',
                                        '2008-10-23 05:53:05', 'nan', '14.015318782639952', 'nan'],
                                       ['1', '39.98419952392578', '116.31932067871094',
                                        '2008-10-23 05:53:06', '14.015318782639952',
                                        '7.345483960534693', '20.082061827224607'],
                                       ['1', '39.984222412109375', '116.31940460205078',
                                        '2008-10-23 05:53:11', '7.345483960534693',
                                        '1.6286216204832726', '5.929779944096936'],
                                       ['1', '39.98421859741211', '116.31941986083984',
                                        '2008-10-23 05:53:21', '2.4484945931533275', 'nan', 'nan']])
    df_move.set_index('id', inplace=True)
    outliers = filters.outliers(move_data=df_move, jump_coefficient=1)
    assert(move_df.index.name is None )


def test_clean_duplicates():
    df_move = MoveDataFrame(data=list_data, latitude="lat", longitude="lon", datetime="datetime", traj_id="id")
    duplicates = filters.clean_duplicates(df_move)
    assert_array_equal(duplicates, [[39.984092712402344, 116.3192367553711,
                                 pd.Timestamp('2008-10-23 05:53:05'), 1],
                                [39.98419952392578, 116.31932067871094,
                                 pd.Timestamp('2008-10-23 05:53:06'), 1],
                                [39.984222412109375, 116.31940460205078,
                                 pd.Timestamp('2008-10-23 05:53:11'), 2]])
    duplicates = filters.clean_duplicates(duplicates, subset='id')
    assert_array_equal(duplicates, [[39.984092712402344, 116.3192367553711,
                                 pd.Timestamp('2008-10-23 05:53:05'), 1],
                                [39.984222412109375, 116.31940460205078,
                                 pd.Timestamp('2008-10-23 05:53:11'), 2]])
    indexes = filters.clean_duplicates(df_move, subset='id', inplace = True)
    assert_array_equal(df_move, [[39.984092712402344, 116.3192367553711,
                                pd.Timestamp('2008-10-23 05:53:05'), 1],
                               [39.984222412109375, 116.31940460205078,
                                pd.Timestamp('2008-10-23 05:53:11'), 2]])
    assert(indexes is None)


def test_clean_nan_values():
    move = MoveDataFrame(data=list_data, latitude="lat", longitude="lon", datetime="datetime", traj_id="id")
    move.loc[3, 'id'] = np.nan
    assert(move.len() == 4)
    filters.clean_nan_values(move, inplace=True)
    assert(move.len() == 3)


def test_clean_gps_jumps_by_distance():
    df_move = MoveDataFrame(data=list_data_test, latitude="latp", longitude="lon", datetime="datetime")
    filter_df = filters.clean_gps_jumps_by_distance(df_move, jump_coefficient=1, threshold=0.5)

    assert_array_equal(filter_df.values.astype(str), [['1', '39.984092712402344', '116.3192367553711',
                                                     '2008-10-23 05:53:05', 'nan', '14.015318782639952', 'nan'],
                                                    ['1', '39.98419952392578', '116.31932067871094',
                                                     '2008-10-23 05:53:06', '14.015318782639952',
                                                     '7.345483960534693', '20.082061827224607'],
                                                    ['1', '39.984222412109375', '116.31940460205078',
                                                     '2008-10-23 05:53:11', '7.345483960534693',
                                                     '1.6286216204832726', '5.929779944096936'],
                                                    ['1', '39.98421859741211', '116.31941986083984',
                                                     '2008-10-23 05:53:21', '2.4484945931533275', 'nan', 'nan']])

    filters.clean_gps_jumps_by_distance(df_move, jump_coefficient=1, threshold=0.5, inplace = True)

    assert_array_equal(df_move.values.astype(str), [['1', '39.984092712402344', '116.3192367553711',
                                                     '2008-10-23 05:53:05', 'nan', '14.015318782639952', 'nan'],
                                                    ['1', '39.98419952392578', '116.31932067871094',
                                                     '2008-10-23 05:53:06', '14.015318782639952',
                                                     '7.345483960534693', '20.082061827224607'],
                                                    ['1', '39.984222412109375', '116.31940460205078',
                                                     '2008-10-23 05:53:11', '7.345483960534693',
                                                     '1.6286216204832726', '5.929779944096936'],
                                                    ['1', '39.98421859741211', '116.31941986083984',
                                                     '2008-10-23 05:53:21', '2.4484945931533275', 'nan', 'nan']])


def test_clean_gps_nearby_points_by_distances():
    df_move = MoveDataFrame(data=list_data_test, latitude="lat", longitude="lon", datetime="datetime", traj_id="id")

    filter_df = filters.clean_gps_nearby_points_by_distances(df_move, radius_area=10.0)
    assert_array_equal(filter_df.values.astype(str), [['1', '39.984092712402344', '116.3192367553711',
                                                     '2008-10-23 05:53:05', 'nan', '14.015318782639952', 'nan'],
                                                    ['1', '39.98419952392578', '116.31932067871094',
                                                     '2008-10-23 05:53:06', '14.015318782639952',
                                                     '7.345483960534693', '20.082061827224607']])

    filters.clean_gps_nearby_points_by_distances(df_move, radius_area=10.0, inplace = True)
    assert_array_equal(df_move.values.astype(str), [['1', '39.984092712402344', '116.3192367553711',
                                 '2008-10-23 05:53:05', 'nan', '14.015318782639952', 'nan'],
                                 ['1', '39.98419952392578', '116.31932067871094',
                                 '2008-10-23 05:53:06', '14.015318782639952',
                                 '7.345483960534693', '20.082061827224607']])


def test_clean_gps_nearby_points_by_speed():
    df_move = MoveDataFrame(data=list_data_test, latitude="lat", longitude="lon", datetime="datetime", traj_id="id")
    filter_df = filters.clean_gps_nearby_points_by_speed(df_move, speed_radius=20.0)
    assert_array_equal(filter_df.values.astype(str), [['1', '39.984092712402344', '116.3192367553711',
                                                     '2008-10-23 05:53:05', 'nan', 'nan', 'nan']])

    filters.clean_gps_nearby_points_by_speed(df_move, speed_radius=20.0, inplace = True)
    assert_array_equal(df_move.values.astype(str), [['1', '39.984092712402344', '116.3192367553711',
                                                     '2008-10-23 05:53:05', 'nan', 'nan', 'nan']])


def test_clean_gps_speed_max_radius():
    df_move = MoveDataFrame(data=list_data_test, latitude="lat", longitude="lon", datetime="datetime", traj_id="id")
    filter_df = filters.clean_gps_speed_max_radius(df_move, speed_max=0.33)
    assert_array_equal(filter_df.values.astype(str), [['1', '39.984092712402344', '116.3192367553711',
                                                    '2008-10-23 05:53:05', 'nan', 'nan', 'nan'],
                                                   ['1', '39.98421096801758', '116.31938934326172',
                                                    '2008-10-23 05:53:16', '1.6286216204832726', '5.0',
                                                    '0.3257243240966545']])

    filters.clean_gps_speed_max_radius(df_move, speed_max=0.33, inplace = True)
    assert_array_equal(df_move.values.astype(str), [['1', '39.984092712402344', '116.3192367553711',
                                                     '2008-10-23 05:53:05', 'nan', 'nan', 'nan'],
                                                    ['1', '39.98421096801758', '116.31938934326172',
                                                     '2008-10-23 05:53:16', '1.6286216204832726', '5.0',
                                                     '0.3257243240966545']])

def test_clean_trajectories_with_few_points():
    df_move = MoveDataFrame(data=list_data_test, latitude="lat", longitude="lon", datetime="datetime", traj_id="id")
    filter_df = filters.clean_trajectories_with_few_points(df_move, min_points_per_trajectory=6)
    assert(filter_df.len() == 0)

    filters.clean_trajectories_with_few_points(df_move, min_points_per_trajectory=6, inplace = True)
    assert (df_move.len() == 0)


def test_clean_id_by_time_max():
    df_move = MoveDataFrame(data=list_data, latitude="lat", longitude="lon", datetime="datetime", traj_id="id")
    df_move.generate_dist_time_speed_features()
    indexes = filters.clean_id_by_time_max(df_move, time_max =1.0)
    assert_array_equal(indexes['datetime'].astype(str), ['2008-10-23 05:53:05', '2008-10-23 05:53:06'])

