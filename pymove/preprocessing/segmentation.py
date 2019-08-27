# TODO: Arina
import numpy as np
import pandas as pd
import time

from pymove import utils as ut
from pymove import gridutils


def bbox_split(bbox, number_grids):
    """
        split bound box in N grids of the same size
    """
    lat_min = bbox[0]
    lon_min = bbox[1]
    lat_max = bbox[2]
    lon_max = bbox[3]
    
    const_lat =  abs(abs(lat_max) - abs(lat_min))/number_grids
    const_lon =  abs(abs(lon_max) - abs(lon_min))/number_grids
    print('const_lat: {}\nconst_lon: {}'.format(const_lat, const_lon))

    df = pd.DataFrame(columns=['lat_min', 'lon_min', 'lat_max', 'lon_max'])
    for i in range(number_grids):
        df = df.append({'lat_min':lat_min, 'lon_min': lon_min + (const_lon * i), 'lat_max': lat_max, 'lon_max':lon_min + (const_lon * (i + 1))}, ignore_index=True)
    return df

""" segment trajectory based on threshold for each ID object"""

def segment_trajectory_by_dist_time_speed(df_, label_id=dic_labels['id'], max_dist_between_adj_points=3000, max_time_between_adj_points=7200,
                      max_speed_between_adj_points=50.0, drop_single_points=True, label_new_tid='tid_part'):
    """
    index_name is the current id.
    label_new_id is the new splitted id.
    time, dist, speeed features must be updated after split.
    """
        
    print('\nSplit trajectories')
    print('...max_time_between_adj_points:', max_time_between_adj_points)
    print('...max_dist_between_adj_points:', max_dist_between_adj_points)
    print('...max_speed:', max_speed_between_adj_points)
    
    try:
        if df_.index.name is None:
            print('...setting {} as index'.format(label_id))
            df_.set_index(label_id, inplace=True)

        curr_tid = 0
        if label_new_tid not in df_:
            df_[label_new_tid] = curr_tid

        ids = df_.index.unique()
        count = 0
        df_size = df_.shape[0]
        curr_perc_int = -1
        start_time = time.time()

        for idx in ids:
            curr_tid += 1
            
            filter_ = (df_.at[idx, dic_features_label['time_to_prev']] > max_time_between_adj_points) | \
                        (df_.at[idx, dic_features_label['dist_to_prev']] > max_dist_between_adj_points) | \
                        (df_.at[idx, dic_features_label['speed_to_prev']] > max_speed_between_adj_points)        

            """ check if object have only one point to be removed """
            if filter_.shape == ():
                # trajectories with only one point is useless for interpolation and so they must be removed.
                count += 1
                df_.at[idx, label_new_tid] = -1
                curr_tid += -1
            else:
                tids = np.empty(filter_.shape[0], dtype=np.int64)
                tids.fill(curr_tid)
                for i, has_problem in enumerate(filter_):
                    if has_problem:
                        curr_tid += 1
                        tids[i:] = curr_tid
                count += tids.shape[0]
                df_.at[idx, label_new_tid] = tids
            
            curr_perc_int, est_time_str = ut.progress_update(count, df_size, start_time, curr_perc_int, step_perc=20)

        if label_id == label_new_tid:
            df_.reset_index(drop=True, inplace=True)
            print('... label_id = label_new_id, then reseting and drop index')
        else:
            df_.reset_index(inplace=True)
            print('... Reseting index\n')
        
        if drop_single_points:
            shape_before_drop = df_.shape
            idx = df_[ df_[label_new_tid] == -1 ].index
            if idx.shape[0] > 0:
                print('...Drop Trajectory with a unique GPS point\n')
                ids_before_drop = df_[label_id].unique().shape[0]
                df_.drop(index=idx, inplace=True)
                print('...Object - before drop: {} - after drop: {}'.format(ids_before_drop, df_[label_id].unique().shape[0]))
                print('...Shape - before drop: {} - after drop: {}'.format(shape_before_drop, df_.shape))
                create_update_dist_time_speed_features(df_, label_id, dic_labels)
            else:
                print('...No trajs with only one point.', df_.shape)

    except Exception as e:
        raise e

def segment_trajectory_by_speed(df_, label_id=dic_labels['id'], max_speed_between_adj_points=50.0, drop_single_points=True, label_new_tid='tid_speed'):
    """ Index_name is the current id.
    label_new_id is the new splitted id.
    Speed features must be updated after split.
    """     
    print('\nSplit trajectories by max_speed_between_adj_points:', max_speed_between_adj_points) 
    try:
        if df_.index.name is None:
            print('...setting {} as index'.format(label_id))
            df_.set_index(label_id, inplace=True)

        curr_tid = 0
        if label_new_tid not in df_:
            df_[label_new_tid] = curr_tid

        ids = df_.index.unique()
        count = 0
        df_size = df_.shape[0]
        curr_perc_int = -1
        start_time = time.time()

        for idx in ids:            
            """ increment index to trajectory"""
            curr_tid += 1

            """ filter speed max"""
            speed = (df_.at[idx, dic_features_label['speed_to_prev']] > max_speed_between_adj_points)        
                     
            """ check if object have only one point to be removed """
            if speed.shape == ():
                count += 1
                df_.at[idx, label_new_tid] = -1 # set object  = -1 to remove ahead
                curr_tid += -1
            else: 
                tids = np.empty(speed.shape[0], dtype=np.int64)
                tids.fill(curr_tid)
                for i, has_problem in enumerate(speed):
                    if has_problem:
                        curr_tid += 1
                        tids[i:] = curr_tid
                count += tids.shape[0]
                df_.at[idx, label_new_tid] = tids

            curr_perc_int, est_time_str = ut.progress_update(count, df_size, start_time, curr_perc_int, step_perc=20)

        if label_id == label_new_tid:
            df_.reset_index(drop=True, inplace=True)
            print('... label_id = label_new_id, then reseting and drop index')
        else:
            df_.reset_index(inplace=True)
            print('... Reseting index\n')
       
        if drop_single_points:
            shape_before_drop = df_.shape
            idx = df_[df_[label_new_tid] == -1].index
            if idx.shape[0] > 0:
                print('...Drop Trajectory with a unique GPS point\n')
                ids_before_drop = df_[label_id].unique().shape[0]
                df_.drop(index=idx, inplace=True)
                print('...Object - before drop: {} - after drop: {}'.format(ids_before_drop, df_[label_id].unique().shape[0]))
                print('...Shape - before drop: {} - after drop: {}'.format(shape_before_drop, df_.shape))
            else:
                print('...No trajs with only one point.', df_.shape)
    except Exception as e:
        raise e

def segment_trajectory_by_time(df_, label_id=dic_labels['id'], max_time_between_adj_points=900.0, drop_single_points=True, label_new_tid='tid_time'):
    """
    index_name is the current id.
    label_new_id is the new splitted id.
    Speed features must be updated after split.
    """     
    print('\nSplit trajectories by max_time_between_adj_points:', max_time_between_adj_points) 
    try:
        if df_.index.name is None:
            print('...setting {} as index'.format(label_id))
            df_.set_index(label_id, inplace=True)

        curr_tid = 0
        if label_new_tid not in df_:
            df_[label_new_tid] = curr_tid

        ids = df_.index.unique()
        count = 0
        df_size = df_.shape[0]
        curr_perc_int = -1
        start_time = time.time()

        for idx in ids:            
            """ increment index to trajectory"""
            curr_tid += 1

            """ filter time max"""
            times = (df_.at[idx, dic_features_label['time_to_prev']] > max_time_between_adj_points)        
                     
            """ check if object have only one point to be removed """
            if times.shape == ():
                count += 1
                df_.at[idx, label_new_tid] = -1 # set object  = -1 to remove ahead
                curr_tid += -1
            else: 
                tids = np.empty(times.shape[0], dtype=np.int64)
                tids.fill(curr_tid)
                for i, has_problem in enumerate(times):
                    if has_problem:
                        curr_tid += 1
                        tids[i:] = curr_tid
                count += tids.shape[0]
                df_.at[idx, label_new_tid] = tids

            curr_perc_int, est_time_str = ut.progress_update(count, df_size, start_time, curr_perc_int, step_perc=20)

        if label_id == label_new_tid:
            df_.reset_index(drop=True, inplace=True)
            print('... label_id = label_new_id, then reseting and drop index')
        else:
            df_.reset_index(inplace=True)
            print('... Reseting index\n')
       
        if drop_single_points:
            shape_before_drop = df_.shape
            idx = df_[ df_[label_new_tid] == -1 ].index
            if idx.shape[0] > 0:
                print('...Drop Trajectory with a unique GPS point\n')
                ids_before_drop = df_[label_id].unique().shape[0]
                df_.drop(index=idx, inplace=True)
                print('...Object - before drop: {} - after drop: {}'.format(ids_before_drop, df_[label_id].unique().shape[0]))
                print('...Shape - before drop: {} - after drop: {}'.format(shape_before_drop, df_.shape))
                create_update_dist_time_speed_features(df_, label_id, dic_labels)
            else:
                print('...No trajs with only one point.', df_.shape)
    except Exception as e:
        raise e


