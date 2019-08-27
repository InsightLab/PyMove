# TODO: Arina
import numpy as np
import pandas as pd
import time
from scipy.interpolate import interp1d

from pymove import utils as ut
from pymove import gridutils

""" Fuction to solve problems after Map-matching"""
def check_time_dist(df, index_name='tid', tids=None, max_dist_between_adj_points=5000, max_time_between_adj_points=900, max_speed=30):
    try:
        if df.index.name is not None:
            print('reseting index...')
            df.reset_index(inplace=True)
        
        if tids is None:
            tids = df[index_name].unique()
        
        size = df.shape[0]
        if df.index.name is None:
            print('creating index...')
            df.set_index(index_name, inplace=True)        
        
        count = 0
        curr_perc_int = -1
        start_time = time.time()
        size_id = 0
        print('checking ascending distance and time...')
        for tid in tids:
            filter_ = (df.at[tid,'isNode'] != 1)

            # be sure that distances are in ascending order
            dists = df.at[tid, 'distFromTrajStartToCurrPoint'][filter_]
            assert np.all(dists[:-1] < dists[1:]), 'distance feature is not in ascending order'
            
            # be sure that times are in ascending order
            times = df.at[tid, 'time'][filter_].astype(np.float64)
            assert np.all(times[:-1] < times[1:]), 'time feature is not in ascending order'
            
            size_id = 1 if filter_.shape == () else filter_.shape[0]
            count += size_id
            curr_perc_int, est_time_str = ut.progress_update(count, size, start_time, curr_perc_int, step_perc=20)
            

        count = 0
        curr_perc_int = -1
        start_time = time.time()
        size_id = 0
        print('checking delta_times, delta_dists and speeds...')
        for tid in tids:
            filter_ = (df.at[tid,'isNode'] != 1)

            dists = df.at[tid, 'distFromTrajStartToCurrPoint'][filter_]
            delta_dists = (ut.shift(dists, -1) - dists)[:-1]   # do not use last element (np.nan)
                          
            assert np.all(delta_dists <= max_dist_between_adj_points), \
                          'delta_dists must be <= {}'.format(max_dist_between_adj_points)
            
            times = df.at[tid, 'time'][filter_].astype(np.float64)
            delta_times = ((ut.shift(times, -1) - times) / 1000.0)[:-1] # do not use last element (np.nan)
                          
            assert np.all(delta_times <= max_time_between_adj_points), \
                          'delta_times must be <= {}'.format(max_time_between_adj_points)
            
            assert np.all(delta_times > 0), 'delta_times must be > 0'

            assert np.all(delta_dists > 0), 'delta_dists must be > 0'
            
            speeds = delta_dists / delta_times
            assert np.all(speeds <= max_speed), 'speeds > {}'.format(max_speed)
            
            size_id = 1 if filter_.shape == () else filter_.shape[0]
            count += size_id
            curr_perc_int, est_time_str = ut.progress_update(count, size, start_time, curr_perc_int, step_perc=20)
            

        df.reset_index(inplace=True)
    
    except Exception as e:
        print('{}: {} - size: {}'.format(index_name, tid, size_id))
        raise e
        
def fix_time_not_in_ascending_order_id(df, tid, index_name='tid'):
    if 'deleted' not in df:
        df['deleted'] = False
        
    if df.index.name is None:
        print('creating index...')
        df.set_index(index_name, inplace=True)
    
    filter_ = (df.at[tid,'isNode'] != 1) & (~df.at[tid,'deleted'])
    
    # be sure that distances are in ascending order
    dists = df.at[tid, 'distFromTrajStartToCurrPoint'][filter_]
    assert np.all(dists[:-1] <= dists[1:]), 'distance feature is not in ascending order'
    
    if filter_.shape == ():     # do not use trajectories with only 1 point.
        size_id = 1
        df.at[tid, 'deleted'] = True
    else:
        size_id = filter_.shape[0]
        times = df.at[tid, 'time'][filter_]
        idx_not_in_ascending_order = np.where(times[:-1] >= times[1:])[0] + 1

        if idx_not_in_ascending_order.shape[0] > 0:
            #print(tid, 'idx_not_in_ascending_order:', idx_not_in_ascending_order, 'times.shape', times.shape)

            ut.change_df_feature_values_using_filter_and_indexes(df, tid, 'deleted', filter_, idx_not_in_ascending_order, True)
            # equivalent of: df.at[tid, 'deleted'][filter_][idx_not_in_ascending_order] = True

            fix_time_not_in_ascending_order_id(df, tid, index_name=index_name)
    
    return size_id        
        
def fix_time_not_in_ascending_order_all(df, index_name='tid', drop_marked_to_delete=False):
    try:
        if df.index.name is not None:
            print('reseting index...')
            df.reset_index(inplace=True)
        
        print('dropping duplicate distances... shape before:', df.shape)
        df.drop_duplicates(subset=[index_name, 'isNode', 'distFromTrajStartToCurrPoint'], keep='first', inplace=True)
        print('shape after:', df.shape)
        
        print('sorting by id and distance...')
        df.sort_values(by=[index_name, 'distFromTrajStartToCurrPoint'], inplace=True)
        print('sorting done')
        
        tids = df[index_name].unique()
        df['deleted'] = False

        print('starting fix...')
        size = df.shape[0]
        count = 0
        curr_perc_int = -1
        start_time = time.time()
        for tid in tids:
            size_id = fix_time_not_in_ascending_order_id(df, tid, index_name)

            count += size_id
            curr_perc_int, est_time_str = ut.progress_update(count, size, start_time, curr_perc_int, step_perc=20)

        df.reset_index(inplace=True)
        idxs = df[ df['deleted'] ].index
        print('{} rows marked for deletion.'.format(idxs.shape[0]))

        if idxs.shape[0] > 0 and drop_marked_to_delete:
            print('shape before dropping: {}'.format(df.shape))
            df.drop(index=idxs, inplace=True )
            df.drop(labels='deleted', axis=1, inplace=True)
            print('shape after dropping: {}'.format(df.shape))
    
    except Exception as e:
        print('{}: {} - size: {}'.format(index_name, tid, size_id))
        raise e
       
def interpolate_add_deltatime_speed_features(df, label_id='tid', max_time_between_adj_points=900, 
                                             max_dist_between_adj_points=5000, max_speed=30):
    """
    interpolate distances (x) to find times (y).
    max_time_between_adj_points, max_dist_between_adj_points and max_speed are used only for verification.
    """
    if df.index.name is not None:
        print('reseting index...')
        df.reset_index(inplace=True)

    tids = df[label_id].unique()
    #tids = [2]

    if df.index.name is None:
        print('creating index...')
        df.set_index(label_id, inplace=True)

    drop_trajectories = []    
    size = df.shape[0]
    count = 0
    curr_perc_int = -1
    start_time = time.time()

    df['delta_time'] = np.nan
    df['speed'] = np.nan

    try:
        for tid in tids:
            filter_nodes = (df.at[tid,'isNode'] == 1)
            times = df.at[tid, 'time'][filter_nodes]
            size_id = 1 if filter_nodes.shape == () else filter_nodes.shape[0]
            count += size_id

            # y - time of snapped points
            y_ = df.at[tid,'time'][~filter_nodes]
            if y_.shape[0] < 2:
                #print('traj: {} - insuficient points ({}) for interpolation. adding to drop list...'.format(tid,  y_.shape[0]))
                drop_trajectories.append(tid)
                curr_perc_int, est_time_str = ut.progress_update(count, size, start_time, curr_perc_int, step_perc=20)
                continue

            assert np.all(y_[1:] >= y_[:-1]), 'time feature is not in ascending order'

            # x - distance from traj start to snapped points
            x_ = df.at[tid, 'distFromTrajStartToCurrPoint'][~filter_nodes]

            assert np.all(x_[1:] >= x_[:-1]), 'distance feature is not in ascending order'

            # remove duplicates in distances to avoid np.inf in future interpolation results
            idx_duplicates = np.where(x_[1:] == x_[:-1])[0]
            if idx_duplicates.shape[0] > 0:
                x_ = np.delete(x_, idx_duplicates)
                y_ = np.delete(y_, idx_duplicates)

            if y_.shape[0] < 2:
                #print('traj: {} - insuficient points ({}) for interpolation. adding to drop list...'.format(tid,  y_.shape[0]))
                drop_trajectories.append(tid)
                curr_perc_int, est_time_str = ut.progress_update(count, size, start_time, curr_perc_int, step_perc=20)
                continue

            # compute delta_time and distance between points
            #values = (ut.shift(df.at[tid, 'time'][filter_nodes].astype(np.float64), -1) - df.at[tid, 'time'][filter_nodes]) / 1000
            #ut.change_df_feature_values_using_filter(df, tid, 'delta_time', filter_nodes, values)
            delta_time = ( (ut.shift(y_.astype(np.float64), -1) - y_) / 1000.0 )[:-1]
            dist_curr_to_next = (ut.shift(x_, -1) - x_)[:-1]
            speed = (dist_curr_to_next / delta_time)[:-1]

            assert np.all(delta_time <= max_time_between_adj_points), 'delta_time between points cannot be more than {}'.format(max_time_between_adj_points)
            assert np.all(dist_curr_to_next <= max_dist_between_adj_points), 'distance between points cannot be more than {}'.format(max_dist_between_adj_points)
            assert np.all(speed <= max_speed), 'speed between points cannot be more than {}'.format(max_speed)   

            assert np.all(x_[1:] >= x_[:-1]), 'distance feature is not in ascending order'

            f_intp = interp1d(x_, y_, fill_value='extrapolate')

            x2_ = df.at[tid, 'distFromTrajStartToCurrPoint'][filter_nodes]
            assert np.all(x2_[1:] >= x2_[:-1]), 'distances in nodes are not in ascending order'

            intp_result = f_intp(x2_) #.astype(np.int64)
            assert np.all(intp_result[1:] >= intp_result[:-1]), 'resulting times are not in ascending order'

            assert ~np.isin(np.inf, intp_result), 'interpolation results with np.inf value(s)'

            # update time features for nodes. initially they are empty.
            values = intp_result.astype(np.int64)
            ut.change_df_feature_values_using_filter(df, tid, 'time', filter_nodes, values)

            # create delta_time feature
            values = (ut.shift(df.at[tid, 'time'][filter_nodes].astype(np.float64), -1) - df.at[tid, 'time'][filter_nodes]) / 1000
            ut.change_df_feature_values_using_filter(df, tid, 'delta_time', filter_nodes, values)

            # create speed feature
            values = df.at[tid, 'edgeDistance'][filter_nodes] / df.at[tid, 'delta_time'][filter_nodes]
            ut.change_df_feature_values_using_filter(df, tid, 'speed', filter_nodes, values)

            curr_perc_int, est_time_str = ut.progress_update(count, size, start_time, curr_perc_int, step_perc=20)
            
    except Exception as e:
        print('{}: {} - size: {} - count: {}'.format(label_id, tid, size_id, count))
        raise e
        
    print(count, size)
    print('we still need to drop {} trajectories with only 1 gps point'.format(len(drop_trajectories)))
    df.reset_index(inplace=True)
    idxs_drop = df[ df[label_id].isin(drop_trajectories) ].index.values
    print('dropping {} rows in {} trajectories with only 1 gps point'.format(idxs_drop.shape[0], 
            len(drop_trajectories)))
    if idxs_drop.shape[0] > 0:
        print('shape before dropping: {}'.format(df.shape))
        df.drop(index=idxs_drop, inplace=True )
        print('shape after dropping: {}'.format(df.shape))
