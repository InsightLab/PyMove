# TODO: Arina
import numpy as np
import pandas as pd
import time
from scipy.interpolate import interp1d

from pymove import utils as ut
from pymove import gridutils

def filter_bbox(df_, bbox, filter_out=False, dic_labels=dic_labels, inplace=False):
    """
    Filter bounding box.
    Example: 
        filter_bbox(df_, [-3.90, -38.67, -3.68, -38.38]) -> Fortaleza
            lat_down =  bbox[0], lon_left =  bbox[1], lat_up = bbox[2], lon_right = bbox[3]
    """
    try:
        filter_ = (df_[dic_labels['lat']] >=  bbox[0]) & (df_[dic_labels['lat']] <= bbox[2]) & (df_[dic_labels['lon']] >= bbox[1]) & (df_[dic_labels['lon']] <= bbox[3])
        if filter_out:
            filter_ = ~filter_

        if inplace:
            df_.drop( index=df_[ ~filter_ ].index, inplace=True )
            return df_
        else:
            return df_.loc[ filter_ ]
    except Exception as e: 
            raise e

def filter_by_datetime(df_, startDatetime=None, endDatetime=None, dic_labels=dic_labels, filter_out=False):
    
    try:
        if startDatetime is not None and endDatetime is not None:
            filter_ = (df_[dic_labels['datetime']] > startDatetime) & (df_[dic_labels['datetime']] <= endDatetime)
        elif endDatetime is not None:
            filter_ = (df_[dic_labels['datetime']] <= endDatetime)
        else:
            filter_ = (df_[dic_labels['datetime']] > startDatetime)
        
        if filter_out:
            filter_ = ~filter_
        
        return df_[filter_]

    except Exception as e:
        raise e

def filter_by_label(df_, value, label_name, filter_out=False):
    try:
        filter_ = (df_[label_name] == value)

        if filter_out:
            filter_ = ~filter_
        
        return df_[filter_]
    
    except Exception as e:
        raise e

def filter_by_id(df_, id_=None, label_id=dic_labels['id'], filter_out=False):
    """
        filter dataset from id
    """
    return filter_by_label(df_, id_, label_id, filter_out)

def filter_by_tid(df_, tid_=None, label_tid=dic_features_label['tid'], filter_out=False):
    """
        filter dataset from id
    """
    return filter_by_label(df_, tid_, label_tid, filter_out)

def filter_jumps(df_, jump_coefficient=3.0, threshold = 1, filter_out=False):
    
    if df_.index.name is not None:
        print('...Reset index for filtering\n')
        df_.reset_index(inplace=True)
    
    if dic_features_label['dist_to_prev'] in df_ and dic_features_label['dist_to_next'] and dic_features_label['dist_prev_to_next'] in df_:
        filter_ = (df_[dic_features_label['dist_to_next']] > threshold) & (df_[dic_features_label['dist_to_prev']] > threshold) & (df_[dic_features_label['dist_prev_to_next']] > threshold) & \
        (jump_coefficient * df_[dic_features_label['dist_prev_to_next']] < df_[dic_features_label['dist_to_next']]) & \
        (jump_coefficient * df_[dic_features_label['dist_prev_to_next']] < df_[dic_features_label['dist_to_prev']])  

        if filter_out:
            filter_ = ~filter_

        print('...Filtring jumps \n')
        return df_[filter_]
    
    else:
        print('...Distances features were not created')
        return df_

"""----------------------  FUCTIONS TO DATA CLEANING   ----------------------------------- """ 

def clean_duplicates(df_, subset=None, keep='first', inplace=False, sort=True, return_idx=False):
    """
    Return DataFrame with duplicate rows removed, optionally only considering certain columns.
    """
    print('\nRemove rows duplicates by subset')
    if sort is True:
        print('...Sorting by {} and {} to increase performance\n'.format(dic_labels['id'], dic_labels['datetime']))
        df_.sort_values([dic_labels['id'], dic_labels['datetime']], inplace=True)
    
    idx = df_.duplicated(subset=subset )
    tam_drop = df_[idx].shape[0] 

    if tam_drop > 0:
        df_.drop_duplicates(subset, keep, inplace)
        print('...There are {} GPS points duplicated'.format(tam_drop))
    else:
        print('...There are no GPS points duplicated')

    if return_idx:
        return return_idx

def clean_consecutive_duplicates(df, subset=None, keep='first', inplace=False):
    if keep == 'first':
        n = 1
    else:
        n = -1
        
    if subset is None:
        filter_ = (df.shift(n) != df).any(axis=1)
    else:
        filter_ = (df[subset].shift(n) != df[subset]).any(axis=1)

    if inplace:
        df.drop( index=df[~filter_].index, inplace=True )
        return df
    else:
        return df.loc[ filter_ ]

def clean_NaN_values(df_, axis=0, how='any', thresh=None, subset=None, inplace=True):
    #df.isna().sum()
    df_.dropna(axis=axis, how=how, thresh=thresh, subset=None, inplace=inplace)
         
def clean_gps_jumps_by_distance(df_, label_id=dic_labels['id'], jump_coefficient=3.0, threshold = 1, dic_labels=dic_labels, label_dtype=np.float64, sum_drop=0):

    create_update_dist_features(df_, label_id, dic_labels, label_dtype=label_dtype)

    try:
        print('\nCleaning gps jumps by distance to jump_coefficient {}...\n'.format(jump_coefficient))
        df_jumps = filter_jumps(df_, jump_coefficient, threshold)
        rows_to_drop = df_jumps.shape[0]

        if rows_to_drop > 0:
            print('...Dropping {} rows of gps points\n'.format(rows_to_drop))
            shape_before = df_.shape[0]
            df_.drop(index=df_jumps.index, inplace=True)
            sum_drop = sum_drop + rows_to_drop
            print('...Rows before: {}, Rows after:{}, Sum drop:{}\n'.format(shape_before, df_.shape[0], sum_drop))
            clean_gps_jumps_by_distance(df_, label_id, jump_coefficient, threshold, dic_labels, label_dtype, sum_drop)  
        else:
            print('{} GPS points were dropped'.format(sum_drop))    

    except Exception as e:
       raise e

def clean_gps_nearby_points_by_distances(df_, label_id=dic_labels['id'], dic_labels=dic_labels, radius_area=10.0, label_dtype=np.float64):

    create_update_dist_features(df_, label_id, dic_labels, label_dtype)
    try:
        print('\nCleaning gps points from radius of {} meters\n'.format(radius_area))
        if df_.index.name is not None:
            print('...Reset index for filtering\n')
            df_.reset_index(inplace=True)
    
        if dic_features_label['dist_to_prev'] in df_:
            filter_nearby_points = (df_[dic_features_label['dist_to_prev']] <= radius_area)

            idx = df_[filter_nearby_points].index
            print('...There are {} gps points to drop\n'.format(idx.shape[0]))
            if idx.shape[0] > 0:
                print('...Dropping {} gps points\n'.format(idx.shape[0]))
                shape_before = df_.shape[0]
                df_.drop(index=idx, inplace=True)
                print('...Rows before: {}, Rows after:{}\n'.format(shape_before, df_.shape[0]))
                clean_gps_nearby_points_by_distances(df_, label_id, dic_labels, radius_area, label_dtype)
        else:
            print('...{} is not in the dataframe'.format(dic_features_label['dist_to_prev']))
    except Exception as e:
       raise e

def clean_gps_nearby_points_by_speed(df_, label_id=dic_labels['id'], dic_labels=dic_labels, speed_radius=0.0, label_dtype=np.float64):

    create_update_dist_time_speed_features(df_, label_id, dic_labels, label_dtype)
    try:
        print('\nCleaning gps points using {} speed radius\n'.format(speed_radius))
        if df_.index.name is not None:
            print('...Reset index for filtering\n')
            df_.reset_index(inplace=True)
    
        if dic_features_label['speed_to_prev'] in df_:
            filter_nearby_points = (df_[dic_features_label['speed_to_prev']] <= speed_radius)

            idx = df_[filter_nearby_points].index
            print('...There are {} gps points to drop\n'.format(idx.shape[0]))
            if idx.shape[0] > 0:
                print('...Dropping {} gps points\n'.format(idx.shape[0]))
                shape_before = df_.shape[0]
                df_.drop(index=idx, inplace=True)
                print('...Rows before: {}, Rows after:{}\n'.format(shape_before, df_.shape[0]))
                clean_gps_nearby_points_by_speed(df_, label_id, dic_labels, speed_radius, label_dtype)
        else:
            print('...{} is not in the dataframe'.format(dic_features_label['dist_to_prev']))
    except Exception as e:
       raise e

def clean_gps_speed_max_radius(df_, label_id=dic_labels['id'], dic_labels=dic_labels, speed_max=50.0, label_dtype=np.float64):

    create_update_dist_time_speed_features(df_, label_id, dic_labels=dic_labels, label_dtype=label_dtype)

    print('\nClean gps points with speed max > {} meters by seconds'.format(speed_max))

    if dic_features_label['speed_to_prev'] in df_:
        filter_ = (df_[dic_features_label['speed_to_prev']] > speed_max) | (df_[dic_features_label['speed_to_next']] > speed_max)
    
        idx = df_[filter_].index
    
        print('...There {} gps points with speed_max > {}\n'.format(idx.shape[0], speed_max))
        if idx.shape[0] > 0:
            print('...Dropping {} rows of jumps by speed max\n'.format(idx.shape[0]))
            shape_before = df_.shape[0]
            df_.drop(index=idx, inplace=True)
            print('...Rows before: {}, Rows after:{}\n'.format(shape_before, df_.shape[0]))
            clean_gps_speed_max_radius(df_, label_id, dic_labels, speed_max, label_dtype)

def clean_trajectories_with_few_points(df_, label_tid=dic_features_label['tid'], dic_labels=dic_labels, min_points_per_trajectory=2, label_dtype=np.float64):
    if df_.index.name is not None:
        print('\n...Reset index for filtering\n')
        df_.reset_index(inplace=True)

    df_count_tid = df_.groupby(by= label_tid).size()
    
    tids_with_few_points = df_count_tid[ df_count_tid < min_points_per_trajectory ].index
    
    print('\n...There are {} ids with few points'.format(tids_with_few_points.shape[0])) 
    shape_before_drop = df_.shape
    idx = df_[ df_[label_tid].isin(tids_with_few_points) ].index
    if idx.shape[0] > 0:
        print('\n...Tids before drop: {}'.format(df_[label_tid].unique().shape[0]))
        df_.drop(index=idx, inplace=True)
        print('\n...Tids after drop: {}'.format(df_[label_tid].unique().shape[0]))
        print('\n...Shape - before drop: {} - after drop: {}'.format(shape_before_drop, df_.shape))
        create_update_dist_time_speed_features(df_, label_tid, dic_labels, label_dtype)      

def clean_trajectories_short_and_few_points_(df_,  label_id=dic_features_label['tid'], dic_labels=dic_labels, min_trajectory_distance=100, min_points_per_trajectory=2, label_dtype=np.float64):
    # remove_tids_with_few_points must be performed before updating features, because 
    # those features only can be computed with at least 2 points per trajactories
    print('\nRemove short trajectories...')
    clean_trajectories_with_few_points(df_, label_id, dic_labels, min_points_per_trajectory, label_dtype)
    
    create_update_dist_time_speed_features(df_, label_id, dic_labels, label_dtype)

    if df_.index.name is not None:
        print('reseting index')
        df_.reset_index(inplace=True)
        
    print('\n...Dropping unnecessary trajectories...')
    df_agg_tid = df_.groupby(by=label_id).agg({dic_features_label['dist_to_prev']:'sum'})

    filter_ = (df_agg_tid[dic_features_label['dist_to_prev']] < min_trajectory_distance)    
    tid_selection = df_agg_tid[ filter_ ].index
    print('\n...short trajectories and trajectories with a minimum distance ({}): {}'.format(df_agg_tid.shape[0], min_trajectory_distance))
    print('\n...There are {} tid do drop'.format(tid_selection.shape[0]))
    shape_before_drop = df_.shape
    idx = df_[ df_[label_id].isin(tid_selection) ].index
    if idx.shape[0] > 0:
        tids_before_drop = df_[label_id].unique().shape[0]
        df_.drop(index=idx, inplace=True)
        print('\n...Tids - before drop: {} - after drop: {}'.format(tids_before_drop, df_[label_id].unique().shape[0]))
        print('\n...Shape - before drop: {} - after drop: {}'.format(shape_before_drop, df_.shape))
        clean_trajectories_short_and_few_points_(df_, dic_labels, min_trajectory_distance, min_points_per_trajectory, label_dtype)    
