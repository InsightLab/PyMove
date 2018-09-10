from time import time

import numpy as np
from scipy.interpolate import interp1d

from utils import timeutils
from utils import numpyutils as npu
from utils import pandasutils as pdu
from utils import progressutils as pru



def filter_bbox(df, lat_down, lon_left, lat_up, lon_right, filter_out=False, inplace=False):
    """
    Filter bounding box.
    Example: 
    filter_bbox(df_, -3.90, -38.67, -3.68, -38.38) -> Fortaleza
    """
    filter_ = (df['lat'] >= lat_down) & (df['lat'] <= lat_up) & (df['lon'] >= lon_left) & (df['lon'] <= lon_right)
    
    if filter_out:
        filter_ = ~filter_

    if inplace:
        df.drop( index=df[ ~filter_ ].index, inplace=True )
        return df
    else:
        return df.loc[ filter_ ]


def remove_duplicates(df, subset=None, inplace=False):
    return df.drop_duplicates(subset, inplace)


def bbox(df):
    return (df['lat'].min(), df['lon'].min(), df['lat'].max(), df['lon'].max())


def lon2XSpherical(lon):
    """
    From Longitude to X EPSG:3857 WGS 84 / Pseudo-Mercator
    https://epsg.io/transform
    @param longitude in degrees
    @return X offset from your original position in meters.
    -38.501597 -> -4285978.17
    """
    return 6378137 * np.radians(lon)


def lat2YSpherical(lat):
    """
    From Latitude to Y EPSG:3857 WGS 84 / Pseudo-Mercator
    @param latitude in degrees
    @return Y offset from your original position in meters.
    -3.797864 -> -423086.22
    """
    return 6378137 * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2.0))


def x2LonSpherical(x):
    """
    From X to Longitude.
    -4285978.17 -> -38.501597
    """
    return np.degrees(x / 6378137.0)


def y2LatSpherical(y):
    """
    From Y to Longitude.
    -423086.22 -> -3.797864 
    """
    return np.degrees(np.arctan(np.sinh(y / 6378137.0)))


def haversine(lat1, lon1, lat2, lon2, to_radians=True, earth_radius=6371):
    """
    Vectorized haversine function.
    https://stackoverflow.com/questions/43577086/pandas-calculate-haversine-distance-within-each-group-of-rows
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees or in radians).
    All (lat, lon) coordinates must have numeric dtypes and be of equal length.
    Result in meters.
    """
    if to_radians:
        lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])

    a = np.sin((lat2-lat1)/2.0)**2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin((lon2-lon1)/2.0)**2

    return earth_radius * 2 * np.arcsin(np.sqrt(a)) * 1000  # result in meters (* 1000)


def create_update_distance_features(df_, index_name='id'):
    """
    for better performance the trajectory id must be the index.
    """
    print('create_update_distance_features')
    try:
        if df_.index.name is None:
            df_.set_index(index_name, inplace=True)
        
        if 'dist_curr_to_next' not in df_:
            df_['dist_curr_to_next'] = -1.0
        if 'dist_prev_to_curr' not in df_:
            df_['dist_prev_to_curr'] = -1.0
        if 'dist_prev_to_next' not in df_:
            df_['dist_prev_to_next'] = -1.0

        ids = df_.index.unique()
        size = df_.shape[0]
        count = 0
        curr_perc_int = -1;
        start_time = time()
        deltatime_str = ''
        for id_ in ids:
            #if label_id == 'index':
            #    filter_ = df_.index == id_
            #else:    
            #    filter_ = df_[label_id] == id_
                
            curr_lat = df_.at[id_, 'lat']
            curr_lon = df_.at[id_, 'lon']
            prev_lat = numpyutils.shift(curr_lat, 1)
            prev_lon = numpyutils.shift(curr_lon, 1)
            next_lat = numpyutils.shift(curr_lat, -1)
            next_lon = numpyutils.shift(curr_lon, -1)
            # using pandas shift in a large dataset: 7min 21s
            # using numpy shift above: 33.6 s
            
            # compute distance to next point
            #df_.loc[id_, 'dist_curr_to_next'] = 
            df_.at[id_, 'dist_curr_to_next'] = haversine(curr_lat, curr_lon, next_lat, next_lon)

            # compute distance from previous to current point
            #df_.loc[id_, 'dist_prev_to_curr'] = 
            df_.at[id_, 'dist_prev_to_curr'] = numpyutils.shift(df_.at[id_, 'dist_curr_to_next'], 1)

            # use distance from previous to next
            #df_.loc[id_, 'dist_prev_to_next'] = 
            df_.at[id_, 'dist_prev_to_next'] = haversine(prev_lat, prev_lon, next_lat, next_lon)
            
            count += curr_lat.shape[0]
            curr_perc_int, est_time_str = pru.progress_update(count, size, start_time, curr_perc_int, step_perc=20)

        return deltatime_str

    except Exception as e:
        print('{}: {}'.format(index_name, id_))
        raise e

        
def create_update_space_time_features(df_, index_name='id'):
    try:
        if 'dist_curr_to_next' not in df_:
            create_update_distance_features(df_, index_name)

        if df_.index.name is None:
            df_.set_index(index_name, inplace=True)

        print('create_update_space_time_features')
        if 'time_delta' not in df_:
            df_['time_delta'] = -1.0
        if 'speed' not in df_:
            df_['speed'] = -1.0

        ids = df_.index.unique()

        size = df_.shape[0]
        count = 0
        curr_perc_int = -1;
        start_time = time()
        for id_ in ids:
            time_ = df_.at[id_, 'time'].astype(np.float64)
            if type(time_) == np.float64:
                size_id = 1
                raise Exception("Trajectory must have at least 2 gps points.")
            else:
                size_id = time_.shape[0] 
            
            time_delta_ = numpyutils.shift(time_, -1) - time_

            #time_delta_ = df_.loc[filter_, 'time'].shift(-1).values - time_millis_
            df_.at[id_, 'time_delta'] = time_delta_
            df_.at[id_, 'speed'] = df_.at[id_, 'dist_curr_to_next'] / (time_delta_ / 1000)  # unit: m/s

            count += size_id
            curr_perc_int, est_time_str = pru.progress_update(count, size, start_time, curr_perc_int, step_perc=20)
                
    except Exception as e:
        print('{}: {} - size: {}'.format(index_name, id_, size_id))
        raise e

        
def add_curr_to_next_dist_time_speed_features(df_, index_name='tid'):
    """
    for better performance the trajectory id must be the index.
    """
    print('create_update_distance_features')
    try:
        if df_.index.name is not None and df_.index.name != index_name:
            df_.reset_index(inplace=True)
            
        if df_.index.name is None:
            df_.set_index(index_name, inplace=True)
        
        df_['delta_time'] = np.nan
        df_['delta_dist'] = np.nan
        df_['speed'] = np.nan
        
        ids = df_.index.unique()
        size = df_.shape[0]
        count = 0
        curr_perc_int = -1;
        start_time = time()
        deltatime_str = ''
        for id_ in ids:
            filter_points = (df_.at[id_,'isNode'] != 1)
            size_id = 1 if filter_points.shape == () else filter_points.shape[0]
            times = df_.at[id_, 'time'][filter_points].astype(np.float64)
            dists = df_.at[id_, 'distFromTrajStartToCurrPoint'][filter_points]
            
            delta_times = (npu.shift(times, -1) - times) / 1000.0
            delta_dists = npu.shift(dists, -1) - dists
            speeds = delta_dists / delta_times

            pdu.change_df_feature_values_using_filter(df_, id_, 'delta_time', filter_points, delta_times)
            pdu.change_df_feature_values_using_filter(df_, id_, 'delta_dist', filter_points, delta_dists)
            pdu.change_df_feature_values_using_filter(df_, id_, 'speed', filter_points, speeds)
            
            count += size_id
            curr_perc_int, est_time_str = pru.progress_update(count, size, start_time, curr_perc_int, step_perc=20)
            
        print('we still need to drop rows with infinite speeds or delta_dist==0 or delta_time == 0')
        df_.reset_index(inplace=True)
        idxs_drop = df_[ np.isinf(df_['speed']) | (df_['delta_dist'] == 0) | (df_['delta_time'] == 0) ].index.values
        print('dropping {} rows...'.format(idxs_drop.shape[0]) )
        if idxs_drop.shape[0] > 0:
            print('shape before dropping: {}'.format(df_.shape))
            df_.drop(index=idxs_drop, inplace=True)
            print('shape after dropping: {}'.format(df_.shape))
        else:
            print('no need to drop. shape: {}'.format(df_.shape))

    except Exception as e:
        print('{}: {}'.format(index_name, id_))
        raise e
        
        
def remove_trajectories_with_few_points(df_, label_id='id', min_points_per_trajectory=2):
    print('remove_trajectories_with_few_points')
    if df_.index.name is not None:
        print('reseting index')
        df_.reset_index(inplace=True)

    df_count_tid = df_.groupby(by=label_id).size()
    
    tids_with_few_points = df_count_tid[ df_count_tid < min_points_per_trajectory ].index
    print('#ids_with_few_points: {}'.format(tids_with_few_points.shape[0]))
    
    shape_before_drop = df_.shape
    idx = df_[ df_[label_id].isin(tids_with_few_points) ].index
    print('#ids before drop: {}'.format(df_[label_id].unique().shape[0]))
    df_.drop(index=idx, inplace=True)
    print('#ids after drop: {}'.format(df_[label_id].unique().shape[0]))
    print('shape - before drop: {} - after drop: {}'.format(shape_before_drop, df_.shape)) 

    
def clean_gps_jumps(df_, label_id='id'):
    print('clean_gps_jumps')
    create_update_distance_features(df_)
    
    if df_.index.name is not None:
        print('reseting index')
        df_.reset_index(inplace=True)
        
    filter_ = (df_['dist_curr_to_next'] > 1) & (df_['dist_prev_to_curr'] > 1) & \
       (3 * df_['dist_prev_to_next'] < df_['dist_curr_to_next']) & \
       (3 * df_['dist_prev_to_next'] < df_['dist_prev_to_curr'])  
    idx = df_[filter_].index
    
    if idx.shape[0] > 0:
        print('dropping {} gps points'.format(idx.shape[0]))
        df_.drop(index=idx, inplace=True)
        print('remove_gps_errors - shape after removal:', df_.shape)
    
        clean_gps_jumps(df_)
    else:
        print('remove_gps_errors - nothing to remove:', df_.shape)    


def clean_zero_distances(df_, label_id='id', label_dist='dist_curr_to_next'):
    print('clean_zero_distances')
    
    if df_.index.name is not None:
        print('reseting index')
        df_.reset_index(inplace=True)
        
    filter_ = df_[label_dist] == 0 | df_[label_dist].isnull()
    idx = df_[filter_].index
    shape_before = df_.shape
    df_.drop(index=idx, inplace=True)
    print('clean_zero_distance - shape before:{}, shape after:{}'.format(shape_before, df_.shape))


def split_trajectories_by_time(df_, index_name='id', max_time_between_adj_points=900, label_new_id='tid'):
    """
    index_name is the current id.
    label_new_id is the new splitted id.
    delta_time must be updated after split.
    """
    print('max_time_between_adj_points:', max_time_between_adj_points)

    print('split_trajectories')
    if df_.index.name is None:
        df_.set_index(index_name, inplace=True)

    # one vehicle id can generate many trajectory ids (tids).
    # we segment the vehicle trajectory into several trajectories.
    curr_tid = 0
    if label_new_id not in df_:
        df_[label_new_id] = curr_tid

    vids = df_.index.unique()

    count = 0
    size = df_.shape[0]
    curr_perc_int = -1
    start_time = time()

    for id_ in vids:
        
        ## novo id_ deve gerar e alterar novo tid2 (curr_tid)
        curr_tid += 1
        times = df_.at[id_, 'time'].astype(np.float64)
        if times.shape == ():
            # trajectories with only one point is useless for interpolation and so they must be removed.
            count += 1
            df_.at[id_, label_new_id] = -1
            curr_tid += -1
        else:
            delta_times = (npu.shift(times, -1) - times) / 1000.0
            # df_.at[id_, 'delta_time'] = delta_times

            filter_ = (delta_times > max_time_between_adj_points)

            tids = np.empty(filter_.shape[0], dtype=np.int64)
            tids.fill(curr_tid)
            for i, has_problem in enumerate(filter_):
                if has_problem:
                    curr_tid += 1
                    tids[i+1:] = curr_tid
            count += tids.shape[0]
            df_.at[id_, label_new_id] = tids
        
        curr_perc_int, est_time_str = pru.progress_update(count, size, start_time, curr_perc_int, step_perc=20)

    if index_name == label_new_id:
        df_.reset_index(drop=True, inplace=True)
    else:
        df_.reset_index(inplace=True)

    shape_before_drop = df_.shape
    idx = df_[ df_[label_new_id] == -1 ].index
    if idx.shape[0] > 0:
        tids_before_drop = df_['tid'].unique().shape[0]
        df_.drop(index=idx, inplace=True)
        print('#{} - before drop: {} - after drop: {}'.format(label_new_id, tids_before_drop,
                                                              df_[label_new_id].unique().shape[0]))
        print('shape - before drop: {} - after drop: {}'.format(shape_before_drop, df_.shape))
    else:
        print('no trajs with only one point - nothing to change:', df_.shape)


def split_trajectories(df_, index_name='id', max_dist_between_adj_points=1000, max_time_between_adj_points=120000,
                      max_speed=25, label_new_id='tid', label_delta_time='time_delta', 
                      label_delta_dist='dist_curr_to_next', label_speed='speed'):
    """
    index_name is the current id.
    label_new_id is the new splitted id.
    time, dist, speeed features must be updated after split.
    """
    print('max_time_between_adj_points:', max_time_between_adj_points)
    print('max_dist_between_adj_points:', max_dist_between_adj_points)
    print('max_speed:', max_speed)
    # remove speed between points higher than max_speed (m/s)
    # 50m/s = 180km/h
    # 30m/s = 108km/h
    # 25m/s =  90km/h
    
    print('split_trajectories')
    if df_.index.name is None:
        df_.set_index(index_name, inplace=True)

    # one vehicle id can generate many trajectory ids (tids).
    # we segment the vehicle trajectory into several trajectories.
    curr_tid = 0
    if label_new_id not in df_:
        df_[label_new_id] = curr_tid

    vids = df_.index.unique()

    count = 0
    size = df_.shape[0]
    curr_perc_int = -1
    start_time = time()

    for id_ in vids:
        
        ## novo id_ deve gerar e alterar novo tid2 (curr_tid)
        curr_tid += 1
        
        filter_ = (df_.at[id_, label_delta_time] > max_time_between_adj_points) | \
                    (df_.at[id_, label_delta_dist] > max_dist_between_adj_points) | \
                    (df_.at[id_, label_speed] > max_speed)        
        
        # remove speed between points higher than max_speed (m/s)
        # 50m/s = 180km/h
        # 30m/s = 108km/h
        # 25m/s =  90km/h        
                    
        if filter_.shape == ():
            # trajectories with only one point is useless for interpolation and so they must be removed.
            count += 1
            #print('problem', index_name, id_, label_new_id, curr_tid, label_delta_time, df_.at[id_, label_delta_time], \
            #      label_delta_dist, df_.at[id_, label_delta_dist], label_speed, df_.at[id_, label_speed] )
            df_.at[id_, label_new_id] = -1
            curr_tid += -1
        else:
            tids = np.empty(filter_.shape[0], dtype=np.int64)
            tids.fill(curr_tid)
            for i, has_problem in enumerate(filter_):
                if has_problem:
                    curr_tid += 1
                    tids[i+1:] = curr_tid
            count += tids.shape[0]
            df_.at[id_, label_new_id] = tids
        
        
        curr_perc_int, est_time_str = pru.progress_update(count, size, start_time, curr_perc_int, step_perc=20)

    if index_name == label_new_id:
        df_.reset_index(drop=True, inplace=True)
    else:
        df_.reset_index(inplace=True)

    shape_before_drop = df_.shape
    idx = df_[ df_[label_new_id] == -1 ].index
    if idx.shape[0] > 0:
        tids_before_drop = df_['tid'].unique().shape[0]
        df_.drop(index=idx, inplace=True)
        print('#tids - before drop: {} - after drop: {}'.format(tids_before_drop, df_['tid'].unique().shape[0]))
        print('shape - before drop: {} - after drop: {}'.format(shape_before_drop, df_.shape))
    else:
        print('no trajs with only one point - nothing to change:', df_.shape)

def remove_short_trajs_and_trajs_with_few_points(df_, label_id='tid', min_trajectory_distance=100):
    # remove_tids_with_few_points must be performed before updating features, because 
    # those features only can be computed with at least 2 points per trajactories
    print('remove_tids_with_few_points...')
    remove_trajectories_with_few_points(df_, label_id=label_id)
    
    print('create_update_distance_features...')
    create_update_distance_features(df_, index_name=label_id)
    
    print('create_update_space_time_features...')
    create_update_space_time_features(df_, index_name=label_id)

    if df_.index.name is not None:
        print('reseting index')
        df_.reset_index(inplace=True)
        
    print('dropping unnecessary trajectories...')
    df_agg_tid = df_.groupby(by='tid').agg({'dist_curr_to_next':'sum'})
    filter_ = (df_agg_tid['dist_curr_to_next'] < min_trajectory_distance)    
    tid_selection = df_agg_tid[ filter_ ].index
    print('#short trajectories and trajectories with a minimum distance ({}): {}'.format(df_agg_tid.shape[0], 
            min_trajectory_distance))
    
    shape_before_drop = df_.shape
    idx = df_[ df_['tid'].isin(tid_selection) ].index
    if idx.shape[0] > 0:
        tids_before_drop = df_['tid'].unique().shape[0]
        df_.drop(index=idx, inplace=True)
        print('#tids - before drop: {} - after drop: {}'.format(tids_before_drop, df_['tid'].unique().shape[0]))
        print('shape - before drop: {} - after drop: {}'.format(shape_before_drop, df_.shape))
        remove_short_trajs_and_trajs_with_few_points(df_)
    else:
        print('remove_short_trajs_and_trajs_with_few_points - nothing to change:', df_.shape)        

        
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
        start_time = time()
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
            curr_perc_int, est_time_str = pru.progress_update(count, size, start_time, curr_perc_int, step_perc=20)
            

        count = 0
        curr_perc_int = -1
        start_time = time()
        size_id = 0
        print('checking delta_times, delta_dists and speeds...')
        for tid in tids:
            filter_ = (df.at[tid,'isNode'] != 1)

            dists = df.at[tid, 'distFromTrajStartToCurrPoint'][filter_]
            delta_dists = (npu.shift(dists, -1) - dists)[:-1]   # do not use last element (np.nan)
                          
            assert np.all(delta_dists <= max_dist_between_adj_points), \
                          'delta_dists must be <= {}'.format(max_dist_between_adj_points)
            
            times = df.at[tid, 'time'][filter_].astype(np.float64)
            delta_times = ((npu.shift(times, -1) - times) / 1000.0)[:-1] # do not use last element (np.nan)
                          
            assert np.all(delta_times <= max_time_between_adj_points), \
                          'delta_times must be <= {}'.format(max_time_between_adj_points)
            
            assert np.all(delta_times > 0), 'delta_times must be > 0'

            assert np.all(delta_dists > 0), 'delta_dists must be > 0'
            
            speeds = delta_dists / delta_times
            assert np.all(speeds <= max_speed), 'speeds > {}'.format(max_speed)
            
            size_id = 1 if filter_.shape == () else filter_.shape[0]
            count += size_id
            curr_perc_int, est_time_str = pru.progress_update(count, size, start_time, curr_perc_int, step_perc=20)
            

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

            pdu.change_df_feature_values_using_filter_and_indexes(df, tid, 'deleted', filter_, idx_not_in_ascending_order, True)
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
        start_time = time()
        for tid in tids:
            size_id = fix_time_not_in_ascending_order_id(df, tid, index_name)

            count += size_id
            curr_perc_int, est_time_str = pru.progress_update(count, size, start_time, curr_perc_int, step_perc=20)

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
    start_time = time()

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
                curr_perc_int, est_time_str = pru.progress_update(count, size, start_time, curr_perc_int, step_perc=20)
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
                curr_perc_int, est_time_str = pru.progress_update(count, size, start_time, curr_perc_int, step_perc=20)
                continue

            # compute delta_time and distance between points
            #values = (npu.shift(df.at[tid, 'time'][filter_nodes].astype(np.float64), -1) - df.at[tid, 'time'][filter_nodes]) / 1000
            #pdu.change_df_feature_values_using_filter(df, tid, 'delta_time', filter_nodes, values)
            delta_time = ( (npu.shift(y_.astype(np.float64), -1) - y_) / 1000.0 )[:-1]
            dist_curr_to_next = (npu.shift(x_, -1) - x_)[:-1]
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
            pdu.change_df_feature_values_using_filter(df, tid, 'time', filter_nodes, values)

            # create delta_time feature
            values = (npu.shift(df.at[tid, 'time'][filter_nodes].astype(np.float64), -1) - df.at[tid, 'time'][filter_nodes]) / 1000
            pdu.change_df_feature_values_using_filter(df, tid, 'delta_time', filter_nodes, values)

            # create speed feature
            values = df.at[tid, 'edgeDistance'][filter_nodes] / df.at[tid, 'delta_time'][filter_nodes]
            pdu.change_df_feature_values_using_filter(df, tid, 'speed', filter_nodes, values)

            curr_perc_int, est_time_str = pru.progress_update(count, size, start_time, curr_perc_int, step_perc=20)
            
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

