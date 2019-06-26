from time import time

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


from utils import timeutils
from utils import numpyutils as npu
from utils import pandasutils as pdu
from utils import progressutils as pru
from utils import trajutils
"""

import timeutils
import numpyutils as npu
import pandasutils as pdu
import progressutils as pru

"""
"""main labels """
gl_label_id = 'id'
gl_label_lat = 'lat'
gl_label_lon = 'lon'
gl_label_date = 'date'

""" labels to distances"""
gl_label_dist_prev_next = 'dist_prev_to_next'
gl_label_dist_to_prev = 'dist_to_prev'
gl_label_dist_to_next = 'dist_to_next'

    
""" labels to space and time"""
gl_label_time_to_prev = 'time_to_prev'
gl_label_time_to_next = 'time_to_next'
gl_label_speed_to_prev = 'speed_to_prev' 
gl_label_speed_to_next = 'speed_to_next'

""" labels to move and stop"""
gl_situation = 'situation'

def format_labels(df_, current_id, current_lat, current_lon, current_date,  inplace=False):
    """ 
    Format the labels for the PyRoad lib pattern
    """ 
    if inplace:
        return df_.rename(columns= {current_id: gl_label_id, current_lat: gl_label_lat, current_lon: gl_label_lon, current_date: gl_label_date}, inplace=True)
    else:
        return df_.rename(columns= {current_id: gl_label_id, current_lat: gl_label_lat, current_lon: gl_label_lon, current_date: gl_label_date}, inplace=False) 
        
def format_trajectories_data(df_, id=gl_label_id, label_date=gl_label_date, label_lat=gl_label_lat, label_lon=gl_label_lon, setIndex=False):
    """ 
    Format the data to an appropriate format
    """
    try:
        if setIndex:
            df_.set_index(id, inplace=True)
        
        df_[label_lat] = df_[label_lat].astype('float64')
        df_[label_lon] = df_[label_lon].astype('float64')
        df_[label_date] = pd.to_datetime(df_[label_date])
    except Exception as e:
        raise e

def show_data_info(df):
    try:
        print('\n======================= INFORMATION ABOUT DATASET =======================\n')
        print('Number of Rows: {}\nNumber of Ids: {} '.format(df.shape[0], df['id'].nunique()))
        print('Data between {} and {}'.format(df['date'].min(), df['date'].max()))
        print('Bounding Box:', trajutils.get_bbox(df)) # bbox return =  Lat_min , Long_min, Lat_max, Long_max) 
        print('\n=========================================================================\n')
    except Exception as e:
        raise e    

def bbox_split(bbox, total_grids):
    lat_min = bbox[0]
    lon_min = bbox[1]
    lat_max = bbox[2]
    lon_max = bbox[3]
    const_lat =  abs(abs(lat_max) - abs(lat_min))/total_grids
    const_lon =  abs(abs(lon_max) - abs(lon_min))/total_grids
    print('const_lat: {}\nconst_lon: {}'.format(const_lat, const_lon))

    df = pd.DataFrame(columns=['lat_min', 'lon_min', 'lat_max', 'lon_max'])
    for i in range(total_grids):
        df = df.append({'lat_min':lat_min, 'lon_min': lon_min + (const_lon * i), 'lat_max': lat_max, 'lon_max':lon_min + (const_lon * (i + 1))}, ignore_index=True)
    
    return df

def get_bbox(df_, label_lat = gl_label_lat, label_lon = gl_label_lon):
    """
    A bounding box (usually shortened to bbox) is an area defined by two longitudes and two latitudes, where:
    Latitude is a decimal number between -90.0 and 90.0. Longitude is a decimal number between -180.0 and 180.0.
    They usually follow the standard format of: 
    bbox = left,bottom,right,top 
    bbox = min Longitude , min Latitude , max Longitude , max Latitude 
    """
    try:
        return (df_[label_lat].min(), df_[label_lon].min(), df_[label_lat].max(), df_[label_lon].max())
    except Exception as e:
        raise e

def filter_by_date_id(df_, index=None, label_id=gl_label_id, label_date = gl_label_date, startDate=None, endDate=None):
    if (startDate is not None and endDate is not None):
        return df_[(df_[label_id] == index) & (df_[label_date] > startDate) & (df_[label_date] <= endDate)]
    elif (startDate is None):
        return df_[(df_[label_id] == index) & (df_[label_date] <= endDate)]
    else:
        return df_[(df_[label_id] == index) & (df_[label_date] > startDate)]  

def filter_by_date(df_, label_date = gl_label_date, startDate=None, endDate=None):
    if (startDate is not None and endDate is not None):
        return df_[(df_[label_date] > startDate) & (df_[label_date] <= endDate)]
    elif (startDate is None):
        return df_[(df_[label_date] <= endDate)]
    else:
        return df_[(df_[label_date] > startDate)]  

def filter_bbox(df_, lat_down, lon_left, lat_up, lon_right, filter_out=False, label_lat = gl_label_lat, label_lon = gl_label_lon, inplace=False):
    """
    Filter bounding box.
    Example: 
    filter_bbox(df_, -3.90, -38.67, -3.68, -38.38) -> Fortaleza
    """
    try:
        filter_ = (df_[label_lat] >= lat_down) & (df_[label_lat] <= lat_up) & (df_[label_lon] >= lon_left) & (df_[label_lon] <= lon_right)
        if filter_out:
            filter_ = ~filter_

        if inplace:
            df_.drop( index=df_[ ~filter_ ].index, inplace=True )
            return df_
        else:
            return df_.loc[ filter_ ]
    except Exception as e: 
            raise e

""" ===================================================================================
----------------------  FUCTIONS TO LAT AND LONG OPERATIONS ---------------------------
======================================================================================= """ 

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
        a = np.sin((lat2-lat1)/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2-lon1)/2.0)**2
    return earth_radius * 2 * np.arcsin(np.sqrt(a)) * 1000  # result in meters (* 1000)


""" ===================================================================================
----------------------  FUCTIONS TO CREATE NEW FEATURES IN DATASET   -----------------------------------
======================================================================================= """ 


def create_update_distance_features(df_, label_id=gl_label_id, label_lat=gl_label_lat, label_lon=gl_label_lon):
    print('\nCreating or updating distance features in meters...\n')
    try:
        if df_.index.name is None:
            print('...Set {} as index to better peformance\n'.format(label_id))
            df_.set_index(label_id, inplace=True)

        """ create ou update columns"""
        if gl_label_dist_to_next not in df_:
            df_[gl_label_dist_to_next] = -1.0
        if gl_label_dist_to_prev not in df_:
            df_[gl_label_dist_to_prev] = -1.0
        if gl_label_dist_prev_next not in df_:
            df_[gl_label_dist_prev_next] = -1.0

        ids = df_.index.unique()
        size = df_.shape[0]
        count = 0
        curr_perc_int = -1
        start_time = time()
        deltatime_str = ''

        for id_ in ids:
            #if label_id == 'index':
            #    filter_ = df_.index == id_
            #else:    
            #    filter_ = df_[label_id] == id_
                
            curr_lat = df_.at[id_, label_lat]
            curr_lon = df_.at[id_, label_lon]

            prev_lat = npu.shift(curr_lat, 1)
            prev_lon = npu.shift(curr_lon, 1)
            
            next_lat = npu.shift(curr_lat, -1)
            next_lon = npu.shift(curr_lon, -1)
            # using pandas shift in a large dataset: 7min 21s
            # using numpy shift above: 33.6 s
            
            # compute distance to next point
            #df_.loc[id_, 'dist_curr_to_next'] = 
            df_.at[id_, gl_label_dist_to_next] = haversine(curr_lat, curr_lon, next_lat, next_lon)

            # compute distance from previous to current point
            #df_.loc[id_, 'dist_prev_to_curr'] = 
            df_.at[id_, gl_label_dist_to_prev] = haversine(prev_lat, prev_lon, curr_lat, curr_lon)

            # use distance from previous to next
            #df_.loc[id_, 'dist_prev_to_next'] = 
            #
            df_.at[id_, gl_label_dist_prev_next] = haversine(prev_lat, prev_lon, next_lat, next_lon)
            
            count += curr_lat.shape[0]
            curr_perc_int, est_time_str = pru.progress_update(count, size, start_time, curr_perc_int, step_perc=20)
        df_.reset_index(inplace=True)
        print('...Reset index\n')
        print(deltatime_str)
    except Exception as e:
        print('{} = {}, size: {}'.format(label_id, id_, size))
        raise e


def create_update_dist_time_speed_features(df_, label_id=gl_label_id, label_date=gl_label_date):
    try:
        print('\nCreating or updating space and time features in meters by seconds\n')
       
        """update or create distance features"""
        create_update_distance_features(df_, label_id)   

        if df_.index.name is None:
            print('...Set {} as index to better peformance\n'.format(label_id))
            df_.set_index(label_id, inplace=True)

        """create new feature to time"""
        if gl_label_time_to_prev not in df_:
            df_[gl_label_time_to_prev] = -1.0
        if gl_label_time_to_next not in df_:
            df_[gl_label_time_to_next] = -1.0

        """create new feature to speed"""
        if gl_label_speed_to_prev not in df_:
            df_[gl_label_speed_to_prev] = -1.0
        if gl_label_speed_to_next not in df_:
            df_[gl_label_speed_to_next] = -1.0

        ids = df_.index.unique()
        size = df_.shape[0]
        count = 0
        curr_perc_int = -1
        start_time = time()
        for id_ in ids:
            time_ = df_.at[id_, label_date].astype(np.float64)
            if type(time_) == np.float64:
                size_id = 1
                raise Exception("Trajectory must have at least 2 gps points.")
            else:
                size_id = time_.shape[0] 
            
            """time_to_prev = current_date - prev_date 
            the time_delta must be in nanosecond, then we multiplie by 10-⁹ to tranform in seconds """
            time_delta_prev = (time_ - npu.shift(time_, 1))*(10**-9)
            df_.at[id_, gl_label_time_to_prev] = time_delta_prev

            """ set time_to_next"""
            time_delta_next = (npu.shift(time_, -1) - time_)*(10**-9)
            df_.at[id_, gl_label_time_to_next] = time_delta_next
            
            "set Speed features"
            df_.at[id_, gl_label_speed_to_prev] = df_.at[id_, gl_label_dist_to_prev] / (time_delta_prev)  # unit: m/s
            df_.at[id_, gl_label_speed_to_next] = df_.at[id_, gl_label_dist_to_next] / (time_delta_next)  # unit: m/s

            count += size_id
            curr_perc_int, est_time_str = pru.progress_update(count, size, start_time, curr_perc_int, step_perc=20)
        print('...Reset index...\n')
        df_.reset_index(inplace=True)
    except Exception as e:
        print('{}: {} - size: {}'.format(label_id, id_, size_id))
        raise e

def create_update_move_and_stop_feature(df_, label_id=gl_label_id, new_label=gl_situation, time_radius_to_stop=10, update_features=True):
    if (update_features) is True:
        """update or create distance features"""
        create_update_dist_time_speed_features(df_, label_id)   
    try:
        print('\nCreating or updating features MOVE and STOPS...\n')
        if(new_label in df_):
            del df_[new_label]
        conditions = (df_[gl_label_time_to_next] < time_radius_to_stop), (df_[gl_label_time_to_next] >= time_radius_to_stop)
        choices = ['move', 'stop']
        df_[new_label] = np.select(conditions, choices, 'undefined')      
        print('\n....There are {} stops to this parameters\n'.format(df_[df_[new_label] == 'stop'].shape[0]))
    except Exception as e:
        raise e


""" ===================================================================================
----------------------  FUCTIONS TO DATA CLEANING   -----------------------------------
======================================================================================= """ 


def remove_duplicates(df_, subset=None, inplace=False):
    """
    Return DataFrame with duplicate rows removed, optionally only considering certain columns.
    """
    print('Remove rows duplicates by subset')
    return df_.drop_duplicates(subset, inplace)
    
""" This fuction is the union between clean_gps_nearby_points and clean_gps_jumps_by_distance fuctions 
def clean_gps_nearby_points_and_jumps_by_distance(df_, label_id=gl_label_id, radius_area=10.0, jump_coefficient=3.0):
    create_update_distance_features(df_, label_id)
    
    filter_nearby_points = (df_[gl_label_dist_to_prev] <= radius_area)
    
    filter_jumpy = (df_[gl_label_dist_to_next] > 1) & (df_[gl_label_dist_to_prev] > 1) & \
       (jump_coefficient * df_[gl_label_dist_prev_next] < df_[gl_label_dist_to_next]) & \
       (jump_coefficient * df_[gl_label_dist_prev_next] < df_[gl_label_dist_to_prev])  

    idx_nearby = df_[filter_nearby_points].index
    idx_jumps= df_[filter_jumpy].index

    if idx_jumps.shape[0] > 0:
        print('...Dropping {} rows of Jumps points\n'.format(idx_jumps.shape[0]))
        shape_before = df_.shape[0]
        df_.drop(index=idx_jumps, inplace=True)
        print('...Rows before: {}, Rows after:{}\n'.format(shape_before, df_.shape[0])) 
        clean_gps_nearby_points_and_jumps_by_distance(df_, label_id, radius_area, jump_coefficient)
    elif idx_nearby.shape[0] > 0:
        print('...Dropping {} gps Nearby points\n'.format(idx_nearby.shape[0]))
        shape_before = df_.shape[0]
        df_.drop(index=idx_nearby, inplace=True)
        print('...Rows before: {}, Rows after:{}\n'.format(shape_before, df_.shape[0]))
        clean_gps_nearby_points_and_jumps_by_distance(df_, label_id, radius_area, jump_coefficient)
    else:
        print("...There is no gps points to drop\n")
"""
def clean_gps_nearby_points(df_, label_id=gl_label_id, radius_area=10.0, update_features=True):
    if(update_features):
       create_update_distance_features(df_, label_id)
   
    print('\nCleaning gps points from radius of {} meters\n'.format(radius_area))
   
    if df_.index.name is not None:
        print('...Reset index for filtering\n')
        df_.reset_index(inplace=True)
    
    filter_nearby_points = (df_[gl_label_dist_to_prev] <= radius_area)
    
    idx = df_[filter_nearby_points].index
    
    print('...There are {} gps points to drop\n'.format(idx.shape[0]))
    if idx.shape[0] > 0:
        print('...Dropping {} gps points\n'.format(idx.shape[0]))
        shape_before = df_.shape[0]
        df_.drop(index=idx, inplace=True)
        print('...Rows before: {}, Rows after:{}\n'.format(shape_before, df_.shape[0]))
        clean_gps_nearby_points(df_, label_id, radius_area, update_features)


def clean_gps_jumps_by_distance(df_, label_id=gl_label_id, jump_coefficient=3.0, update_features=True):
    if(update_features is True):
       create_update_distance_features(df_, label_id)

    print('\nCleaning gps jumps by distance to jump_coefficient {}...\n'.format(jump_coefficient))

    if df_.index.name is not None:
        print('...Reset index for filtering\n')
        df_.reset_index(inplace=True)
    
    filter_jumpy = (df_[gl_label_dist_to_next] > 1) & (df_[gl_label_dist_to_prev] > 1) & \
       (jump_coefficient * df_[gl_label_dist_prev_next] < df_[gl_label_dist_to_next]) & \
       (jump_coefficient * df_[gl_label_dist_prev_next] < df_[gl_label_dist_to_prev])  

    idx = df_[filter_jumpy].index
    
    print('...There are {} gps points to drop \n'.format(idx.shape[0]))
    if  idx.shape[0] > 0:
        print('...Dropping {} rows of gps points\n'.format(idx.shape[0]))
        shape_before = df_.shape[0]
        df_.drop(index=idx, inplace=True)
        print('...Rows before: {}, Rows after:{}\n'.format(shape_before, df_.shape[0]))
        clean_gps_jumps_by_distance(df_, label_id, jump_coefficient, update_features)    
  

def clean_gps_speed_max_radius(df_, label_id=gl_label_id, speed_max=50.0, update_features=True):
    if(update_features):
        create_update_dist_time_speed_features(df_, label_id)

    print('\nClean gps points with speed max > {} meters by seconds'.format(speed_max))

    filter_ = (df_[gl_label_speed_to_prev] > speed_max) | (df_[gl_label_speed_to_next] > speed_max)
    
    idx = df_[filter_].index
    
    print('...There {} gps points with speed_max > {}\n'.format(idx.shape[0], speed_max))
    if idx.shape[0] > 0:
        print('...Dropping {} rows of jumps by speed max\n'.format(idx.shape[0]))
        shape_before = df_.shape[0]
        df_.drop(index=idx, inplace=True)
        print('...Rows before: {}, Rows after:{}\n'.format(shape_before, df_.shape[0]))
        clean_gps_speed_max_radius(df_, label_id, speed_max)


""" ============================================================"""
""" All of the above functions have been standardized and tested"""
""" ============================================================"""
"""
def create_dist_time_speed_features(df_, index_name='tid'):
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
        curr_perc_int = -1
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
"""

def remove_trajectories_with_few_points(df_, label_id='id', min_points_per_trajectory=2):
    print('remove_trajectories_with_few_points')
    if df_.index.name is not None:
        print('\nReset index for filtering\n')
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
    remove_trajectories_with_few_points(df_, label_id)
    
    print('create_update_distance_features...')
    create_update_distance_features(df_, label_id=label_id)
    
    print('create_update_space_time_features...')
    """ create_update_space_time_features(df_, label_id)"""

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

