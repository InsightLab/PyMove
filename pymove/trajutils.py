import numpy as np
import pandas as pd
import time
from scipy.interpolate import interp1d

from pymove import utils as ut


"""main labels """
dic_labels = {"id" : 'id', 'lat' : 'lat', 'lon' : 'lon', 'datetime' : 'datetime'}

dic_features_label = {'tid' : 'tid', 'dist_to_prev' : 'dist_to_prev', "dist_to_next" : 'dist_to_next', 'dist_prev_to_next' : 'dist_prev_to_next', 
                    'time_to_prev' : 'time_to_prev', 'time_to_next' : 'time_to_next', 'speed_to_prev': 'speed_to_prev', 'speed_to_next': 'speed_to_next',
                    'period': 'period', 'day': 'day', 'index_grid_lat': 'index_grid_lat', 'index_grid_lon' : 'index_grid_lon',
                    'situation':'situation'}

def format_labels(df_, current_id, current_lat, current_lon, current_datetime):
    """ 
    Format the labels for the PyRoad lib pattern 
        labels output = lat, lon and datatime
    """ 
    dic_labels['id'] = current_id
    dic_labels['lon'] = current_lon
    dic_labels['lat'] = current_lat
    dic_labels['datetime'] = current_datetime
    return dic_labels
    
def show_trajectories_info(df_, dic_labels=dic_labels):
    """
        show dataset information from dataframe, this is number of rows, datetime interval, and bounding box 
    """
    try:
        print('\n======================= INFORMATION ABOUT DATASET =======================\n')
        print('Number of Points: {}\n'.format(df_.shape[0]))
        if dic_labels['id'] in df_:
            print('Number of IDs objects: {}\n'.format(df_[dic_labels['id']].nunique()))
        if dic_features_label['tid'] in df_:
            print('Number of IDs trajectory: {}\n'.format(df_[dic_features_label['tid']].nunique()))
        if dic_labels['datetime'] in df_:
            print('Start Date:{}     End Date:{}\n'.format(df_[dic_labels['datetime']].min(), df_[dic_labels['datetime']].max()))
        if dic_labels['lat'] and dic_labels['lon'] in df_:
            print('Bounding Box:', get_bbox(df_, dic_labels)) # bbox return =  Lat_min , Long_min, Lat_max, Long_max) 
        print('\n=========================================================================\n')
    except Exception as e:
        raise e    

def get_bbox(df_, dic_labels=dic_labels):
    """
    A bounding box (usually shortened to bbox) is an area defined by two longitudes and two latitudes, where:
    Latitude is a decimal number between -90.0 and 90.0. Longitude is a decimal number between -180.0 and 180.0.
    They usually follow the standard format of: 
    bbox = left,bottom,right,top 
    bbox = min Longitude , min Latitude , max Longitude , max Latitude 
    """
    try:
        return (df_[dic_labels['lat']].min(), df_[dic_labels['lon']].min(), df_[dic_labels['lat']].max(), df_[dic_labels['lon']].max())
    except Exception as e:
        raise e

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


""" ----------------------  FUCTIONS TO LAT AND LONG COORDINATES --------------------------- """ 

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
    Vectorized haversine function: https://stackoverflow.com/questions/43577086/pandas-calculate-haversine-distance-within-each-group-of-rows
    About distance between two points: https://janakiev.com/blog/gps-points-distance-python/
    Calculate the great circle distance between two points on the earth (specified in decimal degrees or in radians).
    All (lat, lon) coordinates must have numeric dtypes and be of equal length.
    Result in meters. Use 3956 in earth radius for miles
    """
    try:
        if to_radians:
            lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
            a = np.sin((lat2-lat1)/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2-lon1)/2.0)**2
        #return earth_radius * 2 * np.arcsin(np.sqrt(a)) * 1000  # result in meters (* 1000)
        return 2 * 1000 * earth_radius * np.arctan2(a ** 0.5, (1-a) ** 0.5)
        #np.arctan2(np.sqrt(a), np.sqrt(1-a)) 

    except Exception as e:
        print('\nError Haverside fuction')
        raise e


""" ----------------------  FUCTIONS TO CREATE NEW FEATURES BASED ON DATATIME  ----------------------------- """

def create_update_tid_based_on_id_datatime(df_, dic_labels=dic_labels, sort=True):
    """
        Create or update trajectory id  
            Exampĺe: ID = M00001 and datetime = 2019-04-28 00:00:56  -> tid = M0000120190428
    """
    try:
        print('\nCreating or updating tid feature...\n')
        if sort is True:
            print('...Sorting by {} and {} to increase performance\n'.format(dic_labels['id'], dic_labels['datetime']))
            df_.sort_values([dic_labels['id'], dic_labels['datetime']], inplace=True)


        df_[dic_features_label['tid']] = df_[dic_labels['id']].astype(str) + df_[dic_labels['datetime']].dt.date.astype(str)
        #strftime("%Y%m%d")  
        print('\n...tid feature was created...\n')      
    except Exception as e:
        raise e

def create_update_day_of_the_week_features(df_, dic_labels=dic_labels):
    """
        Create or update a feature day of the week from datatime
            Exampĺe: datetime = 2019-04-28 00:00:56  -> day = Sunday
    """
    try:
        print('\nCreating or updating day of the week feature...\n')
        df_[dic_features_label['day']] = df_[dic_labels['datetime']].dt.day_name()
        print('...the day of the week feature was created...\n')
    except Exception as e:
        raise e

def create_update_time_of_day_features(df_, dic_labels=dic_labels):
    """
        Create a feature time of day or period from datatime
            Examples: 
                datetime1 = 2019-04-28 02:00:56 -> period = early morning
                datetime2 = 2019-04-28 08:00:56 -> period = morning
                datetime3 = 2019-04-28 14:00:56 -> period = afternoon
                datetime4 = 2019-04-28 20:00:56 -> period = evening
    """
    try:
        print('\nCreating or updating period feature\n...early morning from 0H to 6H\n...morning from 6H to 12H\n...afternoon from 12H to 18H\n...evening from 18H to 24H')
        conditions =   [(df_[dic_labels['datetime']].dt.hour >= 0) & (df_[dic_labels['datetime']].dt.hour < 6), 
                        (df_[dic_labels['datetime']].dt.hour >= 6) & (df_[dic_labels['datetime']].dt.hour < 12),
                        (df_[dic_labels['datetime']].dt.hour >= 12) & (df_[dic_labels['datetime']].dt.hour < 18),  
                        (df_[dic_labels['datetime']].dt.hour >= 18) & (df_[dic_labels['datetime']].dt.hour < 24)]
        choices = ['early morning', 'morning', 'afternoon', 'evening']
        df_[dic_features_label['period']] = np.select(conditions, choices, 'undefined')      
        print('\n...the period of day feature was created')
    except Exception as e:
        raise e

""" ----------------------  FUCTIONS TO CREATE NEW FEATURES BASED ON LAT AND LON COORDINATES  ----------------------------- """
def create_update_dist_features(df_, label_id=dic_labels['id'], dic_labels=dic_labels, label_dtype = np.float64, sort=True):
    """
        Create three distance in meters to an GPS point P (lat, lon)
            Example:
                P to P.next = 2 meters
                P to P.previous = 1 meter
                P.previous to P.next = 1 meters
    """
    try:
        print('\nCreating or updating distance features in meters...\n')
        start_time = time.time()

        if sort is True:
            print('...Sorting by {} and {} to increase performance\n'.format(label_id, dic_labels['datetime']))
            df_.sort_values([label_id, dic_labels['datetime']], inplace=True)

        if df_.index.name is None:
            print('...Set {} as index to increase attribution performance\n'.format(label_id))
            df_.set_index(label_id, inplace=True)
        
        """ create ou update columns"""
        df_[dic_features_label['dist_to_prev']] = label_dtype(-1.0)
        df_[dic_features_label['dist_to_next']] = label_dtype(-1.0)
        df_[dic_features_label['dist_prev_to_next']]= label_dtype(-1.0)

        ids = df_.index.unique()
        df_size = df_.shape[0]
        curr_perc_int = -1
        start_time = time.time()
        deltatime_str = ''
        sum_size_id = 0
        size_id = 0
        for idx in ids:
            curr_lat = df_.at[idx, dic_labels['lat']]
            curr_lon = df_.at[idx, dic_labels['lon']]

            size_id = curr_lat.size
            
            if size_id <= 1:
                print('...id:{}, must have at least 2 GPS points\n'.format(idx))
                #df_.at[idx, dic_features_label['dist_to_next']] = np.nan
                df_.at[idx, dic_features_label['dist_to_prev']] = np.nan
                #df_.at[idx, dic_features_label['dist_prev_to_next']] = np.nan    
                
            else:
                prev_lat = ut.shift(curr_lat, 1)
                prev_lon = ut.shift(curr_lon, 1)
                # compute distance from previous to current point
                df_.at[idx, dic_features_label['dist_to_prev']] = haversine(prev_lat, prev_lon, curr_lat, curr_lon)
                
                next_lat = ut.shift(curr_lat, -1)
                next_lon = ut.shift(curr_lon, -1)
                # compute distance to next point
                df_.at[idx, dic_features_label['dist_to_next']] = haversine(curr_lat, curr_lon, next_lat, next_lon)
                
                # using pandas shift in a large dataset: 7min 21s
                # using numpy shift above: 33.6 s

                # use distance from previous to next
                df_.at[idx, dic_features_label['dist_prev_to_next']] = haversine(prev_lat, prev_lon, next_lat, next_lon)
                
                sum_size_id += size_id
                curr_perc_int, est_time_str = ut.progress_update(sum_size_id, df_size, start_time, curr_perc_int, step_perc=20)
        df_.reset_index(inplace=True)
        print('...Reset index\n')
        print('..Total Time: {}'.format((time.time() - start_time)))
    except Exception as e:
        print('label_id:{}\nidx:{}\nsize_id:{}\nsum_size_id:{}'.format(label_id, idx, size_id, sum_size_id))
        raise e

def create_update_dist_time_speed_features(df_, label_id=dic_labels['id'], dic_labels=dic_labels, label_dtype = np.float64, sort=True):
    """
    Firstly, create three distance to an GPS point P (lat, lon)
    After, create two feature to time between two P: time to previous and time to next 
    Lastly, create two feature to speed using time and distance features
    Example:
        dist_to_prev =  248.33 meters, dist_to_prev 536.57 meters
        time_to_prev = 60 seconds, time_prev = 60.0 seconds
        speed_to_prev = 4.13 m/s, speed_prev = 8.94 m/s.
    """
    try:

        print('\nCreating or updating distance, time and speed features in meters by seconds\n') 
        start_time = time.time()

        if sort is True:
            print('...Sorting by {} and {} to increase performance\n'.format(label_id, dic_labels['datetime']))
            df_.sort_values([label_id, dic_labels['datetime']], inplace=True)
            #time_sort = time.time()
 
        if df_.index.name is None:
            print('...Set {} as index to a higher peformance\n'.format(label_id))
            df_.set_index(label_id, inplace=True)
           # time_index = time.time()

        """create new feature to time"""
        df_[dic_features_label['dist_to_prev']] = label_dtype(-1.0)

        """create new feature to time"""
        #df_[dic_features_label['time_to_next']] = label_dtype(-1.0)
        df_[dic_features_label['time_to_prev']] = label_dtype(-1.0)

        """create new feature to speed"""
        #df_[dic_features_label['speed_to_next']] = label_dtype(-1.0)
        df_[dic_features_label['speed_to_prev']] = label_dtype(-1.0)

        ids = df_.index.unique()
        df_size = df_.shape[0]
        curr_perc_int = -1
        sum_size_id = 0
        size_id = 0

        for idx in ids:
            curr_lat = df_.at[idx, dic_labels['lat']]
            curr_lon = df_.at[idx, dic_labels['lon']]

            size_id = curr_lat.size
            
            if size_id <= 1:
                print('...id:{}, must have at least 2 GPS points\n'.format(idx))
                df_.at[idx, dic_features_label['dist_to_prev']] = np.nan 
                df_.at[idx, dic_features_label['time_to_prev']] = np.nan
                df_.at[idx, dic_features_label['speed_to_prev']] = np.nan   
            else:
                prev_lat = ut.shift(curr_lat, 1)
                prev_lon = ut.shift(curr_lon, 1)
                # compute distance from previous to current point
                df_.at[idx, dic_features_label['dist_to_prev']] = haversine(prev_lat, prev_lon, curr_lat, curr_lon)
                
                
            #""" if data is numpy array, then it is a datetime object with size <= 1"""
            #if type(df_.at[idx, dic_labels['datetime']]) is not np.ndarray:
                #size_id = 1
                #print('...id:{}, must have at least 2 GPS points\n'.format(idx))
                #df_.at[idx, dic_features_label['time_to_prev']] = np.nan
                #df_.at[idx, dic_features_label['speed_to_prev']] = np.nan   
            #else:
                #"""time_to_prev = current_datetime - prev_datetime 
                #the time_delta must be in nanosecond, then we multiplie by 10-⁹ to tranform in seconds """
                #size_id = df_.at[idx, dic_labels['datetime']].size

                time_ = df_.at[idx, dic_labels['datetime']].astype(label_dtype)
                time_prev = (time_ - ut.shift(time_, 1))*(10**-9)
                df_.at[idx, dic_features_label['time_to_prev']] = time_prev

                """ set time_to_next"""
                #time_next = (ut.shift(time_, -1) - time_)*(10**-9)
                #df_.at[idx, dic_features_label['time_to_next']] = time_next
                
                "set Speed features"
                df_.at[idx, dic_features_label['speed_to_prev']] = df_.at[idx, dic_features_label['dist_to_prev']] / (time_prev)  # unit: m/s
                #df_.at[idx, dic_features_label['speed_to_next']] = df_.at[idx, dic_features_label['dist_to_next']] / (time_next)  # unit: m/s


                #ut.change_df_feature_values_using_filter(df_, id_, 'delta_time', filter_points, delta_times)
                #ut.change_df_feature_values_using_filter(df_, id_, 'delta_dist', filter_points, delta_dists)
                #ut.change_df_feature_values_using_filter(df_, id_, 'speed', filter_points, speeds)

                sum_size_id  += size_id
                curr_perc_int, est_time_str = ut.progress_update(sum_size_id , df_size, start_time, curr_perc_int, step_perc=20)
        print('...Reset index...\n')
        df_.reset_index(inplace=True)
        print('..Total Time: {:.3f}'.format((time.time() - start_time)))
    except Exception as e:
        print('label_id:{}\nidx:{}\nsize_id:{}\nsum_size_id:{}'.format(label_id, idx, size_id, sum_size_id ))
        raise e

def create_update_move_and_stop_by_radius(df_, radius=0, target_label='dist_to_prev', new_label=dic_features_label['situation']):
    
    try:
        print('\nCreating or updating features MOVE and STOPS...\n')
        conditions = (df_[target_label] > radius), (df_[target_label] <= radius)
        choices = ['move', 'stop']

        df_[new_label] = np.select(conditions, choices, np.nan)      
        print('\n....There are {} stops to this parameters\n'.format(df_[df_[new_label] == 'stop'].shape[0]))
    except Exception as e:
        raise e

def create_update_index_grid_feature(df_, dic_grid=None, dic_labels=dic_labels, label_dtype=np.int64, sort=True):
    print('\nCreating or updating index of the grid feature..\n')
    try:
        if dic_grid is not None:
            if sort:
                df_.sort_values([dic_labels['id'], dic_labels['datetime']], inplace=True)

            lat_, lon_ = gridutils.point_to_index_grid(df_[dic_labels['lat'] ], df_[dic_labels['lon'] ], dic_grid)
            df_[dic_features_label['index_grid_lat']] = label_dtype(lat_)
            df_[dic_features_label['index_grid_lon']] = label_dtype(lon_)   
        else:
            print('... inform a grid virtual dictionary\n')
    except Exception as e:
        raise e


"""----------------------  FUCTIONS TO DATA CLEANING   ----------------------------------- """ 


def clean_duplicates(df_, subset=None, keep='first', inplace=False, sort=True, return_idx=True):
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
        df_jumps = filter_jumpy
        rows_to_drop = idx.size

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
                clean_gps_nearby_points(df_, label_id, dic_labels, radius_area, label_dtype)
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

def clean_trajectories_with_few_points(df_, label_id=dic_features_label['tid'], dic_labels=dic_labels, min_points_per_trajectory=2, label_dtype=np.float64):

    create_update_dist_time_speed_features(df_, label_id, dic_labels, label_dtype)

    if df_.index.name is not None:
        print('\n...Reset index for filtering\n')
        df_.reset_index(inplace=True)

    df_count_tid = df_.groupby(by= label_id).size()
    
    tids_with_few_points = df_count_tid[ df_count_tid < min_points_per_trajectory ].index
    
    print('\n...there are {} ids with few points'.format(tids_with_few_points.shape[0]))
    
    shape_before_drop = df_.shape
    idx = df_[ df_[ dic_features_label['tid']].isin(tids_with_few_points) ].index
    if  idx.shape[0] > 0:
        print('\n...tids before drop: {}'.format(df_[ dic_features_label['tid']].unique().shape[0]))
        df_.drop(index=idx, inplace=True)
        print('\n...tids after drop: {}'.format(df_[ dic_features_label['tid']].unique().shape[0]))
        print('\n...shape - before drop: {} - after drop: {}'.format(shape_before_drop, df_.shape)) 

def clean_short_and_few_points_trajectories(df_,  label_id=dic_features_label['tid'], dic_labels=dic_labels, min_trajectory_distance=100, min_points_per_trajectory=2, label_dtype=np.float64):
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
        clean_short_and_few_points_trajectories(df_, dic_labels, min_trajectory_distance, min_points_per_trajectory, label_dtype)    

def split_trajectories(df_, label_id=dic_features_label['tid'], max_dist_between_adj_points=1000, max_time_between_adj_points=120000,
                      max_speed=25, label_new_id='tid_part'):
    """
    index_name is the current id.
    label_new_id is the new splitted id.
    time, dist, speeed features must be updated after split.
    """
        
    print('\nSplit trajectories')
    print('...max_time_between_adj_points:', max_time_between_adj_points)
    print('...max_dist_between_adj_points:', max_dist_between_adj_points)
    print('...max_speed:', max_speed)
    # remove speed between points higher than max_speed (m/s)
    # 50m/s = 180km/h
    # 30m/s = 108km/h
    # 25m/s =  90km/h

    if df_.index.name is None:
        print('...setting {} as index'.format(label_id))
        df_.set_index(label_id, inplace=True)

    # one vehicle id can generate many trajectory ids (tids).
    # we segment the vehicle trajectory into several trajectories.
    curr_tid = 0
    if label_new_id not in df_:
        df_[label_new_id] = curr_tid

    ids = df_.index.unique()

    count = 0
    size = df_.shape[0]
    curr_perc_int = -1
    start_time = time.time()

    for idx in ids:
        
        ## novo id_ deve gerar e alterar novo tid2 (curr_tid)
        curr_tid += 1
        
        filter_ = (df_.at[idx, dic_features_label['time_to_prev']] > max_time_between_adj_points) | \
                    (df_.at[idx, dic_features_label['dist_to_prev']] > max_dist_between_adj_points) | \
                    (df_.at[idx, dic_features_label['speed_to_prev']] > max_speed)        
        
        # remove speed between points higher than max_speed (m/s)
        # 50m/s = 180km/h
        # 30m/s = 108km/h
        # 25m/s =  90km/h        
                    
        if filter_.shape == ():
            # trajectories with only one point is useless for interpolation and so they must be removed.
            count += 1
            #print('problem', index_name, id_, label_new_id, curr_tid, label_delta_time, df_.at[id_, label_delta_time], \
            #      label_delta_dist, df_.at[id_, label_delta_dist], label_speed, df_.at[id_, label_speed] )
            df_.at[idx, label_new_id] = -1
            curr_tid += -1
        else:
            tids = np.empty(filter_.shape[0], dtype=np.int64)
            tids.fill(curr_tid)
            for i, has_problem in enumerate(filter_):
                if has_problem:
                    curr_tid += 1
                    tids[i+1:] = curr_tid
            count += tids.shape[0]
            df_.at[idx, label_new_id] = tids
        
        
        curr_perc_int, est_time_str = ut.progress_update(count, size, start_time, curr_perc_int, step_perc=20)

    if label_id == label_new_id:
        df_.reset_index(drop=True, inplace=True)
    else:
        df_.reset_index(inplace=True)

    shape_before_drop = df_.shape
    idx = df_[ df_[label_new_id] == -1 ].index
    if idx.shape[0] > 0:
        tids_before_drop = df_[dic_features_label['tid']].unique().shape[0]
        df_.drop(index=idx, inplace=True)
        print('#tids - before drop: {} - after drop: {}'.format(tids_before_drop, df_[dic_features_label['tid']].unique().shape[0]))
        print('shape - before drop: {} - after drop: {}'.format(shape_before_drop, df_.shape))
    else:
        print('no trajs with only one point - nothing to change:', df_.shape)

def split_trajectories_by_time(df_, label_id=dic_features_label['tid'], max_time_between_adj_points=900, label_new_id='tid_part'):
    """
    index_name is the current tid.
    label_new_id is the new splitted tid.
    delta_time must be updated after split.
    """
    print('\nSplit trajectories by max time between adj points:', max_time_between_adj_points)

    if df_.index.name is None:
        df_.set_index(label_id, inplace=True)

    # one vehicle id can generate many trajectory ids (tids).
    # we segment the vehicle trajectory into several trajectories.
    curr_tid = 0
    if label_new_id not in df_:
        df_[label_new_id] = curr_tid

    ids = df_.index.unique()

    count = 0
    size = df_.shape[0]
    curr_perc_int = -1
    start_time = time.time()

    for id_ in ids:
        ## novo id_ deve gerar e alterar novo tid2 (curr_tid)
        curr_tid += 1
        times = df_.at[id_, dic_features_label['time_to_prev']].astype(np.float64)
        if times.shape == ():
            # trajectories with only one point is useless for interpolation and so they must be removed.
            count += 1
            df_.at[id_, label_new_id] = -1
            curr_tid += -1
        else:
            delta_times = (ut.shift(times, -1) - times) / 1000.0

            filter_ = (delta_times > max_time_between_adj_points)

            tids = np.empty(filter_.shape[0], dtype=np.int64)
            tids.fill(curr_tid)
            for i, has_problem in enumerate(filter_):
                if has_problem:
                    curr_tid += 1
                    tids[i+1:] = curr_tid
            count += tids.shape[0]
            df_.at[id_, label_new_id] = tids
        
        curr_perc_int, est_time_str = ut.progress_update(count, size, start_time, curr_perc_int, step_perc=20)

    if label_id == label_new_id:
        df_.reset_index(drop=True, inplace=True)
    else:
        df_.reset_index(inplace=True)

    shape_before_drop = df_.shape
    idx = df_[ df_[label_new_id] == -1 ].index
    if idx.shape[0] > 0:
        tids_before_drop = df_[dic_features_label['tid']].unique().shape[0]
        df_.drop(index=idx, inplace=True)
        print('...before drop: {} - after drop: {}'.format(tids_before_drop, df_[label_new_id].unique().shape[0]))
        print('...shape - before drop: {} - after drop: {}'.format(shape_before_drop, df_.shape))
    else:
        print('...no trajs with only one point - nothing to change:', df_.shape)

""" transform speed """
def transform_speed_from_ms_to_kmh(df_, label_speed=dic_features_label['speed_to_prev'], new_label = None):
    try:
        df_[label_speed] = df_[label_speed].transform(lambda row: row*3.6)
        if new_label is not None:
            df_.rename(columns = {label_speed: new_label}, inplace=True) 
    except Exception as e: 
        raise e
   
def transform_speed_from_kmh_to_ms(df_, label_speed=dic_features_label['speed_to_prev'], new_label = None):
    try:
        df_[label_speed] = df_[label_speed].transform(lambda row: row/3.6)
        if new_label is not None:
            df_.rename(columns = {label_speed: new_label}, inplace=True) 
    except Exception as e: 
        raise e

""" transform distances """
def transform_dist_from_meters_to_kilometers(df_, label_distance=dic_features_label['dist_to_prev'], new_label=None):
    try:
        df_[label_distance] = df_[label_distance].transform(lambda row: row/1000)
        if new_label is not None:
            df_.rename(columns = {label_distance: new_label}, inplace=True) 
    except Exception as e: 
        raise e

def transform_dist_from_to_kilometers_to_meters(df_, label_distance=dic_features_label['dist_to_prev'], new_label=None):
    try:
        df_[label_distance] = df_[label_distance].transform(lambda row: row*1000)
        if new_label is not None:
            df_.rename(columns = {label_distance: new_label}, inplace=True) 
    except Exception as e: 
        raise e

""" transform time """
def transform_time_from_seconds_to_minutes(df_, label_time=dic_features_label['time_to_prev'], new_label=None):
    try:
        df_[label_time] = df_[label_time].transform(lambda row: row/60.0)
        if new_label is not None:
            df_.rename(columns = {label_time: new_label}, inplace=True) 
    except Exception as e: 
        raise e 

def transform_time_from_minute_to_seconds(df_, label_time=dic_features_label['time_to_prev'], new_label=None):
    try:
        df_[label_time] = df_[label_time].apply(lambda row: row*60.0)
        if new_label is not None:
            df_.rename(columns = {label_time: new_label}, inplace=True) 
    except Exception as e: 
        raise e 

def transform_time_from_minute_to_hours(df_, label_time=dic_features_label['time_to_prev'], new_label=None):
    try:
        df_[label_time] = df_[label_time].apply(lambda row: row/60.0)
        if new_label is not None:
            df_.rename(columns = {label_time: new_label}, inplace=True) 
    except Exception as e: 
        raise e  

def transform_time_from_hours_to_minute(df_, label_time=dic_features_label['time_to_prev'], new_label=None):
    try:
        df_[label_time] = df_[label_time].apply(lambda row: row*60.0)
        if new_label is not None:
            df_.rename(columns = {label_time: new_label}, inplace=True) 
    except Exception as e:
        raise e

def transform_time_from_seconds_to_hours(df_, label_time=dic_features_label['time_to_prev'], new_label=None):
    try:
        df_[label_time] = df_[label_time].apply(lambda row: row/3600.0)
        if new_label is not None:
            df_.rename(columns = {label_time: new_label}, inplace=True) 
    except Exception as e:
        raise e

def transform_time_from_hours_to_seconds(df_, label_time=dic_features_label['time_to_prev'], new_label=None):
    try:
        df_[label_time] = df_[label_time].apply(lambda row: row*3600.0)
        if new_label is not None:
            df_.rename(columns = {label_time: new_label}, inplace=True) 
    except Exception as e:
        raise e


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
