# TODO: Andreza
import numpy as np
import pandas as pd
import time
from scipy.interpolate import interp1d
from pymove.utils.utils import dic_labels, dic_features_label

""" ----------------------  FUCTIONS TO LAT AND LONG COORDINATES --------------------------- """ 

def lon2XSpherical(lon):
    """
    Convert longitude to to X EPSG:3857 WGS 84 / Pseudo-Mercator 

    Parameters
    ----------
    lon : float
        Represents longitude.

    Returns
    -------
    xspherical : float
        X offset from your original position in meters.
     
    Examples
    --------
    >>> from pymove.utils.transformations import lon2XSpherical
    >>> lon2XSpherical(-38.501597 )
    -4285978.17

    References
    ----------
    https://epsg.io/transform

    """
    xspherical = 6378137 * np.radians(lon)
    return xspherical

def lat2YSpherical(lat):
    """
    Convert latitude to Y EPSG:3857 WGS 84 / Pseudo-Mercator 

    Parameters
    ----------
    lat : float
        Represents latitude.

    Returns
    -------
    yspherical : float
        Y offset from your original position in meters.
     
    Examples
    --------
    >>> from pymove.utils.transformations import lat2YSpherical
    >>> lat2YSpherical(-3.797864)
    -423086.2213610324

    References
    ----------
    https://epsg.io/transform

    """
    yspherical = 6378137 * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2.0))
    return yspherical

def x2LonSpherical(x):
    """
    Convert X EPSG:3857 WGS 84 / Pseudo-Mercator to longitude.  

    Parameters
    ----------
    x : float
        X offset from your original position in meters.

    Returns
    -------
    lon : float
        Represents longitude.
     
    Examples
    --------
    >>> from pymove.utils.transformations import x2LonSpherical
    >>> x2LonSpherical(-4285978.17)
    -38.501597

    References
    ----------
    https://epsg.io/transform

    """
    lon = np.degrees(x / 6378137.0)
    return lon

def y2LatSpherical(y):
    """
    Convert Y EPSG:3857 WGS 84 / Pseudo-Mercator to latitude.

    Parameters
    ----------
    y : float
        Y offset from your original position in meters.

    Returns
    -------
    lat : float
        Represents latitude.
     
    Examples
    --------
    >>> from pymove.utils.transformations import y2LatSpherical
    >>> y2LatSpherical(-423086.22)
    -3.797864 

    References
    ----------
    https://epsg.io/transform

    """
    lat = np.degrees(np.arctan(np.sinh(y / 6378137.0)))
    return lat

def haversine(lat1, lon1, lat2, lon2, to_radians=True, earth_radius=6371):
    """
    Calculate the great circle distance between two points on the earth (specified in decimal degrees or in radians).
    All (lat, lon) coordinates must have numeric dtypes and be of equal length.
    Result in meters. Use 3956 in earth radius for miles

    Parameters
    ----------
    lat1 : float
        Y offset from your original position in meters.
    
    lon1 : float
        Y offset from your original position in meters.

    lat2 : float
        Y offset from your original position in meters.

    lon2 : float
        Y offset from your original position in meters.

    to_radians : boolean
        Y offset from your original position in meters.

    earth_radius : int
        Y offset from your original position in meters.

    Returns
    -------
    lat : float
        Represents latitude.
     
    Examples
    --------
    >>> from pymove.utils.transformations import haversine
    >>> haversine(-423086.22)
    -3.797864 

    References
    ----------
    Vectorized haversine function: https://stackoverflow.com/questions/43577086/pandas-calculate-haversine-distance-within-each-group-of-rows
    About distance between two points: https://janakiev.com/blog/gps-points-distance-python/

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

# """ ----------------------  FUCTIONS TO CREATE NEW FEATURES BASED ON DATATIME  ----------------------------- """

# def create_update_tid_based_on_id_datatime(df_, dic_labels=dic_labels, str_format="%Y%m%d%H", sort=True):
#     """
#         Create or update trajectory id  
#             Exampĺe: ID = M00001 and datetime = 2019-04-28 00:00:56  -> tid = M000012019042800
#     """
#     try:
#         print('\nCreating or updating tid feature...\n')
#         if sort is True:
#             print('...Sorting by {} and {} to increase performance\n'.format(dic_labels['id'], dic_labels['datetime']))
#             df_.sort_values([dic_labels['id'], dic_labels['datetime']], inplace=True)


#         df_[dic_features_label['tid']] = df_[dic_labels['id']].astype(str) + df_[dic_labels['datetime']].dt.strftime(str_format)  
#         #%.dt.date.astype(str)
        
#         print('\n...tid feature was created...\n')      
#     except Exception as e:
#         raise e

# def create_update_date_features(df_, dic_labels=dic_labels):
#     try:
#         print('Creating date features...')
#         if dic_labels['datetime'] in df_:
#             df_['date'] = df_[dic_labels['datetime']].dt.date
#             print('..Date features was created...\n')
#     except Exception as e:
#         raise e
    
# def create_update_hour_features(df_, dic_labels=dic_labels):    
#     try:
#         print('\nCreating or updating a feature for hour...\n')
#         if dic_labels['datetime'] in df_:
#             df_['hour'] = df_[dic_labels['datetime']].dt.hour
#             print('...Hour feature was created...\n')
#     except Exception as e:
#         raise e

# def create_update_day_of_the_week_features(df_, dic_labels=dic_labels):
#     """
#         Create or update a feature day of the week from datatime
#             Exampĺe: datetime = 2019-04-28 00:00:56  -> day = Sunday
#     """
#     try:
#         print('\nCreating or updating day of the week feature...\n')
#         df_[dic_features_label['day']] = df_[dic_labels['datetime']].dt.day_name()
#         print('...the day of the week feature was created...\n')
#     except Exception as e:
#         raise e

# def create_update_time_of_day_features(df_, dic_labels=dic_labels):
#     """
#         Create a feature time of day or period from datatime
#             Examples: 
#                 datetime1 = 2019-04-28 02:00:56 -> period = early morning
#                 datetime2 = 2019-04-28 08:00:56 -> period = morning
#                 datetime3 = 2019-04-28 14:00:56 -> period = afternoon
#                 datetime4 = 2019-04-28 20:00:56 -> period = evening
#     """
#     try:
#         print('\nCreating or updating period feature\n...early morning from 0H to 6H\n...morning from 6H to 12H\n...afternoon from 12H to 18H\n...evening from 18H to 24H')
#         conditions =   [(df_[dic_labels['datetime']].dt.hour >= 0) & (df_[dic_labels['datetime']].dt.hour < 6), 
#                         (df_[dic_labels['datetime']].dt.hour >= 6) & (df_[dic_labels['datetime']].dt.hour < 12),
#                         (df_[dic_labels['datetime']].dt.hour >= 12) & (df_[dic_labels['datetime']].dt.hour < 18),  
#                         (df_[dic_labels['datetime']].dt.hour >= 18) & (df_[dic_labels['datetime']].dt.hour < 24)]
#         choices = ['early morning', 'morning', 'afternoon', 'evening']
#         df_[dic_features_label['period']] = np.select(conditions, choices, 'undefined')      
#         print('...the period of day feature was created')
#     except Exception as e:
#         raise e

# """ ----------------------  FUCTIONS TO CREATE NEW FEATURES BASED ON LAT AND LON COORDINATES  ----------------------------- """
# def create_update_dist_features(df_, label_id=dic_labels['id'], dic_labels=dic_labels, label_dtype = np.float64, sort=True):
#     """
#         Create three distance in meters to an GPS point P (lat, lon)
#             Example:
#                 P to P.next = 2 meters
#                 P to P.previous = 1 meter
#                 P.previous to P.next = 1 meters
#     """
#     try:
#         print('\nCreating or updating distance features in meters...\n')
#         start_time = time.time()

#         if sort is True:
#             print('...Sorting by {} and {} to increase performance\n'.format(label_id, dic_labels['datetime']))
#             df_.sort_values([label_id, dic_labels['datetime']], inplace=True)

#         if df_.index.name is None:
#             print('...Set {} as index to increase attribution performance\n'.format(label_id))
#             df_.set_index(label_id, inplace=True)
        
#         """ create ou update columns"""
#         df_[dic_features_label['dist_to_prev']] = label_dtype(-1.0)
#         df_[dic_features_label['dist_to_next']] = label_dtype(-1.0)
#         df_[dic_features_label['dist_prev_to_next']]= label_dtype(-1.0)

#         ids = df_.index.unique()
#         df_size = df_.shape[0]
#         curr_perc_int = -1
#         start_time = time.time()
#         deltatime_str = ''
#         sum_size_id = 0
#         size_id = 0
#         for idx in ids:
#             curr_lat = df_.at[idx, dic_labels['lat']]
#             curr_lon = df_.at[idx, dic_labels['lon']]

#             size_id = curr_lat.size
            
#             if size_id <= 1:
#                 print('...id:{}, must have at least 2 GPS points\n'.format(idx))
#                 #df_.at[idx, dic_features_label['dist_to_next']] = np.nan
#                 df_.at[idx, dic_features_label['dist_to_prev']] = np.nan
#                 #df_.at[idx, dic_features_label['dist_prev_to_next']] = np.nan    
                
#             else:
#                 prev_lat = ut.shift(curr_lat, 1)
#                 prev_lon = ut.shift(curr_lon, 1)
#                 # compute distance from previous to current point
#                 df_.at[idx, dic_features_label['dist_to_prev']] = haversine(prev_lat, prev_lon, curr_lat, curr_lon)
                
#                 next_lat = ut.shift(curr_lat, -1)
#                 next_lon = ut.shift(curr_lon, -1)
#                 # compute distance to next point
#                 df_.at[idx, dic_features_label['dist_to_next']] = haversine(curr_lat, curr_lon, next_lat, next_lon)
                
#                 # using pandas shift in a large dataset: 7min 21s
#                 # using numpy shift above: 33.6 s

#                 # use distance from previous to next
#                 df_.at[idx, dic_features_label['dist_prev_to_next']] = haversine(prev_lat, prev_lon, next_lat, next_lon)
                
#                 sum_size_id += size_id
#                 curr_perc_int, est_time_str = ut.progress_update(sum_size_id, df_size, start_time, curr_perc_int, step_perc=20)
#         df_.reset_index(inplace=True)
#         print('...Reset index\n')
#         print('..Total Time: {}'.format((time.time() - start_time)))
#     except Exception as e:
#         print('label_id:{}\nidx:{}\nsize_id:{}\nsum_size_id:{}'.format(label_id, idx, size_id, sum_size_id))
#         raise e

# def create_update_dist_time_speed_features(df_, label_id=dic_labels['id'], dic_labels=dic_labels, label_dtype = np.float64, sort=True):
#     """
#     Firstly, create three distance to an GPS point P (lat, lon)
#     After, create two feature to time between two P: time to previous and time to next 
#     Lastly, create two feature to speed using time and distance features
#     Example:
#         dist_to_prev =  248.33 meters, dist_to_prev 536.57 meters
#         time_to_prev = 60 seconds, time_prev = 60.0 seconds
#         speed_to_prev = 4.13 m/s, speed_prev = 8.94 m/s.
#     """
#     try:

#         print('\nCreating or updating distance, time and speed features in meters by seconds\n') 
#         start_time = time.time()

#         if sort is True:
#             print('...Sorting by {} and {} to increase performance\n'.format(label_id, dic_labels['datetime']))
#             df_.sort_values([label_id, dic_labels['datetime']], inplace=True)
#             #time_sort = time.time()
 
#         if df_.index.name is None:
#             print('...Set {} as index to a higher peformance\n'.format(label_id))
#             df_.set_index(label_id, inplace=True)
#            # time_index = time.time()

#         """create new feature to time"""
#         df_[dic_features_label['dist_to_prev']] = label_dtype(-1.0)

#         """create new feature to time"""
#         #df_[dic_features_label['time_to_next']] = label_dtype(-1.0)
#         df_[dic_features_label['time_to_prev']] = label_dtype(-1.0)

#         """create new feature to speed"""
#         #df_[dic_features_label['speed_to_next']] = label_dtype(-1.0)
#         df_[dic_features_label['speed_to_prev']] = label_dtype(-1.0)

#         ids = df_.index.unique()
#         df_size = df_.shape[0]
#         curr_perc_int = -1
#         sum_size_id = 0
#         size_id = 0

#         for idx in ids:
#             curr_lat = df_.at[idx, dic_labels['lat']]
#             curr_lon = df_.at[idx, dic_labels['lon']]

#             size_id = curr_lat.size
            
#             if size_id <= 1:
#                 print('...id:{}, must have at least 2 GPS points\n'.format(idx))
#                 df_.at[idx, dic_features_label['dist_to_prev']] = np.nan 
#                 df_.at[idx, dic_features_label['time_to_prev']] = np.nan
#                 df_.at[idx, dic_features_label['speed_to_prev']] = np.nan   
#             else:
#                 prev_lat = ut.shift(curr_lat, 1)
#                 prev_lon = ut.shift(curr_lon, 1)
#                 # compute distance from previous to current point
#                 df_.at[idx, dic_features_label['dist_to_prev']] = haversine(prev_lat, prev_lon, curr_lat, curr_lon)
                
                
#             #""" if data is numpy array, then it is a datetime object with size <= 1"""
#             #if type(df_.at[idx, dic_labels['datetime']]) is not np.ndarray:
#                 #size_id = 1
#                 #print('...id:{}, must have at least 2 GPS points\n'.format(idx))
#                 #df_.at[idx, dic_features_label['time_to_prev']] = np.nan
#                 #df_.at[idx, dic_features_label['speed_to_prev']] = np.nan   
#             #else:
#                 #"""time_to_prev = current_datetime - prev_datetime 
#                 #the time_delta must be in nanosecond, then we multiplie by 10-⁹ to tranform in seconds """
#                 #size_id = df_.at[idx, dic_labels['datetime']].size

#                 time_ = df_.at[idx, dic_labels['datetime']].astype(label_dtype)
#                 time_prev = (time_ - ut.shift(time_, 1))*(10**-9)
#                 df_.at[idx, dic_features_label['time_to_prev']] = time_prev

#                 """ set time_to_next"""
#                 #time_next = (ut.shift(time_, -1) - time_)*(10**-9)
#                 #df_.at[idx, dic_features_label['time_to_next']] = time_next
                
#                 "set Speed features"
#                 df_.at[idx, dic_features_label['speed_to_prev']] = df_.at[idx, dic_features_label['dist_to_prev']] / (time_prev)  # unit: m/s
#                 #df_.at[idx, dic_features_label['speed_to_next']] = df_.at[idx, dic_features_label['dist_to_next']] / (time_next)  # unit: m/s


#                 #ut.change_df_feature_values_using_filter(df_, id_, 'delta_time', filter_points, delta_times)
#                 #ut.change_df_feature_values_using_filter(df_, id_, 'delta_dist', filter_points, delta_dists)
#                 #ut.change_df_feature_values_using_filter(df_, id_, 'speed', filter_points, speeds)

#                 sum_size_id  += size_id
#                 curr_perc_int, est_time_str = ut.progress_update(sum_size_id , df_size, start_time, curr_perc_int, step_perc=20)
#         print('...Reset index...\n')
#         df_.reset_index(inplace=True)
#         print('..Total Time: {:.3f}'.format((time.time() - start_time)))
#     except Exception as e:
#         print('label_id:{}\nidx:{}\nsize_id:{}\nsum_size_id:{}'.format(label_id, idx, size_id, sum_size_id ))
#         raise e

# def create_update_move_and_stop_by_radius(df_, radius=0, target_label='dist_to_prev', new_label=dic_features_label['situation']):
    
#     try:
#         print('\nCreating or updating features MOVE and STOPS...\n')
#         conditions = (df_[target_label] > radius), (df_[target_label] <= radius)
#         choices = ['move', 'stop']

#         df_[new_label] = np.select(conditions, choices, np.nan)      
#         print('\n....There are {} stops to this parameters\n'.format(df_[df_[new_label] == 'stop'].shape[0]))
#     except Exception as e:
#         raise e

# """ transform speed """
# def transform_speed_from_ms_to_kmh(df_, label_speed=dic_features_label['speed_to_prev'], new_label = None):
#     try:
#         df_[label_speed] = df_[label_speed].transform(lambda row: row*3.6)
#         if new_label is not None:
#             df_.rename(columns = {label_speed: new_label}, inplace=True) 
#     except Exception as e: 
#         raise e
   
# def transform_speed_from_kmh_to_ms(df_, label_speed=dic_features_label['speed_to_prev'], new_label = None):
#     try:
#         df_[label_speed] = df_[label_speed].transform(lambda row: row/3.6)
#         if new_label is not None:
#             df_.rename(columns = {label_speed: new_label}, inplace=True) 
#     except Exception as e: 
#         raise e

# """ transform distances """
# def transform_dist_from_meters_to_kilometers(df_, label_distance=dic_features_label['dist_to_prev'], new_label=None):
#     try:
#         df_[label_distance] = df_[label_distance].transform(lambda row: row/1000)
#         if new_label is not None:
#             df_.rename(columns = {label_distance: new_label}, inplace=True) 
#     except Exception as e: 
#         raise e

# def transform_dist_from_to_kilometers_to_meters(df_, label_distance=dic_features_label['dist_to_prev'], new_label=None):
#     try:
#         df_[label_distance] = df_[label_distance].transform(lambda row: row*1000)
#         if new_label is not None:
#             df_.rename(columns = {label_distance: new_label}, inplace=True) 
#     except Exception as e: 
#         raise e

# """ transform time """
# def transform_time_from_seconds_to_minutes(df_, label_time=dic_features_label['time_to_prev'], new_label=None):
#     try:
#         df_[label_time] = df_[label_time].transform(lambda row: row/60.0)
#         if new_label is not None:
#             df_.rename(columns = {label_time: new_label}, inplace=True) 
#     except Exception as e: 
#         raise e 

# def transform_time_from_minute_to_seconds(df_, label_time=dic_features_label['time_to_prev'], new_label=None):
#     try:
#         df_[label_time] = df_[label_time].apply(lambda row: row*60.0)
#         if new_label is not None:
#             df_.rename(columns = {label_time: new_label}, inplace=True) 
#     except Exception as e: 
#         raise e 

# def transform_time_from_minute_to_hours(df_, label_time=dic_features_label['time_to_prev'], new_label=None):
#     try:
#         df_[label_time] = df_[label_time].apply(lambda row: row/60.0)
#         if new_label is not None:
#             df_.rename(columns = {label_time: new_label}, inplace=True) 
#     except Exception as e: 
#         raise e  

# def transform_time_from_hours_to_minute(df_, label_time=dic_features_label['time_to_prev'], new_label=None):
#     try:
#         df_[label_time] = df_[label_time].apply(lambda row: row*60.0)
#         if new_label is not None:
#             df_.rename(columns = {label_time: new_label}, inplace=True) 
#     except Exception as e:
#         raise e

# def transform_time_from_seconds_to_hours(df_, label_time=dic_features_label['time_to_prev'], new_label=None):
#     try:
#         df_[label_time] = df_[label_time].apply(lambda row: row/3600.0)
#         if new_label is not None:
#             df_.rename(columns = {label_time: new_label}, inplace=True) 
#     except Exception as e:
#         raise e

# def transform_time_from_hours_to_seconds(df_, label_time=dic_features_label['time_to_prev'], new_label=None):
#     try:
#         df_[label_time] = df_[label_time].apply(lambda row: row*3600.0)
#         if new_label is not None:
#             df_.rename(columns = {label_time: new_label}, inplace=True) 
#     except Exception as e:
#         raise e

