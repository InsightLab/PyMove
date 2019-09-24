# TODO: Andreza
import numpy as np
import pandas as pd
import time
from scipy.interpolate import interp1d
from pymove.utils.traj_utils import shift, progress_update
from pymove.utils.constants import LATITUDE, LONGITUDE, DATETIME, TRAJ_ID, TID, PERIOD, DATE, HOUR, DAY, DIST_PREV_TO_NEXT, DIST_TO_PREV, SITUATION


""" ----------------------  FUCTIONS TO LAT AND LONG COORDINATES --------------------------- """ 

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

""" ----------------------  FUCTIONS TO CREATE NEW FEATURES BASED ON DATATIME  ----------------------------- """
#TODO botar o check pra replace
#TODO trocar nome da func
def create_update_tid_based_on_id_datatime(df_,  str_format="%Y%m%d%H", sort=True):
    """
    Create or update trajectory id based on id e datetime.  

    Parameters
    ----------
    df_ : pandas.core.frame.DataFrame
        Represents the dataset with contains lat, long and datetime.

    str_format : String
        Contains informations about virtual grid, how
            - lon_min_x: longitude mínima.
            - lat_min_y: latitude miníma. 
            - grid_size_lat_y: tamanho da grid latitude. 
            - grid_size_lon_x: tamanho da longitude da grid.
            - cell_size_by_degree: tamanho da célula da Grid.
        If value is none, the function ask user by dic_grid.

    sort : boolean
        Represents the state of dataframe, if is sorted. By default it's true.

    Returns
    -------
    

    Examples
    --------
    ID = M00001 and datetime = 2019-04-28 00:00:56  -> tid = M000012019042800

    >>> from pymove.utils.transformations import create_update_tid_based_on_id_datatime
    >>> create_update_tid_based_on_id_datatime(df)

    """
    try:
        print('\nCreating or updating tid feature...\n')
        if sort is True:
            print('...Sorting by {} and {} to increase performance\n'.format(TRAJ_ID, DATETIME))
            df_.sort_values([TRAJ_ID, DATETIME], inplace=True)

        df_[TID] = df_[TRAJ_ID].astype(str) + df_[DATETIME].dt.strftime(str_format)  
        print('\n...tid feature was created...\n')   

    except Exception as e:
        raise e

#TODO complementar oq ela faz
#TODO botar o check pra replace
#TODO trocar nome da func
def create_update_date_features(df_):
    """
    Create or update date feature.  

    Parameters
    ----------
    df_ : pandas.core.frame.DataFrame
        Represents the dataset with contains lat, long and datetime.

    Returns
    -------
    

    Examples
    --------
    

    >>> from pymove.utils.transformations import create_update_date_features
    >>> create_update_date_features(df)

    """
    try:
        print('Creating date features...')
        if DATETIME in df_:
            df_['date'] = df_[DATETIME].dt.date
            print('..Date features was created...\n')
    except Exception as e:
        raise e
    
#TODO complementar oq ela faz
#TODO botar o check pra replace
#TODO trocar nome da func
def create_update_hour_features(df_):    
    """
    Create or update hour feature.  

    Parameters
    ----------
    df_ : pandas.core.frame.DataFrame
        Represents the dataset with contains lat, long and datetime.

    Returns
    -------
    

    Examples
    --------
    

    >>> from pymove.utils.transformations import create_update_hour_features
    >>> create_update_date_features(df)

    """
    try:
        print('\nCreating or updating a feature for hour...\n')
        if DATETIME in df_:
            df_['hour'] = df_[DATETIME].dt.hour
            print('...Hour feature was created...\n')
    except Exception as e:
        raise e


#TODO botar o check pra replace
#TODO trocar nome da func
def create_update_day_of_the_week_features(df_):
    """
    Create or update a feature day of the week from datatime.  

    Parameters
    ----------
    df_ : pandas.core.frame.DataFrame
        Represents the dataset with contains lat, long and datetime.

    Returns
    -------
    

    Examples
    --------
    Exampĺe: datetime = 2019-04-28 00:00:56  -> day = Sunday

    >>> from pymove.utils.transformations import create_update_day_of_the_week_features
    >>> create_update_day_of_the_week_features(df)

    """
    try:
        print('\nCreating or updating day of the week feature...\n')
        df_[DAY] = df_[DATETIME].dt.day_name()
        print('...the day of the week feature was created...\n')
    except Exception as e:
        raise e


#TODO botar o check pra replace
#TODO trocar nome da func
def create_update_time_of_day_features(df_):
    """
    Create a feature time of day or period from datatime.

    Parameters
    ----------
    df_ : pandas.core.frame.DataFrame
        Represents the dataset with contains lat, long and datetime.

    Returns
    -------
    

    Examples
    --------
    - datetime1 = 2019-04-28 02:00:56 -> period = early morning
    - datetime2 = 2019-04-28 08:00:56 -> period = morning
    - datetime3 = 2019-04-28 14:00:56 -> period = afternoon
    - datetime4 = 2019-04-28 20:00:56 -> period = evening

    >>> from pymove.utils.transformations import create_update_time_of_day_features
    >>> create_update_time_of_day_features(df)

    """
    try:
        print('\nCreating or updating period feature\n...early morning from 0H to 6H\n...morning from 6H to 12H\n...afternoon from 12H to 18H\n...evening from 18H to 24H')
        conditions =   [(df_[DATETIME].dt.hour >= 0) & (df_[DATETIME].dt.hour < 6), 
                        (df_[DATETIME].dt.hour >= 6) & (df_[DATETIME].dt.hour < 12),
                        (df_[DATETIME].dt.hour >= 12) & (df_[DATETIME].dt.hour < 18),  
                        (df_[DATETIME].dt.hour >= 18) & (df_[DATETIME].dt.hour < 24)]
        choices = ['early morning', 'morning', 'afternoon', 'evening']
        df_[PERIOD] = np.select(conditions, choices, 'undefined')      
        print('...the period of day feature was created')
    except Exception as e:
        raise e

""" ----------------------  FUCTIONS TO CREATE NEW FEATURES BASED ON LAT AND LON COORDINATES  ----------------------------- """
#TODO complementar oq ela faz
#TODO botar o check pra replace
#TODO trocar nome da func
def create_update_dist_features(df_, label_id=TRAJ_ID, label_dtype = np.float64, sort=True):
    """
     Create three distance in meters to an GPS point P (lat, lon).

    Parameters
    ----------
    df_ : pandas.core.frame.DataFrame
        Represents the dataset with contains lat, long and datetime.

    label_id : String
        Represents name of column of trajectore's id. By default it's 'id'.

    label_dtype : String
        Represents column id type. By default it's np.float64.

    sort : boolean
        Represents the state of dataframe, if is sorted. By default it's true.

    Returns
    -------
    

    Examples
    --------
    Example:    P to P.next = 2 meters
                P to P.previous = 1 meter
                P.previous to P.next = 1 meters

    >>> from pymove.utils.transformations import create_update_dist_features
    >>> create_update_dist_features(df)

    """
    try:
        print('\nCreating or updating distance features in meters...\n')
        start_time = time.time()

        if sort is True:
            print('...Sorting by {} and {} to increase performance\n'.format(label_id, DATETIME))
            df_.sort_values([label_id, DATETIME], inplace=True)

        if df_.index.name is None:
            print('...Set {} as index to increase attribution performance\n'.format(label_id))
            df_.set_index(label_id, inplace=True)
        
        """ create ou update columns"""
        df_[DIST_TO_PREV] = label_dtype(-1.0)
        df_[DIST_TO_NEXT] = label_dtype(-1.0)
        df_[DIST_PREV_TO_NEXT]= label_dtype(-1.0)

        ids = df_.index.unique()
        df_size = df_.shape[0]
        curr_perc_int = -1
        start_time = time.time()
        deltatime_str = ''
        sum_size_id = 0
        size_id = 0
        for idx in ids:
            curr_lat = df_.at[idx, LATITUDE]
            curr_lon = df_.at[idx, LONGITUDE]

            size_id = curr_lat.size
            
            if size_id <= 1:
                print('...id:{}, must have at least 2 GPS points\n'.format(idx))
                df_.at[idx, DIST_TO_PREV] = np.nan  
                
            else:
                prev_lat = shift(curr_lat, 1)
                prev_lon = shift(curr_lon, 1)
                # compute distance from previous to current point
                df_.at[idx, DIST_TO_PREV] = haversine(prev_lat, prev_lon, curr_lat, curr_lon)
                
                next_lat = shift(curr_lat, -1)
                next_lon = shift(curr_lon, -1)
                # compute distance to next point
                df_.at[idx, DIST_TO_NEXT] = haversine(curr_lat, curr_lon, next_lat, next_lon)
                
                # using pandas shift in a large dataset: 7min 21s
                # using numpy shift above: 33.6 s

                # use distance from previous to next
                df_.at[idx, DIST_PREV_TO_NEXT] = haversine(prev_lat, prev_lon, next_lat, next_lon)
                
                sum_size_id += size_id
                curr_perc_int, est_time_str = progress_update(sum_size_id, df_size, start_time, curr_perc_int, step_perc=20)
        df_.reset_index(inplace=True)
        print('...Reset index\n')
        print('..Total Time: {}'.format((time.time() - start_time)))
    except Exception as e:
        print('label_id:{}\nidx:{}\nsize_id:{}\nsum_size_id:{}'.format(label_id, idx, size_id, sum_size_id))
        raise e

#TODO botar o check pra replace
#TODO trocar nome da func
def create_update_dist_time_speed_features(df_, label_id=TRAJ_ID,  label_dtype = np.float64, sort=True):
    """
    Firstly, create three distance to an GPS point P (lat, lon)
    After, create two feature to time between two P: time to previous and time to next 
    Lastly, create two feature to speed using time and distance features

    Parameters
    ----------
    df_ : pandas.core.frame.DataFrame
        Represents the dataset with contains lat, long and datetime.

    label_id : String
        Represents name of column of trajectore's id. By default it's 'id'.

    label_dtype : String
        Represents column id type. By default it's np.float64.

    sort : boolean
        Represents the state of dataframe, if is sorted. By default it's true.

    Returns
    -------
    

    Examples
    --------
    Example:    dist_to_prev =  248.33 meters, dist_to_prev 536.57 meters
                time_to_prev = 60 seconds, time_prev = 60.0 seconds
                speed_to_prev = 4.13 m/s, speed_prev = 8.94 m/s.

    >>> from pymove.utils.transformations import create_update_dist_time_speed_features
    >>> create_update_dist_time_speed_features(df)

    """
    try:

        print('\nCreating or updating distance, time and speed features in meters by seconds\n') 
        start_time = time.time()

        if sort is True:
            print('...Sorting by {} and {} to increase performance\n'.format(label_id, DATETIME))
            df_.sort_values([label_id, DATETIME], inplace=True)
 
        if df_.index.name is None:
            print('...Set {} as index to a higher peformance\n'.format(label_id))
            df_.set_index(label_id, inplace=True)

        """create new feature to time"""
        df_[DIST_TO_PREV] = label_dtype(-1.0)

        """create new feature to time"""
        df_[TIME_TO_PREV] = label_dtype(-1.0)

        """create new feature to speed"""
        df_[SPEED_TO_PREV] = label_dtype(-1.0)

        ids = df_.index.unique()
        df_size = df_.shape[0]
        curr_perc_int = -1
        sum_size_id = 0
        size_id = 0

        for idx in ids:
            curr_lat = df_.at[idx, LATITUDE]
            curr_lon = df_.at[idx, LONGITUDE]

            size_id = curr_lat.size
            
            if size_id <= 1:
                print('...id:{}, must have at least 2 GPS points\n'.format(idx))
                df_.at[idx, DIST_TO_PREV] = np.nan 
                df_.at[idx, TIME_TO_PREV] = np.nan
                df_.at[idx, SPEED_TO_PREV] = np.nan   
            else:
                prev_lat = shift(curr_lat, 1)
                prev_lon = shift(curr_lon, 1)
                prev_lon = shift(curr_lon, 1)
                # compute distance from previous to current point
                df_.at[idx, DIST_TO_PREV] = haversine(prev_lat, prev_lon, curr_lat, curr_lon)
                
                time_ = df_.at[idx, DATETIME].astype(label_dtype)
                time_prev = (time_ - shift(time_, 1))*(10**-9)
                df_.at[idx, TIME_TO_PREV] = time_prev

                """ set time_to_next"""
                #time_next = (ut.shift(time_, -1) - time_)*(10**-9)
                #df_.at[idx, dic_features_label['time_to_next']] = time_next
                
                "set Speed features"
                df_.at[idx, SPEED_TO_PREV] = df_.at[idx, DIST_TO_PREV] / (time_prev)  # unit: m/s
            
                sum_size_id  += size_id
                curr_perc_int, est_time_str = progress_update(sum_size_id , df_size, start_time, curr_perc_int, step_perc=20)
        print('...Reset index...\n')
        df_.reset_index(inplace=True)
        print('..Total Time: {:.3f}'.format((time.time() - start_time)))
    except Exception as e:
        print('label_id:{}\nidx:{}\nsize_id:{}\nsum_size_id:{}'.format(label_id, idx, size_id, sum_size_id ))
        raise e

#TODO complementar oq ela faz
#TODO botar o check pra replace
#TODO trocar nome da func
def create_update_move_and_stop_by_radius(df_, radius=0, target_label=DIST_TO_PREV, new_label=SITUATION):
    """
    ?
    Create or update move and stop by radius.

    Parameters
    ----------
    df_ : pandas.core.frame.DataFrame
        Represents the dataset with contains lat, long and datetime.

    radius : float
        Represents the radius.

    target_label : String
        ?. By default it's "dist_to_prev".

    new_label : String
        ?. By default it's "situation".

    Returns
    -------
    

    Examples
    --------
    -

    >>> from pymove.utils.transformations import create_update_move_and_stop_by_radius
    >>> create_update_move_and_stop_by_radius(df)

    """
    try:
        print('\nCreating or updating features MOVE and STOPS...\n')
        conditions = (df_[target_label] > radius), (df_[target_label] <= radius)
        choices = ['move', 'stop']

        df_[new_label] = np.select(conditions, choices, np.nan)      
        print('\n....There are {} stops to this parameters\n'.format(df_[df_[new_label] == 'stop'].shape[0]))
    except Exception as e:
        raise e

#TODO complementar oq ela faz
#TODO botar o check pra replace
#TODO trocar nome da func
def change_df_feature_values_using_filter(df, id_, feature_name, filter_, values):
    """
    ?
    equivalent of: df.at[id_, feature_name][filter_] = values
    e.g. df.at[tid, 'time'][filter_nodes] = intp_result.astype(np.int64)
    dataframe must be indexed by id_: df.set_index(index_name, inplace=True)

    Parameters
    ----------
    df_ : pandas.core.frame.DataFrame
        Represents the dataset with contains lat, long and datetime.

    id_ : String
        ?

    feature_name : String
        ?. 

    filter_ : ?
        ?. 

    values : ?
        ?.

    Returns
    -------
    

    Examples
    --------
    -

    >>> from pymove.utils.transformations import change_df_feature_values_using_filter
    >>> change_df_feature_values_using_filter(df, -, -, -, -)

    """
    """
    
    """
    values_feature = df.at[id_, feature_name]
    if filter_.shape == ():
        df.at[id_, feature_name] = values
    else:
        values_feature[filter_] = values
        df.at[id_, feature_name] = values_feature

#TODO complementar oq ela faz
#TODO botar o check pra replace
#TODO trocar nome da func
def change_df_feature_values_using_filter_and_indexes(df, id_, feature_name, filter_, idxs, values):
    """
    ?
    Create or update move and stop by radius.

    Parameters
    ----------
    df_ : pandas.core.frame.DataFrame
        Represents the dataset with contains lat, long and datetime.
    
    id_ : String
        ?

    feature_name : String
        ?. 

    filter_ : ?
        ?. 

    idxs: ?
        ?.

    values : ?
        ?.

   
    Returns
    -------
    

    Examples
    --------
    -

    >>> from pymove.utils.transformations import change_df_feature_values_using_filter_and_indexes
    >>> change_df_feature_values_using_filter_and_indexes(df)

    """
    values_feature = df.at[id_, feature_name]
    values_feature_filter = values_feature[filter_]
    values_feature_filter[idxs] = values
    values_feature[filter_] = values_feature_filter
    df.at[id_, feature_name] = values_feature
