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
