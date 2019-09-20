# TODO: Andreza e Arina
from __future__ import division
import time
import math
import folium
import datetime
import numpy as np
import pandas as pd

from IPython.display import display
from ipywidgets import IntProgress, HTML, VBox
from pandas._libs.tslibs.timestamps import Timestamp

"""main labels """
dic_labels = {"id": 'id', 'lat': 'lat', 'lon': 'lon', 'datetime': 'datetime'}

dic_features_label = {'tid': 'tid', 'dist_to_prev': 'dist_to_prev', "dist_to_next": 'dist_to_next',
                      'dist_prev_to_next': 'dist_prev_to_next',
                      'time_to_prev': 'time_to_prev', 'time_to_next': 'time_to_next', 'speed_to_prev': 'speed_to_prev',
                      'speed_to_next': 'speed_to_next',
                      'period': 'period', 'day': 'day', 'index_grid_lat': 'index_grid_lat',
                      'index_grid_lon': 'index_grid_lon',
                      'situation': 'situation'}


# Esses aqui seriam o traj_utils
def get_bbox(df_, dic_labels=dic_labels):
    """
    A bounding box (usually shortened to bbox) is an area defined by two longitudes and two latitudes, where:
        - Latitude is a decimal number between -90.0 and 90.0.
        - Longitude is a decimal number between -180.0 and 180.0.
    They usually follow the standard format of:
    - bbox = left, bottom, right, top
    - bbox = min Longitude , min Latitude , max Longitude , max Latitude

    Parameters
    ----------
    df_ : pandas.core.frame.DataFrame
        Represents the dataset with contains lat, long and datetime.

    dic_labels : dict
        Represents mapping of column's header between values passed on params.

    Returns
    -------
    bbox : tuple
        Represents a bound box, that is a tuple of 4 values with the min and max limits of latitude e longitude.


    Examples
    --------
    >>> from pymove.utils.utils import get_bbox
    >>> get_bbox(df, dic_labels)
    (22.147577, 113.54884299999999, 41.132062, 121.156224)

    """
    try:
        bbox = (df_[dic_labels['lat']].min(), df_[dic_labels['lon']].min(), df_[dic_labels['lat']].max(),
                df_[dic_labels['lon']].max())
        return bbox
    except Exception as e:
        raise e


def save_bbox(bbox_tuple, file, tiles='OpenStreetMap', color='red'):
    """
    Save bbox as file .html using Folium.

    Parameters
    ----------
    bbox_tuple : tuple
        Represents a bound box, that is a tuple of 4 values with the min and max limits of latitude e longitude.


    file : String
        Represents filename.

    tiles : String
        Represents tyles's type.
        Example: 'openstreetmap', 'cartodbpositron', 'stamentoner', 'stamenterrain', 'mapquestopen', 'MapQuest Open Aerial', 'Mapbox Control Room' and 'Mapbox Bright'.

    color : String
        Represents color of trajectorys on map.

    Returns
    -------


    Examples
    --------
    >>> from pymove.utils.utils import save_bbox
    >>> bbox = (22.147577, 113.54884299999999, 41.132062, 121.156224)
    >>> save_bbox(bbox, 'bbox.html')

    """
    m = folium.Map(tiles=tiles)
    m.fit_bounds([[bbox_tuple[0], bbox_tuple[1]], [bbox_tuple[2], bbox_tuple[3]]])
    points_ = [(bbox_tuple[0], bbox_tuple[1]), (bbox_tuple[0], bbox_tuple[3]),
               (bbox_tuple[2], bbox_tuple[3]), (bbox_tuple[2], bbox_tuple[1]),
               (bbox_tuple[0], bbox_tuple[1])]
    folium.PolyLine(points_, weight=3, color=color).add_to(m)
    m.save(file)


def show_trajectories_info(df_, dic_labels=dic_labels):
    """
    Show dataset information from dataframe, this is number of rows, datetime interval, and bounding box.

    Parameters
    ----------
    df_ : pandas.core.frame.DataFrame
        Represents the dataset with contains lat, long and datetime.

    dic_labels : dict
        Represents mapping of column's header between values passed on params.

    Returns
    -------


    Examples
    --------
    >>> from pymove.utils.utils import show_trajectories_info
    >>> show_trajectories_info(df_, dic_labels)
    ======================= INFORMATION ABOUT DATASET =======================

    Number of Points: 217654

    Number of IDs objects: 2

    Start Date:2008-10-23 05:53:05     End Date:2009-03-19 05:46:37

    Bounding Box:(22.147577, 113.54884299999999, 41.132062, 121.156224)


    =========================================================================

    """
    try:
        print('\n======================= INFORMATION ABOUT DATASET =======================\n')
        print('Number of Points: {}\n'.format(df_.shape[0]))
        if dic_labels['id'] in df_:
            print('Number of IDs objects: {}\n'.format(df_[dic_labels['id']].nunique()))
        if dic_features_label['tid'] in df_:
            print('Number of TIDs trajectory: {}\n'.format(df_[dic_features_label['tid']].nunique()))
        if dic_labels['datetime'] in df_:
            print('Start Date:{}     End Date:{}\n'.format(df_[dic_labels['datetime']].min(),
                                                           df_[dic_labels['datetime']].max()))
        if dic_labels['lat'] and dic_labels['lon'] in df_:
            print('Bounding Box:{}\n'.format(
                get_bbox(df_, dic_labels)))  # bbox return =  Lat_min , Long_min, Lat_max, Long_max)
        if dic_features_label['time_to_prev'] in df_:
            print(
                'Gap time MAX:{}     Gap time MIN:{}\n'.format(round(df_[dic_features_label['time_to_prev']].max(), 3),
                                                               round(df_[dic_features_label['time_to_prev']].min(), 3)))
        if dic_features_label['speed_to_prev'] in df_:
            print('Speed MAX:{}    Speed MIN:{}\n'.format(round(df_[dic_features_label['speed_to_prev']].max(), 3),
                                                          round(df_[dic_features_label['speed_to_prev']].min(), 3)))
        if dic_features_label['dist_to_prev'] in df_:
            print('Distance MAX:{}    Distance MIN:{}\n'.format(round(df_[dic_features_label['dist_to_prev']].max(), 3),
                                                                round(df_[dic_features_label['dist_to_prev']].min(),
                                                                      3)))

        print('\n=========================================================================\n')
    except Exception as e:
        raise e

    #  CONVERSIONS


def change_df_feature_values_using_filter(df, id_, feature_name, filter_, values):
    """
    equivalent of: df.at[id_, feature_name][filter_] = values
    e.g. df.at[tid, 'time'][filter_nodes] = intp_result.astype(np.int64)
    dataframe must be indexed by id_: df.set_index(index_name, inplace=True)
    """
    values_feature = df.at[id_, feature_name]
    if filter_.shape == ():
        df.at[id_, feature_name] = values
    else:
        values_feature[filter_] = values
        df.at[id_, feature_name] = values_feature


def change_df_feature_values_using_filter_and_indexes(df, id_, feature_name, filter_, idxs, values):
    """
    equivalent of: df.at[id_, feature_name][filter_][idxs] = values
    e.g. df.at[tid, 'deleted'][filter_][idx_not_in_ascending_order] = True
    dataframe must be indexed by id_: df.set_index(index_name, inplace=True)
    """
    values_feature = df.at[id_, feature_name]
    values_feature_filter = values_feature[filter_]
    values_feature_filter[idxs] = values
    values_feature[filter_] = values_feature_filter
    df.at[id_, feature_name] = values_feature


# duvida se era para ficar aqui ou no math_utils
def interpolation(x0, y0, x1, y1, x):
    """Perfomers interpolation and extrapolation

    Parameters
    ----------
    x0 : float
        The coordinate of the first point on the x axis

    y0 : float
        The coordinate of the first point on the y axis

    x1 : float
        The coordinate of the second point on the x axis

    y1 : float
        The coordinate of the second point on the y axis

    x : float
        A value in the interval (x0, x1)

    Returns
    -------
    y : float
        Is the interpolated  or extrapolated value.

    Examples
    --------
    interpolation 1: (30, 3, 40, 5, 37) -> 4.4
    interpolation 2: (30, 3, 40, 5, 35) -> 4.0
    extrapolation 1: (30, 3, 40, 5, 25) -> 2.0
    extrapolation 2: (30, 3, 40, 5, 45) -> 6.0
    """
    y = y0 + (y1 - y0) * ((x - x0) / (x1 - x0))
    return y
