# TODO: Andreza e Arina
from __future__ import division
import time
import folium
import numpy as np
import pandas as pd

from pymove.utils.time_utils import deltatime_str
from IPython.display import display
from ipywidgets import IntProgress, HTML, VBox
from pymove.utils.constants import LATITUDE, LONGITUDE
# """main labels """
# dic_labels = {"id" : 'id', 'lat' : 'lat', 'lon' : 'lon', 'datetime' : 'datetime'}

# dic_features_label = {'tid' : 'tid', 'dist_to_prev' : 'dist_to_prev', "dist_to_next" : 'dist_to_next', 'dist_prev_to_next' : 'dist_prev_to_next',
#                     'time_to_prev' : 'time_to_prev', 'time_to_next' : 'time_to_next', 'speed_to_prev': 'speed_to_prev', 'speed_to_next': 'speed_to_next',
#                     'period': 'period', 'day': 'day', 'index_grid_lat': 'index_grid_lat', 'index_grid_lon' : 'index_grid_lon',
#                     'situation':'situation'}


def log_progress(sequence, every=None, size=None, name='Items'):
    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = u'{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = "{name}: {index}".format(
            name=name,
            index=str(index or '?')
        )


def progress_update(size_processed, size_all, start_time, curr_perc_int, step_perc=1):
    """
    update and print current progress.
    e.g.
    curr_perc_int, _ = pu.progress_update(size_processed, size_all, start_time, curr_perc_int)
    returns: curr_perc_int_new, deltatime_str
    """
    curr_perc_new = size_processed*100.0 / size_all
    curr_perc_int_new = int(curr_perc_new)
    if curr_perc_int_new != curr_perc_int and curr_perc_int_new % step_perc == 0:
        deltatime = time.time() - start_time
        deltatime_str_ = deltatime_str(deltatime)
        est_end = deltatime / curr_perc_new * 100
        est_time_str = deltatime_str(est_end - deltatime)
        print('({}/{}) {}% in {} - estimated end in {}'.format(size_processed, size_all, curr_perc_int_new, deltatime_str_, est_time_str))
        return curr_perc_int_new, deltatime_str # aqui era pra ser deltatime_str_ não?
    else:
        return curr_perc_int_new, None


def shift(arr, num, fill_value=np.nan):
    """Shifts the elements of the given array by the number of periods specified.

    Parameters
    ----------
    arr : array
        The array to be shifed.

    num : Integer
        Number of periods to shift. Can be positive or negative. If posite, the elements will be pulled down, and pulled
        up otherwise.

    fill_value : Integer, optional(np.nan by default)
        The scalar value used for newly introduced missing values.

    Returns
    -------
    result : array
        A new array with the same shape and type as the initial given array, but with the indexes shifted.

    Notes
    -----
        Similar to pandas shift, but faster.

    See also
    --------
        https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array
    """

    result = np.empty_like(arr)

    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result = arr
    return result


# erro se tentar converter int para str e funcao n verifica isso
def fill_list_with_new_values(original_list, new_list_values):
    """ Copies elements from one list to another. The elements will be positioned in the same position in the new list as
    they were in their original list.

    Parameters
    ----------
    original_list : array
    The list to which the elements will be copied

    new_list_values : array
    The list from which elements will be copied

    """
    for i in range(len(new_list_values)):
        type1 = type(original_list[i])
        if type1 == int:
            original_list[i] = int(new_list_values[i])
        elif type1 == float:
            original_list[i] = float(new_list_values[i])
        else:
            original_list[i] = new_list_values[i]


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

def get_bbox(df_):
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
    
    Returns
    -------
    bbox : tuple
        Represents a bound box, that is a tuple of 4 values with the min and max limits of latitude e longitude.
    Examples
    --------
    >>> from pymove.utils.utils import get_bbox
    >>> get_bbox(df)
    (22.147577, 113.54884299999999, 41.132062, 121.156224)
    """
    try:
        bbox = (df_[LATITUDE].min(), df_[LONGITUDE].min(), df_[LATITUDE].max(), df_[LONGITUDE].max())
        return bbox
    except Exception as e:
        raise e

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
            print('Start Date:{}     End Date:{}\n'.format(df_[dic_labels['datetime']].min(), df_[dic_labels['datetime']].max()))
        if dic_labels['lat'] and dic_labels['lon'] in df_:
            print('Bounding Box:{}\n'.format(get_bbox(df_, dic_labels))) # bbox return =  Lat_min , Long_min, Lat_max, Long_max) 
        if dic_features_label['time_to_prev'] in df_:            
            print('Gap time MAX:{}     Gap time MIN:{}\n'.format(round(df_[dic_features_label['time_to_prev']].max(),3), round(df_[dic_features_label['time_to_prev']].min(), 3)))
        if dic_features_label['speed_to_prev'] in df_:            
            print('Speed MAX:{}    Speed MIN:{}\n'.format(round(df_[dic_features_label['speed_to_prev']].max(), 3), round(df_[dic_features_label['speed_to_prev']].min(), 3))) 
        if dic_features_label['dist_to_prev'] in df_:            
            print('Distance MAX:{}    Distance MIN:{}\n'.format(round(df_[dic_features_label['dist_to_prev']].max(),3), round(df_[dic_features_label['dist_to_prev']].min(), 3))) 
            
        print('\n=========================================================================\n')
    except Exception as e:
        raise e    