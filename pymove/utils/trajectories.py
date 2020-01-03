from __future__ import division
import time
import folium
import numpy as np
import pandas as pd
from IPython.display import display
from ipywidgets import IntProgress, HTML, VBox
from pymove.utils.datetime import deltatime_str
from pymove.utils.constants import LATITUDE, LONGITUDE, DATETIME, TRAJ_ID


def read_csv(
    filename,
    sep=',',
    encoding="utf-8",
    latitude=LATITUDE,
    longitude=LONGITUDE,
    datetime=DATETIME,
    traj_id=TRAJ_ID,
    type_="pandas",
    n_partitions=1
):
    """
    Reads a .csv file and structures the data into the desired structure supported by PyMove.

    Parameters

    ----------
    filename : String.
        Represents coordinates lat, lon which will be the center of the map.

    sep : String, optional, default ','.
        Represents .

    encoding : String, optional, default 'utf-8'.
        Represents .

    latitude : String, optional, default 'lat'.
        Represents the column name of feature latitude.

    longitude : String, optional, default 'lon'.
        Represents the column name of feature longitude.

    datetime : String, optional, default 'datetime'.
        Represents the column name of feature datetime.

    traj_id : String, optional, default 'id'.
        Represents the column name of feature id trajectory.

    type_ : String, optional, default 'pandas'.
        Represents the type of                    \

    n_partitions : int, optional, default 1.
        Represents .

    Returns
    -------
    pymove.core.MoveDataFrameAbstract subclass.
        Trajectory data.

    """
    df = pd.read_csv(filename, sep=sep, encoding=encoding, parse_dates=['datetime'])

    from pymove import PandasMoveDataFrame as pm
    from pymove import DaskMoveDataFrame as dm

    if type_ == 'pandas':
        return pm(df, latitude, longitude, datetime, traj_id)
    if type_ == 'dask':
        return dm(df, latitude, longitude, datetime, traj_id, n_partitions)


def format_labels(move_data, current_id, current_lat, current_lon, current_datetime):
    """
    Format the labels for the PyMove lib pattern labels output = lat, lon and datatime.

    Parameters
    ----------
    move_data : pymove.core.MoveDataFrameAbstract subclass.
        Input trajectory data.

    current_id : String.
        Represents the column name of feature id.

    current_lat : String.
        Represents the column name of feature latitude.

    current_lon : String.
        Represents the column name of feature longitude.

    current_datetime : String.
         Represents the column name of feature datetime.

    Returns
    -------
    dic_labels : dict.
        Represents a dict with mapping current columns of data to format of PyMove column.

    """ 
    dic_labels = {}
    dic_labels[current_id] = TRAJ_ID
    dic_labels[current_lon] = LONGITUDE
    dic_labels[current_lat] = LATITUDE
    dic_labels[current_datetime] = DATETIME
    return dic_labels


#TODO: COmpletar as infos
def log_progress(sequence, every=None, size=None, name='Items'):
    """
    Make and display a progress bar.

    Parameters
    ----------
    sequence : list.
        Represents a elements sequence.

    every : ?, optional, default None.
        Represents the column name of feature id.

    size : int, optional, default None.
        Represents the size/number elements in sequence.

    name : String, optional, default 'Items'.
        Represents the name of ?.

    Returns
    -------

    """
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
                every = int(size/200)
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
                    label.value = '{name}: {index} / ?'.format(name=name, index=index)
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
        label.value = "{name}: {index}".format(name=name, index=str(index or '?'))


#TODO: COmpletar as infos
def progress_update(size_processed, size_all, start_time, curr_perc_int, step_perc=1):
    """
    Update and print current progress.

    Parameters
    ----------
    size_processed : int.
        Represents a number of elements already processed.

    size_all : int.
        Represents the number of elements.

    start_time : int, optional, default None.
        Represents the size/number elements in sequence.

    curr_perc_int : ?
        Represents the name of ?.

    step_perc : int, optional, default 1.
        Represents the name of ?.

    Returns
    -------
    curr_perc_int_new : ?
        Represents ?.

    deltatime_str : ?
        Represents ?.

    """
    curr_perc_new = size_processed*100.0/size_all
    curr_perc_int_new = int(curr_perc_new)
    if curr_perc_int_new != curr_perc_int and curr_perc_int_new % step_perc == 0:
        deltatime = time.time() - start_time
        deltatime_str_ = deltatime_str(deltatime)
        est_end = deltatime / curr_perc_new * 100
        est_time_str = deltatime_str(est_end - deltatime)
        print('({}/{}) {}% in {} - estimated end in {}'.format(size_processed, size_all, curr_perc_int_new, deltatime_str_, est_time_str))
        return curr_perc_int_new, deltatime_str
    else:
        return curr_perc_int_new, None


def shift(arr, num, fill_value=np.nan):
    """
    Shifts the elements of the given array by the number of periods specified.

    Parameters
    ----------
    arr : array.
        The array to be shifed.

    num : int.
        Number of periods to shift. Can be positive or negative. If posite, the elements will be pulled down, and pulled
        up otherwise.

    fill_value : int, optional, default np.nan.
        The scalar value used for newly introduced missing values.

    Returns
    -------
    result : array.
        A new array with the same shape and type as the initial given array, but with the indexes shifted.

    Notes
    -----
        Similar to pandas shift, but faster.

    References
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


# TODO: erro se tentar converter int para str e funcao n verifica isso
# def fill_list_with_new_values(original_list, new_list_values):
#     """
#     Copies elements from one list to another. The elements will be positioned in the same position in the new list as
#     they were in their original list.
#
#     Parameters
#     ----------
#     original_list : list.
#         The list to which the elements will be copied.
#
#     new_list_values : list.
#         The list from which elements will be copied.
#
#     Returns
#     -------
#
#     """
#     for i in range(len(new_list_values)):
#         type1 = type(original_list[i])
#         if type1 == int:
#             original_list[i] = int(new_list_values[i])
#         elif type1 == float:
#             original_list[i] = float(new_list_values[i])
#         else:
#             original_list[i] = new_list_values[i]


def save_bbox(bbox_tuple, file, tiles='OpenStreetMap', color='red'):
    """
    Save bbox as file .html using Folium.

    Parameters
    ----------
    bbox_tuple : tuple.
        Represents a bound box, that is a tuple of 4 values with the min and max limits of latitude e longitude.

    file : String.
        Represents filename.

    tiles : String, optional, default 'OpenStreetMap'.
        Represents tyles's type.
        Example: 'openstreetmap', 'cartodbpositron', 'stamentoner', 'stamenterrain', 'mapquestopen',
        'MapQuest Open Aerial', 'Mapbox Control Room' and 'Mapbox Bright'.

    color : String, optional, default 'red'.
        Represents color of lines on map.

    Returns
    -------


    Examples
    --------
    >>> from pymove.trajectories import save_bbox
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