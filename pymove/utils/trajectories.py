from __future__ import division

from itertools import chain

import folium
import numpy as np
import pandas as pd

from pymove.utils.constants import (
    DATETIME,
    LATITUDE,
    LONGITUDE,
    TILES,
    TRAJ_ID,
    TYPE_DASK,
    TYPE_PANDAS,
)


def read_csv(
    filename,
    sep=',',
    encoding='utf-8',
    header='infer',
    names=None,
    latitude=LATITUDE,
    longitude=LONGITUDE,
    datetime=DATETIME,
    traj_id=TRAJ_ID,
    type_=TYPE_PANDAS,
    n_partitions=1,
):
    """
    Reads a .csv file and structures the data into the desired structure
    supported by PyMove.

    Parameters

    ----------
    filename : String.
        Represents coordinates lat, lon which will be the center of the map.
    sep : String, optional, default ','.
        Delimiter to use.
    encoding : String, optional, default 'utf-8'.
        Encoding to use for UTF when reading/writing
    header: int, list of int, default ‘infer’
        Row number(srs) to use as the column names, and the start of the data.
        Default behavior is to infer the column names: if no names are passed
        the behavior is identical to header=0 and column names are inferred from
        the first line of the file, if column names are passed explicitly then
        the behavior is identical to header=None
    names: array-like, optional
        List of column names to use. If the file contains a header row,
        then you should explicitly pass header=0 to override the column names.
        Duplicates in this list are not allowed.
    latitude : String, optional, default 'lat'.
        Represents the column name of feature latitude.
    longitude : String, optional, default 'lon'.
        Represents the column name of feature longitude.
    datetime : String, optional, default 'datetime'.
        Represents the column name of feature datetime.
    traj_id : String, optional, default 'id'.
        Represents the column name of feature id trajectory.
    type_ : String, optional, default 'pandas'.
        Represents the type of the MoveDataFrame
    n_partitions : int, optional, default 1.
        Represents number of partitions for DaskMoveDataFrame

    Returns
    -------
    pymove.core.MoveDataFrameAbstract subclass.
        Trajectory data.

    """

    df = pd.read_csv(
        filename,
        sep=sep,
        encoding=encoding,
        header=header,
        names=names,
        parse_dates=[datetime],
    )

    from pymove import PandasMoveDataFrame as pm
    from pymove import DaskMoveDataFrame as dm

    if type_ == TYPE_PANDAS:
        return pm(df, latitude, longitude, datetime, traj_id)
    if type_ == TYPE_DASK:
        return dm(df, latitude, longitude, datetime, traj_id, n_partitions)


def format_labels(current_id, current_lat, current_lon, current_datetime):
    """
    Format the labels for the PyMove lib pattern labels output
    lat, lon and datatime.

    Parameters
    ----------
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
    dict
        Represents a dict with mapping current columns of data
        to format of PyMove column.

    """

    return {
        current_id: TRAJ_ID,
        current_lon: LONGITUDE,
        current_lat: LATITUDE,
        current_datetime: DATETIME
    }


def flatten_dict(d, parent_key='', sep='_'):
    """
    Flattens a nested dictionary.

    Parameters
    ----------
    d: dict
        Dictionary to be flattened.
    parent_key: str, optional, default ''
        Key of the parent dictionary.
    sep: str, optional, default '_'
        Separator for the parent and child keys

    Returns
    -------
    dict
        Flattened dictionary

    References
    ----------
    https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys

    Examples
    --------
    >>> d = { 'a': 1, 'b': { 'c': 2, 'd': 3}}
    >>> flatten_dict(d)
    { 'a': 1, 'b_c': 2, 'b_d': 3 }

    """
    if not isinstance(d, dict):
        return {parent_key: d}
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def flatten_columns(df, columns):
    """
    Transforms columns containing dictionaries in individual columns

    Parameters
    ----------
    df: dataframe
        Dataframe with columns to be flattened
    columns: list
        List of columns from dataframe containing dictionaries

    Returns
    -------
    dataframe
        Dataframe with the new columns from the flattened dictionary columns

    References
    ----------
    https://stackoverflow.com/questions/51698540/import-nested-mongodb-to-pandas

    Examples
    --------
    >>> d = {'a': 1, 'b': {'c': 2, 'd': 3}}
    >>>> df = pd.DataFrame({'col1': [1], 'col2': [d]})
    >>>> flatten_columns(df, ['col2'])
       col1  col2_b_d  col2_a  col2_b_c
    0     1         3       1         2

    """
    for col in columns:
        df[f'{col}_'] = df[f'{col}'].apply(flatten_dict)
        keys = set(chain(*df[f'{col}_'].apply(lambda column: column.keys())))
        for key in keys:
            column_name = f'{col}_{key}'.lower()
            df[column_name] = df[f'{col}_'].apply(
                lambda cell: cell[key] if key in cell.keys() else np.NaN
            )
    cols_to_drop = [(f'{col}', f'{col}_') for col in columns]
    return df.drop(columns=list(chain(*cols_to_drop)))


def shift(arr, num, fill_value=np.nan):
    """
    Shifts the elements of the given array by the number of periods specified.

    Parameters
    ----------
    arr : array.
        The array to be shifted.
    num : int.
        Number of periods to shift. Can be positive or negative.
        If posite, the elements will be pulled down, and pulled up otherwise.
    fill_value : int, optional, default np.nan.
        The scalar value used for newly introduced missing values.

    Returns
    -------
    array
        A new array with the same shape and type_ as the initial given array,
        but with the indexes shifted.

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


def fill_list_with_new_values(original_list, new_list_values):
    """
    Copies elements from one list to another. The elements will be positioned in
    the same position in the new list as they were in their original list.

    Parameters
    ----------
    original_list : list.
        The list to which the elements will be copied.
    new_list_values : list.
        The list from which elements will be copied.

    """

    n = len(new_list_values)
    original_list[:n] = new_list_values


def save_bbox(
        bbox_tuple, file='bbox.html', tiles=TILES[0], color='red', return_map=False
):
    """
    Save bbox as file .html using Folium.

    Parameters
    ----------
    bbox_tuple : tuple.
        Represents a bound box, that is a tuple of 4 values with the
        min and max limits of latitude e longitude.
    file : String, optional, default 'bbox.html'.
        Represents filename.
    tiles : String, optional, default 'OpenStreetMap'.
        Represents tyles'srs type_.
        Example: 'openstreetmap', 'cartodbpositron',
                'stamentoner', 'stamenterrain',
                'mapquestopen', 'MapQuest Open Aerial',
                'Mapbox Control Room' and 'Mapbox Bright'.
    color : String, optional, default 'red'.
        Represents color of lines on map.
    return_map: Boolean, optional, default False.
        Wether to return the bbox folium map.

    Examples
    --------
    >>> from pymove.trajectories import save_bbox
    >>> bbox = (22.147577, 113.54884299999999, 41.132062, 121.156224)
    >>> save_bbox(bbox, 'bbox.html')

    """

    m = folium.Map(tiles=tiles)
    m.fit_bounds(
        [[bbox_tuple[0], bbox_tuple[1]], [bbox_tuple[2], bbox_tuple[3]]]
    )
    points_ = [
        (bbox_tuple[0], bbox_tuple[1]),
        (bbox_tuple[0], bbox_tuple[3]),
        (bbox_tuple[2], bbox_tuple[3]),
        (bbox_tuple[2], bbox_tuple[1]),
        (bbox_tuple[0], bbox_tuple[1]),
    ]
    folium.PolyLine(points_, weight=3, color=color).add_to(m)
    m.save(file)
    if return_map:
        return m
