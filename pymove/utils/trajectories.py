from __future__ import division

from itertools import chain

import folium
import numpy as np
import pandas as pd

from pymove.core.dataframe import MoveDataFrame
from pymove.utils.constants import (
    DATETIME,
    LATITUDE,
    LONGITUDE,
    TILES,
    TRAJ_ID,
    TYPE_DASK,
    TYPE_PANDAS,
)
from pymove.utils.log import progress_bar
from pymove.utils.math import is_number


def read_csv(
    filename,
    latitude=LATITUDE,
    longitude=LONGITUDE,
    datetime=DATETIME,
    traj_id=TRAJ_ID,
    type_=TYPE_PANDAS,
    n_partitions=1,
    sep=',',
    encoding='utf-8',
    header='infer',
    names=None,
    index_col=None,
    usecols=None,
    dtype=None,
    nrows=None,
):
    """
    Reads a .csv file and structures the data into the desired structure
    supported by PyMove.

    Parameters

    ----------
    filename : String.
        Represents coordinates lat, lon which will be the center of the map.
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
    sep : String, optional, default ','.
        Delimiter to use.
    encoding : String, optional, default 'utf-8'.
        Encoding to use for UTF when reading/writing
    header : int, list of int, default ‘infer’
        Row number(srs) to use as the column names, and the start of the data.
        Default behavior is to infer the column names: if no names are passed
        the behavior is identical to header=0 and column names are inferred from
        the first line of the file, if column names are passed explicitly then
        the behavior is identical to header=None
    names : array-like, optional
        List of column names to use. If the file contains a header row,
        then you should explicitly pass header=0 to override the column names.
        Duplicates in this list are not allowed.
    index_col : int, str, sequence of int / str, or False, default None
        Column(s) to use as the row labels of the DataFrame, either given as
        string name or column index.
        If a sequence of int / str is given, a MultiIndex is used.
    usecols : list-like or callable, optional, default None
        Return a subset of the columns. If list-like, all elements must either
        be positional (i.e. integer indices into the document columns) or strings
        that correspond to column names provided either by the user in names or
        inferred from the document header row(s).
    dtype : Type name or dict of column -> type, optional, default None
        Data type for data or columns.
        E.g. {‘a’: np.float64, ‘b’: np.int32, ‘c’: ‘Int64’}
        Use str or object together with suitable na_values settings to
        preserve and not interpret dtype.
    nrows : int, optional, default None
        Number of rows of file to read. Useful for reading pieces of large files.

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
        index_col=index_col,
        usecols=usecols,
        dtype=dtype,
        nrows=nrows
    )

    return MoveDataFrame(
        df, latitude, longitude, datetime, traj_id, type_, n_partitions
    )


def invert_dict(dict_):
    """
    Inverts the key:value relation of a dictionary

    Parameters
    ----------
    dict_ : dict
        dictionary to be inverted

    Returns
    -------
    dict
        inverted dict

    """

    return {v: k for k, v in dict_.items()}


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


def object_for_array(object_):
    """
    Transforms an object into an array.

    Parameters
    ----------
    object : String
        object representing a list of integers or strings.

    Returns
    -------
    array
        object converted to a list
    """

    if object_ is None:
        return object_

    try:
        conv = np.array(object_[1:-1].split(', '))

        if is_number(conv[0]):
            return conv.astype(np.float32)
        else:
            return conv.astype('object_')

    except Exception as e:
        raise e


def column_to_array(df_, label_conversion):
    """
    Transforms all columns values to list.

    Parameters
    ----------
    df_ : dataframe
        The input trajectory data.

    label_conversion : Object
        Label of df_ referring to the column for conversion.
    """
    try:

        if label_conversion not in df_:
            raise KeyError(
                'Dataframe must contain a %s column' % label_conversion
            )

        arr = np.full(df_.shape[0], None, dtype=np.ndarray)
        for idx, row in progress_bar(df_.iterrows(), total=df_.shape[0]):
            arr[idx] = object_for_array(row[label_conversion])

        df_[label_conversion] = arr

    except Exception as e:
        raise e


def plot_bbox(
        bbox_tuple,
        tiles=TILES[0],
        color='red',
        save_map=False,
        file='bbox.html'
):
    """
    Plots a bbox using Folium.

    Parameters
    ----------
    bbox_tuple : tuple.
        Represents a bound box, that is a tuple of 4 values with the
        min and max limits of latitude e longitude.
    tiles : String, optional, default 'OpenStreetMap'.
        Represents tyles'srs type_.
        Example: 'openstreetmap', 'cartodbpositron',
                'stamentoner', 'stamenterrain',
                'mapquestopen', 'MapQuest Open Aerial',
                'Mapbox Control Room' and 'Mapbox Bright'.
    color : String, optional, default 'red'.
        Represents color of lines on map.
    file : String, optional, default 'bbox.html'.
        Represents filename.
    save_map: Boolean, optional, default False.
        Wether to save the bbox folium map.

    Returns
    --------
    folium map with bounding box

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
    polygon = folium.PolyLine(points_, weight=3, color=color)
    polygon.add_to(m)

    if save_map:
        m.save(file)

    return m
