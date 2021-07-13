"""
Data operations.

read_csv,
invert_dict,
flatten_dict,
flatten_columns,
shift,
fill_list_with_new_values,
object_for_array,
column_to_array

"""
from __future__ import annotations

from itertools import chain
from typing import Any

import numpy as np
from numpy import ndarray
from pandas import DataFrame, Series
from pandas import read_csv as _read_csv
from pandas._typing import FilePathOrBuffer

from pymove.core.dataframe import MoveDataFrame
from pymove.utils.constants import DATETIME, LATITUDE, LONGITUDE, TRAJ_ID, TYPE_PANDAS
from pymove.utils.math import is_number


def read_csv(
    filepath_or_buffer: FilePathOrBuffer,
    latitude: str = LATITUDE,
    longitude: str = LONGITUDE,
    datetime: str = DATETIME,
    traj_id: str = TRAJ_ID,
    type_: str = TYPE_PANDAS,
    n_partitions: int = 1,
    **kwargs
) -> MoveDataFrame:
    """
    Reads a `csv` file and structures the data.

    Parameters
    ----------
    filepath_or_buffer : str or path object or file-like object
        Any valid string path is acceptable. The string could be a URL.
        Valid URL schemes include http, ftp, s3, gs, and file.
        For file URLs, a host is expected.
        A local file could be: file://localhost/path/to/table.csv.
        If you want to pass in a path object, pandas accepts any os.PathLike.
        By file-like object, we refer to objects with a read() method,
        such as a file handle (e.g. via builtin open function) or StringIO.
    latitude : str, optional
        Represents the column name of feature latitude, by default 'lat'
    longitude : str, optional
        Represents the column name of feature longitude, by default 'lon'
    datetime : str, optional
        Represents the column name of feature datetime, by default 'datetime'
    traj_id : str, optional
        Represents the column name of feature id trajectory, by default 'id'
    type_ : str, optional
        Represents the type of the MoveDataFrame, by default 'pandas'
    n_partitions : int, optional
        Represents number of partitions for DaskMoveDataFrame, by default 1
    **kwargs : Pandas read_csv arguments
        https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html?highlight=read_csv#pandas.read_csv

    Returns
    -------
    MoveDataFrameAbstract subclass
        Trajectory data

    Examples
    --------
    >>> from pymove.utils.trajectories import read_csv
    >>> move_df = read_csv('geolife_sample.csv')
    >>> move_df.head()
              lat          lon              datetime  id
    0   39.984094   116.319236   2008-10-23 05:53:05   1
    1   39.984198   116.319322   2008-10-23 05:53:06   1
    2   39.984224   116.319402   2008-10-23 05:53:11   1
    3   39.984211   116.319389   2008-10-23 05:53:16   1
    4   39.984217   116.319422   2008-10-23 05:53:21   1
    >>> type(move_df)
    <class 'pymove.core.pandas.PandasMoveDataFrame'>
    """
    data = _read_csv(
        filepath_or_buffer,
        **kwargs
    )

    return MoveDataFrame(
        data, latitude, longitude, datetime, traj_id, type_, n_partitions
    )


def invert_dict(d: dict) -> dict:
    """
    Inverts the key:value relation of a dictionary.

    Parameters
    ----------
    d : dict
        dictionary to be inverted

    Returns
    -------
    dict
        inverted dict

    Examples
    --------
    >>> from pymove.utils.trajectories import invert_dict
    >>> traj_dict = {'a': 1, 'b': 2}
    >>> invert_dict(traj_dict)
    {1: 'a, 2: 'b'}
    """
    return {v: k for k, v in d.items()}


def flatten_dict(
    d: dict,
    parent_key: str = '',
    sep: str = '_'
) -> dict:
    """
    Flattens a nested dictionary.

    Parameters
    ----------
    d: dict
        Dictionary to be flattened
    parent_key: str, optional
        Key of the parent dictionary, by default ''
    sep: str, optional
        Separator for the parent and child keys, by default '_'

    Returns
    -------
    dict
        Flattened dictionary

    References
    ----------
    https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys

    Examples
    --------
    >>> from pymove.utils.trajectories import flatten_dict
    >>> d = {'a': 1, 'b': {'c': 2, 'd': 3}}
    >>> flatten_dict(d)
    {'a': 1, 'b_c': 2, 'b_d': 3}
    """
    if not isinstance(d, dict):
        return {parent_key: d}
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f'{parent_key}{sep}{k}' if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def flatten_columns(data: DataFrame, columns: list) -> DataFrame:
    """
    Transforms columns containing dictionaries in individual columns.

    Parameters
    ----------
    data: DataFrame
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
    >>> from pymove.utils.trajectories import flatten_columns
    >>> move_df
              lat          lon              datetime  id           dict_column
    0   39.984094   116.319236   2008-10-23 05:53:05   1              {'a': 1}
    1   39.984198   116.319322   2008-10-23 05:53:06   1              {'b': 2}
    2   39.984224   116.319402   2008-10-23 05:53:11   1      {'c': 3, 'a': 4}
    3   39.984211   116.319389   2008-10-23 05:53:16   1              {'b': 2}
    4   39.984217   116.319422   2008-10-23 05:53:21   1      {'a': 3, 'c': 2}
    >>> flatten_columns(move_df, columns='dict_column')
              lat            lon               datetime   id \
    dict_column_b         dict_column_c   dict_column_a
    0   39.984094      116.319236   2008-10-23 05:53:05    1 \
              NaN                   NaN             1.0
    1   39.984198      116.319322   2008-10-23 05:53:06    1 \
              2.0                   NaN             NaN
    2   39.984224      116.319402   2008-10-23 05:53:11    1 \
              NaN                   3.0             4.0
    3   39.984211      116.319389   2008-10-23 05:53:16    1 \
              2.0                   NaN             NaN
    4   39.984217      116.319422   2008-10-23 05:53:21    1 \
              NaN                   2.0             3.0
    """
    data = data.copy()
    if not isinstance(columns, list):
        columns = [columns]
    for col in columns:
        data[f'{col}_'] = data[f'{col}'].apply(flatten_dict)
        keys = set(chain(*data[f'{col}_'].apply(lambda column: column.keys())))
        for key in keys:
            column_name = f'{col}_{key}'.lower()
            data[column_name] = data[f'{col}_'].apply(
                lambda cell: cell[key] if key in cell.keys() else np.NaN
            )
    cols_to_drop = [(f'{col}', f'{col}_') for col in columns]
    return data.drop(columns=list(chain(*cols_to_drop)))


def shift(
    arr: list | Series | ndarray,
    num: int,
    fill_value: Any | None = None
) -> ndarray:
    """
    Shifts the elements of the given array by the number of periods specified.

    Parameters
    ----------
    arr : array
        The array to be shifted
    num : int
        Number of periods to shift. Can be positive or negative
        If positive, the elements will be pulled down, and pulled up otherwise
    fill_value : float, optional
        The scalar value used for newly introduced missing values, by default np.nan

    Returns
    -------
    array
        A new array with the same shape and type_ as the initial given array,
        but with the indexes shifted.

    Notes
    -----
        Similar to pandas shift, but faster.

    References
    ----------
    https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array

    Examples
    --------
    >>> from pymove.utils.trajectories import shift
    >>> array = [1, 2, 3, 4, 5, 6, 7]
    >>> shift(array, 1)
    [0 1 2 3 4 5 6]
    >>> shift(array, 0)
    [1, 2, 3, 4, 5, 6, 7]
    >>> shift(array, -1)
    [2 3 4 5 6 7 0]
    """
    result = np.empty_like(arr)
    arr = np.array(arr)

    if fill_value is None:
        dtype = result.dtype
        if np.issubdtype(dtype, np.bool_):
            fill_value = False
        elif np.issubdtype(dtype, np.integer):
            fill_value = 0
        else:
            fill_value = np.nan

    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result = arr
    return result


def fill_list_with_new_values(original_list: list, new_list_values: list):
    """
    Copies elements from one list to another.

    The elements will be positioned in
    the same position in the new list as they were in their original list.

    Parameters
    ----------
    original_list : list.
        The list to which the elements will be copied
    new_list_values : list.
        The list from which elements will be copied

    Example
    -------
    >>> from pymove.utils.trajectories import fill_list_with_new_values
    >>> lst = [1, 2, 3, 4]
    >>> fill_list_with_new_values(lt, ['a','b'])
    >>> print(lst)
    ['a', 'b', 3, 4]
    """
    n = len(new_list_values)
    original_list[:n] = new_list_values


def object_for_array(object_: str) -> ndarray:
    """
    Transforms an object into an array.

    Parameters
    ----------
    object : str
        object representing a list of integers or strings

    Returns
    -------
    array
        object converted to a list

    Examples
    --------
    >>> from pymove.utils.trajectories import object_for_array
    >>> list_str = '[1,2,3,4,5]'
    >>> object_for_array(list_str)
    array([1., 2., 3., 4., 5.], dtype=float32)
    """
    if object_ is None:
        return object_

    conv = np.array([*map(str.strip, object_[1:-1].split(','))])

    if is_number(conv[0]):
        return conv.astype(np.float32)
    else:
        return conv.astype('object_')


def column_to_array(data: DataFrame, column: str) -> DataFrame:
    """
    Transforms all columns values to list.

    Parameters
    ----------
    data : dataframe
        The input trajectory data

    column : str
        Label of data referring to the column for conversion

    Returns
    -------
    dataframe
        Dataframe with the selected column converted to list

    Example
    -------
    >>> from pymove.utils.trajectories import column_to_array
    >>> move_df
              lat          lon              datetime  id   list_column
    0   39.984094   116.319236   2008-10-23 05:53:05   1        '[1,2]'
    1   39.984198   116.319322   2008-10-23 05:53:06   1        '[3,4]'
    2   39.984224   116.319402   2008-10-23 05:53:11   1        '[5,6]'
    3   39.984211   116.319389   2008-10-23 05:53:16   1        '[7,8]'
    4   39.984217   116.319422   2008-10-23 05:53:21   1       '[9,10]'
    >>> column_to_array(move_df, column='list_column')
              lat          lon              datetime  id    list_column
    0   39.984094   116.319236   2008-10-23 05:53:05   1      [1.0,2.0]
    1   39.984198   116.319322   2008-10-23 05:53:06   1      [3.0,4.0]
    2   39.984224   116.319402   2008-10-23 05:53:11   1      [5.0,6.0]
    3   39.984211   116.319389   2008-10-23 05:53:16   1      [7.0,8.0]
    4   39.984217   116.319422   2008-10-23 05:53:21   1     [9.0,10.0]
    """
    data = data.copy()
    if column not in data:
        raise KeyError(
            'Dataframe must contain a %s column' % column
        )

    data[column] = data[column].apply(object_for_array)
    return data
