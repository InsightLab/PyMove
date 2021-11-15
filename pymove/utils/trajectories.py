"""
Data operations.

read_csv,
invert_dict,
flatten_dict,
flatten_columns,
shift,
fill_list_with_new_values,
append_trajectory,
split_trajectory,
object_for_array,
column_to_array

"""
from __future__ import annotations

from ast import literal_eval
from itertools import chain
from typing import Any, Generator, Text

import numpy as np
import pandas as pd
from networkx.classes.digraph import DiGraph
from numpy import ndarray
from pandas import DataFrame, Series
from pandas import read_csv as _read_csv
from pandas._typing import FilePathOrBuffer

from pymove.core.dataframe import MoveDataFrame
from pymove.utils.constants import (
    DATETIME,
    LATITUDE,
    LOCAL_LABEL,
    LONGITUDE,
    TRAJ_ID,
    TYPE_PANDAS,
)
from pymove.utils.networkx import graph_to_dict


def read_csv(
    filepath_or_buffer: FilePathOrBuffer,
    latitude: str = LATITUDE,
    longitude: str = LONGITUDE,
    datetime: str = DATETIME,
    traj_id: str = TRAJ_ID,
    type_: str = TYPE_PANDAS,
    n_partitions: int = 1,
    **kwargs
):
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


def append_trajectory(
    data: DataFrame,
    trajectory: list,
    graph: DiGraph
):
    """
    Inserts a trajectory in the data set.

    Inserts the trajectory retrieved from the
    transition graph in the trajectory data set.

    Parameters
    ----------
    data: DataFrame
        Trajectory data in sequence format
    trajectory: list
        Trajectory recovered from the transition graph
    graph: DiGraph
        Transition graph constructed from trajectory data

    Example
    -------
    >>> from pymove.utils.data_augmentation import append_trajectory
    >>>
    >>> traj_df.to_dict()
    {'id': [[1, 1, 1], [2, 2, 2, 2]],
     'datetime': [['2017-09-02 22:00:27', '2017-09-02 22:01:36',
                  '2017-09-02 22:03:08'],
                 ['2017-09-02 23:03:46', '2017-09-02 23:07:19',
                  '2017-09-02 23:07:40', '2017-09-02 23:09:10']],
     'local_label': [[85, 673, 394], [263, 224, 623, 394]],
     'lat': [[-3.8347478, -3.8235834, -3.813889],
            [-3.9067654, -3.8857223, -3.8828723, -3.9939834]],
     'lon': [[-38.592189, -38.590389, -38.5904445],
            [-38.5907723, -38.5928892, -38.5929789, -38.70409]]}
    >>>
    >>> trajectory = [263, 224, 623]
    >>> graph = build_transition_graph_from_df(traj_df)
    >>>
    >>> append_trajectory(traj_df, trajectory, graph)
    >>> traj_df.iloc[-1]
    id                                                              [3, 3, 3]
    datetime  [2017-09-02 23:03:46, 2017-09-02 23:07:19, 2017-09-02 23:07:40]
    local_label                                               [263, 224, 623]
    lat                                  [-3.9067654, -3.8857223, -3.8828723]
    lon                               [-38.5907723, -38.5928892, -38.5929789]
    Name: 2, dtype: object

    """
    source = str(trajectory[0])
    dict_graph = graph_to_dict(graph)

    dt = np.random.choice(dict_graph['nodes']['datetime'][source])
    datetimes = [pd.Timestamp(str(dt))]

    coords = dict_graph['nodes']['coords']
    lats, lons = [coords[source][0]], [coords[source][1]]

    for idx, edge in enumerate(zip(trajectory[:-1], trajectory[1:])):
        u, v = str(edge[0]), str(edge[1])
        mean_times = dict_graph['edges'][u][v]['mean_times']

        datetime = pd.Timestamp(str(datetimes[idx])) + pd.Timedelta(mean_times)
        datetimes.append(datetime)

        lats.append(coords[v][0])
        lons.append(coords[v][1])

    prev_id = data.loc[data.shape[0] - 1, TRAJ_ID][0]
    ids = np.full(len(trajectory), prev_id + 1, dtype=np.int32).tolist()

    path = np.array(trajectory, dtype=np.float32).tolist()

    data.loc[data.shape[0], [
        DATETIME, TRAJ_ID, LOCAL_LABEL, LATITUDE, LONGITUDE
    ]] = [datetimes, ids, path, lats, lons]


def split_trajectory(
    row: Series,
    size_window: int = 6,
    size_jump: int = 3,
    label_local: Text = LOCAL_LABEL,
    columns: list = None
) -> Generator[Series, None, None]:
    """
    It breaks the trajectory in stretches.

    Extracts all possible sub-trajectories, according to the specified
    window size and jump.

    Parameters
    ----------
    row: Series
        Line of the trajectory dataframe
    size_window: int, optional
        Sliding window size, by default 6
    size_jump: int, optional
        Size of the jump in the trajectory, by default 3
    label_local: str, optional
        Name of the column referring to the trajectories, by default LOCAL_LABEL
    columns: list, optional
        Columns to which the split will be applied, by default None

    Return
    ------
    Generator of Series
        Series with the stretches recovered from the observed trajectory.

    Example
    -------
    >>> from pymove.utils.trajectories import split_trajectories
    >>>
    >>> trajectory
    id                           [1, 1, 1, 1, 1, 1, 1, 1]
    local_label    [85, 673, 394, 85, 224, 623, 394, 263]
    dtype: object
    >>>
    >>> split = split_trajectory(trajectory, size_jump=1)
    >>> split

                        id 	                  local_label
    0   [1, 1, 1, 1, 1, 1]  [ 85, 673, 394, 85, 224, 623]
    1   [1, 1, 1, 1, 1, 1]  [673, 394, 85, 224, 623, 394]
    2   [1, 1, 1, 1, 1, 1]  [394, 85, 224, 623, 394, 263]

    """
    if columns is None:
        columns = row.index

    size_t = len(row[label_local])

    return pd.concat([
        pd.Series(
            {col: row[col][i:i + size_window] for col in columns}
        ) for i in range(0, size_t, size_jump) if (size_t - i) > size_window - 1
    ], axis=1).T


def object_for_array(object_: str) -> ndarray:
    """
    Transforms an object into an array.

    Parameters
    ----------
    object_ : str
        object representing a list of integers or strings

    Returns
    -------
    array
        object converted to a list

    Example
    -------
    >>> from pymove.utils.trajectories import object_for_array
    >>>
    >>> object_1, object_2, object_3
    ('[1, 2, 3]', '[1.5, 2.5, 3.5]', "['event', 'event']")
    >>>
    >>> object_for_array(object_1)
    [1, 2, 3]
    >>> object_for_array(object_2)
    [1.5, 2.5, 3.5]
    >>> object_for_array(object_3)
    ['event', 'event'   ]
    """
    if object_ is None:
        return object_

    return literal_eval('[' + object_ + ']')[0]


def columns_to_array(
    traj_df: DataFrame,
    columns: list | None = None
):
    """
    Transforms all columns values to list.

    Parameters
    ----------
    traj_df : DataFrame
        The input trajectory data.
    columns : list, optional
        List of the columns for conversion.

    Example
    -------
    >>> from pymove.utils.trajectories import columns_to_array
    >>>
    >>> traj_df
                  ids                     descritions                   price
    0     '[1, 1, 1]'   "['event', 'event', 'event']"    '[10.5, 20.5, 13.5]'
    1     '[2, 2, 2]'      "['bike', 'bike', 'bike']"    '[50.2, 33.4, 90.0]'
    2  '[3, 3, 3, 3]'  "['car', 'car', 'car', 'car']"  '[1.0, 2.9, 3.4, 8.4]'
    3        '[4, 4]'            "['house', 'house']"        '[100.4, 150.5]'
    >>>
    >>> columns_to_array(traj_df)
    >>> traj_df
                ids            descritions                 price
    0     [1, 1, 1]  [event, event, event]    [10.5, 20.5, 13.5]
    1     [2, 2, 2]     [bike, bike, bike]    [50.2, 33.4, 90.0]
    2  [3, 3, 3, 3]   [car, car, car, car]  [1.0, 2.9, 3.4, 8.4]
    3        [4, 4]         [house, house]        [100.4, 150.5]
    """
    if columns is None:
        columns = list(traj_df.columns)

    f = {col: object_for_array for col in columns}
    traj_df[columns] = traj_df[columns].agg(f)
