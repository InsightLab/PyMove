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


import random
import uuid
from itertools import chain
from typing import Any, Dict, Generator, List, Optional, Text, Union

import numpy as np
import pandas as pd
from networkx.classes.digraph import DiGraph
from numpy import ndarray
from pandas import read_csv as _read_csv
from pandas._typing import FilePathOrBuffer
from pandas.core.frame import DataFrame
from pandas.core.series import Series

from pymove.core.dataframe import MoveDataFrame
from pymove.utils.constants import (
    DATETIME,
    LATITUDE,
    LOCAL_LABEL,
    LONGITUDE,
    PREV_LOCAL,
    TID_STAT,
    TRAJ_ID,
    TYPE_PANDAS,
)
from pymove.utils.networkx import graph_to_dict


def read_csv(
    filepath_or_buffer: FilePathOrBuffer,
    latitude: Optional[Text] = LATITUDE,
    longitude: Optional[Text] = LONGITUDE,
    datetime: Optional[Text] = DATETIME,
    traj_id: Optional[Text] = TRAJ_ID,
    type_: Optional[Text] = TYPE_PANDAS,
    n_partitions: Optional[int] = 1,
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
        Represents the column name of feature latitude, by default LATITUDE
    longitude : str, optional
        Represents the column name of feature longitude, by default LONGITUDE
    datetime : str, optional
        Represents the column name of feature datetime, by default DATETIME
    traj_id : str, optional
        Represents the column name of feature id trajectory, by default TRAJ_ID
    type_ : str, optional
        Represents the type of the MoveDataFrame, by default TYPE_PANDAS
    n_partitions : int, optional
        Represents number of partitions for DaskMoveDataFrame, by default 1
    **kwargs : Pandas read_csv arguments
        https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html?highlight=read_csv#pandas.read_csv

    Returns
    -------
    MoveDataFrameAbstract subclass
        Trajectory data

    """
    data = _read_csv(
        filepath_or_buffer,
        **kwargs
    )

    return MoveDataFrame(
        data, latitude, longitude, datetime, traj_id, type_, n_partitions
    )


def invert_dict(d: Dict) -> Dict:
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

    """
    return {v: k for k, v in d.items()}


def flatten_dict(
    d: Dict, parent_key: Optional[Text] = '', sep: Optional[Text] = '_'
) -> Dict:
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


def flatten_columns(data: DataFrame, columns: List) -> DataFrame:
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
    >>> d = {'a': 1, 'b': {'c': 2, 'd': 3}}
    >>>> data = pd.DataFrame({'col1': [1], 'col2': [d]})
    >>>> flatten_columns(data, ['col2'])
       col1  col2_b_d  col2_a  col2_b_c
    0     1         3       1         2

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
    arr: Union[List, Series, ndarray],
    num: int,
    fill_value: Optional[Any] = None
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

    """
    result = np.empty_like(arr)
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


def fill_list_with_new_values(original_list: List, new_list_values: List):
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

    """
    n = len(new_list_values)
    original_list[:n] = new_list_values


def append_trajectory(
    data: DataFrame,
    trajectory: List,
    graph: DiGraph,
    label_tid: Optional[Text] = TID_STAT
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
    label_tid: str, optional
        Column name for trajectory IDs, by default TID_STAT

    Example
    -------
    >>> from pymove.utils.data_augmentation import append_trajectory
    >>> traj_df
        id               datetime    local           lat            lon    prev  tid_stat
    0  [1,  [2017-09-02 22:00:27,    [ 85,  [-3.8347478,  [-38.5921890,   [nan,       [1,
    .   1,   2017-09-02 22:01:36,     673,   -3.8235834,   -38.5903890,     85,        1,
    .   1]   2017-09-02 22:03:08]     394]   -3.8138890]   -38.5904445]    673]        1]
    1  [2,  [2017-09-02 23:03:46,    [263,  [-3.9067654,  [-38.5907723,   [nan,       [2,
    .   2,   2017-09-02 23:07:19,     224,   -3.8857223,   -38.5928892,    263,        2,
    .   2,   2017-09-02 23:07:40,     623,   -3.8828723,   -38.5929789,    224,        2,
    .   2]   2017-09-02 23:09:10]     394]   -3.9939834]   -38.7040900,    623]        2]
    >>>
    >>> trajectory = [263, 224, 623]
    >>> append_trajectory(traj_df, trajectory, graph)
    >>> traj_df
        id               datetime    local           lat            lon    prev  tid_stat
    0  [1,  [2017-09-02 22:00:27,    [ 85,  [-3.8347478,  [-38.5921890,   [nan,       [1,
    .   1,   2017-09-02 22:01:36,     673,   -3.8235834,   -38.5903890,     85,        1,
    .   1]   2017-09-02 22:03:08]     394]   -3.8138890]   -38.5904445]    673]        1]
    1  [2,  [2017-09-02 23:03:46,    [263,  [-3.9067654,  [-38.5907723,   [nan,       [2,
    .   2,   2017-09-02 23:07:19,     224,   -3.8857223,   -38.5928892,    263,        2,
    .   2,   2017-09-02 23:07:40,     623,   -3.8828723,   -38.5929789,    224,        2,
    .   2]   2017-09-02 23:09:10]     394]   -3.9939834]   -38.7040900,    623]        2]
    2  [3,  [2017-09-02 23:07:19,  [224.0,  [-3.8857223,  [-38.5928892,   [nan,       [3,
    .   3,   2017-09-02 23:07:40,   623.0,   -3.8828723,   -38.5929789,  224.0,        3,
    .   3]   2017-09-02 23:09:10]   394.0]   -3.9939834]   -38.7040900]  623.0]        3]
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

    prev_tid = data.loc[data.shape[0] - 1, label_tid][0]
    tids = np.full(len(trajectory), prev_tid + 1, dtype=np.int32).tolist()

    rd = random.Random()
    rd.seed(tids[0])
    traj_ids = np.full(len(trajectory),
                       uuid.UUID(int=rd.getrandbits(128)).hex,
                       dtype=np.object).tolist()

    path = np.array(trajectory, dtype=np.float32).tolist()
    prev_locals = [np.nan] + path[:-1]

    data.loc[data.shape[0], [
        DATETIME, TRAJ_ID, LOCAL_LABEL, LATITUDE, LONGITUDE, PREV_LOCAL, label_tid
    ]] = [datetimes, traj_ids, path, lats, lons, prev_locals, tids]


def split_trajectory(
    row: Series,
    size_window: Optional[int] = 6,
    size_jump: Optional[int] = 3,
    label_local: Optional[Text] = LOCAL_LABEL,
    columns: Optional[List] = None
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
    id                                     [1, 1, 1, 1, 1, 1]
    datetime       [2017-09-02 22:00:27, 2017-09-02 22:01:36, \
                    2017-09-02 22:03:08, 2017-09-02 22:03:46, \
                    2017-09-02 22:07:19, 2017-09-02 22:07:40]
    local_label                 [85, 673, 394, 263, 224, 623]
    lat                  [-3.8347478, -3.8235834, -3.8138890, \
                          -3.9067654, -3.8857223, -3.8828723]
    lon               [-38.5921890, -38.5903890, -38.5904445, \
                       -38.5907723, -38.5928892, -38.5929789]
    tid               [12017090222, 12017090222, 12017090222, \
                       12017090222, 12017090222, 12017090222]
    Name: 0, dtype: object
    dtype: object
    >>>
    >>> split = split_trajectory(trajectory)
    >>> split
        id               datetime  local           lat            lon             tid
    0  [1,  [2017-09-02 22:00:27,   [ 85,  [-3.8347478,   [-38.5921890,  [12017090222,
    .   1,   2017-09-02 22:01:36,    673,   -3.8235834,    -38.5903890,   12017090222,
    .   1,   2017-09-02 22:03:08,    394,   -3.8138890,    -38.5904445,   12017090222,
    .   1,   2017-09-02 22:03:46,    263,   -3.9067654,    -38.5907723,   12017090222,
    .   1,   2017-09-02 22:07:19,    224,   -3.8857223,    -38.5928892,   12017090222,
    .   1]   2017-09-02 22:07:40]    623]   -3.8828723]    -38.5929789]   12017090222]
    1  [1,  [2017-09-02 22:03:46,   [263,  [-3.9067654,   [-38.5907723,  [12017090222,
    .   1,   2017-09-02 22:07:19,    224,   -3.8857223,    -38.5928892,   12017090222,
    .   1]   2017-09-02 22:07:40]    623]   -3.8828723]    -38.5929789]   12017090222]
    """
    if columns is None:
        columns = row.index

    return pd.concat(
        [pd.Series(
            {col: row[col][i:i + size_window] for col in columns}
        ) for i in range(0, len(row[label_local]), size_jump)], axis=1
    ).T


def object_for_array(object_: Text) -> ndarray:
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
    ('[1, 2, 3]', '[1.5, 2.5, 3.5]', '[event, event]')
    >>>
    >>> object_for_array(object_1)
    [1, 2, 3]
    >>> object_for_array(object_2)
    [1.5, 2.5, 3.5]
    >>> object_for_array(object_3)
    [event, event]
    """
    if object_ is None:
        return object_

    return eval('[' + object_ + ']', {'nan': np.nan})[0]


def columns_to_array(
    traj_df: DataFrame,
    columns: Optional[List] = None
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
