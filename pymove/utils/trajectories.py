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

import networkx as nx
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
from pymove.utils.math import is_number


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
        Column name for trajectory IDs, by default 'tid_stat'

    """
    rd = random.Random()

    datetimes, lats, lons = [], [], []
    node = trajectory[0]

    datetimes.append(
        np.random.choice(
            nx.get_node_attributes(graph, 'datetime')[node]
        )
    )
    coords = nx.get_node_attributes(graph, 'coords')

    lats.append(coords[node][0])
    lons.append(coords[node][1])

    for i in range(1, len(trajectory)):
        edge = (trajectory[i - 1], trajectory[i])
        mean_times = nx.get_edge_attributes(graph, 'mean_times')[edge]

        dt_curr = pd.Timestamp(str(datetimes[i - 1])) + pd.Timedelta(mean_times)
        datetimes.append(str(dt_curr))

        lats.append(coords[trajectory[i]][0])
        lons.append(coords[trajectory[i]][1])

    prev_tid = data.loc[data.shape[0] - 1, label_tid][0]
    tids = np.full(len(trajectory), prev_tid + 1, dtype=np.int32).tolist()

    rd.seed(tids[0])
    ids = np.full(len(trajectory),
                  uuid.UUID(int=rd.getrandbits(128)).hex,
                  dtype=np.object).tolist()

    path = np.array(trajectory, dtype=np.float32).tolist()

    prev_locals = [np.nan]
    prev_locals.extend(path[:-1])

    data.loc[data.shape[0], [
        DATETIME, TRAJ_ID, LOCAL_LABEL, LATITUDE, LONGITUDE, PREV_LOCAL, label_tid
    ]] = [datetimes, ids, path, lats, lons, prev_locals, tids]


def split_trajectory(
    row: Series,
    size_window: Optional[int] = 6,
    size_jump: Optional[int] = 3,
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
    columns: list, optional
        Columns to which the split will be applied, by default None

    Return
    ------
    Generator of Series
        Series with the stretches recovered from the observed trajectory.

    """
    if columns is None:
        columns = row.index

    sequence = row[columns[0]]
    for i in range(0, len(sequence), size_jump):
        row_ = row.copy()
        for col in columns:
            row_ = row_.append(
                pd.Series({
                    'window_' + col: row_[col][i:i + size_window]
                })
            )
        yield row_


def object_for_array(object_: Text) -> ndarray:
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

    """
    if object_ is None:
        return object_

    conv = np.array([*map(str.strip, object_[1:-1].split(','))])

    if is_number(conv[0]):
        return conv.astype(np.float32)
    else:
        return conv.astype('object_')


def column_to_array(data: DataFrame, column: Text):
    """
    Transforms all columns values to list.

    Parameters
    ----------
    data : DataFrame
        The input trajectory data
    column : str
        Label of data referring to the column for conversion

    """
    data = data.copy()
    if column not in data:
        raise KeyError(
            'Dataframe must contain a %s column' % column
        )

    data[column] = data[column].apply(object_for_array)
    return data
