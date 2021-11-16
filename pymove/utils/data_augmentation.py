"""
Data augmentation operations.

append_row,
generate_trajectories_df,
split_crossover,
_augmentation,
flatten_trajectories_dataframe,
instance_crossover_augmentation,
sliding_window,
get_all_paths,
transition_graph_augmentation_all_vertex

"""
from __future__ import annotations

from typing import TYPE_CHECKING, Text

import networkx as nx
import numpy as np
import pandas as pd
from networkx.classes.digraph import DiGraph
from pandas.core.frame import DataFrame
from pandas.core.series import Series

from pymove.utils.constants import DESTINY, LOCAL_LABEL, START, TID
from pymove.utils.log import progress_bar
from pymove.utils.networkx import build_transition_graph_from_df
from pymove.utils.trajectories import append_trajectory, split_trajectory

if TYPE_CHECKING:
    from pymove.core.dask import DaskMoveDataFrame
    from pymove.core.pandas import PandasMoveDataFrame


def append_row(
    data: DataFrame,
    row: Series | None = None,
    columns: dict | None = None
):
    """
    Insert a new line in the dataframe with the information passed by parameter.

    Parameters
    ----------
    data : DataFrame
        The input trajectories data.
    row : Series, optional
        The row of a dataframe, by default None
    columns : dict, optional
        Dictionary containing the values to be added, by default None

    Example
    -------
    >>> from pymove.utils.data_augmentation import append_row
    >>>
    >>> df
      id            datetime  local        lat         lon          tid
    0  1  2017-09-02 21:59:34    162  -3.843132  -38.593314  12017090221
    1  1  2017-09-02 22:00:27     85  -3.834748  -38.592189  12017090222
    2  1  2017-09-02 22:01:36    673  -3.823583  -38.590389  12017090222
    3  1  2017-09-02 22:03:08    394  -3.813889  -38.590444  12017090222
    4  1  2017-09-02 22:03:46    263  -3.906765  -38.590772  12017090222
    5  1  2017-09-02 22:07:19    224  -3.885722  -38.592889  12017090222
    >>>
    >>> row
    id                            1
    datetime    2017-09-02 22:07:40
    local                       623
    lat                    -3.88287
    lon                     -38.593
    tid                 12017090222
    dtype: object
    >>>
    >>> append_row(df, row)
    >>> df
      id             datetime  local        lat         lon          tid
    0  1  2017-09-02 21:59:34    162  -3.843132  -38.593314  12017090221
    1  1  2017-09-02 22:00:27     85  -3.834748  -38.592189  12017090222
    2  1  2017-09-02 22:01:36    673  -3.823583  -38.590389  12017090222
    3  1  2017-09-02 22:03:08    394  -3.813889  -38.590444  12017090222
    4  1  2017-09-02 22:03:46    263  -3.906765  -38.590772  12017090222
    5  1  2017-09-02 22:07:19    224  -3.885722  -38.592889  12017090222
    6  1  2017-09-02 22:07:40    623  -3.882872  -38.592979  12017090222

    """
    if row is not None:
        keys = row.index.tolist()
        data.at[data.shape[0], keys] = row.values
    else:
        if isinstance(columns, dict):
            keys = list(columns.keys())
            values = [np.array(v).tolist() for v in list(columns.values())]
            data.at[data.shape[0], keys] = values


def generate_trajectories_df(
    data: 'PandasMoveDataFrame' | 'DaskMoveDataFrame',
    label_tid: Text = TID,
    min_points_traj: int = 3
) -> DataFrame:
    """
    Generates a dataframe with the sequence of location points of a trajectory.

    Parameters
    ----------
    data : DataFrame
        The input trajectory data.
    label_tid: String, optional
        Label referring to the ID of the trajectories, by default TID
    min_points_traj: Number, optional
        Minimum points per trajectory, by default 3

    Return
    ------
    DataFrame
        DataFrame of the trajectories

    Example
    -------
    >>> from pymove.utils.data_augmentation import generate_trajectories_df
    >>>
    >>> df
      id             datetime  local         lat          lon          tid
    0  1  2017-09-02 21:59:34    162  -3.8431323  -38.5933142  12017090221
    1  1  2017-09-02 22:00:27     85  -3.8347478  -38.5921890  12017090222
    2  1  2017-09-02 22:01:36    673  -3.8235834  -38.5903890  12017090222
    3  1  2017-09-02 22:03:08    394  -3.8138890  -38.5904445  12017090222
    4  1  2017-09-02 22:03:46    263  -3.9067654  -38.5907723  12017090222
    5  1  2017-09-02 22:07:19    224  -3.8857223  -38.5928892  12017090222
    6  1  2017-09-02 22:07:40    623  -3.8828723  -38.5929789  12017090222
    >>>
    >>> traj_df = generate_trajectories_df(df)
    >>> traj_df.local
    0    [85, 673, 394, 263, 224, 623]
    Name: local, dtype: object

    """
    if label_tid not in data:
        raise ValueError(
            '{} not in DataFrame'.format(label_tid)
        )

    frames = []
    tids = data[label_tid].unique()

    desc = 'Gererating Trajectories DataFrame'
    for tid in progress_bar(tids, desc=desc, total=len(tids)):
        frame = data[data[label_tid] == tid]

        if frame.shape[0] >= min_points_traj:
            frames.append(frame.T.values.tolist())

    return pd.DataFrame(frames, columns=data.columns)


def split_crossover(
    sequence_a: list, sequence_b: list, frac: float = 0.5
) -> tuple[list, list]:
    """
    Divides two arrays in the indicated ratio and exchange their halves.

    Parameters
    ----------
    sequence_a : list or ndarray
        Array any
    sequence_b : list or ndarray
        Array any
    frac : float, optional
        Represents the percentage to be exchanged, by default 0.5

    Returns
    -------
    tuple[list, list]
        Arrays with the halves exchanged.

    Example
    -------
    >>> from pymove.utils.data_augmentation import split_crossover
    >>>
    >>> sequence_a, sequence_b
    ([0, 2, 4, 6, 8], [1, 3, 5, 7, 9])
    >>>
    >>> sequence_a, sequence_b = split_crossover(sequence_a, sequence_b)
    >>> sequence_a, sequence_b
    ([0, 2, 5, 7, 9], [1, 3, 4, 6, 8])

    """
    size_a = int(len(sequence_a) * frac)
    size_b = int(len(sequence_b) * frac)

    sequence_a1 = sequence_a[:size_a]
    sequence_a2 = sequence_a[size_a:]

    sequence_b1 = sequence_b[:size_b]
    sequence_b2 = sequence_b[size_b:]

    sequence_a = np.concatenate((sequence_a1, sequence_b2))
    sequence_b = np.concatenate((sequence_b1, sequence_a2))

    return sequence_a, sequence_b


def _augmentation(
    traj_df: DataFrame, frac: float = 0.5
):
    """
    Generates new data with unobserved trajectories.

    Parameters
    ----------
    data : DataFrame
        The input trajectories data.
    frac : float, optional
        Represents the percentage to be exchanged, by default 0.5

    Return
    ------
    DataFrame
        Increased data set.

    Example
    -------
    >>> from pymove.utils.data_augmentation import _augmentation
    >>>
    >>> traj_df
                  id                 local
    0      [1, 1, 1]        [85, 673, 394]
    1 	[2, 2, 2, 2]  [263, 224, 623, 515]
    >>>
    >>> _augmentations(traj_df, frac=0.5)
                  id                 local
    0 	   [1, 1, 1] 	    [85, 673, 394]
    1 	[2, 2, 2, 2]  [263, 224, 623, 515]
    2 	   [1, 2, 2]    	[85, 623, 515]
    3 	[2, 2, 1, 1]  [263, 224, 673, 394]

    """
    traj_df.reset_index(drop=True, inplace=True)

    frames = {}
    for idx, row in traj_df.iterrows():
        if idx + 1 < traj_df.shape[0]:
            series = {}
            for column in traj_df.columns:
                series[column] = pd.Series(
                    traj_df[idx + 1:][column].apply(
                        lambda x: split_crossover(row[column], x, frac)
                    ).values[0], name=column,
                )
            frames[idx] = pd.concat([series[col] for col in traj_df.columns], axis=1)

    aug_df = pd.concat(
        [frames[i] for i in range(len(frames))], axis=1
    )
    return pd.concat([traj_df, aug_df], ignore_index=True)


def flatten_trajectories_dataframe(traj_df: DataFrame) -> DataFrame:
    """
    Extracts information from trajectories.

    Parameters
    ----------
    traj_df : DataFrame
        The input trajectories data

    Return
    ------
    DataFrames
        Flat trajectories.

    Example
    -------
    >>> from pymove.utils.data_augmentation import flatten_trajectories_dataframe
    >>>
    >>> traj_df
                 id                 local
    0     [1, 1, 1]        [85, 673, 394]
    1  [2, 2, 2, 2]  [263, 224, 623, 515]
    >>>
    >>> flatten_trajectories_dataframe(traj_df)
       id  local
    0   1     85
    1   1    673
    2   1    394
    3   2    263
    4   2    224
    5   2    623
    6   2    515

    """
    frames = {}
    for idx, row in progress_bar(traj_df.iterrows(), total=traj_df.shape[0]):
        frames[idx] = pd.DataFrame(row.to_dict())

    return pd.concat([frames[i] for i in range(len(frames))], ignore_index=True)


def instance_crossover_augmentation(
    data: DataFrame,
    restriction: str = 'destination only',
    label_local: Text = LOCAL_LABEL,
    frac: float = 0.5,
) -> DataFrame:
    """
    Generates new data from unobserved trajectories, with a specific restriction.

    By default, the algorithm uses the same destination constraint
    as the route and inserts the points on the
    original dataframe.

    Parameters
    ----------
    data : DataFrame
        The input trajectories data
    restriction : str, optional
        Constraint used to generate new data, by default 'destination only'
    label_local : str, optional
        Label of the points sequences, by default LOCAL_LABEL
    frac : float, optional
        Represents the percentage to be exchanged, by default 0.5

    Example
    -------
    >>> from pymove.utils.data_augmentation import instance_crossover_augmentation
    >>>
    >>> df
                 id 	     local_label
    0     [1, 1, 1]       [85, 673, 394]
    1  [2, 2, 2, 2]  [85, 224, 623, 394]
    2     [3, 3, 3]      [263, 673, 394]
    >>>
    >>> aug_df = instance_crossover_augmentation(df)
    >>> aug_df
                 id 	     local_label
    0     [1, 1, 1]       [85, 673, 394]
    1  [2, 2, 2, 2]  [85, 224, 623, 394]
    2     [3, 3, 3]      [263, 673, 394]
    3     [1, 2, 2]       [85, 623, 394]
    4  [2, 2, 1, 1]  [85, 224, 673, 394]
    5  [2, 2, 3, 3]  [85, 224, 673, 394]
    6     [3, 2, 2]      [263, 623, 394]

    """
    df = data.copy()

    df[DESTINY] = df[label_local].apply(lambda x: x[-1])
    df[START] = df[label_local].apply(lambda x: x[0])

    frames = {}
    destinations = df[DESTINY].unique()
    for idx, dest in progress_bar(enumerate(destinations), total=len(destinations)):
        filter_ = df[df[DESTINY] == dest]

        if restriction == 'departure and destination':
            starts = filter_[START].unique()

            for st in progress_bar(starts, total=len(starts)):
                filter_ = filter_[filter_[START] == st]

                if filter_.shape[0] >= 2:
                    frames[idx] = _augmentation(filter_.iloc[:, :-2], frac=frac)
        else:
            if filter_.shape[0] >= 2:
                frames[idx] = _augmentation(filter_.iloc[:, :-2], frac=frac)

    return pd.concat([frames[i] for i in range(len(frames))], axis=0, ignore_index=True)


def sliding_window(
    data: DataFrame,
    size_window: int = 6,
    size_jump: int = 3,
    label_local: Text = LOCAL_LABEL,
    columns: list = None,
) -> DataFrame:
    """
    Sliding window technique.

    Performs an increase in the trajectory data by sliding a window
    Over each sequence to a specified size n, skipping m points.
    This process inserts sub-trajectories in the data set.

    Parameters
    ----------
    data: DataFrame
        Trajectory data in sequence format
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
    DataFrame
        Increased data set.

    Example
    -------
    >>> from pymove.utils.data_augmentation import sliding_window
    >>>
    >>> traj_df
                             id                                   local
    0  [1, 1, 1, 1, 1, 1, 1, 1]  [85, 673, 394, 85, 224, 623, 394, 263]
    1        [2, 2, 2, 2, 2, 2]      [85, 224, 623, 394, 263, 673, 394]
    2     [3, 3, 3, 3, 3, 3, 3]  [263, 673, 394, 85, 673, 394, 85, 224]
    >>>
    >>> sliding_window(traj_df, size_jump=1)
                        id 	                  local_label
    0   [1, 1, 1, 1, 1, 1] 	[ 85, 673, 394, 85, 224, 623]
    1   [1, 1, 1, 1, 1, 1] 	[673, 394, 85, 224, 623, 394]
    2   [1, 1, 1, 1, 1, 1] 	[394, 85, 224, 623, 394, 263]
    3   [2, 2, 2, 2, 2, 2] 	[85, 224, 623, 394, 263, 673]
    4   [3, 3, 3, 3, 3, 3] 	[263, 673, 394, 85, 673, 394]
    5   [3, 3, 3, 3, 3, 3] 	[673, 394, 85, 673, 394,  85]

    """
    if columns is None:
        columns = data.columns

    frames = {}
    desc = 'Sliding Window...'
    for idx, row in progress_bar(data.iterrows(), desc=desc, total=data.shape[0]):
        frames[idx] = split_trajectory(row, size_window, size_jump, label_local, columns)

    return pd.concat([frame for frame in frames.values()], ignore_index=True)


def get_all_paths(
    traj_df: DataFrame, graph: DiGraph, source: str, target: str,
    min_path_size: int = 3, max_path_size: int = 6,
    max_sampling_source: int = 100,
    max_sampling_target: int = 100,
    label_local: str = LOCAL_LABEL,
    simple_paths: bool = False
):
    """
    Generate All Paths.

    Retrieves all paths in the graph between the past source and
    destination, if any. The number of paths returned is limited
    by the max_sampling_source and max_sampling_target
    parameters, and the path size is limited by the
    min_path_size and max_path_size parameters.

    Parameters
    ----------
    traj_df: DataFrame
        Trajectory data in sequence format.
    graph: DiGraph
        Transition graph constructed from trajectory data.
    source: Node
        Sequence source node.
    target: Node
        Sequence destination node.
    min_path_size: int, optional
        Minimum number of points for the trajectory, by default 3
    max_path_size: int, optional
        Maximum number of points for the trajectory, by default 6
    max_sampling_source: int, optional
        Maximum number of paths to be returned,
        considering the observed origin, by default 10
    max_sampling_target: int, optional
        Maximum number of paths to be returned,
        considering the observed destination, by default 10
    label_local: str, optional
        Name of the column referring to the trajectories, by default LOCAL_LABEL
    simple_paths: bool, optional
        If true, use the paths with the most used sections
        Otherwise, use paths with less used sections, by default False

    Example
    -------
    >>> from pymove.utils.data_augmentation import get_all_paths
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
    >>> graph = build_transition_graph_from_df(traj_df)
    >>>
    >>> get_all_paths(traj_df, graph, 224, 394)
    [224.0, 623.0, 394.0]

    """
    if not nx.has_path(graph, source, target):
        return []

    param: int | None = None

    if simple_paths:
        all_paths = nx.all_simple_paths
        param = max_path_size - 1

    else:
        all_paths = nx.shortest_simple_paths

    for path in all_paths(graph, source, target, param):
        freq_source = nx.get_node_attributes(graph, 'freq_source')[source]
        freq_target = nx.get_node_attributes(graph, 'freq_target')[target]

        if freq_source >= max_sampling_source:
            break

        if freq_target >= max_sampling_target:
            break

        if len(path) > max_path_size and simple_paths is False:
            break

        if len(path) >= min_path_size:
            path_ = np.array(path, dtype='float32').tolist()
            if path_ not in traj_df[label_local].values.tolist():

                print(path_)
                append_trajectory(traj_df, path, graph)

                freq_source += 1
                freq_target += 1

                graph.add_node(source, freq_source=freq_source)
                graph.add_node(target, freq_target=freq_target)


def transition_graph_augmentation_all_vertex(
    traj_df: DataFrame,
    graph: DiGraph | None = None,
    min_path_size: int = 3,
    max_path_size: int = 6,
    max_sampling_source: int = 10,
    max_sampling_target: int = 10,
    source: dict | None = None,
    target: dict | None = None,
    label_local: Text = LOCAL_LABEL,
    simple_paths: bool = False,
    inplace: bool = True
) -> DataFrame:
    """
    Transition Graph Data Augmentation.

    Performs the data increase from the transition graph.

    Parameters
    ----------
    traj_df: DataFrame
        Trajectory data in sequence format
    graph: DiGraph
        Transition graph constructed from trajectory data
    min_path_size: int, optional
        Minimum number of points for the trajectory, by default 3
    max_path_size: int, optional
        Maximum number of points for the trajectory, by default 6
    max_sampling_source: int, optional
        Maximum number of paths to be returned,
        considering the observed origin, by default 10
    max_sampling_target: int, optional
        Maximum number of paths to be returned,
        considering the observed destination, by default 10
    source: dict, optional
        Degree of entry of each node in the graph, by default None
        Example: {node: degree-of-entry}
    target: dict, optional
        Degree of output of each node in the graph, by default None
        Example: {node: degree-of-output}
    label_local: str, optional
        Name of the column referring to the trajectories, by default LOCAL_LABEL
    label_tid: str, optional
        Column name for trajectory IDs, by default TID_STAT
    simple_paths: boolean, optional
        If true, use the paths with the most used sections
        Otherwise, use paths with less used sections, by default False
    inplace : boolean, optional
        if set to true the original dataframe will be altered to contain the result
        of the augmentation, otherwise a copy will be returned, by default True

    Return
    ------
    DataFrame
        Increased data set.

    Example
    -------
    >>> from pymove.utils.data_augmentation import (
            transition_graph_augmentation_all_vertex
        )
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
    >>> transition_graph_augmentation_all_vertex(traj_df)
    [263.0, 224.0, 623.0]
    [224.0, 623.0, 394.0]

    """
    if inplace:
        traj_df_ = traj_df
    else:
        traj_df_ = traj_df.copy()

    if graph is None:
        graph = build_transition_graph_from_df(traj_df_)

    if source is None:
        source = dict(graph.nodes)
        source = {key: value['freq_source'] for key, value in source.items()}

    if target is None:
        target = dict(graph.nodes)
        target = {key: value['freq_source'] for key, value in target.items()}

    targets = sorted(target.items(), key=lambda x: x[1], reverse=True)
    sources = sorted(source.items(), key=lambda x: x[1], reverse=True)

    [[get_all_paths(
        traj_df_, graph, s, t, min_path_size, max_path_size,
        max_sampling_source, max_sampling_target, label_local, simple_paths
    ) for s, _ in sources] for t, _ in targets]

    if not inplace:
        return traj_df_
