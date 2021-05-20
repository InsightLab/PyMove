"""
Data augmentation operations.

append_row,
generate_trajectories_df,
generate_start_feature,
generate_destiny_feature,
split_crossover,
augmentation_trajectories_df,
insert_points_in_df,
instance_crossover_augmentation,
sliding_window,
transition_graph_augmentation_all_vertex

"""

from typing import TYPE_CHECKING, Dict, List, Optional, Text, Tuple, Union

import numpy as np
import pandas as pd
from networkx.classes.digraph import DiGraph
from pandas.core.frame import DataFrame
from pandas.core.series import Series

from pymove.utils.constants import DESTINY, LOCAL_LABEL, START, TID, TID_STAT, TRAJECTORY
from pymove.utils.log import progress_bar
from pymove.utils.networkx import build_transition_graph_from_df, get_all_paths
from pymove.utils.trajectories import split_trajectory

if TYPE_CHECKING:
    from pymove.core.dask import DaskMoveDataFrame
    from pymove.core.pandas import PandasMoveDataFrame


def append_row(
    data: DataFrame, row: Optional[Series] = None, columns: Optional[Dict] = None
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
    data: Union['PandasMoveDataFrame', 'DaskMoveDataFrame']
) -> DataFrame:
    """
    Generates a dataframe with the sequence of location points of a trajectory.

    Parameters
    ----------
    data : DataFrame
        The input trajectory data.

    Return
    ------
    DataFrame
        DataFrame of the trajectories

    """
    if TID not in data:
        data.generate_tid_based_on_id_datetime()
        data.reset_index(drop=True, inplace=True)

    tids = data[TID].unique()
    new_df = pd.DataFrame(
        columns=data.columns
    )

    for tid in progress_bar(tids, total=len(tids)):
        filter_ = data[data[TID] == tid]
        filter_.reset_index(drop=True, inplace=True)

        if filter_.shape[0] > 4:

            values = []
            for col in filter_.columns:
                if filter_[col].nunique() == 1:
                    values.append(filter_.at[0, col])
                else:
                    values.append(
                        np.array(
                            filter_[col], dtype=type(filter_.at[0, col])
                        ).tolist()
                    )

            row = pd.Series(values, filter_.columns)
            append_row(new_df, row=row)

    return new_df


def generate_start_feature(
    data: DataFrame, label_trajectory: Optional[Text] = TRAJECTORY
):
    """
    Removes the last point from the trajectory and adds it in a new column 'destiny'.

    Parameters
    ----------
    data : DataFrame
        The input trajectory data.
    label_trajectory : str, optional
        Label of the points sequences, by default TRAJECTORY

    """
    if START not in data:
        data[START] = data[label_trajectory].apply(
            lambda x: np.int64(x[0])
        )


def generate_destiny_feature(
    data: DataFrame, label_trajectory: Optional[Text] = TRAJECTORY
):
    """
    Removes the first point from the trajectory and adds it in a new column 'start'.

    Parameters
    ----------
    data : DataFrame
        The input trajectory data.
    label_trajectory : str, optional
        Label of the points sequences, by default TRAJECTORY

    """
    if DESTINY not in data:
        data[DESTINY] = data[label_trajectory].apply(
            lambda x: np.int64(x[-1])
        )


def split_crossover(
    sequence_a: List, sequence_b: List, frac: Optional[float] = 0.5
) -> Tuple[List, List]:
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
    tuple(list, list)
        Arrays with the halves exchanged.

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


def _augmentation(data: DataFrame, aug_df: DataFrame, frac: Optional[float] = 0.5):
    """
    Generates new data with unobserved trajectories.

    Parameters
    ----------
    data : DataFrame
        The input trajectories data.
    aug_df : DataFrame
        The dataframe with new trajectories
    frac : float, optional
        Represents the percentage to be exchanged, by default 0.5

    """
    data.reset_index(drop=True, inplace=True)

    for idx in range(data.shape[0] - 1):
        for idx_ in range(idx + 1, data.shape[0]):
            sequences1 = []
            sequences2 = []

            columns = data.columns

            for col in columns:
                if (isinstance(
                    data.at[idx, col], list
                ) or isinstance(
                    data.at[idx, col], np.ndarray
                )) and (isinstance(
                    data.at[idx_, col], list
                ) or isinstance(
                    data.at[idx_, col], np.ndarray
                )):
                    seq1, seq2 = split_crossover(
                        data.at[idx, col],
                        data.at[idx_, col],
                        frac=frac
                    )
                    sequences1.append(seq1)
                    sequences2.append(seq2)
                else:
                    value1 = data.at[idx, col]
                    value2 = data.at[idx_, col]

                    if isinstance(value1, str) and isinstance(value2, str):
                        sequences1.append(value1 + '_' + value2)
                        sequences2.append(value2 + '_' + value1)
                    else:
                        sequences1.append(value1)
                        sequences2.append(value2)

            row = pd.Series(sequences1, index=columns)
            append_row(aug_df, row=row)

            row = pd.Series(sequences2, index=columns)
            append_row(aug_df, row=row)


def augmentation_trajectories_df(
    data: Union['PandasMoveDataFrame', 'DaskMoveDataFrame'],
    restriction: Optional[Text] = 'destination only',
    label_trajectory: Optional[Text] = TRAJECTORY,
    insert_at_df: Optional[bool] = False,
    frac: Optional[float] = 0.5,
) -> DataFrame:
    """
    Generates new data from unobserved trajectories, given a specific restriction.

    By default, the algorithm uses the same route destination constraint.

    Parameters
    ----------
    data : DataFrame
        The input trajectories data.
    restriction : str, optional
        Constraint used to generate new data, by default 'destination only'
    label_trajectory : str, optional
        Label of the points sequences, by default TRAJECTORY
    insert_at_df : boolean, optional
        Whether to return a new DataFrame, by default False
        If True then value of copy is ignored.
    frac : float, optional
        Represents the percentage to be exchanged, by default 0.5

    Returns
    -------
    DataFrame
        Dataframe with the new data generated

    """
    if DESTINY not in data:
        generate_destiny_feature(data, label_trajectory=label_trajectory)

    if restriction == 'departure and destination':
        generate_start_feature(data)

    if insert_at_df:
        aug_df = data
    else:
        aug_df = pd.DataFrame(columns=data.columns)

    destinations = data[DESTINY].unique()
    for dest in progress_bar(destinations, total=len(destinations)):
        filter_ = data[data[DESTINY] == dest]

        if restriction == 'departure and destination':
            starts = filter_[START].unique()

            for st in progress_bar(starts, total=len(starts)):
                f_filter_ = filter_[filter_[START] == st]

                if f_filter_.shape[0] >= 2:
                    _augmentation(f_filter_, aug_df, frac=frac)

        else:
            if filter_.shape[0] >= 2:
                _augmentation(filter_, aug_df, frac=frac)

    return aug_df


def insert_points_in_df(data: DataFrame, aug_df: DataFrame):
    """
    Inserts the points of the generated trajectories to the original data sets.

    Parameters
    ----------
    data : DataFrame
        The input trajectories data
    aug_df : DataFrame
        The data of unobserved trajectories

    """
    for _, row in progress_bar(aug_df.iterrows(), total=aug_df.shape[0]):

        keys = row.index.tolist()
        values = row.values.tolist()

        row_df = pd.DataFrame()

        for k, v in zip(keys, values):
            if k in data:
                if isinstance(v, list) or isinstance(v, np.ndarray):
                    row_df[k] = v

        for k, v in zip(keys, values):
            if k in data:
                if not isinstance(v, list) and not isinstance(v, np.ndarray):
                    row_df[k] = v

        for _, row_ in row_df.iterrows():
            append_row(data, row=row_)


def instance_crossover_augmentation(
    data: DataFrame,
    restriction: Optional[Text] = 'destination only',
    label_trajectory: Optional[Text] = TRAJECTORY,
    frac: Optional[float] = 0.5
):
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
    label_trajectory : str, optional
        Label of the points sequences, by default TRAJECTORY
    frac : float, optional
        Represents the percentage to be exchanged, by default 0.5

    """
    traj_df = generate_trajectories_df(data)

    generate_destiny_feature(traj_df, label_trajectory=label_trajectory)

    if restriction == 'departure and destination':
        generate_start_feature(traj_df, label_trajectory=label_trajectory)

    aug_df = augmentation_trajectories_df(
        traj_df, restriction=restriction, frac=frac
    )
    insert_points_in_df(data, aug_df)


def sliding_window(
    data: DataFrame,
    size_window: Optional[int] = 6,
    size_jump: Optional[int] = 3,
    label_local: Optional[Text] = LOCAL_LABEL,
    columns: Optional[List] = None,
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
        Name of the column referring to the trajectories, by default 'local_label'
    columns: list, optional
        Columns to which the split will be applied, by default None

    Return
    ------
    DataFrame
        Increased data set.
    """
    if columns is None:
        columns = data.columns

    frames = {}
    desc = 'Sliding Window...'
    for idx, row in progress_bar(data.iterrows(), desc=desc, total=data.shape[0]):
        frames[idx] = split_trajectory(row, size_window, size_jump, label_local, columns)

    return pd.concat(
        [frame for frame in frames.values()],
        ignore_index=True
    )


def transition_graph_augmentation_all_vertex(
    data: DataFrame,
    graph: Optional[DiGraph] = None,
    min_path_size: Optional[int] = 3,
    max_path_size: Optional[int] = 6,
    max_sampling_source: Optional[int] = 10,
    max_sampling_target: Optional[int] = 10,
    source: Optional[Dict] = None,
    target: Optional[Dict] = None,
    label_local: Optional[Text] = LOCAL_LABEL,
    label_tid: Optional[Text] = TID_STAT,
    simple_paths: Optional[bool] = False,
    inplace: Optional[bool] = True
) -> DataFrame:
    """
    Transition Graph Data Augmentation.

    Performs the data increase from the transition graph.

    Parameters
    ----------
    data: DataFrame
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
    """
    if inplace:
        data_ = data
    else:
        data_ = data.copy()

    if graph is None:
        graph = build_transition_graph_from_df(data)

    if source is None:
        source = dict(graph.nodes)
        source = {key: value['freq_source'] for key, value in source.items()}

    if target is None:
        target = dict(graph.nodes)
        target = {key: value['freq_source'] for key, value in target.items()}

    targets = sorted(target, key=target.get, reverse=True)
    desc = 'Trajectory OverSampling'
    for t in progress_bar(targets, desc=desc, total=len(targets)):
        for s in sorted(source, key=source.get, reverse=True):
            get_all_paths(
                data=data, graph=graph, source=s, target=t, min_path_size=min_path_size,
                max_path_size=max_path_size, max_sampling_source=max_sampling_source,
                max_sampling_target=max_sampling_target, label_local=label_local,
                label_tid=label_tid, simple_paths=simple_paths
            )

    if not inplace:
        return data_
