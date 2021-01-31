from typing import TYPE_CHECKING, Dict, List, Optional, Text, Tuple, Union

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from pandas.core.series import Series

from pymove.utils.constants import DESTINY, START, TID, TRAJECTORY
from pymove.utils.log import progress_bar

if TYPE_CHECKING:
    from pymove.core.dask import DaskMoveDataFrame
    from pymove.core.pandas import PandasMoveDataFrame


def append_row(
    data: DataFrame, row: Optional[Series] = None, columns: Optional[Dict] = None
):
    """
    Insert a new line in the dataframe with
    the information passed by parameter.

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
    Generates a dataframe with the sequence of
    location points of a trajectory.

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
    Removes the last point from the trajectory and
    adds it in a new column called 'destiny'.

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
    Removes the first point from the trajectory and
    adds it in a new column called 'start'.

    Parameters
    ----------
    data : DataFrame
        The input trajectory data.
    label_trajectory : str, optional
        Label of the points sequences, by default 'trajectory'

    """
    if DESTINY not in data:
        data[DESTINY] = data[label_trajectory].apply(
            lambda x: np.int64(x[-1])
        )


def split_crossover(
    sequence_a: List, sequence_b: List, frac: Optional[float] = 0.5
) -> Tuple[List, List]:
    """
    Divide two arrays in the indicated ratio
    and exchange their halves.

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
    Tuple[List, List]
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
    frac : number, optional
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
    Generate new data from unobserved trajectories,
    given a specific restriction. By default, the
    algorithm uses the same route destination constraint.

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
    frac : number, optional
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
    Inserts the points of the generated trajectories
    to the original data sets.

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
    Generate new data from unobserved trajectories,
    with a specific restriction. By default, the
    algorithm uses the same destination constraint
    as the route and inserts the points on the
    original dataframe.

    Parameters
    ----------
    data : DataFrame
        The input trajectories data
    restriction : str, optional
        Constraint used to generate new data, by default 'destination only'
    label_trajectory : str, optional
        Label of the points sequences, by default 'trajectory'
    frac : number, optional
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
