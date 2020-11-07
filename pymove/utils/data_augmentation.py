import numpy as np
import pandas as pd

from pymove.utils.constants import (
    DATETIME,
    DESTINY,
    LABEL,
    LATITUDE,
    LOCAL_LABEL,
    LONGITUDE,
    START,
    TID,
    TRAJ_ID,
    TRAJECTORY,
)
from pymove.utils.log import progress_bar


def append_row(df_, row=None, columns=None):
    """
    Insert a new line in the dataframe with
    the information passed by parameter.

    Parameters
    ----------
    df_ : dataframe
        The input trajectories data.
    row : series, optional, default None
        The row of a dataframe.
    columns : dict, optional, default None
        Dictionary containing the values to be added.
    """
    if row is not None:
        keys = row.index.tolist()
        df_.at[df_.shape[0], keys] = row.values
    else:
        if isinstance(columns, dict):
            keys = list(columns.keys())
            values = [np.array(v).tolist() for v in list(columns.values())]
            df_.at[df_.shape[0], keys] = values


def generate_trajectories_df(df_):
    """
    Generates a dataframe with the sequence of
    location points of a trajectory.

    Parameters
    ----------
    df_ : dataframe
        The input trajectory data.

    Return
    ------
    dataframe
        DataFrame of the trajectories
    """
    if TID not in df_:
        df_.generate_tid_based_on_id_datetime()
        df_.reset_index(drop=True, inplace=True)

    tids = df_[TID].unique()
    new_df = pd.DataFrame(
        columns=df_.columns
    )

    for tid in progress_bar(tids, total=len(tids)):
        filter_ = df_[df_[TID] == tid]
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


def generate_start_feature(df_, label_trajectory=TRAJECTORY):
    """
    Removes the last point from the trajectory and
    adds it in a new column called 'destiny'.

    Parameters
    ----------
    df_ : dataframe
        The input trajectory data.
    label_trajectory : str, optional, default 'trajectory'
        Label of the points sequences
    """
    if START not in df_:
        df_[START] = df_[label_trajectory].apply(
            lambda x: np.int64(x[0])
        )


def generate_destiny_feature(df_, label_trajectory=TRAJECTORY):
    """
    Removes the first point from the trajectory and
    adds it in a new column called 'start'.

    Parameters
    ----------
    df_ : dataframe
        The input trajectory data.
    label_trajectory : str, optional, default 'trajectory'
        Label of the points sequences
    """
    if DESTINY not in df_:
        df_[DESTINY] = df_[label_trajectory].apply(
            lambda x: np.int64(x[-1])
        )


def split_crossover(sequence_a, sequence_b, frac=0.5):
    """
    Divide two arrays in the indicated ratio
    and exchange their halves.

    Parameters
    ----------
    sequence_a : list, np.ndarray
        Array any
    sequence_b : list, np.ndarray
        Array any
    frac : number, optional, default 0.5
        Represents the percentage to be exchanged.

    Returns
    -------
    arrays
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


def _augmentation(df_, aug_df, frac=0.5):
    """
    Generates new data with unobserved trajectories.

    Parameters
    ----------
    df_ : dataframe
        The input trajectories data.
    aug_df : dataframe
        The dataframe with new trajectories
    frac : number, optional, default 0.5
        Represents the percentage to be exchanged.
    """
    df_.reset_index(drop=True, inplace=True)

    for idx in range(df_.shape[0] - 1):
        for idx_ in range(idx + 1, df_.shape[0]):
            sequences1 = []
            sequences2 = []

            columns = df_.columns

            for col in columns:
                if (isinstance(
                    df_.at[idx, col], list
                ) or isinstance(
                    df_.at[idx, col], np.ndarray
                )) and (isinstance(
                    df_.at[idx_, col], list
                ) or isinstance(
                    df_.at[idx_, col], np.ndarray
                )):
                    seq1, seq2 = split_crossover(
                        df_.at[idx, col],
                        df_.at[idx_, col],
                        frac=frac
                    )
                    sequences1.append(seq1)
                    sequences2.append(seq2)
                else:
                    value1 = df_.at[idx, col]
                    value2 = df_.at[idx_, col]

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
    df_,
    restriction='destination only',
    label_trajectory=TRAJECTORY,
    insert_at_df=False,
    frac=0.5,
):
    """
    Generate new data from unobserved trajectories,
    given a specific restriction. By default, the
    algorithm uses the same route destination constraint.

    Parameters
    ----------
    df_ : dataframe
        The input trajectories data.
    restriction : str, optional, default 'destination only'
        Constraint used to generate new data.
    label_trajectory : str, optional, default 'trajectory'
        Label of the points sequences.
    insert_at_df : boolean, optional, default False
        Whether to return a new DataFrame.
        If True then value of copy is ignored.
    frac : number, optional, default 0.5
        Represents the percentage to be exchanged.

    Returns
    -------
    DataFrame or None
        Dataframe with the new data generated.
    """

    if DESTINY not in df_:
        generate_destiny_feature(df_, label_trajectory=label_trajectory)

    if restriction == 'departure and destination':
        generate_start_feature(df_)

    if insert_at_df:
        aug_df = df_
    else:
        aug_df = pd.DataFrame(columns=df_.columns)

    destinations = df_[DESTINY].unique()
    for dest in progress_bar(destinations, total=len(destinations)):
        filter_ = df_[df_[DESTINY] == dest]

        if restriction == 'departure and destination':
            starts = filter_[START].unique()

            for st in progress_bar(starts, total=len(starts)):
                ffilter_ = filter_[filter_[START] == st]

                if ffilter_.shape[0] >= 2:
                    _augmentation(ffilter_, aug_df, frac=frac)

        else:
            if filter_.shape[0] >= 2:
                _augmentation(filter_, aug_df, frac=frac)

    return aug_df


def insert_points_in_df(df_, aug_df):
    """
    Inserts the points of the generated trajectories
    to the original data sets.

    Parameters
    ----------
    df_ : dataframe
        The input trajectories data.
    aug_df : dataframe
        The data of unobserved trajectories.
    """
    for _, row in progress_bar(aug_df.iterrows(), total=aug_df.shape[0]):

        keys = row.index.tolist()
        values = row.values.tolist()

        row_df = pd.DataFrame()

        for k, v in zip(keys, values):
            if k in df_:
                if isinstance(v, list) or isinstance(v, np.ndarray):
                    row_df[k] = v

        for k, v in zip(keys, values):
            if k in df_:
                if not isinstance(v, list) and not isinstance(v, np.ndarray):
                    row_df[k] = v

        for _, row_ in row_df.iterrows():
            append_row(df_, row=row_)


def instance_crossover_augmentation(
    df_,
    restriction='destination only',
    label_trajectory=TRAJECTORY,
    frac=0.5
):
    """
    Generate new data from unobserved trajectories,
    with a specific restriction. By default, the
    algorithm uses the same destination constraint
    as the route and inserts the points on the
    original dataframe.

    Parameters
    ----------
    df_ : dataframe
        The input trajectories data.
    restriction : str, optional, default 'destination only'
        Constraint used to generate new data.
    label_trajectory : str, optional, default 'trajectory'
        Label of the points sequences.
    frac : number, optional, default 0.5
        Represents the percentage to be exchanged.

    Returns
    -------
    DataFrame or None
        Dataframe with the new data generated.
    """
    try:
        traj_df = generate_trajectories_df(df_)

        generate_destiny_feature(traj_df, label_trajectory=label_trajectory)

        if restriction == 'departure and destination':
            generate_start_feature(traj_df, label_trajectory=label_trajectory)

        aug_df = augmentation_trajectories_df(
            traj_df, restriction=restriction, frac=frac
        )
        insert_points_in_df(df_, aug_df)

    except Exception as e:
        raise e
