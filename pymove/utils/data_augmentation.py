import numpy as np
import pandas as pd

from pymove.utils.constants import (
    DATETIME,
    LABEL,
    LATITUDE,
    LOCAL_LABEL,
    LONGITUDE,
    TID,
    TRAJ_ID,
    TRAJECTORY,
)
from pymove.utils.log import progress_bar


def generate_trajectories_df(df_, label_local=LOCAL_LABEL):
    """
    Generates a dataframe with the sequence of
    location points of a trajectory.

    Parameters
    ----------
    df_ : dataframe
        The input trajectory data.
    label_local : str, optional, default 'local_label'
        The name of the with id of the local

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
        columns=[TRAJ_ID, TRAJECTORY, DATETIME, LATITUDE, LONGITUDE, TID]
    )

    for tid in progress_bar(tids, total=len(tids)):
        filter_ = df_[df_[TID] == tid]
        filter_.reset_index(drop=True, inplace=True)

        if filter_.shape[0] > 4:
            new_df.at[new_df.shape[0]] = [
                filter_.at[0, TRAJ_ID],
                np.array(filter_[label_local], dtype=np.int32).tolist(),
                np.array(filter_[DATETIME], dtype='object_').tolist(),
                np.array(filter_[LATITUDE], dtype=np.float32).tolist(),
                np.array(filter_[LONGITUDE], dtype=np.float32).tolist(),
                np.array(filter_[TID], dtype='object_').tolist()
            ]

    return new_df


def generate_target_feature(df_):
    """
    Removes the last point from the trajectory and
    adds it in a new column called 'target'.

    Parameters
    ----------
    df_ : dataframe
        The input trajectory data.
    """
    if LABEL not in df_:
        df_[LABEL] = df_[TRAJECTORY].apply(
            lambda x: np.int32(x[-1])
        )
        df_[TRAJECTORY] = df_[TRAJECTORY].apply(
            lambda x: np.array(x[:-1], dtype=np.int32).tolist()
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


def append_row(
    df_,
    row=None,
    columns=None
):
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


def augmentation_trajectories_df(df_, frac=0.5):
    """
    Generates a new dataframe with unobserved trajectories.

    Parameters
    ----------
    df_ : dataframe
        The input trajectories data.
    frac : number, optional, default 0.5
        Represents the percentage to be exchanged.

    Return
    ------
    dataframe
        unobserved trajectories
    """
    new_df = pd.DataFrame(
        columns=[TRAJ_ID, TRAJECTORY, DATETIME, LATITUDE, LONGITUDE, TID, LABEL]
    )

    i = 0
    for _, row1 in progress_bar(df_.iterrows(), total=df_.shape[0]):
        for _, row2 in df_.iterrows():
            if (
                row1[LABEL] == row2[LABEL]
                and str(row1[TRAJECTORY]) != str(row2[TRAJECTORY])
            ):

                t1, t2 = split_crossover(row1[TRAJECTORY], row2[TRAJECTORY], frac)
                d1, d2 = split_crossover(row1[DATETIME], row2[DATETIME], frac)
                lat1, lat2 = split_crossover(row1[LATITUDE], row2[LATITUDE], frac)
                lon1, lon2 = split_crossover(row1[LONGITUDE], row2[LONGITUDE], frac)

                traj_id = row1[TRAJ_ID] + '_' + row2[TRAJ_ID]
                tid1 = [row1[TRAJ_ID] + x.date().strftime('%Y%m%d') + str(i) for x in d1]
                i += 1

                append_row(
                    new_df,
                    columns={
                        TRAJ_ID: traj_id,
                        TRAJECTORY: t1,
                        DATETIME: d1,
                        LATITUDE: lat1,
                        LONGITUDE: lon1,
                        TID: tid1,
                        LABEL: row1[LABEL]
                    }
                )

                traj_id = row2[TRAJ_ID] + '_' + row1[TRAJ_ID]
                tid2 = [row2[TRAJ_ID] + x.date().strftime('%Y%m%d') + str(i) for x in d2]
                i += 1

                append_row(
                    new_df,
                    columns={
                        TRAJ_ID: traj_id,
                        TRAJECTORY: t2,
                        DATETIME: d2,
                        LATITUDE: lat2,
                        LONGITUDE: lon2,
                        TID: tid2,
                        LABEL: row2[LABEL]
                    }
                )

    new_df[LABEL] = new_df[LABEL].astype(np.int64)
    return new_df


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
        traj = row[TRAJECTORY]
        date = row[DATETIME][:-1]
        lat = row[LATITUDE][:-1]
        lon = row[LONGITUDE][:-1]
        tid = row[TID][:-1]

        for t, d, l1, l2, t_id in zip(traj, date, lat, lon, tid):
            df_.at[
                df_.shape[0], [
                    TRAJ_ID,
                    LOCAL_LABEL,
                    DATETIME,
                    LATITUDE,
                    LONGITUDE,
                    TID]
            ] = [row[TRAJ_ID], t, d, l1, l2, t_id]

        date_ = row[DATETIME][-1]
        df_.at[
            df_.shape[0], [
                TRAJ_ID,
                LOCAL_LABEL,
                DATETIME,
                LATITUDE,
                LONGITUDE,
                TID]
        ] = [row[TRAJ_ID], row[LABEL], date_,
             row[LATITUDE][-1],
             row[LONGITUDE][-1],
             row[TID][-1]]


def instance_crossover(df_, frac=0.5):
    """
    Technique for generating unobserved data
    based on the crossing of instances.

    Parameters
    ----------
    df_ : dataframe
        The input trajectories data.
    frac : number, optional, default 0.5
        Represents the percentage to be exchanged.
    """
    try:
        traj_df = generate_trajectories_df(df_)
        generate_target_feature(traj_df)
        aug_df = augmentation_trajectories_df(traj_df, frac=frac)
        insert_points_in_df(df_, aug_df)

    except Exception as e:
        raise e
