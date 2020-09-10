import numpy as np
import pandas as pd

from pymove import distances


def range_query(
    traj,
    move_df,
    move_df_id='id',
    range=1000,
    distance='MEDP',
    label_lat='lat',
    label_lon='lon',
    label_time='datetime'
):
    """
    Given a distance, a trajectory, and a MoveDataFrame
    with several trajectories, it returns all trajectories that
    have a distance equal to or less than the informed
    trajectory.

    Parameters
    ----------
    traj: dataframe
        The input of one trajectory.

    move_df: dataframe
        The input trajectory data.

    move_df_id: string ("id" by default)
        Label of the trajectories dataframe referring to the MoveDataFrame id.

    range: float (1000 by default)
        Minimum distance measure.

    distance: string ("MEDP" by default)
        Distance measure type.

    label_lat: string ("lat" by default)
        Label of the trajectories dataframe referring to the latitude.

    label_lon: string ("lon" by default)
        Label of the trajectories dataframe referring to the longitude.

    label_time: string ("datetime" by default)
        Label of the trajectories dataframe referring to the timestamp.
    """

    result = traj.copy()
    result.drop(result.index, inplace=True)
    if (distance == 'MEDP'):
        for move_traj in move_df[move_df_id].unique():
            this = move_df.loc[move_df[move_df_id] == move_traj]
            if (distances.MEDP(traj, this, label_lat, label_lon) < range):
                result = result.append(this)
    elif (distance == 'MEDT'):
        for move_traj in move_df[move_df_id].unique():
            this = move_df.loc[move_df[move_df_id] == move_traj]
            if (distances.MEDT(traj, this, label_lat, label_lon, label_time) < range):
                result = result.append(this)
    return result


def knn_query(
    traj,
    move_df,
    move_df_id='id',
    k=5,
    label_lat='lat',
    label_lon='lon',
    label_time='datetime',
    distance='MEDP'
):
    """
    Given a k, a trajectory and a
    MoveDataFrame with multiple paths, it returns
    the k neighboring trajectories closest to the trajectory.

    Parameters
    ----------
    traj: dataframe
        The input of one trajectory.

    move_df: dataframe
        The input trajectory data.

    move_df_id: string ("id" by default)
        Label of the trajectories dataframe referring to the MoveDataFrame id.

    range: float (1000 by default)
        Minimum similarity rate.

    distance: string ("MEDP" by default)
        Similarity measure type.

    label_lat: string ("lat" by default)
        Label of the trajectories dataframe referring to the latitude.

    label_lon: string ("lon" by default)
        Label of the trajectories dataframe referring to the longitude.

    label_time: string ("datetime" by default)
        Label of the trajectories dataframe referring to the timestamp.
    """

    k_list = pd.DataFrame([[np.Inf, 'empty']] * k, columns=['distance', 'traj_id'])

    if (distance == 'MEDP'):
        for traj_id in move_df[move_df_id].unique():
            if (traj_id != traj[move_df_id].values[0]):
                this = move_df.loc[move_df[move_df_id] == traj_id]
                this_distance = distances.MEDP(traj, this, label_lat, label_lon)
                n = 0
                for n in range(k):
                    if (this_distance < k_list.loc[n, 'distance']):
                        k_list.loc[n, 'distance'] = this_distance
                        k_list.loc[n, 'traj_id'] = traj_id
                        break
                    n = n + 1

    elif (distance == 'MEDT'):
        for traj_id in move_df[move_df_id].unique():
            if (traj_id != traj[move_df_id].values[0]):
                this = move_df.loc[move_df[move_df_id] == traj_id]
                this_distance = distances.MEDT(
                    traj, this, label_lat, label_lon, label_time)
                n = 0
                for n in range(k):
                    if (this_distance < k_list.loc[n, 'distance']):
                        k_list.loc[n, 'distance'] = this_distance
                        k_list.loc[n, 'traj_id'] = traj_id
                        break
                    n = n + 1

    result = traj.copy()
    print('Gerando DataFrame com as k trajetórias mais próximas')
    for n in range(k):
        result = result.append(
            move_df.loc[move_df[move_df_id] == k_list.loc[n, 'traj_id']])

    return result
