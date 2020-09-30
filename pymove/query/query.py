import numpy as np
import pandas as pd

from pymove.utils import distances
from pymove.utils.constants import DATETIME, LATITUDE, LONGITUDE, MEDP, MEDT, TRAJ_ID
from pymove.utils.log import progress_bar


def range_query(
    traj,
    move_df,
    id=TRAJ_ID,
    range=1000,
    distance=MEDP,
    latitude=LATITUDE,
    longitude=LONGITUDE,
    datetime=DATETIME
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

    id: string ("id" by default)
        Label of the trajectories dataframe referring to the MoveDataFrame id.

    range: float (1000 by default)
        Minimum distance measure.

    distance: string ("MEDP" by default)
        Distance measure type.

    latitude: string ("lat" by default)
        Label of the trajectories dataframe referring to the latitude.

    longitude: string ("lon" by default)
        Label of the trajectories dataframe referring to the longitude.

    datetime: string ("datetime" by default)
        Label of the trajectories dataframe referring to the timestamp.

    Raises
    ------
        ValueError: if distance measure is invalid

    """

    result = traj.copy()
    result.drop(result.index, inplace=True)

    if (distance == MEDP):
        def dist_measure(traj, this, latitude, longitude, datetime):
            return distances.MEDP(
                traj, this, latitude, longitude
            )
    elif (distance == MEDT):
        def dist_measure(traj, this, latitude, longitude, datetime):
            return distances.MEDT(
                traj, this, latitude, longitude, datetime
            )
    else:
        raise ValueError('Unknown distance measure. Use MEDP or MEDT')

    for traj_id in progress_bar(
        move_df[id].unique(), desc='Querying range by {}'.format(distance)
    ):
        this = move_df.loc[move_df[id] == traj_id]
        if dist_measure(traj, this, latitude, longitude, datetime) < range:
            result = result.append(this)

    return result


def knn_query(
    traj,
    move_df,
    k=5,
    id=TRAJ_ID,
    latitude=LATITUDE,
    longitude=LONGITUDE,
    datetime=DATETIME,
    distance=MEDP
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

    id: string ("id" by default)
        Label of the trajectories dataframe referring to the MoveDataFrame id.

    range: float (1000 by default)
        Minimum similarity rate.

    distance: string ("MEDP" by default)
        Similarity measure type.

    latitude: string ("lat" by default)
        Label of the trajectories dataframe referring to the latitude.

    longitude: string ("lon" by default)
        Label of the trajectories dataframe referring to the longitude.

    datetime: string ("datetime" by default)
        Label of the trajectories dataframe referring to the timestamp.

    Raises
    ------
        ValueError: if distance measure is invalid

    """

    k_list = pd.DataFrame([[np.Inf, 'empty']] * k, columns=['distance', TRAJ_ID])

    if (distance == MEDP):
        def dist_measure(traj, this, latitude, longitude, datetime):
            return distances.MEDP(
                traj, this, latitude, longitude
            )
    elif (distance == MEDT):
        def dist_measure(traj, this, latitude, longitude, datetime):
            return distances.MEDT(
                traj, this, latitude, longitude, datetime
            )
    else:
        raise ValueError('Unknown distance measure. Use MEDP or MEDT')

    for traj_id in progress_bar(
        move_df[id].unique(), desc='Querying knn by {}'.format(distance)
    ):
        if (traj_id != traj[id].values[0]):
            this = move_df.loc[move_df[id] == traj_id]
            this_distance = dist_measure(
                traj, this, latitude, longitude, datetime
            )
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
            move_df.loc[move_df[id] == k_list.loc[n, 'traj_id']])

    return result
