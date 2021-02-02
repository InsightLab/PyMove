from typing import Optional, Text

import numpy as np
import pandas as pd
from pandas import DataFrame

from pymove.utils import distances
from pymove.utils.constants import DATETIME, LATITUDE, LONGITUDE, MEDP, MEDT, TRAJ_ID
from pymove.utils.log import progress_bar


def range_query(
    traj: DataFrame,
    move_df: DataFrame,
    _id: Optional[Text] = TRAJ_ID,
    min_dist: Optional[float] = 1000,
    distance: Optional[Text] = MEDP,
    latitude: Optional[Text] = LATITUDE,
    longitude: Optional[Text] = LONGITUDE,
    datetime: Optional[Text] = DATETIME
) -> DataFrame:
    """
    Given a distance, a trajectory, and a DataFrame
    with several trajectories, it returns all trajectories that
    have a distance equal to or less than the informed
    trajectory.

    Parameters
    ----------
    traj: dataframe
        The input of one trajectory.
    move_df: dataframe
        The input trajectory data.
    _id: str, optional
        Label of the trajectories dataframe user id, by default TRAJ_ID
    min_dist: float, optional
        Minimum distance measure, by default 1000
    distance: string, optional
        Distance measure type, by default MEDP
    latitude: string, optional
        Label of the trajectories dataframe referring to the latitude,
        by default LATITUDE
    longitude: string, optional
        Label of the trajectories dataframe referring to the longitude,
        by default LONGITUDE
    datetime: string, optional
        Label of the trajectories dataframe referring to the timestamp,
        by default DATETIME

    Returns
    -------
    DataFrame
        dataframe with near trajectories

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
        move_df[_id].unique(), desc='Querying range by {}'.format(distance)
    ):
        this = move_df.loc[move_df[_id] == traj_id]
        if dist_measure(traj, this, latitude, longitude, datetime) < min_dist:
            result = result.append(this)

    return result


def knn_query(
    traj: DataFrame,
    move_df: DataFrame,
    k: Optional[int] = 5,
    id_: Optional[Text] = TRAJ_ID,
    distance: Optional[Text] = MEDP,
    latitude: Optional[Text] = LATITUDE,
    longitude: Optional[Text] = LONGITUDE,
    datetime: Optional[Text] = DATETIME
) -> DataFrame:
    """
    Given a k, a trajectory and a
    DataFrame with multiple paths, it returns
    the k neighboring trajectories closest to the trajectory.

    Parameters
    ----------
    traj: dataframe
        The input of one trajectory.
    move_df: dataframe
        The input trajectory data.
    k: int, optional
        neighboring trajectories, by default 5
    id_: str, optional
        Label of the trajectories dataframe user id, by default TRAJ_ID
    distance: string, optional
        Distance measure type, by default MEDP
    latitude: string, optional
        Label of the trajectories dataframe referring to the latitude,
        by default LATITUDE
    longitude: string, optional
        Label of the trajectories dataframe referring to the longitude,
        by default LONGITUDE
    datetime: string, optional
        Label of the trajectories dataframe referring to the timestamp,
        by default DATETIME

    Returns
    -------
    DataFrame
        dataframe with near trajectories


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
        move_df[id_].unique(), desc='Querying knn by {}'.format(distance)
    ):
        if (traj_id != traj[id_].values[0]):
            this = move_df.loc[move_df[id_] == traj_id]
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
            move_df.loc[move_df[id_] == k_list.loc[n, 'traj_id']]
        )

    return result
