"""
Query operations.

range_query,
knn_query,
query_all_points_by_range,

"""
from datetime import timedelta

import numpy as np
import pandas as pd
from pandas import DataFrame

from pymove.utils import distances
from pymove.utils.constants import DATETIME, LATITUDE, LONGITUDE, MEDP, MEDT, TRAJ_ID
from pymove.utils.log import logger, progress_bar


def range_query(
    traj: DataFrame,
    move_df: DataFrame,
    _id: str = TRAJ_ID,
    min_dist: float = 1000,
    distance: str = MEDP,
    latitude: str = LATITUDE,
    longitude: str = LONGITUDE,
    datetime: str = DATETIME
) -> DataFrame:
    """
    Returns all trajectories that have a distance equal to or less than the trajectory.

    Given a distance, a trajectory, and a DataFrame with several trajectories.

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

    Examples
    --------
    >>> from pymove.query.query import range_query
    >>> traj_df
         lat      lon              datetime  id
    0   16.4    -54.9   2014-10-11 18:00:00   1
    1   16.4    -55.9   2014-10-12 00:00:00   1
    2   16.4    -56.9   2014-10-12 06:00:00   1
    >>> move_df
         lat      lon              datetime  id
    0   33.1    -77.0   2012-05-19 00:00:00   2
    1   32.8    -77.1   2012-05-19 06:00:00   3
    2   32.5    -77.3   2012-05-19 12:00:00   4
    >>> range_query(
    >>>    traj_df, move_df, min_dist=80.5
    >>> )
         lat      lon              datetime  id
    1   32.8	-77.1	2012-05-19 06:00:00	  3
    2	32.5	-77.3	2012-05-19 12:00:00	  4
    """
    result = traj.copy()
    result.drop(result.index, inplace=True)

    if (distance == MEDP):
        def dist_measure(traj, this, latitude, longitude, datetime):
            return distances.medp(
                traj, this, latitude, longitude
            )
    elif (distance == MEDT):
        def dist_measure(traj, this, latitude, longitude, datetime):
            return distances.medt(
                traj, this, latitude, longitude, datetime
            )
    else:
        raise ValueError('Unknown distance measure. Use MEDP or MEDT')

    for traj_id in progress_bar(
        move_df[_id].unique(), desc=f'Querying range by {distance}'
    ):
        this = move_df.loc[move_df[_id] == traj_id]
        if dist_measure(traj, this, latitude, longitude, datetime) < min_dist:
            result = result.append(this)

    return result


def knn_query(
    traj: DataFrame,
    move_df: DataFrame,
    k: int = 5,
    id_: str = TRAJ_ID,
    distance: str = MEDP,
    latitude: str = LATITUDE,
    longitude: str = LONGITUDE,
    datetime: str = DATETIME
) -> DataFrame:
    """
    Returns the k neighboring trajectories closest to the trajectory.

    Given a k, a trajectory and a DataFrame with multiple paths.

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

    Examples
    --------
    >>> from pymove.query.query import knn_query
    >>> traj_df
         lat      lon              datetime  id
    0   16.4    -54.9   2014-10-11 18:00:00   1
    1   16.4    -55.9   2014-10-12 00:00:00   1
    2   16.4    -56.9   2014-10-12 06:00:00   1
    >>> move_df
         lat      lon              datetime  id
    0   33.1    -77.0   2012-05-19 00:00:00   2
    1   32.8    -77.1   2012-05-19 06:00:00   3
    2   32.5    -77.3   2012-05-19 12:00:00   4
    >>> knn_query(
    >>>    traj_df, move_df, k=1
    >>> )
         lat      lon              datetime  id
    0	16.4	-54.9	2014-10-11 18:00:00   1
    1	16.4	-55.9	2014-10-12 00:00:00	  1
    2	16.4	-56.9	2014-10-12 06:00:00	  1
    2	32.5	-77.3	2012-05-19 12:00:00	  4
    """
    k_list = pd.DataFrame([[np.Inf, 'empty']] * k, columns=['distance', TRAJ_ID])

    if (distance == MEDP):
        def dist_measure(traj, this, latitude, longitude, datetime):
            return distances.medp(
                traj, this, latitude, longitude
            )
    elif (distance == MEDT):
        def dist_measure(traj, this, latitude, longitude, datetime):
            return distances.medt(
                traj, this, latitude, longitude, datetime
            )
    else:
        raise ValueError('Unknown distance measure. Use MEDP or MEDT')

    for traj_id in progress_bar(
        move_df[id_].unique(), desc=f'Querying knn by {distance}'
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
    logger.debug('Generating DataFrame with k nearest trajectories.')
    for n in range(k):
        result = result.append(
            move_df.loc[move_df[id_] == k_list.loc[n, 'traj_id']]
        )

    return result


def _datetime_filter(
    row: DataFrame,
    move_df: DataFrame,
    minimum_distance: timedelta
) -> DataFrame:
    """
    Returns all the points of the DataFrame which are in a temporal distance.

    Given a row referencing to a point, a DataFrame with
    multiple points and a minimum distance, it returns
    all the points of the DataFrame which are in a temporal
    distance equal or smaller than the minimum distance
    parameter.

    Parameters
    ----------
    row: dataframe
        The input of one point of a trajectory.
    move_df: dataframe
        The input trajectory data.
    minimum_distance: datetime.timedelta
        the minimum temporal distance between the points.

    Returns
    -------
    DataFrame
        dataframe with all the points of move_df which are in
        a temporal distance equal or smaller than the minimum
        distance parameter.

    Examples
    --------
    >>> from pymove.query.query import _datetime_filter
    >>>> point
         lat      lon              datetime  id
    0   16.4    -54.9   2014-10-11 18:00:00   1
    >>> move_df
         lat      lon              datetime  id
    0   33.1    -77.0   2012-05-19 00:00:00   2
    1   32.8    -77.1   2012-05-19 06:00:00   3
    2   32.5    -77.3   2012-05-19 12:00:00   4
    >>> _datetime_filter(point, move_df, timedelta(hours=21010))
         lat      lon              datetime  id   temporal_distance\
        target_id   target_lat  target_lon        target_datetime
    0   32.5    -77.3   2012-05-19 12:00:00   4   875 days 06:00:00\
                1          16.4       -54.9    2014-10-11 18:00:00
    """
    datetime = row['datetime']
    move_df['temporal_distance'] = (move_df['datetime'] - datetime).abs()
    filtered = move_df[
        (move_df['temporal_distance'] < minimum_distance)
        & (move_df['temporal_distance'] > -minimum_distance)
    ]

    if (filtered.shape[0] > 0):
        filtered['target_id'] = row['id']
        filtered['target_lat'] = row['lat']
        filtered['target_lon'] = row['lon']
        filtered['target_datetime'] = row['datetime']

    return filtered


def _meters_filter(
    row: DataFrame,
    move_df: DataFrame,
    minimum_distance: float
) -> DataFrame:
    """
    Returns all the points of the DataFrame which are in a spatial distance.

    Given a row referencing to a point, a DataFrame with
    multiple points and a minimum distance, it returns
    all the points of the DataFrame which are in a spatial
    distance (in meters) equal or smaller than the minimum distance
    parameter.

    Parameters
    ----------
    row: dataframe
        The input of one point of a trajectory.
    move_df: dataframe
        The input trajectory data.
    minimum_distance: float
        the minimum spatial distance between the points in meters.

    Returns
    -------
    DataFrame
        dataframe with all the points of move_df which are in
        a spatial distance equal or smaller than the minimum
        distance parameter.

    Examples
    --------
    >>> from pymove.query.query import _meters_filter
    >>> point
         lat      lon              datetime  id
    0   16.4    -54.9   2014-10-11 18:00:00   1
    >>> move_df
         lat      lon              datetime  id
    0   33.1    -77.0   2012-05-19 00:00:00   2
    1   32.8    -77.1   2012-05-19 06:00:00   3
    2   32.5    -77.3   2012-05-19 12:00:00   4
    >>> _meters_filter(firstpoint, move_df, 3190000)
         lat      lon              datetime  id spatial_distance\
        target_id   target_lat  target_lon      target_datetime
    0   32.5    -77.3   2012-05-19 12:00:00   4     3.182834e+06\
                1         16.4       -54.9  2014-10-11 18:00:00
    """
    lat = row[LATITUDE]
    lon = row[LONGITUDE]
    move_df['spatial_distance'] = distances.euclidean_distance_in_meters(
        lat1=lat, lon1=lon, lat2=move_df[LATITUDE], lon2=move_df[LONGITUDE]
    )
    filtered = move_df[move_df['spatial_distance'] < minimum_distance]

    if (filtered.shape[0] > 0):
        filtered['target_id'] = row[TRAJ_ID]
        filtered['target_lat'] = row[LATITUDE]
        filtered['target_lon'] = row[LONGITUDE]
        filtered['target_datetime'] = row[DATETIME]

    return filtered


def query_all_points_by_range(
    traj1: DataFrame,
    move_df: DataFrame,
    minimum_meters: float = 100,
    minimum_time: timedelta = None
) -> DataFrame:
    """
    Queries closest point within a spatial range based on meters and a temporal range.

    Selects only the points between two Move Dataframes
    that have the closest point within a spatial range
    based on meters and a temporal range.

    Parameters
    ----------
    traj1: dataframe
        The input of a trajectory data.
    move_df: dataframe
        The input of another trajectory data.
    minimum_meters: float, optional
        the minimum spatial distance, based in meters, between the points, by default 100
    minimum_time: datetime.timedelta, optional
        the minimum temporal distance between the points, by default timedelta(minutes=2)
    datetime_label: string, optional
        the label that refers to the datetime label of the dataframes, by default DATETIME

    Returns
    -------
    DataFrame
        dataframe with all the points of move_df which are in
        a spatial distance and temporal distance equal or smaller
        than the minimum distance parameters.

    Examples
    --------
    >>> from pymove.query.query import query_all_points_by_range
    >>> traj_df
         lat      lon              datetime  id
    0   16.4    -54.9   2014-10-11 18:00:00   1
    1   16.4    -55.9   2014-10-12 00:00:00   1
    2   16.4    -56.9   2014-10-12 06:00:00   1
    >>> move_df
         lat      lon              datetime  id
    0   33.1    -77.0   2012-05-19 00:00:00   2
    1   32.8    -77.1   2012-05-19 06:00:00   3
    2   32.5    -77.3   2012-05-19 12:00:00   4
    >>> query_all_points_by_range(
    >>>    traj_df, move_df, minimum_meters=3190000, minimum_time=timedelta(hours=21010)
    >>> )
         lat      lon              datetime  id    spatial_distance target_id\
        target_lat  target_lon        target_datetime   temporal_distance
    0   32.5    -77.3   2012-05-19 12:00:00   4        3.182834e+06         1\
             16.4       -54.9    2014-10-11 18:00:00   875 days 06:00:00
    """
    if minimum_time is None:
        minimum_time = timedelta(minutes=2)

    result = DataFrame([])
    total = traj1.shape[0]
    for _, row in progress_bar(
        traj1.iterrows(),
        desc='Querying all points by temporal and spatial distance',
        total=total
    ):
        coinc_points = _meters_filter(row, move_df, minimum_meters)
        coinc_points = _datetime_filter(row, coinc_points, minimum_time)
        result = coinc_points.append(result)

    return result
