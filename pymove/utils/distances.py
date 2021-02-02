from typing import Optional, Text, Union

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas.core.frame import DataFrame
from scipy.spatial import distance

from pymove import utils
from pymove.utils.constants import DATETIME, EARTH_RADIUS, LATITUDE, LONGITUDE


def haversine(
    lat1: Union[float, ndarray],
    lon1: Union[float, ndarray],
    lat2: Union[float, ndarray],
    lon2: Union[float, ndarray],
    to_radians: Optional[bool] = True,
    earth_radius: Optional[float] = EARTH_RADIUS
) -> Union[float, ndarray]:
    """
    Calculate the great circle distance between two points on the earth
    (specified in decimal degrees or in radians). All (lat, lon) coordinates
    must have numeric dtypes and be of equal length. Result in meters. Use 3956
    in earth radius for miles.

    Parameters
    ----------
    lat1 : float or array
        latitute of point 1
    lon1 : float or array
        longitude of point 1
    lat2 : float or array
        latitute of point 2
    lon2 : float or array
        longitude of point 2
    to_radians : boolean
        Wether to convert the values to radians, by default True
    earth_radius : int
        Radius of sphere, by default EARTH_RADIUS

    Returns
    -------
    float or ndarray
        Represents distance between points in meters

    References
    ----------
    Vectorized haversine function:
        https://stackoverflow.com/questions/43577086/pandas-calculate-haversine-distance-within-each-group-of-rows
    About distance between two points:
        https://janakiev.com/blog/gps-points-distance-python/

    """

    if to_radians:
        lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    a = (
        np.sin((lat2 - lat1) / 2.0)
        ** 2 + np.cos(lat1)
        * np.cos(lat2)
        * np.sin((lon2 - lon1) / 2.0) ** 2
    )
    return (earth_radius * 2 * np.arctan2(a ** 0.5, (1 - a) ** 0.5)) * 1000


def euclidean_distance_in_meters(
    lat1: Union[float, ndarray],
    lon1: Union[float, ndarray],
    lat2: Union[float, ndarray],
    lon2: Union[float, ndarray]
) -> Union[float, ndarray]:
    """
    Calculate the euclidean distance in meters between two points.

    Parameters
    ----------
    ----------
    lat1 : float or array
        latitute of point 1
    lon1 : float or array
        longitude of point 1
    lat2 : float or array
        latitute of point 2
    lon2 : float or array
        longitude of point 2

    Returns
    -------
    float or ndarray
        euclidean distance in meters between the two points.
    """

    meters_by_radians = 6371
    dist_eucl = np.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)
    dist_eucl_meters = dist_eucl * meters_by_radians

    return dist_eucl_meters


def nearest_points(
    traj1: DataFrame,
    traj2: DataFrame,
    latitude: Optional[Text] = LATITUDE,
    longitude: Optional[Text] = LONGITUDE,
) -> DataFrame:
    """
    For each point on a trajectory, it returns the point closest to
    another trajectory based on the Euclidean distance.

    Parameters
    ----------
    traj1: dataframe
        The input of one trajectory.
    traj2: dataframe
        The input of another trajectory.
    latitude: str, optional
        Label of the trajectories dataframe referring to the latitude,
        by default LATITUDE
    longitude: str, optional
        Label of the trajectories dataframe referring to the longitude,
        by default LONGITUDE

    Returns
    -------
    DataFrame
        dataframe with closest points

    """

    result = pd.DataFrame(columns=traj1.columns)

    for _, t1 in traj1.iterrows():
        round_result = np.Inf
        round_traj = []
        for _, t2 in traj2.iterrows():
            this_distance = distance.euclidean(
                (t1[latitude], t1[longitude]),
                (t2[latitude], t2[longitude]),
            )
            if this_distance < round_result:
                round_result = this_distance
                round_traj = t2
        result = result.append(round_traj)

    return result


def MEDP(
    traj1: DataFrame,
    traj2: DataFrame,
    latitude: Optional[Text] = LATITUDE,
    longitude: Optional[Text] = LONGITUDE
) -> float:
    """
    Returns the Mean Euclidian Distance Predictive between
    two trajectories, which considers only the spatial
    dimension for the similarity measure.

    Parameters
    ----------
    traj1: dataframe
        The input of one trajectory.
    traj2: dataframe
        The input of another trajectory.
    latitude: str, optional
        Label of the trajectories dataframe referring to the latitude,
        by default LATITUDE
    longitude: str, optional
        Label of the trajectories dataframe referring to the longitude,
        by default LONGITUDE

    Returns
    -------
    float
        total distance
    """

    soma = 0
    traj2 = nearest_points(traj1, traj2, latitude, longitude)
    for (_, t1), (_, t2) in zip(traj1.iterrows(), traj2.iterrows()):
        this_distance = distance.euclidean(
            (t1[latitude], t1[longitude]),
            (t2[latitude], t2[longitude])
        )
        soma = soma + this_distance
    return soma


def MEDT(
    traj1: DataFrame,
    traj2: DataFrame,
    latitude: Optional[Text] = LATITUDE,
    longitude: Optional[Text] = LONGITUDE,
    datetime: Optional[Text] = DATETIME
) -> float:
    """
    Returns the Mean Euclidian Distance Trajectory between two
    trajectories, which considers the spatial dimension and the
    temporal dimension when measuring similarity.

    Parameters
    ----------
    traj1: dataframe
        The input of one trajectory.
    traj2: dataframe
        The input of another trajectory.
    latitude: str, optional
        Label of the trajectories dataframe referring to the latitude,
        by default LATITUDE
    longitude: str, optional
        Label of the trajectories dataframe referring to the longitude,
        by default LONGITUDE
    datetime: str, optional
        Label of the trajectories dataframe referring to the timestamp,
        by default DATETIME

    Returns
    -------
    float
        total distance

    """

    soma = 0
    proportion = 1000000000
    if(len(traj2) < len(traj1)):
        traj1, traj2 = traj2, traj1

    for i in range(0, len(traj1)):
        this_distance = distance.euclidean(
            (traj1[latitude].iloc[i],
                traj1[longitude].iloc[i],
                float(utils.datetime.timestamp_to_millis(
                    traj1[datetime].iloc[i]
                )) / proportion),
            (traj2[latitude].iloc[i],
                traj2[longitude].iloc[i],
                float(utils.datetime.timestamp_to_millis(
                    traj2[datetime].iloc[i]
                )) / proportion),
        )
        soma = soma + this_distance
    for j in range(len(traj1) + 1, len(traj2)):
        soma = soma + \
            float(utils.datetime.timestamp_to_millis(
                traj2[datetime].iloc[j])) / proportion
    return soma
