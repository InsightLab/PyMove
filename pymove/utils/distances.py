"""
Distances operations.

haversine,
euclidean_distance_in_meters,
nearest_points,
medp,
medt

"""
from typing import Text, Union

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
    to_radians: bool = True,
    earth_radius: float = EARTH_RADIUS
) -> Union[float, ndarray]:
    """
    Calculates the great circle distance between two points on the earth.

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

    Example
    -------
    >>> from pymove.utils.distances import haversine
    >>> lat_fortaleza, lon_fortaleza = [-3.71839 ,-38.5434]
    >>> lat_quixada, lon_quixada = [-4.979224744401671, -39.056434302570665]
    >>> haversine(lat_fortaleza, lon_fortaleza, lat_quixada, lon_quixada)
    151298.02548428564

    References
    ----------
    Vectorized haversine function:
        https://stackoverflow.com/questions/43577086/pandas-calculate-haversine-distance-within-each-group-of-rows
    About distance between two points:
        https://janakiev.com/blog/gps-points-distance-python/

    """
    if to_radians:
        lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])  # type: ignore
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

    Example
    -------
    >>> from pymove.utils.distances import euclidean_distance_in_meters
    >>> lat_fortaleza, lon_fortaleza = [-3.71839 ,-38.5434]
    >>> lat_quixada, lon_quixada = [-4.979224744401671, -39.056434302570665]
    >>> euclidean_distance_in_meters(
    >>>    lat_fortaleza, lon_fortaleza, lat_quixada, lon_quixada
    >>> )
    151907.9670136588
    """
    y1 = utils.conversions.lat_to_y_spherical(lat=lat1)
    y2 = utils.conversions.lat_to_y_spherical(lat=lat2)
    x1 = utils.conversions.lon_to_x_spherical(lon=lon1)
    x2 = utils.conversions.lon_to_x_spherical(lon=lon2)

    dist_eucl_in_meters = ((y1 - y2)**2 + (x1 - x2)**2)**0.5

    return dist_eucl_in_meters


def nearest_points(
    traj1: DataFrame,
    traj2: DataFrame,
    latitude: Text = LATITUDE,
    longitude: Text = LONGITUDE,
) -> DataFrame:
    """
    Returns the point closest to another trajectory based on the Euclidean distance.

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

    Example
    -------
    >>> from pymove.utils.distances import nearest_points
    >>> df_a
             lat          lon               datetime    id
    0   39.984198   116.319322   2008-10-23 05:53:06     1
    1   39.984224   116.319402   2008-10-23 05:53:11     1
    >>> df_b
              lat          lon              datetime    id
    0   39.984211   116.319389   2008-10-23 05:53:16     1
    1   39.984217   116.319422   2008-10-23 05:53:21     1
    >>> nearest_points(df_a,df_b)
              lat          lon              datetime    id
    0   39.984211   116.319389   2008-10-23 05:53:16     1
    1   39.984211   116.319389   2008-10-23 05:53:16     1
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


def medp(
    traj1: DataFrame,
    traj2: DataFrame,
    latitude: Text = LATITUDE,
    longitude: Text = LONGITUDE
) -> float:
    """
    Returns the Mean Euclidian Distance Predictive between two trajectories.

    Considers only the spatial dimension for the similarity measure.

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

    Example
    -------
    >>> from pymove.utils.distances import medp
    >>> traj_1
                lat          lon           datetime     id
    0   39.98471   116.319865   2008-10-23 05:53:23      1
    >>> traj_2
                lat        lon             datetime     id
    0   39.984674   116.31981   2008-10-23 05:53:28      1
    >>> medp(traj_1, traj_2)
    6.573431370981577e-05
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


def medt(
    traj1: DataFrame,
    traj2: DataFrame,
    latitude: Text = LATITUDE,
    longitude: Text = LONGITUDE,
    datetime: Text = DATETIME
) -> float:
    """
    Returns the Mean Euclidian Distance Trajectory between two trajectories.

    Considers the spatial dimension and the
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

    Example
    -------
    >>> from pymove.utils.distances import medt
    >>> traj_1
             lat          lon              datetime  id
    0   39.98471   116.319865   2008-10-23 05:53:23   1
    >>> traj_2
              lat         lon              datetime  id
    0   39.984674   116.31981   2008-10-23 05:53:28   1
    >>> medt(traj_1, traj_2)
    6.592419887747872e-05
    """
    soma = 0.
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
            float(
                utils.datetime.timestamp_to_millis(traj2[datetime].iloc[j])
            ) / proportion
    return soma
