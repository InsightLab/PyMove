import numpy as np
import pandas as pd
from scipy.spatial import distance

from pymove import utils
from pymove.utils.constants import DATETIME, LATITUDE, LONGITUDE


def haversine(lat1, lon1, lat2, lon2, to_radians=True, earth_radius=6371):
    """
    Calculate the great circle distance between two points on the earth
    (specified in decimal degrees or in radians). All (lat, lon) coordinates
    must have numeric dtypes and be of equal length. Result in meters. Use 3956
    in earth radius for miles.

    Parameters
    ----------
    lat1 : float or array
        Y offset from your original position in meters.
    lon1 : float or array
        Y offset from your original position in meters.
    lat2 : float or array
        Y offset from your original position in meters.
    lon2 : float or array
        Y offset from your original position in meters.
    to_radians : boolean
        Y offset from your original position in meters.
    earth_radius : int
        Y offset from your original position in meters.

    Returns
    -------
    float
        Represents latitude.

    References
    ----------
    Vectorized haversine function:
        https://stackoverflow.com/questions/43577086/pandas-calculate-haversine-distance-within-each-group-of-rows
    About distance between two points:
        https://janakiev.com/blog/gps-points-distance-python/

    """

    try:
        if to_radians:
            lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
        a = (
            np.sin((lat2 - lat1) / 2.0)
            ** 2 + np.cos(lat1)
            * np.cos(lat2)
            * np.sin((lon2 - lon1) / 2.0) ** 2
        )
        # return earth_radius * 2 * np.arcsin(np.sqrt(a)) * 1000
        # result in meters (* 1000)
        return (earth_radius * 2 * np.arctan2(a ** 0.5, (1 - a) ** 0.5)) * 1000
    except Exception as e:
        raise e


def nearest_points(traj1, traj2, latitude=LATITUDE, longitude=LONGITUDE):
    """
    For each point on a trajectory, it returns the point closest to
    another trajectory based on the Euclidean distance.

    Parameters
    ----------
    traj1: dataframe
        The input of one trajectory.

    traj2: dataframe
        The input of another trajectory.

    latitude: string ("lat" by default)
        Label of the trajectories dataframe referring to the latitude.

    longitude: string ("lon" by default)
        Label of the trajectories dataframe referring to the longitude.
    """

    result = pd.DataFrame(columns=traj1.columns)

    for i in range(0, len(traj1)):
        round_result = np.Inf
        round_traj = []
        for j in range(0, len(traj2)):
            this_distance = distance.euclidean(
                (traj1[latitude].iloc[i], traj1[longitude].iloc[i]),
                (traj2[latitude].iloc[j], traj2[longitude].iloc[j]),
            )
            if this_distance < round_result:
                round_result = this_distance
                round_traj = traj2.iloc[j]
        result = result.append(round_traj)

    return result


def MEDP(traj1, traj2, latitude=LATITUDE, longitude=LONGITUDE):
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

    latitude: string ("lat" by default)
        Label of the trajectories dataframe referring to the latitude.

    longitude: string ("lon" by default)
        Label of the trajectories dataframe referring to the longitude.
    """

    soma = 0
    traj2 = nearest_points(traj1, traj2, latitude, longitude)
    for i in range(0, len(traj1)):
        this_distance = distance.euclidean(
            (traj1[latitude].iloc[i],
             traj1[longitude].iloc[i]),
            (traj2[latitude].iloc[i],
             traj2[longitude].iloc[i]))
        soma = soma + this_distance
    return soma


def MEDT(traj1, traj2, latitude=LATITUDE, longitude=LONGITUDE, datetime=DATETIME):
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

    latitude: string ("lat" by default)
        Label of the trajectories dataframe referring to the latitude.

    longitude: string ("lon" by default)
        Label of the trajectories dataframe referring to the longitude.

    datetime: string ("datetime" by default)
        Label of the trajectories dataframe referring to the timestamp.
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
