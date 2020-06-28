import numpy as np


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
            np.sin((lat2 - lat1) / 2.0) ** 2
            + np.cos(lat1)
            * np.cos(lat2)
            * np.sin((lon2 - lon1) / 2.0) ** 2
        )
        # return earth_radius * 2 * np.arcsin(np.sqrt(a)) * 1000
        # result in meters (* 1000)
        return (earth_radius * 2 * np.arctan2(a ** 0.5, (1 - a) ** 0.5)) * 1000
    except Exception as e:
        raise e
