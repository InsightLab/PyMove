from typing import Optional, Text, Tuple, Union

import geohash2 as gh
import numpy as np
from matplotlib.pyplot import axes
from numpy import ndarray
from pandas import DataFrame
from shapely.geometry import LineString, MultiLineString
from shapely.geometry.base import BaseGeometry

from pymove.utils.constants import (
    BASE_32,
    BIN_GEOHASH,
    COLORS,
    GEOHASH,
    LATITUDE,
    LATITUDE_DECODE,
    LONGITUDE,
    LONGITUDE_DECODE,
)
from pymove.utils.log import progress_bar

BINARY = [
    np.asarray(
        list('{0:05b}'.format(x)), dtype=int
    ) for x in range(0, len(BASE_32))
]


BASE_32_TO_BIN = dict(zip(BASE_32, BINARY))


def v_color(ob: BaseGeometry) -> Text:
    """
    Returns '#ffcc33' if object crosses otherwise it returns '#6699cc'.

    Parameters
    ----------
    ob : geometry object
        Any geometric object

    Return
    ------
    str
        Geometric object color
     """
    return COLORS[ob.is_simple + 33]


def plot_coords(ax: axes, ob: BaseGeometry, color: Optional[Text] = 'r'):
    """
    Plot the coordinates of each point of the object in a 2D chart.

    Parameters
    ----------
    ax : axes
        Single axes object
    ob : geometry object
        Any geometric object
    color : str, optional
        Sets the geometric object color, by default 'r'
    """
    x, y = ob.xy
    ax.plot(x, y, 'o', color=color, zorder=1)


def plot_bounds(ax: axes, ob: Union[LineString, MultiLineString], color='b'):
    """
    Plot the limites of geometric object.

    Parameters
    ----------
    ax : axes
        Single axes object
    ob : LineString or MultiLineString
        Geometric object formed by lines.
    color : str, optional
        Sets the geometric object color, by default 'b'
    """
    x, y = zip(*list((p.x, p.y) for p in ob.boundary))
    ax.plot(x, y, '-', color=color, zorder=1)


def plot_line(
    ax: axes,
    ob: LineString,
    color: Optional[Text] = 'r',
    alpha: Optional[float] = 0.7,
    linewidth: Optional[float] = 3,
    solid_capstyle: Optional[Text] = 'round',
    zorder: Optional[float] = 2
):
    """
    Plot a LineString

    Parameters
    ----------
    ax : axes
        Single axes object
    ob : LineString
        Sequence of points.
    color : str, optional
        Sets the line color, by default 'r'
    alpha : float, optional
        Defines the opacity of the line, by default 0.7
    linewidth : float, optional
        Defines the line thickness, by default 3
    solid_capstyle : str, optional
        Defines the style of the ends of the line, by default 'round'
    zorder : float, optional
        Determines the default drawing order for the axes, by default 2
    """
    x, y = ob.xy
    ax.plot(
        x, y, color=color, alpha=alpha, linewidth=linewidth,
        solid_capstyle=solid_capstyle, zorder=zorder
    )


def _encode(lat: float, lon: float, precision: Optional[float] = 15) -> Text:
    """
    Encodes latitude/longitude to geohash, either to specified
    precision or to automatically evaluated precision.

    Parameters
    ----------
    lat : float
        Latitude in degrees.
    lon : float
        Longitude in degrees.
    precision : float, optional
        Number of characters in resulting geohash, by default 15

    Return
    ------
    str
        Geohash of supplied latitude/longitude.
    """
    return gh.encode(lat, lon, precision)


def _decode(geohash: Text) -> Tuple[float, float]:
    """
    Decode geohash to latitude/longitude (location is approximate
    centre of geohash cell, to reasonable precision).

    Parameters
    ----------
    geohash : str
        Geohash str to be converted to latitude/longitude.

    Return
    ------
    (lat : float, lon : float)
        Geohashed location.
    """
    return gh.decode(geohash)


def _bin_geohash(lat: float, lon: float, precision: Optional[float] = 15) -> ndarray:
    """
    Transforms a point's geohash into a binary array.

    Parameters
    ----------
    lat : float
        Latitude in degrees
    lon : float
        Longitude in degrees
    precision : float, optional
        Number of characters in resulting geohash, by default 15

    Return
    ------
    array
        Returns a binary geohash array
    """
    hashed = _encode(lat, lon, precision)
    return np.concatenate([BASE_32_TO_BIN[x] for x in hashed])


def _reset_and_create_arrays_none(
    data: DataFrame, reset_index: Optional[bool] = True
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    Reset the df index and create arrays of none values.

    Parameters
    ----------
    data : dataframe
        The input trajectories data
    reset_index : boolean, optional
        Condition to reset the df index, by default True

    Return
    ------
    arrays
        Returns arrays of none values, of the size of the df.
    """
    if reset_index:
        data.reset_index(drop=True, inplace=True)

    latitudes = np.full(
        data.shape[0], None, dtype=np.float64
    )

    longitudes = np.full(
        data.shape[0], None, dtype=np.float64
    )

    geohash = np.full(
        data.shape[0], None, dtype='object_'
    )

    bin_geohash = np.full(
        data.shape[0], None, dtype=np.ndarray
    )

    return latitudes, longitudes, geohash, bin_geohash


def create_geohash_df(data: DataFrame, precision: Optional[float] = 15):
    """
    Create geohash from geographic coordinates
    and integrate with df.

    Parameters
    ----------
    data : dataframe
        The input trajectories data
    precision : float, optional
        Number of characters in resulting geohash, by default 15
    """

    _, _, geohash, _ = _reset_and_create_arrays_none(data)

    for idx, row in progress_bar(
        data[[LATITUDE, LONGITUDE]].iterrows(), total=data.shape[0]
    ):
        geohash[idx] = _encode(row[LATITUDE], row[LONGITUDE], precision)

    data[GEOHASH] = geohash
    print('\n================================================')
    print('\n========> geohash feature was created. <========')
    print('\n================================================')


def create_bin_geohash_df(data: DataFrame, precision: Optional[float] = 15):
    """
    Create trajectory geohash binaries and integrate with df.

    Parameters
    ----------
    data : dataframe
        The input trajectories data
    precision : float, optional
        Number of characters in resulting geohash, by default 15
    """

    _, _, _, bin_geohash = _reset_and_create_arrays_none(data)

    for idx, row in progress_bar(
        data[[LATITUDE, LONGITUDE]].iterrows(), total=data.shape[0]
    ):
        bin_geohash[idx] = _bin_geohash(row[LATITUDE], row[LONGITUDE], precision)

    data[BIN_GEOHASH] = bin_geohash
    print('\n================================================')
    print('\n=====> bin_geohash features was created. <======')
    print('\n================================================')


def decode_geohash_to_latlon(
    data: DataFrame,
    label_geohash: Optional[Text] = GEOHASH,
    reset_index: Optional[bool] = True
):
    """
    Decode feature with hash of trajectories back to
    geographic coordinates.

    Parameters
    ----------
    data : dataframe
        The input trajectories data
    label_geohash : str, optional
        The name of the feature with hashed trajectories, by default GEOHASH
    reset_index : boolean, optional
        Condition to reset the df index, by default True
    """

    if label_geohash not in data:
        raise ValueError('feature {} not in df'.format(label_geohash))

    lat, lon, _, _ = _reset_and_create_arrays_none(data, reset_index=reset_index)

    for idx, row in progress_bar(data[[label_geohash]].iterrows(), total=data.shape[0]):
        lat_lon = _decode(row[label_geohash])
        lat[idx] = lat_lon[0]
        lon[idx] = lat_lon[1]

    data[LATITUDE_DECODE] = lat
    data[LONGITUDE_DECODE] = lon
    print('\n================================================')
    print('\n==> lat and lon decode features was created. <==')
    print('\n================================================')
