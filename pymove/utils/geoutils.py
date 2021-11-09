"""
Geo operations.

v_color,
create_geohash_df,
create_bin_geohash_df,
decode_geohash_to_latlon,

"""
from __future__ import annotations

import geohash2 as gh
import numpy as np
from numpy import ndarray
from pandas import DataFrame
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
        list(f'{x:05b}'), dtype=int
    ) for x in range(0, len(BASE_32))
]
BASE_32_TO_BIN = dict(zip(BASE_32, BINARY))


def v_color(ob: BaseGeometry) -> str:
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

    Example
    -------
    >>> from pymove.utils.geoutils import v_color
    >>> from shapely.geometry import LineString
    >>> case_1 = LineString([(3,3),(4,4), (3,4)])
    >>> case_2 = LineString([(3,3),(4,4), (4,3)])
    >>> case_3 = LineString([(3,3),(4,4), (3,4), (4,3)])
    >>> print(v_color(case_1), type(v_color(case_1)))
    #6699cc <class 'str'>
    >>> print(v_color(case_2), type(v_color(case_2)))
    #6699cc <class 'str'>
    >>> print(v_color(case_3), type(v_color(case_3)))
    #ffcc33 <class 'str'>
    """
    return COLORS[ob.is_simple + 33]


def _encode(lat: float, lon: float, precision: float = 15) -> str:
    """
    Encodes latitude/longitude to geohash.

    Either to specified precision or to automatically evaluated precision.

    Parameters
    ----------
    lat : float
        This represents latitude in degrees.
    lon : float
        This represents longitude in degrees.
    precision : float, optional
        Number of characters in resulting geohash, by default 15

    Return
    ------
    str
        Geohash of supplied latitude/longitude.

    Example
    -------
    >>> from pymove.utils.geoutils import _encode
    >>> lat1, lon1 = -3.777736, -38.547792
    >>> lat2, lon2 = -3.793388, -38.517722
    >>> print(_encode(lat1,lon1), type(_encode(lat1,lon1)))
    7pkddb6356fyzxq <class 'str'>
    >>> print(_encode(lat2,lon2), type(_encode(lat2,lon2)))
    7pkd7t2mbj0z1v7 <class 'str'>
    """
    return gh.encode(lat, lon, precision)


def _decode(geohash: str) -> tuple[float, float]:
    """
    Decode geohash to latitude/longitude.

    Location is approximate centre of geohash cell, to reasonable precision.

    Parameters
    ----------
    geohash : str
        Geohash str to be converted to latitude/longitude.

    Return
    ------
    (lat : float, lon : float)
        Geohashed location.

    Example
    -------
    >>> from pymove.utils.geoutils import _decode
    >>> geoHash_1 = '7pkddb6356fyzxq'
    >>> geoHash_2 = '7pkd7t2mbj0z1v7'
    >>> print(_decode(geoHash_1), type(_decode(geoHash_1)))
    ('-3.777736', '-38.547792') <class 'tuple'>
    >>> print(_decode(geoHash_2), type(_decode(geoHash_2)))
    ('-3.793388', '-38.517722') <class 'tuple'>
    """
    return gh.decode(geohash)


def _bin_geohash(lat: float, lon: float, precision: float = 15) -> ndarray:
    """
    Transforms a point's geohash into a binary array.

    Parameters
    ----------
    lat : float
        This represents latitude in degrees.
    lon : float
        This represents longitude in degrees.
    precision : float, optional
        Number of characters in resulting geohash, by default 15

    Return
    ------
    array
        Returns a binary geohash array

    Example
    -------
    >>> from pymove.utils.geoutils import _bin_geohash
    >>> lat1, lon1 = -3.777736, -38.547792
    >>> lat2, lon2 = -3.793388, -38.517722
    >>> print(_bin_geohash(lat1,lon1), type(_bin_geohash(lat1,lon1)))
    [0 0 1 1 1 1 1 0 0 1 1 0 1 0 0 0 1 1 0 1 0 1 1 0 1 0 1 0 1 1 0 0 1 1 0 0 0
    0 1 1 0 0 1 0 1 0 0 1 1 0 0 1 1 1 1 1 0 0 0 1 0 1 0 0 0 1 1 1 0 0 0 0 1 1
    1 0 1 0] <class 'numpy.ndarray'>
    >>> print(_bin_geohash(lat2,lon2), type(_bin_geohash(lat2,lon2)))
    [0 0 1 1 1 1 1 0 0 1 1 0 1 0 0 0 1 1 0 1 0 0 1 1 1 1 1 1 0 1 0 0 0 1 0 1 0
    1 1 0 0 1 0 1 1 1 0 0 1 1 0 0 0 0 0 1 0 0 0 1 1 0 0 0 0 1 1 1 1 1 1 0 0 1
    1 1] <class 'numpy.ndarray'>
    """
    hashed = _encode(lat, lon, precision)
    return np.concatenate([BASE_32_TO_BIN[x] for x in hashed])


def _reset_and_create_arrays_none(
    data: DataFrame, reset_index: bool = True
) -> tuple[ndarray, ndarray, ndarray, ndarray]:
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

    Example
    -------
    >>> from pymove.utils.geoutils import _reset_and_create_arrays_none
    >>> print(type(_reset_and_create_arrays_none(geoLife_df)))
    >>> _reset_and_create_arrays_none(geoLife_df)
    <class 'tuple'>
    (array([nan, nan, nan, nan, nan]),
     array([nan, nan, nan, nan, nan]),
     array([None, None, None, None, None], dtype=object),
     array([None, None, None, None, None], dtype=object))
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


def create_geohash_df(data: DataFrame, precision: float = 15):
    """
    Create geohash from geographic coordinates and integrate with df.

    Parameters
    ----------
    data : dataframe
        The input trajectories data
    precision : float, optional
        Number of characters in resulting geohash, by default 15

    Return
    ------
    A DataFrame with the additional column 'geohash'

    Example
    -------
    >>> from pymove.utils.geoutils import create_geohash_df, _reset_and_create_arrays_none
    >>> geoLife_df
              lat          lon
    0   39.984094   116.319236
    1   39.984198   116.319322
    2   39.984224   116.319402
    3   39.984211   116.319389
    4   39.984217   116.319422
    >>> print(type (create_geohash_df(geoLife_df)))
    >>> geoLife_df
    <class 'NoneType'>
              lat          lon           geohash
    0   39.984094   116.319236   wx4eqyvh4xkg0xs
    1   39.984198   116.319322   wx4eqyvhudszsev
    2   39.984224   116.319402   wx4eqyvhyx8d9wc
    3   39.984211   116.319389   wx4eqyvhyjnv5m7
    4   39.984217   116.319422   wx4eqyvhyyr2yy8
    """
    _, _, geohash, _ = _reset_and_create_arrays_none(data)

    for idx, row in progress_bar(
        data[[LATITUDE, LONGITUDE]].iterrows(), total=data.shape[0]
    ):
        geohash[idx] = _encode(row[LATITUDE], row[LONGITUDE], precision)

    data[GEOHASH] = geohash


def create_bin_geohash_df(data: DataFrame, precision: float = 15):
    """
    Create trajectory geohash binaries and integrate with df.

    Parameters
    ----------
    data : dataframe
        The input trajectories data
    precision : float, optional
        Number of characters in resulting geohash, by default 15

    Return
    ------
    A DataFrame with the additional column 'bin_geohash'

    Example
    -------
    >>> from pymove.utils.geoutils import create_bin_geohash_df
    >>> geoLife_df
             lat          lon
    0   39.984094   116.319236
    1   39.984198   116.319322
    2   39.984224   116.319402
    3   39.984211   116.319389
    4   39.984217   116.319422
    >>> print(type(create_bin_geohash_df(geoLife_df)))
    >>> geoLife_df
    <class 'NoneType'>
              lat         lon                                         bin_geohash
    0   39.984094   116.319236  [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, ...
    1   39.984198   116.319322  [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, ...
    2   39.984224   116.319402  [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, ...
    3   39.984211   116.319389  [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, ...
    4   39.984217   116.319422  [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, ...
    """
    *_, bin_geohash = _reset_and_create_arrays_none(data)

    for idx, row in progress_bar(
        data[[LATITUDE, LONGITUDE]].iterrows(), total=data.shape[0]
    ):
        bin_geohash[idx] = _bin_geohash(row[LATITUDE], row[LONGITUDE], precision)

    data[BIN_GEOHASH] = bin_geohash


def decode_geohash_to_latlon(
    data: DataFrame,
    label_geohash: str = GEOHASH,
    reset_index: bool = True
):
    """
    Decode feature with hash of trajectories back to geographic coordinates.

    Parameters
    ----------
    data : dataframe
        The input trajectories data
    label_geohash : str, optional
        The name of the feature with hashed trajectories, by default GEOHASH
    reset_index : boolean, optional
        Condition to reset the df index, by default True

    Return
    ------
    A DataFrame with the additional columns 'lat_decode' and 'lon_decode'

    Example
    -------
    >>> from pymove.utils.geoutils import decode_geohash_to_latlon
    >>> geoLife_df
              lat          lon           geohash
    0   39.984094   116.319236   wx4eqyvh4xkg0xs
    1   39.984198   116.319322   wx4eqyvhudszsev
    2   39.984224   116.319402   wx4eqyvhyx8d9wc
    3   39.984211   116.319389   wx4eqyvhyjnv5m7
    4   39.984217   116.319422   wx4eqyvhyyr2yy8
    >>> print(type(decode_geohash_to_latlon(geoLife_df)))
    >>> geoLife_df
    <class 'NoneType'>
              lat          lon           geohash  lat_decode   lon_decode
    0   39.984094   116.319236   wx4eqyvh4xkg0xs   39.984094   116.319236
    1   39.984198   116.319322   wx4eqyvhudszsev   39.984198   116.319322
    2   39.984224   116.319402   wx4eqyvhyx8d9wc   39.984224   116.319402
    3   39.984211   116.319389   wx4eqyvhyjnv5m7   39.984211   116.319389
    4   39.984217   116.319422   wx4eqyvhyyr2yy8   39.984217   116.319422
    """
    if label_geohash not in data:
        raise ValueError(f'feature {label_geohash} not in df')

    lat, lon, _, _ = _reset_and_create_arrays_none(data, reset_index=reset_index)

    for idx, row in progress_bar(data[[label_geohash]].iterrows(), total=data.shape[0]):
        lat_lon = _decode(row[label_geohash])
        lat[idx] = lat_lon[0]
        lon[idx] = lat_lon[1]

    data[LATITUDE_DECODE] = lat
    data[LONGITUDE_DECODE] = lon
