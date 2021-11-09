"""
Unit conversion operations.

lat_meters,
meters_to_eps,
list_to_str,
list_to_csv_str,
list_to_svm_line,
lon_to_x_spherical,
lat_to_y_spherical,
x_to_lon_spherical,
y_to_lat_spherical,
geometry_points_to_lat_and_lon,
lat_and_lon_decimal_degrees_to_decimal,
ms_to_kmh,
kmh_to_ms,
meters_to_kilometers,
kilometers_to_meters,
seconds_to_minutes,
minute_to_seconds,
minute_to_hours,
hours_to_minute,
seconds_to_hours,
hours_to_seconds

"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from numpy import ndarray
from pandas import DataFrame
from shapely.geometry import Point

from pymove.utils.constants import (
    DIST_TO_PREV,
    EARTH_RADIUS,
    GEOMETRY,
    LATITUDE,
    LONGITUDE,
    SPEED_TO_PREV,
    TIME_TO_PREV,
)

if TYPE_CHECKING:
    from pymove.core.dask import DaskMoveDataFrame
    from pymove.core.pandas import PandasMoveDataFrame


def lat_meters(lat: float) -> float:
    """
    Transform latitude degree to meters.

    Parameters
    ----------
    lat : float
        This represent latitude value.

    Returns
    -------
    float
        Represents the corresponding latitude value in meters.

    Examples
    --------
    Latitude in Fortaleza: -3.71839
    >>> from pymove.utils.conversions import lat_meters
    >>> lat_meters(-3.71839)
    110832.75545918777
    """
    rlat = float(lat) * math.pi / 180
    # meter per degree Latitude
    meters_lat = (
        111132.92 - 559.82 * math.cos(2 * rlat) + 1.175 * math.cos(4 * rlat)
    )
    # meter per degree Longitude
    meters_lgn = 111412.84 * math.cos(rlat) - 93.5 * math.cos(3 * rlat)
    meters = (meters_lat + meters_lgn) / 2
    return meters


def meters_to_eps(
    radius_meters: float, earth_radius: float = EARTH_RADIUS
) -> float:
    """
    Converts radius in meters to eps.

    Parameters
    ----------
    radius_meters : float
        radius in meters
    earth_radius : float, optional
        radius of the earth in the location, by default EARTH_RADIUS

    Returns
    -------
    float
        radius in eps

    Example
    -------
    >>> from pymove.utils.conversions import meters_to_eps
    >>> earth_radius = 6371000
    >>> meters_to_eps(earth_radius)
    1000.0
    """
    return radius_meters / earth_radius


def list_to_str(input_list: list, delimiter: str = ',') -> str:
    """
    Concatenates a list elements, joining them by the separator `delimiter`.

    Parameters
    ----------
    input_list : list
        List with elements to be joined.
    delimiter : str, optional
        The separator used between elements, by default ','.

    Returns
    -------
    str
        Returns a string, resulting from concatenation of list elements,
        separeted by the delimiter.

    Example
    -------
    >>> from pymove.utils.conversions import list_to_str
    >>> list = [1,2,3,4,5]
    >>> print(list_to_str(list, 'x'), type(list_to_str(list)))
    1x2x3x4x5 <class 'str'>
    """
    return delimiter.join(
        [x if isinstance(x, str) else repr(x) for x in input_list]
    )


def list_to_csv_str(input_list: list) -> str:
    """
    Concatenates the elements of the list, joining them by ",".

    Parameters
    ----------
    input_list : list
        List with elements to be joined.

    Returns
    -------
    str
        Returns a string, resulting from concatenation of list elements,
        separeted by ",".

    Example
    -------
    >>> from pymove.utils.conversions import list_to_csv_str
    >>> list = [1,2,3,4,5]
    >>> print(list_to_csv_str(list), type(list_to_csv_str(list)))
    1,2,3,4,5 <class 'str'>
    """
    return list_to_str(input_list)


def list_to_svm_line(original_list: list) -> str:
    """
    Concatenates list elements in consecutive element pairs.

    Parameters
    ----------
    original_list : list
        The elements to be joined

    Returns
    -------
    str
        Returns a string, resulting from concatenation of list elements
        in consecutive element pairs, separeted by " ".

    Example
    -------
    >>> from pymove.utils.conversions import list_to_svm_line
    >>> list = [1,2,3,4,5]
    >>> print(list_to_svm_line(list), type(list_to_svm_line(list)))
    1 1:2 2:3 3:4 4:5 <class 'str'>
    """
    list_size = len(original_list)
    svm_line = '%s ' % original_list[0]
    for i in range(1, list_size):
        svm_line += f'{i}:{original_list[i]} '
    return svm_line.rstrip()


def lon_to_x_spherical(lon: float | ndarray) -> float | ndarray:
    """
    Convert longitude to X EPSG:3857 WGS 84/Pseudo-Mercator.

    Parameters
    ----------
    lon : float
        This represents longitude value.

    Returns
    -------
    float
        X offset from your original position in meters.

    Examples
    --------
    >>> from pymove.utils.conversions import lon_to_x_spherical
    >>> lon_fortaleza = -38.5434
    >>> for_x = lon_to_x_spherical(lon_fortaleza)
    >>> print(x_for, type(x_for))
    -4290631.66144146 <class 'numpy.float64'>

    References
    ----------
    https://epsg.io/transform

    """
    return 6378137 * np.radians(lon)


def lat_to_y_spherical(lat: float | ndarray) -> float | ndarray:
    """
    Convert latitude to Y EPSG:3857 WGS 84/Pseudo-Mercator.

    Parameters
    ----------
    lat : float
        This represents latitude value.

    Returns
    -------
    float
        Y offset from your original position in meters.

    Examples
    --------
    >>> from pymove.utils.conversions import lat_to_y_spherical
    >>> lat_fortaleza = -3.71839
    >>> for_y = lat_to_y_spherical(lat_fortaleza)
    >>> print(y_for, type(y_for))
    -414220.15015607665 <class 'numpy.float64'>

    References
    ----------
    https://epsg.io/transform

    """
    return 6378137 * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2.0))


def x_to_lon_spherical(x: float | ndarray) -> float | ndarray:
    """
    Convert X EPSG:3857 WGS 84 / Pseudo-Mercator to longitude.

    Parameters
    ----------
    x : float
        X offset from your original position in meters.

    Returns
    -------
    float
        Represents longitude.

    Examples
    --------
    >>> from pymove.utils.conversions import x_to_lon_spherical
    >>> for_x = -4290631.66144146
    >>> print(x_to_lon_spherical(for_x), type(x_to_lon_spherical(for_x)))
    -38.5434 <class 'numpy.float64'>

    References
    ----------
    https://epsg.io/transform

    """
    return np.degrees(x / 6378137.0)


def y_to_lat_spherical(y: float | ndarray) -> float | ndarray:
    """
    Convert Y EPSG:3857 WGS 84 / Pseudo-Mercator to latitude.

    Parameters
    ----------
    y : float
        Y offset from your original position in meters.

    Returns
    -------
    float
        Represents latitude.

    Examples
    --------
    >>> from pymove.utils.conversions import y_to_lat_spherical
    >>> for_y = -414220.15015607665
    >>> print(y_to_lat_spherical(y_for), type(y_to_lat_spherical(y_for)))
    -3.7183900000000096 <class 'numpy.float64'>

    References
    ----------
    https://epsg.io/transform

    """
    return np.degrees(np.arctan(np.sinh(y / 6378137.0)))


def geometry_points_to_lat_and_lon(
    move_data: DataFrame,
    geometry_label: str = GEOMETRY,
    drop_geometry: bool = False,
    inplace: bool = False
) -> DataFrame:
    """
    Creates lat and lon columns from Points in geometry column.

    Removes geometries that are not of the Point type.

    Parameters
    ----------
    move_data : DataFrame
        Input trajectory data.
    geometry: str, optional
        Represents column name of the geometry column, by default GEOMETRY
    drop_geometry: bool, optional
        Option to drop the geometry column, by default False
    inplace: bool, optional
        Whether the operation will be done in the original dataframe, by default False

    Returns
    -------
    DataFrame
        A new dataframe with the converted feature or None

    Example
    -------
    >>> from pymove.utils.conversions import geometry_points_to_lat_and_lon
    >>> geom_points_df
        id                     geometry
    0    1   POINT (116.36184 39.77529)
    1    2   POINT (116.36298 39.77564)
    2    3   POINT (116.33767 39.83148)
    >>> geometry_points_to_lat_and_lon(geom_points_df)
        id                     geometry        lon       lat
    0    1   POINT (116.36184 39.77529)  116.36184  39.77529
    1    2   POINT (116.36298 39.77564)  116.36298  39.77564
    2    3   POINT (116.33767 39.83148)  116.33767  39.83148
    """
    if not inplace:
        move_data = move_data.copy()

    move_data = move_data[
        move_data[geometry_label].map(type) == Point
    ]
    move_data[LONGITUDE] = move_data[geometry_label].map(lambda p: p.x)
    move_data[LATITUDE] = move_data[geometry_label].map(lambda q: q.y)

    if drop_geometry:
        move_data.drop(geometry_label, axis=1, inplace=True)

    if not inplace:
        return move_data


def lat_and_lon_decimal_degrees_to_decimal(
    move_data: DataFrame,
    latitude: str = LATITUDE,
    longitude: str = LONGITUDE
) -> DataFrame:
    """
    Converts latitude and longitude format from decimal degrees to decimal format.

    Parameters
    ----------
    move_data : DataFrame
        Input trajectory data.
    latitude: str, optional
        Represents column name of the latitude column, by default LATITUDE
    longitude: str, optional
        Represents column name of the longitude column, by default LONGITUDE

    Returns
    -------
    DataFrame
        A new dataframe with the converted feature

    Example
    -------
    >>> from pymove.utils.conversions import lat_and_lon_decimal_degrees_to_decimal
    >>> lat_and_lon_df
       id     lat     lon
    0   0   28.0N   94.8W
    1   1   41.3N   50.4W
    2   1   40.8N   47.5W
    >>> lat_and_lon_decimal_degrees_to_decimal(lat_and_lon_df)
       id    lat      lon
    0   0   28.0    -94.8
    1   1   41.3    -50.4
    2   1   40.8    -47.5
    """
    def _decimal_degree_to_decimal(row):
        if (row[latitude][-1:] == 'N'):
            row[latitude] = float(row[latitude][:-1])
        else:
            row[latitude] = float(row[latitude][:-1]) * -1

        if (row[longitude][-1:] == 'E'):
            row[longitude] = float(row[longitude][:-1])
        else:
            row[longitude] = float(row[longitude][:-1]) * -1
        return row

    return move_data.apply(_decimal_degree_to_decimal, axis=1)


def ms_to_kmh(
    move_data: 'PandasMoveDataFrame' | 'DaskMoveDataFrame',
    label_speed: str = SPEED_TO_PREV,
    new_label: str = None,
    inplace: bool = False,
) -> 'PandasMoveDataFrame' | 'DaskMoveDataFrame' | None:
    """
    Convert values, in ms, in label_speed column to kmh.

    Parameters
    ----------
    move_data : DataFrame
        Input trajectory data.
    label_speed : str, optional
        Represents column name of speed, by default SPEED_TO_PREV
    new_label: str, optional
        Represents a new column that will contain the conversion result, by default None
    inplace: bool, optional
        Whether the operation will be done in the original dataframe, by default False

    Returns
    -------
    DataFrame
        A new dataframe with the converted feature or None

    Example
    -------
    >>> from pymove.utils.conversions import ms_to_kmh
    >>> geo_life_df
              lat          lon             datetime     id
    0   39.984094   116.319236   2008-10-23 05:53:05     1
    1   39.984198   116.319322   2008-10-23 05:53:06     1
    2   39.984224   116.319402   2008-10-23 05:53:11     1
    3   39.984211   116.319389   2008-10-23 05:53:16     1
    4   39.984217   116.319422   2008-10-23 05:53:21     1
    >>> geo_life.generate_dist_time_speed_features(inplace=True)
    >>> geo_life
       id         lat          lon              datetime\
         dist_to_prev    time_to_prev    speed_to_prev
    0   1   39.984094   116.319236   2008-10-23 05:53:05\
                  NaN             NaN              NaN
    1   1   39.984198   116.319322   2008-10-23 05:53:06\
            13.690153             1.0        13.690153
    2   1   39.984224   116.319402   2008-10-23 05:53:11\
             7.403788             5.0         1.480758
    3   1   39.984211   116.319389   2008-10-23 05:53:16\
             1.821083             5.0         0.364217
    4   1   39.984217   116.319422   2008-10-23 05:53:21\
             2.889671             5.0         0.577934
    >>> ms_to_kmh(geo_life, inplace=False)
       id         lat          lon             datetime\
          dist_to_prev    time_to_prev    speed_to_prev
    0   1   39.984094   116.319236   2008-10-23 05:53:05\
                  NaN             NaN              NaN
    1   1   39.984198   116.319322   2008-10-23 05:53:06\
            13.690153             1.0        49.284551
    2   1   39.984224   116.319402   2008-10-23 05:53:11\
             7.403788             5.0         5.330727
    3   1   39.984211   116.319389   2008-10-23 05:53:16\
             1.821083             5.0         1.311180
    4   1   39.984217   116.319422   2008-10-23 05:53:21\
             2.889671             5.0         2.080563
    """
    if not inplace:
        move_data = move_data.copy()

    if label_speed not in move_data:
        move_data.generate_dist_time_speed_features()
    move_data[label_speed] = move_data[label_speed].apply(
        lambda row: row * 3.6
    )
    if new_label is not None:
        move_data.rename(columns={label_speed: new_label}, inplace=True)
    if not inplace:
        return move_data


def kmh_to_ms(
    move_data: 'PandasMoveDataFrame' | 'DaskMoveDataFrame',
    label_speed: str = SPEED_TO_PREV,
    new_label: str | None = None,
    inplace: bool = False,
) -> 'PandasMoveDataFrame' | 'DaskMoveDataFrame' | None:
    """
    Convert values, in kmh, in label_speed column to ms.

    Parameters
    ----------
    move_data : DataFame
        Input trajectory data.
    label_speed : str, optional
        Represents column name of speed, by default SPEED_TO_PREV
    new_label: str, optional
        Represents a new column that will contain the conversion result, by default None
    inplace: bool, optional
        Whether the operation will be done in the original dataframe, by default False

    Returns
    -------
    DataFrame
        A new dataframe with the converted feature or None

    Example
    -------
    >>> from pymove.utils.conversions import kmh_to_ms
    >>> geo_life_df
       id         lat          lon              datetime\
          dist_to_prev    time_to_prev    speed_to_prev
    0   1   39.984094   116.319236   2008-10-23 05:53:05\
                  NaN             NaN              NaN
    1   1   39.984198   116.319322   2008-10-23 05:53:06\
            13.690153             1.0        49.284551
    2   1   39.984224   116.319402   2008-10-23 05:53:11\
             7.403788             5.0         5.330727
    3   1   39.984211   116.319389   2008-10-23 05:53:16\
             1.821083             5.0         1.311180
    4   1   39.984217   116.319422   2008-10-23 05:53:21\
             2.889671             5.0         2.080563
    >>> kmh_to_ms(geo_life, inplace=False)
       id         lat          lon              datetime\
         dist_to_prev    time_to_prev    speed_to_prev
    0   1   39.984094   116.319236   2008-10-23 05:53:05\
                  NaN             NaN              NaN
    1   1   39.984198   116.319322   2008-10-23 05:53:06\
            13.690153             1.0        13.690153
    2   1   39.984224   116.319402   2008-10-23 05:53:11\
             7.403788             5.0         1.480758
    3   1   39.984211   116.319389   2008-10-23 05:53:16\
             1.821083             5.0         0.364217
    4   1   39.984217   116.319422   2008-10-23 05:53:21\
             2.889671             5.0         0.577934

    """
    if not inplace:
        move_data = move_data.copy()

    if label_speed not in move_data:
        move_data.generate_dist_time_speed_features()
        ms_to_kmh(move_data, label_speed)
    move_data[label_speed] = move_data[label_speed].apply(
        lambda row: row / 3.6
    )
    if new_label is not None:
        move_data.rename(columns={label_speed: new_label}, inplace=True)
    if not inplace:
        return move_data


def meters_to_kilometers(
    move_data: 'PandasMoveDataFrame' | 'DaskMoveDataFrame',
    label_distance: str = DIST_TO_PREV,
    new_label: str | None = None,
    inplace: bool = False,
) -> 'PandasMoveDataFrame' | 'DaskMoveDataFrame' | None:
    """
    Convert values, in meters, in label_distance column to kilometers.

    Parameters
    ----------
    move_data : DataFame
        Input trajectory data.
    label_distance : str, optional
        Represents column name of speed, by default DIST_TO_PREV
    new_label: str, optional
        Represents a new column that will contain the conversion result, by default None
    inplace: bool, optional
        Whether the operation will be done in the original dataframe, by default False

    Returns
    -------
    DataFrame
        A new dataframe with the converted feature or None

    Example
    -------
    >>> from pymove.utils.conversions import meters_to_kilometers
    >>> geo_life_df
       id         lat          lon              datetime\
         dist_to_prev    time_to_prev    speed_to_prev
    0   1   39.984094   116.319236   2008-10-23 05:53:05\
                  NaN             NaN              NaN
    1   1   39.984198   116.319322   2008-10-23 05:53:06\
            13.690153             1.0        13.690153
    2   1   39.984224   116.319402   2008-10-23 05:53:11\
             7.403788             5.0         1.480758
    3   1   39.984211   116.319389   2008-10-23 05:53:16\
             1.821083             5.0         0.364217
    4   1   39.984217   116.319422   2008-10-23 05:53:21\
             2.889671             5.0         0.577934
    >>> meters_to_kilometers(geo_life, inplace=False)
       id         lat          lon              datetime\
         dist_to_prev    time_to_prev    speed_to_prev
    0   1   39.984094   116.319236   2008-10-23 05:53:05\
                  NaN             NaN              NaN
    1   1   39.984198   116.319322   2008-10-23 05:53:06\
             0.013690             1.0        13.690153
    2   1   39.984224   116.319402   2008-10-23 05:53:11\
             0.007404             5.0         1.480758
    3   1   39.984211   116.319389   2008-10-23 05:53:16\
             0.001821             5.0         0.364217
    4   1   39.984217   116.319422   2008-10-23 05:53:21\
             0.002890             5.0         0.577934

    """
    if not inplace:
        move_data = move_data.copy()

    if label_distance not in move_data:
        move_data.generate_dist_time_speed_features()
    move_data[label_distance] = move_data[label_distance].apply(
        lambda row: row / 1000
    )
    if new_label is not None:
        move_data.rename(columns={label_distance: new_label}, inplace=True)
    if not inplace:
        return move_data


def kilometers_to_meters(
    move_data: 'PandasMoveDataFrame' | 'DaskMoveDataFrame',
    label_distance: str = DIST_TO_PREV,
    new_label: str | None = None,
    inplace: bool = False,
) -> 'PandasMoveDataFrame' | 'DaskMoveDataFrame' | None:
    """
    Convert values, in kilometers, in label_distance column to meters.

    Parameters
    ----------
    move_data : DataFame
        Input trajectory data.
    label_distance : str, optional
        Represents column name of speed, by default DIST_TO_PREV
    new_label: str, optional
        Represents a new column that will contain the conversion result, by default None
    inplace: bool, optional
        Whether the operation will be done in the original dataframe, by default False

    Returns
    -------
    DataFrame
        A new dataframe with the converted feature or None

    Example
    -------
    >>> from pymove.utils.conversions import kilometers_to_meters
    >>> geo_life_df
       id         lat          lon              datetime\
         dist_to_prev    time_to_prev    speed_to_prev
    0   1   39.984094   116.319236   2008-10-23 05:53:05\
                  NaN             NaN              NaN
    1   1   39.984198   116.319322   2008-10-23 05:53:06\
             0.013690             1.0        13.690153
    2   1   39.984224   116.319402   2008-10-23 05:53:11\
             0.007404             5.0         1.480758
    3   1   39.984211   116.319389   2008-10-23 05:53:16\
             0.001821            5.0         0.364217
    4   1   39.984217   116.319422   2008-10-23 05:53:21\
             0.002890             5.0         0.577934
    >>> kilometers_to_meters(geo_life, inplace=False)
       id         lat          lon              datetime\
         dist_to_prev    time_to_prev    speed_to_prev
    0   1   39.984094   116.319236   2008-10-23 05:53:05\
                  NaN             NaN              NaN
    1   1   39.984198   116.319322   2008-10-23 05:53:06\
            13.690153             1.0        13.690153
    2   1   39.984224   116.319402   2008-10-23 05:53:11\
             7.403788             5.0         1.480758
    3   1   39.984211   116.319389   2008-10-23 05:53:16\
             1.821083             5.0         0.364217
    4   1   39.984217   116.319422   2008-10-23 05:53:21\
             2.889671             5.0         0.577934

    """
    if not inplace:
        move_data = move_data.copy()

    if label_distance not in move_data:
        move_data.generate_dist_time_speed_features()
        meters_to_kilometers(move_data, label_distance)
    move_data[label_distance] = move_data[label_distance].apply(
        lambda row: row * 1000
    )
    if new_label is not None:
        move_data.rename(columns={label_distance: new_label}, inplace=True)
    if not inplace:
        return move_data


def seconds_to_minutes(
    move_data: 'PandasMoveDataFrame' | 'DaskMoveDataFrame',
    label_time: str = TIME_TO_PREV,
    new_label: str | None = None,
    inplace: bool = False,
) -> 'PandasMoveDataFrame' | 'DaskMoveDataFrame' | None:
    """
    Convert values, in seconds, in label_distance column to minutes.

    Parameters
    ----------
    move_data : DataFame
        Input trajectory data.
    label_time : str, optional
        Represents column name of speed, by default TIME_TO_PREV
    new_label: str, optional
        Represents a new column that will contain the conversion result, by default None
    inplace: bool, optional
        Whether the operation will be done in the original dataframe, by default False

    Returns
    -------
    DataFrame
        A new dataframe with the converted feature or None

    Example
    -------
    >>> from pymove.utils.conversions import seconds_to_minutes
    >>> geo_life_df
       id         lat          lon              datetime\
         dist_to_prev    time_to_prev    speed_to_prev
    0   1   39.984094   116.319236   2008-10-23 05:53:05\
                  NaN             NaN              NaN
    1   1   39.984198   116.319322   2008-10-23 05:53:06\
            13.690153             1.0        13.690153
    2   1   39.984224   116.319402   2008-10-23 05:53:11\
             7.403788             5.0         1.480758
    3   1   39.984211   116.319389   2008-10-23 05:53:16\
             1.821083             5.0         0.364217
    4   1   39.984217   116.319422   2008-10-23 05:53:21\
             2.889671             5.0         0.577934
    >>> seconds_to_minutes(geo_life, inplace=False)
       id         lat          lon              datetime\
         dist_to_prev    time_to_prev    speed_to_prev
    0   1   39.984094   116.319236   2008-10-23 05:53:05\
                  NaN             NaN              NaN
    1   1   39.984198   116.319322   2008-10-23 05:53:06\
            13.690153        0.016667        13.690153
    2   1   39.984224   116.319402   2008-10-23 05:53:11\
             7.403788        0.083333         1.480758
    3   1   39.984211   116.319389   2008-10-23 05:53:16\
             1.821083        0.083333         0.364217
    4   1   39.984217   116.319422   2008-10-23 05:53:21\
             2.889671        0.083333         0.577934

    """
    if not inplace:
        move_data = move_data.copy()

    if label_time not in move_data:
        move_data.generate_dist_time_speed_features()
    move_data[label_time] = move_data[label_time].apply(
        lambda row: row / 60.0
    )
    if new_label is not None:
        move_data.rename(columns={label_time: new_label}, inplace=True)
    if not inplace:
        return move_data


def minute_to_seconds(
    move_data: 'PandasMoveDataFrame' | 'DaskMoveDataFrame',
    label_time: str = TIME_TO_PREV,
    new_label: str | None = None,
    inplace: bool = False,
) -> 'PandasMoveDataFrame' | 'DaskMoveDataFrame' | None:
    """
    Convert values, in minutes, in label_distance column to seconds.

    Parameters
    ----------
    move_data : DataFame
        Input trajectory data.
    label_time : str, optional
        Represents column name of speed, by default TIME_TO_PREV
    new_label: str, optional
        Represents a new column that will contain the conversion result, by default None
    inplace: bool, optional
        Whether the operation will be done in the original dataframe, by default False

    Returns
    -------
    DataFrame
        A new dataframe with the converted feature or None

    Example
    -------
    >>> from pymove.utils.conversions import minute_to_seconds
    >>> geo_life_df
       id         lat          lon              datetime\
         dist_to_prev    time_to_prev    speed_to_prev
    0   1   39.984094   116.319236   2008-10-23 05:53:05\
                  NaN             NaN              NaN
    1   1   39.984198   116.319322   2008-10-23 05:53:06\
            13.690153        0.016667        13.690153
    2   1   39.984224   116.319402   2008-10-23 05:53:11\
             7.403788        0.083333         1.480758
    3   1   39.984211   116.319389   2008-10-23 05:53:16\
             1.821083        0.083333         0.364217
    4   1   39.984217   116.319422   2008-10-23 05:53:21\
             2.889671        0.083333         0.577934
    >>> minute_to_seconds(geo_life, inplace=False)
       id         lat          lon              datetime\
         dist_to_prev    time_to_prev    speed_to_prev
    0   1   39.984094   116.319236   2008-10-23 05:53:05\
                  NaN             NaN              NaN
    1   1   39.984198   116.319322   2008-10-23 05:53:06\
            13.690153             1.0        13.690153
    2   1   39.984224   116.319402   2008-10-23 05:53:11\
             7.403788             5.0         1.480758
    3   1   39.984211   116.319389   2008-10-23 05:53:16\
             1.821083             5.0         0.364217
    4   1   39.984217   116.319422   2008-10-23 05:53:21\
             2.889671             5.0         0.577934

    """
    if not inplace:
        move_data = move_data.copy()

    if label_time not in move_data:
        move_data.generate_dist_time_speed_features()
        seconds_to_minutes(move_data, label_time)
    move_data['time_to_prev'] = move_data['time_to_prev'].apply(
        lambda row: row * 60.0
    )
    if new_label is not None:
        move_data.rename(columns={label_time: new_label}, inplace=True)
    if not inplace:
        return move_data


def minute_to_hours(
    move_data: 'PandasMoveDataFrame' | 'DaskMoveDataFrame',
    label_time: str = TIME_TO_PREV,
    new_label: str | None = None,
    inplace: bool = False,
) -> 'PandasMoveDataFrame' | 'DaskMoveDataFrame' | None:
    """
    Convert values, in minutes, in label_distance column to hours.

    Parameters
    ----------
    move_data : DataFame
        Input trajectory data.
    label_time : str, optional
        Represents column name of speed, by default TIME_TO_PREV
    new_label: str, optional
        Represents a new column that will contain the conversion result, by default None
    inplace: bool, optional
        Whether the operation will be done in the original dataframe, by default False

    Returns
    -------
    DataFrame
        A new dataframe with the converted feature or None

    Example
    -------
    >>> from pymove.utils.conversions import minute_to_hours, seconds_to_minutes
    >>> geo_life_df
       id         lat          lon              datetime\
         dist_to_prev    time_to_prev    speed_to_prev
    0   1   39.984094   116.319236   2008-10-23 05:53:05\
                  NaN             NaN              NaN
    1   1   39.984198   116.319322   2008-10-23 05:53:06\
            13.690153             1.0        13.690153
    2   1   39.984224   116.319402   2008-10-23 05:53:11\
             7.403788             5.0         1.480758
    3   1   39.984211   116.319389   2008-10-23 05:53:16\
             1.821083             5.0         0.364217
    4   1   39.984217   116.319422   2008-10-23 05:53:21\
             2.889671             5.0         0.577934
    >>> seconds_to_minutes(geo_life, inplace=True)
    >>> minute_to_hours(geo_life, inplace=False)
       id         lat          lon              datetime\
         dist_to_prev    time_to_prev    speed_to_prev
    0   1   39.984094   116.319236   2008-10-23 05:53:05\
                  NaN             NaN              NaN
    1   1   39.984198   116.319322   2008-10-23 05:53:06\
            13.690153        0.000278        13.690153
    2   1   39.984224   116.319402   2008-10-23 05:53:11\
             7.403788        0.001389         1.480758
    3   1   39.984211   116.319389   2008-10-23 05:53:16\
             1.821083        0.001389         0.364217
    4   1   39.984217   116.319422   2008-10-23 05:53:21\
             2.889671        0.001389         0.577934

    """
    if not inplace:
        move_data = move_data.copy()

    if label_time not in move_data:
        move_data.generate_dist_time_speed_features()
        seconds_to_minutes(move_data, label_time)
    move_data[label_time] = move_data[label_time].apply(
        lambda row: row / 60.0
    )
    if new_label is not None:
        move_data.rename(columns={label_time: new_label}, inplace=True)
    if not inplace:
        return move_data


def hours_to_minute(
    move_data: 'PandasMoveDataFrame' | 'DaskMoveDataFrame',
    label_time: str = TIME_TO_PREV,
    new_label: str | None = None,
    inplace: bool = False,
) -> 'PandasMoveDataFrame' | 'DaskMoveDataFrame' | None:
    """
    Convert values, in hours, in label_distance column to minute.

    Parameters
    ----------
    move_data : DataFame
        Input trajectory data.
    label_time : str, optional
        Represents column name of speed, by default TIME_TO_PREV
    new_label: str, optional
        Represents a new column that will contain the conversion result, by default None
    inplace: bool, optional
        Whether the operation will be done in the original dataframe, by default False

    Returns
    -------
    DataFrame
        A new dataframe with the converted feature or None

    Example
    -------
    >>> from pymove.utils.conversions import hours_to_minute
    >>> geo_life_df
       id         lat          lon              datetime\
         dist_to_prev    time_to_prev    speed_to_prev
    0   1   39.984094   116.319236   2008-10-23 05:53:05\
                  NaN             NaN              NaN
    1   1   39.984198   116.319322   2008-10-23 05:53:06\
            13.690153        0.000278        13.690153
    2   1   39.984224   116.319402   2008-10-23 05:53:11\
             7.403788        0.001389         1.480758
    3   1   39.984211   116.319389   2008-10-23 05:53:16\
             1.821083        0.001389         0.364217
    4   1   39.984217   116.319422   2008-10-23 05:53:21\
             2.889671        0.001389         0.577934
    >>> hours_to_minute(geo_life, inplace=False)
       id         lat          lon              datetime\
         dist_to_prev    time_to_prev    speed_to_prev
    0   1   39.984094   116.319236   2008-10-23 05:53:05\
                  NaN             NaN              NaN
    1   1   39.984198   116.319322   2008-10-23 05:53:06\
            13.690153        0.016667        13.690153
    2   1   39.984224   116.319402   2008-10-23 05:53:11\
             7.403788        0.083333         1.480758
    3   1   39.984211   116.319389   2008-10-23 05:53:16\
             1.821083        0.083333         0.364217
    4   1   39.984217   116.319422   2008-10-23 05:53:21\
             2.889671        0.083333         0.577934

    """
    if not inplace:
        move_data = move_data.copy()

    if label_time not in move_data:
        move_data.generate_dist_time_speed_features()
        seconds_to_hours(move_data, label_time)
    move_data[label_time] = move_data[label_time].apply(
        lambda row: row * 60.0
    )
    if new_label is not None:
        move_data.rename(columns={label_time: new_label}, inplace=True)
    if not inplace:
        return move_data


def seconds_to_hours(
    move_data: 'PandasMoveDataFrame' | 'DaskMoveDataFrame',
    label_time: str = TIME_TO_PREV,
    new_label: str | None = None,
    inplace: bool = False,
) -> 'PandasMoveDataFrame' | 'DaskMoveDataFrame' | None:
    """
    Convert values, in seconds, in label_distance column to hours.

    Parameters
    ----------
    move_data : DataFame
        Input trajectory data.
    label_time : str, optional
        Represents column name of speed, by default TIME_TO_PREV
    new_label: str, optional
        Represents a new column that will contain the conversion result, by default None
    inplace: bool, optional
        Whether the operation will be done in the original dataframe, by default False

    Returns
    -------
    DataFrame
        A new dataframe with the converted feature or None

    Example
    -------
    >>> from pymove.utils.conversions import minute_to_seconds, seconds_to_hours
    >>> geo_life_df
       id         lat          lon              datetime\
         dist_to_prev    time_to_prev    speed_to_prev
    0   1   39.984094   116.319236   2008-10-23 05:53:05\
                  NaN             NaN              NaN
    1   1   39.984198   116.319322   2008-10-23 05:53:06\
            13.690153        0.016667        13.690153
    2   1   39.984224   116.319402   2008-10-23 05:53:11\
             7.403788        0.083333         1.480758
    3   1   39.984211   116.319389   2008-10-23 05:53:16\
             1.821083        0.083333         0.364217
    4   1   39.984217   116.319422   2008-10-23 05:53:21\
             2.889671        0.083333         0.577934
    >>> minute_to_seconds(geo_life, inplace=True)
    >>> seconds_to_hours(geo_life, inplace=False)
       id         lat          lon              datetime\
         dist_to_prev    time_to_prev    speed_to_prev
    0   1   39.984094   116.319236   2008-10-23 05:53:05\
                  NaN             NaN              NaN
    1   1   39.984198   116.319322   2008-10-23 05:53:06\
            13.690153        0.000278        13.690153
    2   1   39.984224   116.319402   2008-10-23 05:53:11\
             7.403788        0.001389         1.480758
    3   1   39.984211   116.319389   2008-10-23 05:53:16\
             1.821083        0.001389         0.364217
    4   1   39.984217   116.319422   2008-10-23 05:53:21\
             2.889671        0.001389         0.577934

    """
    if not inplace:
        move_data = move_data.copy()

    if label_time not in move_data:
        move_data.generate_dist_time_speed_features()
    move_data[label_time] = move_data[label_time].apply(
        lambda row: row / 3600.0
    )
    if new_label is not None:
        move_data.rename(columns={label_time: new_label}, inplace=True)
    if not inplace:
        return move_data


def hours_to_seconds(
    move_data: 'PandasMoveDataFrame' | 'DaskMoveDataFrame',
    label_time: str = TIME_TO_PREV,
    new_label: str | None = None,
    inplace: bool = False,
) -> 'PandasMoveDataFrame' | 'DaskMoveDataFrame' | None:
    """
    Convert values, in hours, in label_distance column to seconds.

    Parameters
    ----------
    move_data : DataFame
        Input trajectory data.
    label_time : str, optional
        Represents column name of speed, by default TIME_TO_PREV
    new_label: str, optional
        Represents a new column that will contain the conversion result, by default None
    inplace: bool, optional
        Whether the operation will be done in the original dataframe, by default False

    Returns
    -------
    DataFrame
        A new dataframe with the converted feature or None

    Example
    -------
    >>> from pymove.utils.conversions import hours_to_seconds
    >>> geo_life_df
       id         lat          lon              datetime\
         dist_to_prev    time_to_prev    speed_to_prev
    0   1   39.984094   116.319236   2008-10-23 05:53:05\
                  NaN             NaN              NaN
    1   1   39.984198   116.319322   2008-10-23 05:53:06\
            13.690153        0.000278        13.690153
    2   1   39.984224   116.319402   2008-10-23 05:53:11\
             7.403788        0.001389         1.480758
    3   1   39.984211   116.319389   2008-10-23 05:53:16\
             1.821083        0.001389         0.364217
    4   1   39.984217   116.319422   2008-10-23 05:53:21\
             2.889671        0.001389         0.577934
    >>> hours_to_seconds(geo_life, inplace=False)
       id         lat          lon              datetime\
         dist_to_prev    time_to_prev    speed_to_prev
    0   1   39.984094   116.319236   2008-10-23 05:53:05\
                  NaN             NaN              NaN
    1   1   39.984198   116.319322   2008-10-23 05:53:06\
            13.690153             1.0        13.690153
    2   1   39.984224   116.319402   2008-10-23 05:53:11\
             7.403788             5.0         1.480758
    3   1   39.984211   116.319389   2008-10-23 05:53:16\
             1.821083             5.0         0.364217
    4   1   39.984217   116.319422   2008-10-23 05:53:21\
             2.889671             5.0         0.577934

    """
    if not inplace:
        move_data = move_data.copy()

    if label_time not in move_data:
        move_data.generate_dist_time_speed_features()
        seconds_to_hours(move_data, label_time)
    move_data[label_time] = move_data[label_time].apply(
        lambda row: row * 3600.0
    )
    if new_label is not None:
        move_data.rename(columns={label_time: new_label}, inplace=True)
    if not inplace:
        return move_data
