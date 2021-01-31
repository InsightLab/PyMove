import math
from typing import TYPE_CHECKING, List, Optional, Text, Union

import numpy as np
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
    from pymove.core.pandas import PandasMoveDataFrame
    from pymove.core.dask import DaskMoveDataFrame


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
    Latitude in Fortaleza: -3.8162973555
    >>> from pymove.utils.conversions import lat_meters
    >>> lat_meters(-3.8162973555)
        110826.6722516857

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
    radius_meters: float, earth_radius: Optional[float] = EARTH_RADIUS
) -> float:
    """
    Converts radius in meters to eps

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
    """
    return radius_meters / earth_radius


def list_to_str(input_list: List, delimiter: Optional[Text] = ',') -> Text:
    """
    Concatenates list elements, joining them by the separator specified by the
    parameter "delimiter".

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

    """

    return delimiter.join(
        [x if isinstance(x, str) else repr(x) for x in input_list]
    )


def list_to_csv_str(input_list: List) -> Text:
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
    >>> from pymove import conversions
    >>> a = [1, 2, 3, 4, 5]
    >>> conversions.list_to_csv_str(a)
    '1 1:2 2:3 3:4 4:5'

    """

    return list_to_str(input_list)


def list_to_svm_line(original_list: List) -> Text:
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
    >>> from pymove import conversions
    >>> a = [1, 2, 3, 4, 5]
    >>> conversions.list_to_svm_line(a)
    '1 1:2 2:3 3:4 4:5'

    """

    list_size = len(original_list)
    svm_line = '%s ' % original_list[0]
    for i in range(1, list_size):
        svm_line += '%s:%s ' % (i, original_list[i])
    return svm_line.rstrip()


def lon_to_x_spherical(lon: float) -> float:
    """
    Convert longitude to X EPSG:3857 WGS 84/Pseudo-Mercator.

    Parameters
    ----------
    lon : float
        Represents longitude.

    Returns
    -------
    float
        X offset from your original position in meters.

    Examples
    --------
    >>> from pymove import conversions
    >>> conversions.lon_to_x_spherical(-38.501597 )
    -4285978.17

    References
    ----------
    https://epsg.io/transform

    """

    return 6378137 * np.radians(lon)


def lat_to_y_spherical(lat: float) -> float:
    """
    Convert latitude to Y EPSG:3857 WGS 84/Pseudo-Mercator.

    Parameters
    ----------
    lat : float
        Represents latitude.

    Returns
    -------
    float
        Y offset from your original position in meters.

    Examples
    --------
    >>> from pymove import conversions
    >>> conversions.lat_to_y_spherical(-3.797864)
    -423086.2213610324

    References
    ----------
    https://epsg.io/transform

    """

    return 6378137 * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2.0))


def x_to_lon_spherical(x: float) -> float:
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
    >>> from pymove import conversions
    >>> conversions.x_to_lon_spherical(-4285978.17)
    -38.501597

    References
    ----------
    https://epsg.io/transform

    """

    return np.degrees(x / 6378137.0)


def y_to_lat_spherical(y: float) -> float:
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
    >>> from pymove import conversions
    >>> conversions.y2_lat_spherical(-423086.22)
    -3.797864

    References
    ----------
    https://epsg.io/transform

    """

    return np.degrees(np.arctan(np.sinh(y / 6378137.0)))


def geometry_points_to_lat_and_lon(
    move_data: DataFrame,
    geometry_label: Optional[Text] = GEOMETRY,
    drop_geometry: Optional[bool] = True,
    inplace: Optional[bool] = True
) -> DataFrame:
    """
    Converts the geometry column to latitude and longitude
    columns (named 'lat' and 'lon'), removing geometries
    that are not of the Point type.

    Parameters
    ----------
    move_data : DataFrame
        Input trajectory data.
    geometry: str, optional
        Represents column name of the geometry column, by default GEOMETRY
    drop_geometry: bool, optional
        Option to drop the geometry column, by default True
    inplace: bool, optional
        Whether the operation will be done in the original dataframe, by default True

    Returns
    -------
    DataFrame
        A new dataframe with the converted feature or None

    """

    if not inplace:
        move_data = move_data[:]

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
    latitude: Optional[Text] = LATITUDE,
    longitude: Optional[Text] = LONGITUDE
) -> DataFrame:
    """
    Converts latitude and longitude format from
    decimal degrees to decimal format.

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
    move_data: Union['PandasMoveDataFrame', 'DaskMoveDataFrame'],
    label_speed: Optional[Text] = SPEED_TO_PREV,
    new_label: Optional[Text] = None,
    inplace: Optional[bool] = True,
) -> Optional[Union['PandasMoveDataFrame', 'DaskMoveDataFrame']]:
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
        Whether the operation will be done in the original dataframe, by default True

    Returns
    -------
    DataFrame
        A new dataframe with the converted feature or None

    """

    if not inplace:
        move_data = move_data[:]
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
    move_data: Union['PandasMoveDataFrame', 'DaskMoveDataFrame'],
    label_speed: Optional[Text] = SPEED_TO_PREV,
    new_label: Optional[Text] = None,
    inplace: Optional[Text] = True,
) -> Optional[Union['PandasMoveDataFrame', 'DaskMoveDataFrame']]:
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
        Whether the operation will be done in the original dataframe, by default True

    Returns
    -------
    DataFrame
        A new dataframe with the converted feature or None

    """

    if not inplace:
        move_data = move_data[:]
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
    move_data: Union['PandasMoveDataFrame', 'DaskMoveDataFrame'],
    label_distance: Optional[Text] = DIST_TO_PREV,
    new_label: Optional[Text] = None,
    inplace: Optional[Text] = True,
) -> Optional[Union['PandasMoveDataFrame', 'DaskMoveDataFrame']]:
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
        Whether the operation will be done in the original dataframe, by default True

    Returns
    -------
    DataFrame
        A new dataframe with the converted feature or None


    """

    if not inplace:
        move_data = move_data[:]
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
    move_data: Union['PandasMoveDataFrame', 'DaskMoveDataFrame'],
    label_distance: Optional[Text] = DIST_TO_PREV,
    new_label: Optional[Text] = None,
    inplace: Optional[Text] = True,
) -> Optional[Union['PandasMoveDataFrame', 'DaskMoveDataFrame']]:
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
        Whether the operation will be done in the original dataframe, by default True

    Returns
    -------
    DataFrame
        A new dataframe with the converted feature or None

    """

    if not inplace:
        move_data = move_data[:]
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
    move_data: Union['PandasMoveDataFrame', 'DaskMoveDataFrame'],
    label_time: Optional[Text] = TIME_TO_PREV,
    new_label: Optional[Text] = None,
    inplace: Optional[Text] = True,
) -> Optional[Union['PandasMoveDataFrame', 'DaskMoveDataFrame']]:
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
        Whether the operation will be done in the original dataframe, by default True

    Returns
    -------
    DataFrame
        A new dataframe with the converted feature or None

    """

    if not inplace:
        move_data = move_data[:]
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
    move_data: Union['PandasMoveDataFrame', 'DaskMoveDataFrame'],
    label_time: Optional[Text] = TIME_TO_PREV,
    new_label: Optional[Text] = None,
    inplace: Optional[Text] = True,
) -> Optional[Union['PandasMoveDataFrame', 'DaskMoveDataFrame']]:
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
        Whether the operation will be done in the original dataframe, by default True

    Returns
    -------
    DataFrame
        A new dataframe with the converted feature or None

    """

    if not inplace:
        move_data = move_data[:]
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
    move_data: Union['PandasMoveDataFrame', 'DaskMoveDataFrame'],
    label_time: Optional[Text] = TIME_TO_PREV,
    new_label: Optional[Text] = None,
    inplace: Optional[Text] = True,
) -> Optional[Union['PandasMoveDataFrame', 'DaskMoveDataFrame']]:
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
        Whether the operation will be done in the original dataframe, by default True

    Returns
    -------
    DataFrame
        A new dataframe with the converted feature or None

    """

    if not inplace:
        move_data = move_data[:]
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
    move_data: Union['PandasMoveDataFrame', 'DaskMoveDataFrame'],
    label_time: Optional[Text] = TIME_TO_PREV,
    new_label: Optional[Text] = None,
    inplace: Optional[Text] = True,
) -> Optional[Union['PandasMoveDataFrame', 'DaskMoveDataFrame']]:
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
        Whether the operation will be done in the original dataframe, by default True

    Returns
    -------
    DataFrame
        A new dataframe with the converted feature or None

    """

    if not inplace:
        move_data = move_data[:]
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
    move_data: Union['PandasMoveDataFrame', 'DaskMoveDataFrame'],
    label_time: Optional[Text] = TIME_TO_PREV,
    new_label: Optional[Text] = None,
    inplace: Optional[Text] = True,
) -> Optional[Union['PandasMoveDataFrame', 'DaskMoveDataFrame']]:
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
        Whether the operation will be done in the original dataframe, by default True

    Returns
    -------
    DataFrame
        A new dataframe with the converted feature or None

    """

    if not inplace:
        move_data = move_data[:]
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
    move_data: Union['PandasMoveDataFrame', 'DaskMoveDataFrame'],
    label_time: Optional[Text] = TIME_TO_PREV,
    new_label: Optional[Text] = None,
    inplace: Optional[Text] = True,
) -> Optional[Union['PandasMoveDataFrame', 'DaskMoveDataFrame']]:
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
        Whether the operation will be done in the original dataframe, by default True

    Returns
    -------
    DataFrame
        A new dataframe with the converted feature or None

    """

    if not inplace:
        move_data = move_data[:]
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
