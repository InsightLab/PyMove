import numpy as np

from pymove.utils import constants

def list_to_str(input_list, delimiter=','):
    """Concatenates the elements of the array, joining them by the separator especified by the parameter "delimiter"

    Parameters
    ----------
    input_list : array
        The elements to be joined

    delimiter : String, optional(',' by default)
        The separator used between elements

    Returns
    -------
        String
            Returns a string, wich is the concatenation of the elements of the array, separeted by the delimiter.
    """
    return delimiter.join([x if type(x) == str else repr(x) for x in input_list])  # list comprehension


def list_to_csv_str(input_list):
    """Concatenates the elements of the array, joining them by ",".

    Parameters
    ----------
    input_list : array
        The elements to be joined

    Returns
    -------
        String
            Returns a string, wich is the concatenation of the elements of the array, separeted by ",".
    """
    return list_to_str(input_list)  # list comprehension


def list_to_svm_line(original_list):
    list_size = len(original_list)
    svm_line = '%s ' % original_list[0]
    for i in range(1, list_size):
        # svm_line += '{}:{} '.format(i, repr(original_list[i]))
        svm_line += '{}:{} '.format(i, original_list[i])
    return svm_line.rstrip()


def lon2XSpherical(lon):
    """
    Convert longitude to to X EPSG:3857 WGS 84 / Pseudo-Mercator

    Parameters
    ----------
    lon : float
        Represents longitude.

    Returns
    -------
    xspherical : float
        X offset from your original position in meters.

    Examples
    --------
    >>> from pymove.utils.transformations import lon2XSpherical
    >>> lon2XSpherical(-38.501597 )
    -4285978.17

    References
    ----------
    https://epsg.io/transform

    """
    xspherical = 6378137 * np.radians(lon)
    return xspherical


def lat2YSpherical(lat):
    """
    Convert latitude to Y EPSG:3857 WGS 84 / Pseudo-Mercator

    Parameters
    ----------
    lat : float
        Represents latitude.

    Returns
    -------
    yspherical : float
        Y offset from your original position in meters.

    Examples
    --------
    >>> from pymove.utils.transformations import lat2YSpherical
    >>> lat2YSpherical(-3.797864)
    -423086.2213610324

    References
    ----------
    https://epsg.io/transform

    """
    yspherical = 6378137 * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2.0))
    return yspherical


def x2LonSpherical(x):
    """
    Convert X EPSG:3857 WGS 84 / Pseudo-Mercator to longitude.

    Parameters
    ----------
    x : float
        X offset from your original position in meters.

    Returns
    -------
    lon : float
        Represents longitude.

    Examples
    --------
    >>> from pymove.utils.transformations import x2LonSpherical
    >>> x2LonSpherical(-4285978.17)
    -38.501597

    References
    ----------
    https://epsg.io/transform

    """
    lon = np.degrees(x / 6378137.0)
    return lon


def y2LatSpherical(y):
    """
    Convert Y EPSG:3857 WGS 84 / Pseudo-Mercator to latitude.

    Parameters
    ----------
    y : float
        Y offset from your original position in meters.

    Returns
    -------
    lat : float
        Represents latitude.

    Examples
    --------
    >>> from pymove.utils.transformations import y2LatSpherical
    >>> y2LatSpherical(-423086.22)
    -3.797864

    References
    ----------
    https://epsg.io/transform

    """
    lat = np.degrees(np.arctan(np.sinh(y / 6378137.0)))
    return lat


""" transform speed """
def ms_to_kmh(df_, label_speed = constants.SPEED_TO_PREV, new_label = None):
    try:
        if label_speed not in df_:
            df_.generate_dist_time_speed_features()
        df_[label_speed] = df_[label_speed].transform(lambda row: row*3.6)
        if new_label is not None:
            df_.rename(columns = {label_speed: new_label}, inplace=True)
    except Exception as e:
        raise e



def kmh_to_ms(df_, label_speed = constants.SPEED_TO_PREV, new_label = None):
    try:
        if label_speed not in df_:
            df_.generate_dist_time_speed_features()
        df_[label_speed] = df_[label_speed].transform(lambda row: row/3.6)
        if new_label is not None:
            df_.rename(columns = {label_speed: new_label}, inplace=True)
    except Exception as e:
        raise e


""" transform distances """
def meters_to_kilometers(df_, label_distance = constants.DIST_TO_PREV, new_label=None):
    try:
        if label_distance not in df_:
            df_.generate_dist_time_speed_features()
        df_[label_distance] = df_[label_distance].transform(lambda row: row/1000)
        if new_label is not None:
            df_.rename(columns = {label_distance: new_label}, inplace=True)
    except Exception as e:
        raise e


def kilometers_to_meters(df_, label_distance = constants.DIST_TO_PREV, new_label=None):
    try:
        if label_distance not in df_:
            df_.generate_dist_time_speed_features()
        df_[label_distance] = df_[label_distance].transform(lambda row: row*1000)
        if new_label is not None:
            df_.rename(columns = {label_distance: new_label}, inplace=True)
    except Exception as e:
        raise e


""" transform time """
def seconds_to_minutes(df_, label_time = constants.TIME_TO_PREV, new_label=None):
    try:
        if label_time not in df_:
            df_.generate_dist_time_speed_features()
        df_[label_time] = df_[label_time].transform(lambda row: row/60.0)
        if new_label is not None:
            df_.rename(columns = {label_time: new_label}, inplace=True)
    except Exception as e:
        raise e


def minute_to_seconds(df_, label_time = constants.TIME_TO_PREV, new_label=None):
    try:
        if label_time not in df_:
            df_.generate_dist_time_speed_features()
        df_[label_time] = df_[label_time].apply(lambda row: row*60.0)
        if new_label is not None:
            df_.rename(columns = {label_time: new_label}, inplace=True)
    except Exception as e:
        raise e


def minute_to_hours(df_, label_time = constants.TIME_TO_PREV, new_label=None):
    """Convertes times features from minutes to hours.

    Parameters
    ----------
    df : dataframe
        The input trajectory data

    label_time : String, optional("dic_features_label['time_to_prev']" by default)
        Indicates the label of the column that contains the time data to be converted.

    new_label : String, optional(None by default)
        The new label of the converted column, if set to none, the original label will be kept
    """
    try:
        if label_time not in df_:
            df_.generate_dist_time_speed_features()
        df_[label_time] = df_[label_time].apply(lambda row: row/60.0)
        if new_label is not None:
            df_.rename(columns = {label_time: new_label}, inplace=True)
    except Exception as e:
        raise e


def hours_to_minute(df_, label_time = constants.TIME_TO_PREV, new_label=None):
    """Convertes time features from hours to minutes.

    Parameters
    ----------
    df : dataframe
        The input trajectory data

    label_time : String, optional("dic_features_label['time_to_prev']" by default)
        Indicates the label of the column that contains the time data to be converted.

    new_label : String, optional(None by default)
        The new label of the converted column, if set to none, the original label will be kept
    """
    try:
        if label_time not in df_:
            df_.generate_dist_time_speed_features()
        df_[label_time] = df_[label_time].apply(lambda row: row*60.0)
        if new_label is not None:
            df_.rename(columns = {label_time: new_label}, inplace=True)
    except Exception as e:
        raise e


def seconds_to_hours(df_, label_time = constants.TIME_TO_PREV, new_label=None):
    """Convertes time features from seconds to hours.

    Parameters
    ----------
    df : dataframe
        The input trajectory data

    label_time : String, optional("dic_features_label['time_to_prev']" by default)
        Indicates the label of the column that contains the time data to be converted.

    new_label : String, optional(None by default)
        The new label of the converted column, if set to none, the original label will be kept
    """
    try:
        if label_time not in df_:
            df_.generate_dist_time_speed_features()
        df_[label_time] = df_[label_time].apply(lambda row: row/3600.0)
        if new_label is not None:
            df_.rename(columns = {label_time: new_label}, inplace=True)
    except Exception as e:
        raise e


def hours_to_seconds(df_, label_time=constants.TIME_TO_PREV, new_label=None):
    """Convertes time features from hours to seconds.

    Parameters
    ----------
    df : dataframe
        The input trajectory data

    label_time : String, optional("dic_features_label['time_to_prev']" by default)
        Indicates the label of the column that contains the time data to be converted.

    new_label : String, optional(None by default)
        The new label of the converted column, if set to none, the original label will be kept
    """
    try:
        if label_time not in df_:
            df_.generate_dist_time_speed_features()
        df_[label_time] = df_[label_time].apply(lambda row: row*3600.0)
        if new_label is not None:
            df_.rename(columns = {label_time: new_label}, inplace=True)
    except Exception as e:
        raise e

