from __future__ import division
import time
import math
import folium
import datetime
import numpy as np
import pandas as pd

from IPython.display import display
from ipywidgets import IntProgress, HTML, VBox
from pandas._libs.tslibs.timestamps import Timestamp


def deltatime_str(deltatime_seconds):
    """
    Convert time in a format appropriate of time.

    Parameters
    ----------
    deltatime_seconds : float
        Represents the dataset with contains lat, long and datetime.

    Returns
    -------
    time_str : String
        Represents time in a format hh:mm:ss:---.

    Examples
    --------
    >>> from pymove.utils.utils import deltatime_str
    >>> deltatime_str(1082.7180936336517)
    '00:18:02.718'

    Notes
    -----
    Output example if more than 24 hours: 25:33:57.123
    https://stackoverflow.com/questions/3620943/measuring-elapsed-time-with-the-time-module

    """
    time_int = int(deltatime_seconds)
    time_dec = int((deltatime_seconds - time_int) * 1000)
    time_str = '{:02d}:{:02d}:{:02d}.{:03d}'.format(time_int // 3600, time_int % 3600 // 60, time_int % 60, time_dec)
    return time_str


def timestamp_to_millis(timestamp):
    """
    Converts a local datetime to a POSIX timestamp in milliseconds (like in Java).

    Parameters
    ----------
    timestamp : String
        Represents a data.

    Returns
    -------
    millis : int
        Represents millisecond results.

    Examples
    --------
    >>> from pymove.utils.utils import timestamp_to_millis
    >>> timestamp_to_millis('2015-12-12 08:00:00.123000')
    1449907200123 (UTC)

    """
    millis = Timestamp(timestamp).value // 1000000
    return millis


def millis_to_timestamp(milliseconds):
    """
    Converts milliseconds to timestamp.

    Parameters
    ----------
    milliseconds : int
        Represents millisecond.

    Returns
    -------
    timestamp : pandas._libs.tslibs.timestamps.Timestamp
        Represents the date corresponding.

    Examples
    --------
    >>> from pymove.utils.utils import millis_to_timestamp
    >>> millis_to_timestamp(1449907200123)
    '2015-12-12 08:00:00.123000'

    """
    timestamp = Timestamp(milliseconds, unit='ms')
    return timestamp


def time_to_str(time):
    """
    Get time, in string's format, from timestamp.

    Parameters
    ----------
    time : pandas._libs.tslibs.timestamps.Timestamp
        Represents a time.

    Returns
    -------
    timestr : String
        Represents the time in string's format.

    Examples
    --------
    >>> from pymove.utils.utils import time_to_str
    >>> time_to_str('2015-12-12 08:00:00.123000')
    '08:00:00'

    """
    timestr = time.strftime('%H:%M:%S')
    return timestr


def str_to_time(dt_str):
    """
    Converts a time in string's format '%H:%M:%S' to datetime's format.

    Parameters
    ----------
    dt_str : String
        Represents a time in string's format.

    Returns
    -------
    datetime_time : datetime.datetime
        Represents a time in datetime's format.

    Examples
    --------
    >>> from pymove.utils.utils import str_to_time
    >>> str_to_time('08:00:00')
    datetime.datetime(1900, 1, 1, 8, 0)

    """

    datetime_time = datetime.datetime.strptime(dt_str, '%H:%M:%S')
    return datetime_time


def elapsed_time_dt(start_time):
    """Computes the elapsed time from a specific start time to the moment the function is called.

    Parameters
    ----------
    start_time : Datetime
        Specifies the start time of the time range to be computed.

    Returns
    -------
        time_dif : Integer
            Represents the time elapsed from the start time to the current time (when the function was called).

    """
    time_dif = diff_time(start_time, datetime.datetime.now())
    return time_dif


def diff_time(start_time, end_time):
    """Computes the elapsed time from the start time to the end time specifed by the user.

    Parameters
    ----------
    start_time : Datetime
        Specifies the start time of the time range to be computed.

    end_time : Datetime
        Specifies the start time of the time range to be computed.

    Returns
    -------
        time_dif : Integer
            Represents the time elapsed from the start time to the current time (when the function was called).

    """

    time_dif = int((end_time - start_time).total_seconds() * 1000)
    return time_dif
