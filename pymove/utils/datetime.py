import datetime

import holidays
from pandas._libs.tslibs.timestamps import Timestamp


def date_to_str(date):
    """
    Get date, in string'srs format, from timestamp.

    Parameters
    ----------
    date : pandas._libs.tslibs.timestamps.Timestamp
        Represents a date.

    Returns
    -------
    str
        Represents the date in string"srs format.

    """

    return date.strftime('%Y-%m-%d')


def str_to_datetime(dt_str):
    """
    Converts a datetime in string"srs format "%Y-%m-%d" or "%Y-%m-%d %H:%M:%S" to
    datetime"srs format.

    Parameters
    ----------
    dt_str : String
        Represents a datetime in string"srs format.

    Returns
    -------
    datetime.datetime
        Represents a datetime in datetime"srs format.

    """

    if len(dt_str) == 10:
        return datetime.datetime.strptime(dt_str, '%Y-%m-%d')
    else:
        return datetime.datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')


def to_str(data):
    """
    Converts a date in datetime'srs format to string"srs format.

    Parameters
    ----------
    data : datetime.datetime
        Represents a datetime in datetime"srs format.

    Returns
    -------
    str
        Represents a datetime in string"srs format "%Y-%m-%d %H:%M:%S".

    """

    return data.strftime('%Y-%m-%d %H:%M:%S')


def to_min(dt):
    """
    Converts a datetime to an int representation in minutes. To do the reverse
    use: min_to_datetime.

    Parameters
    ----------
    dt : datetime.datetime
        Represents a datetime in datetime"srs format.

    Returns
    -------
    int
        Represents minutes from.

    """

    # get an integer time slot from a datetime
    return int(
        (dt - dt.utcfromtimestamp(0)).total_seconds() / 60
    )


def min_to_datetime(min_):
    """
    Converts an int representation in minutes to a datetime. To do the reverse
    use: datetime_to_min.

    Parameters
    ----------
    min_ : int
        Represents minutes.

    Returns
    -------
    datetime.datetime
        Represents minutes in datetime"srs format.

    """

    return datetime.datetime.utcfromtimestamp(min_ * 60)


# TODO: ve o que sao os parametros e tipo dos param
# def slot_of_day_to_time(slot_of_day1, time_window_duration=5):
#     """Converts a slot of day to a time (datetime)
#
#     Parameters
#     ----------
#     slot_of_day1 :
#
#     time_window_duration: Integer, optional(5 by default)
#
#     Returns
#     -------
#     """
#     min1 = slot_of_day1 * time_window_duration
#     return datetime.time(min1 // 60, min1 % 60)
#
#
# #TODO: vê o que são os parametros e tipo dos param
# def slot_of_day(dt1, time_window_duration=5):
#     """Converts
#
#     Parameters
#     ----------
#     slot_of_day1 :
#
#     time_window_duration: Integer, optional(5 by default)
#
#     Returns
#     -------
#     """
#     return (dt1.hour * 60 + dt1.minute) // time_window_duration
#
#
# #TODO: vê o que são os parametros e tipo dos param
# def slot(dt1, time_window_duration=5):
#     """Converts
#
#     Parameters
#     ----------
#     slot_of_day1 :
#
#     time_window_duration: Integer, optional(5 by default)
#
#     Returns
#     -------
#     """
#     minute = (dt1.minute // time_window_duration) * time_window_duration
#     return datetime.datetime(dt1.year, dt1.month, dt1.day, dt1.hour, minute)
#
#
# # TODO: Finalizar
# def str_to_min_slot(dt_str, time_window_duration=5):
#     """Converts a datetime string to an int minute time slot
#     (approximated to the time slot).
#     Same as datetime_str_to_min_slot, but another implementation.
#     To do almost the reverse (consider time slot approximation)
#     use: min_to_datetime.
#
#     Parameters
#     ----------
#     dt_str : datetime.datetime
#         Represents a datetime in datetime"srs format.
#     time_window_duration: int
#
#     Returns
#     -------
#     dt_slot : int
#         Represents minutes from.
#
#     Examples
#     --------
#     >>> from pymove import datetime
#     >>> datetime.str_to_min_slot("2014-01-01 20:56:00)
#     23143495
#     """
#     dt = to_str(dt_str)
#     dt_slot = slot(dt, time_window_duration)
#     return dt_slot


def to_day_of_week_int(date):
    """
    Get day of week of a date. Monday == 0...Sunday == 6.

    Parameters
    ----------
    date : datetime.datetime
        Represents a datetime in datetime"srs format.

    Returns
    -------
    int
        Represents day of week.

    """

    return date.weekday()


def working_day(dt, country='BR', state=None):
    """
    Indices if a day specified by the user is a working day.

    Parameters
    ----------
    dt : str or datetime
        Specifies the day the user wants to know if it is a business day.
    country : String
        Indicates country to check for vacation days.
    state: String
        Indicates state to check for vacation days.

    Returns
    -------
    boolean
        if true, means that the day informed by the user is a working day.
        if false, means that the day is not a working day.

    References
    ----------
    Countries and States names available in https://pypi.org/project/holidays/

    """

    result = True

    if isinstance(dt, str):
        dt = str_to_datetime(dt)

    if isinstance(dt, datetime.datetime):
        dt = datetime.date(dt.year, dt.month, dt.day)

    if dt in holidays.CountryHoliday(country=country, prov=None, state=state):
        result = False
    else:
        dow = to_day_of_week_int(dt)
        # 5 == Saturday, 6 == Sunday
        if dow == 5 or dow == 6:
            result = False

    return result


def now_str():
    """
    Get datetime of now.

    Parameters
    ----------

    Returns
    -------
    str
        Represents a data.

    Examples
    --------
    >>> from pymove import datetime
    >>> datetime.now_str()
    "2019-09-02 13:54:16"

    """

    return to_str(datetime.datetime.now())


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
    >>> from pymove import datetime
    >>> datetime.deltatime_str(1082.7180936336517)
    "18m:02.718s"

    Notes
    -----
    Output example if more than 24 hours: 25:33:57.123
    https://stackoverflow.com/questions/3620943/measuring-elapsed-time-with-the-time-module

    """

    hours, rem = divmod(deltatime_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return '{:0>2}h:{:0>2}m:{:05.2f}s'.format(int(hours), int(minutes), seconds)
    elif minutes:
        return '{:0>2}m:{:05.2f}s'.format(int(minutes), seconds)
    else:
        return '{:05.2f}s'.format(seconds)


def timestamp_to_millis(timestamp):
    """
    Converts a local datetime to a POSIX timestamp in milliseconds (like in
    Java).

    Parameters
    ----------
    timestamp : String
        Represents a data.

    Returns
    -------
    int
        Represents millisecond results.

    Examples
    --------
    >>> from pymove.utils.utils import datetime
    >>> datetime.timestamp_to_millis("2015-12-12 08:00:00.123000")
    1449907200123 (UTC)

    """

    return Timestamp(timestamp).value // 1000000


def millis_to_timestamp(milliseconds):
    """
    Converts milliseconds to timestamp.

    Parameters
    ----------
    milliseconds : int
        Represents millisecond.

    Returns
    -------
    pandas._libs.tslibs.timestamps.Timestamp
        Represents the date corresponding.

    Examples
    --------
    >>> from pymove.utils.utils import datetime
    >>> datetime.millis_to_timestamp(1449907200123)
    "2015-12-12 08:00:00.123000"

    """

    return Timestamp(milliseconds, unit='ms')


def time_to_str(time):
    """
    Get time, in string'srs format, from timestamp.

    Parameters
    ----------
    time : pandas._libs.tslibs.timestamps.Timestamp
        Represents a time.

    Returns
    -------
    str
        Represents the time in string"srs format.

    Examples
    --------
    >>> from pymove.utils.utils import datetime
    >>> datetime.time_to_str("2015-12-12 08:00:00.123000")
    "08:00:00"

    """

    return time.strftime('%H:%M:%S')


def str_to_time(dt_str):
    """
    Converts a time in string'srs format "%H:%M:%S" to datetime'srs format.

    Parameters
    ----------
    dt_str : String
        Represents a time in string"srs format.

    Returns
    -------
    datetime.datetime
        Represents a time in datetime"srs format.

    Examples
    --------
    >>> from pymove.utils.utils import datetime
    >>> datetime.str_to_time("08:00:00")
    datetime.datetime(1900, 1, 1, 8, 0)

    """

    return datetime.datetime.strptime(dt_str, '%H:%M:%S')


def elapsed_time_dt(start_time):
    """
    Computes the elapsed time from a specific start time to the moment the
    function is called.

    Parameters
    ----------
    start_time : Datetime
        Specifies the start time of the time range to be computed.

    Returns
    -------
    int
        Represents the time elapsed from the start time to the current time
        (when the function was called).

    """

    return diff_time(start_time, datetime.datetime.now())


def diff_time(start_time, end_time):
    """
    Computes the elapsed time from the start time to the end time specifed by
    the user.

    Parameters
    ----------
    start_time : Datetime
        Specifies the start time of the time range to be computed.
    end_time : Datetime
        Specifies the start time of the time range to be computed.

    Returns
    -------
    int
        Represents the time elapsed from the start time to the current time
        (when the function was called).

    """

    return int((end_time - start_time).total_seconds() * 1000)
