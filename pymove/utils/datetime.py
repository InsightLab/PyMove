"""
Datetime operations.

date_to_str,
str_to_datetime,
datetime_to_str,
datetime_to_min,
min_to_datetime,
to_day_of_week_int,
working_day,
now_str,
deltatime_str,
timestamp_to_millis,
millis_to_timestamp,
time_to_str,
str_to_time,
elapsed_time_dt,
diff_time,
create_time_slot_in_minute,
generate_time_statistics,
threshold_time_statistics

"""
from __future__ import annotations

from datetime import datetime

import holidays
from pandas import DataFrame, Timestamp

from pymove.utils.constants import (
    COUNT,
    DATETIME,
    LOCAL_LABEL,
    MAX,
    MEAN,
    MIN,
    PREV_LOCAL,
    STD,
    SUM,
    THRESHOLD,
    TIME_SLOT,
    TIME_TO_PREV,
)


def date_to_str(dt: datetime) -> str:
    """
    Get date, in string format, from timestamp.

    Parameters
    ----------
    dt : datetime
        This represents a date

    Returns
    -------
    str
        Represents the date in string format

    Example
    -------
    >>> from datetime import datatime
    >>> from pymove.utils.datetime import date_to_str
    >>> time_now = datetime.now()
    >>> print(time_now)
    '2021-04-29 11:01:29.909340'
    >>> print(type(time_now))
    '<class 'datetime.datetime'>'
    >>> print(date_to_str(time_now), type(time_now))
    '2021-04-29   <class 'str'>'
    """
    return dt.strftime('%Y-%m-%d')


def str_to_datetime(dt_str: str) -> datetime:
    """
    Converts a datetime in string format to datetime format.

    Parameters
    ----------
    dt_str : str
        Represents a datetime in string format, "%Y-%m-%d" or "%Y-%m-%d %H:%M:%S"

    Returns
    -------
    datetime
        Represents a datetime in datetime format

    Example
    -------
    >>> from pymove.utils.datetime import str_to_datetime
    >>> time_1 = '2020-06-29'
    >>> time_2 = '2020-06-29 12:45:59'
    >>> print(type(time_1), type(time_2))
    '<class 'str'> <class 'str'>'
    >>> print( str_to_datetime(time_1), type(str_to_datetime(time_1)))
    '2020-06-29 00:00:00 <class 'datetime.datetime'>'
    >>> print(str_to_datetime(time_2), type(str_to_datetime(time_2)))
    '2020-06-29 12:45:59 <class 'datetime.datetime'>'


    """
    if len(dt_str) == 10:
        return datetime.strptime(dt_str, '%Y-%m-%d')
    else:
        return datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')


def datetime_to_str(dt: datetime) -> str:
    """
    Converts a date in datetime format to string format.

    Parameters
    ----------
    dt : datetime
        Represents a datetime in datetime format.

    Returns
    -------
    str
        Represents a datetime in string format "%Y-%m-%d %H:%M:%S".

    Example:
    -------
    >>> from pymove.utils.datetime import datetime_to_str
    >>> from datetime import datetime
    >>> time_now = datetime.now()
    >>> print(time_now)
    '2021-04-29 14:15:29.708113'
    >>> print(type(time_now))
    '<class 'datetime.datetime'>'
    >>> print(datetime_to_str(time_now), type(datetime_to_str(time_now)))
    '2021-04-29 14:15:29  <class 'str' >'
    """
    return dt.strftime('%Y-%m-%d %H:%M:%S')


def datetime_to_min(dt: datetime) -> int:
    """
    Converts a datetime to an int representation in minutes.

    To do the reverse use: min_to_datetime.

    Parameters
    ----------
    dt : datetime
        Represents a datetime in datetime format

    Returns
    -------
    int
        Represents minutes from

    Example
    -------
    >>> from pymove.utils.datetime import datetime_to_min
    >>> from datetime import datetime
    >>> time_now = datetime.now()
    >>> print(type(datetime_to_min(time_now)))
    '<class 'int'>'
    >>> datetime_to_min(time_now)
    '26996497'
    """
    # get an integer time slot from a datetime
    return int(
        (dt - dt.utcfromtimestamp(0)).total_seconds() / 60
    )


def min_to_datetime(minutes: int) -> datetime:
    """
    Converts an int representation in minutes to a datetime.

    To do the reverse use: datetime_to_min.

    Parameters
    ----------
    minutes : int
        This represents a value in minutes

    Returns
    -------
    datetime
        Represents minutes in datetime format

    Example
    -------
    >>> from pymove.utils.datetime import min_to_datetime
    >>> print(min_to_datetime(26996497), type(min_to_datetime(26996497)))
    '2021-04-30 13:37:00 <class 'datetime.datetime'>'
    """
    return datetime.utcfromtimestamp(minutes * 60)


def to_day_of_week_int(dt: datetime) -> int:
    """
    Get day of week of a date. Monday == 0...Sunday == 6.

    Parameters
    ----------
    dt : datetime
        Represents a datetime in datetime format.

    Returns
    -------
    int
        Represents day of week.

    Example
    -------
    >>> from pymove.utils.datetime import str_to_datetime
    >>> monday = str_to_datetime('2021-05-3 12:00:01')
    >>> friday = str_to_datetime('2021-05-7 12:00:01')
    >>> print(to_day_of_week_int(monday), type(to_day_of_week_int(monday)))
    '0 <class 'int'>'
    >>> print(to_day_of_week_int(friday), type(to_day_of_week_int(friday)))
    '4 <class 'int'>'
    """
    return dt.weekday()


def working_day(
    dt: str | datetime,
    country: str = 'BR',
    state: str | None = None
) -> bool:
    """
    Indices if a day specified by the user is a working day.

    Parameters
    ----------
    dt : str or datetime
        Specifies the day the user wants to know if it is a business day.
    country : str
        Indicates country to check for vacation days, by default 'BR'
    state: str
        Indicates state to check for vacation days, by default None

    Returns
    -------
    boolean
        if true, means that the day informed by the user is a working day.
        if false, means that the day is not a working day.

    Examples
    --------
    >>> from pymove.utils.datetime import str_to_datetime
    >>> independence_day = str_to_datetime('2021-09-7 12:00:01') # Holiday in Brazil
    >>> next_day = str_to_datetime('2021-09-8 12:00:01') # Not a Holiday in Brazil
    >>> print(working_day(independence_day, 'BR'))
    False
    >>> print(type(working_day(independence_day, 'BR')))
    <class 'bool'>
    >>> print(working_day(next_day, 'BR'))
    True
    >>> print(type(working_day(next_day, 'BR')))
    '<class 'bool'>'

    References
    ----------
    Countries and States names available in https://pypi.org/project/holidays/

    """
    result = True
    if isinstance(dt, str):
        dt = str_to_datetime(dt)

    if isinstance(dt, datetime):
        dt = datetime(dt.year, dt.month, dt.day)

    if dt in holidays.CountryHoliday(country=country, prov=None, state=state):
        result = False
    else:
        dow = to_day_of_week_int(dt)
        # 5 == Saturday, 6 == Sunday
        if dow == 5 or dow == 6:
            result = False

    return result


def now_str() -> str:
    """
    Get datetime of now.

    Returns
    -------
    str
        Represents a date

    Examples
    --------
    >>> from pymove.utils.datetime import now_str
    >>> now_str()
    '2019-09-02 13:54:16'
    """
    return datetime_to_str(datetime.now())


def deltatime_str(deltatime_seconds: float) -> str:
    """
    Convert time in a format appropriate of time.

    Parameters
    ----------
    deltatime_seconds : float
        Represents the elapsed time in seconds

    Returns
    -------
    time_str : str
        Represents time in a format hh:mm:ss

    Examples
    --------
    >>> from pymove.utils.datetime import deltatime_str
    >>> deltatime_str(1082.7180936336517)
    '18m:02.718s'

    Notes
    -----
    Output example if more than 24 hours: 25:33:57
    https://stackoverflow.com/questions/3620943/measuring-elapsed-time-with-the-time-module

    """
    hours, rem = divmod(deltatime_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f'{int(hours):0>2}h:{int(minutes):0>2}m:{seconds:05.2f}s'
    elif minutes:
        return f'{int(minutes):0>2}m:{seconds:05.2f}s'
    else:
        return f'{seconds:05.2f}s'


def timestamp_to_millis(timestamp: str) -> int:
    """
    Converts a local datetime to a POSIX timestamp in milliseconds (like in Java).

    Parameters
    ----------
    timestamp : str
        Represents a date

    Returns
    -------
    int
        Represents millisecond results

    Examples
    --------
    >>> from pymove.utils.datetime import timestamp_to_millis
    >>> timestamp_to_millis('2015-12-12 08:00:00.123000')
    1449907200123 (UTC)
    """
    return Timestamp(timestamp).value // 1000000


def millis_to_timestamp(milliseconds: float) -> Timestamp:
    """
    Converts milliseconds to timestamp.

    Parameters
    ----------
    milliseconds : int
        Represents millisecond.

    Returns
    -------
    Timestamp
        Represents the date corresponding.

    Examples
    --------
    >>> from pymove.utils.datetime import millis_to_timestamp
    >>> millis_to_timestamp(1449907200123)
    '2015-12-12 08:00:00.123000'
    """
    return Timestamp(milliseconds, unit='ms')


def time_to_str(time: Timestamp) -> str:
    """
    Get time, in string format, from timestamp.

    Parameters
    ----------
    time : Timestamp
        Represents a time

    Returns
    -------
    str
        Represents the time in string format

    Examples
    --------
    >>> from pymove.utils.datetime import time_to_str
    >>> time_to_str("2015-12-12 08:00:00.123000")
    '08:00:00'
    """
    return time.strftime('%H:%M:%S')


def str_to_time(dt_str: str) -> datetime:
    """
    Converts a time in string format "%H:%M:%S" to datetime format.

    Parameters
    ----------
    dt_str : str
        Represents a time in string format

    Returns
    -------
    datetime
        Represents a time in datetime format

    Examples
    --------
    >>> from pymove.utils.datetime import str_to_time
    >>> str_to_time("08:00:00")
    datetime(1900, 1, 1, 8, 0)
    """
    return datetime.strptime(dt_str, '%H:%M:%S')


def elapsed_time_dt(start_time: datetime) -> int:
    """
    Computes the elapsed time from a specific start time.

    Parameters
    ----------
    start_time : datetime
        Specifies the start time of the time range to be computed

    Returns
    -------
    int
        Represents the time elapsed from the start time to the current time
        (when the function was called).

    Examples
    --------
    >>> from datetime import datetime
    >>> from pymove.utils.datetime import str_to_datetime
    >>> start_time_1 = datetime(2020, 6, 29, 0, 0)
    >>> start_time_2 = str_to_datetime('2020-06-29 12:45:59')
    >>> print(elapsed_time_dt(start_time_1))
    26411808666
    >>> print(elapsed_time_dt(start_time_2))
    26365849667
    """
    return diff_time(start_time, datetime.now())


def diff_time(start_time: datetime, end_time: datetime) -> int:
    """
    Computes the elapsed time from the start time to the end time specified by the user.

    Parameters
    ----------
    start_time : datetime
        Specifies the start time of the time range to be computed
    end_time : datetime
        Specifies the start time of the time range to be computed

    Returns
    -------
    int
        Represents the time elapsed from the start time to the current time
        (when the function was called).

    Examples
    --------
    >>> from datetime import datetime
    >>> from pymove.utils.datetime import str_to_datetime
    >>> time_now = datetime.now()
    >>> start_time_1 = datetime(2020, 6, 29, 0, 0)
    >>> start_time_2 = str_to_datetime('2020-06-29 12:45:59')
    >>> print(diff_time(start_time_1, time_now))
    26411808665
    >>> print(diff_time(start_time_2, time_now))
    26365849665
    """
    return int((end_time - start_time).total_seconds() * 1000)


def create_time_slot_in_minute(
    data: DataFrame,
    slot_interval: int = 15,
    initial_slot: int = 0,
    label_datetime: str = DATETIME,
    label_time_slot: str = TIME_SLOT,
    inplace: bool = False
) -> DataFrame | None:
    """
    Partitions the time in slot windows.

    Parameters
    ----------
    data : DataFrame
        dataframe with datetime column
    slot_interval : int, optional
        size of the slot window in minutes, by default 5
    initial_slot : int, optional
        initial window time, by default 0
    label_datetime : str, optional
        name of the datetime column, by default DATETIME
    label_time_slot : str, optional
        name of the time slot column, by default TIME_SLOT
    inplace : boolean, optional
        wether the operation will be done in the original dataframe,
        by default False

    Returns
    -------
    DataFrame
        data with converted time slots or None

    Examples
    --------
    >>> from pymove.utils.datetime import create_time_slot_in_minute
    >>> from pymove import datetime
    >>> data
              lat          lon              datetime  id
    0   39.984094   116.319236   2008-10-23 05:44:05   1
    1   39.984198   116.319322   2008-10-23 05:56:06   1
    2   39.984224   116.319402   2008-10-23 05:56:11   1
    3   39.984224   116.319402   2008-10-23 06:10:15   1
    >>> datetime.create_time_slot_in_minute(data, inplace=False)
              lat          lon              datetime  id   time_slot
    0   39.984094   116.319236   2008-10-23 05:44:05   1          22
    1   39.984198   116.319322   2008-10-23 05:56:06   1          23
    2   39.984224   116.319402   2008-10-23 05:56:11   1          23
    3   39.984224   116.319402   2008-10-23 06:10:15   1          24
    """
    if data.dtypes[label_datetime] != 'datetime64[ns]':
        raise ValueError(f'{label_datetime} colum must be of type datetime')
    if not inplace:
        data = data.copy()
    minute_day = data[label_datetime].dt.hour * 60 + data[label_datetime].dt.minute
    data[label_time_slot] = minute_day // slot_interval + initial_slot
    if not inplace:
        return data


def generate_time_statistics(
    data: DataFrame,
    local_label: str = LOCAL_LABEL
):
    """
    Calculates time statistics of the pairwise local labels.

    (average, standard deviation, minimum, maximum, sum and count)
    of the pairwise local labels of a symbolic trajectory.

    Parameters
    ----------
    data : DataFrame
        The input trajectories date.
    local_label : str, optional
        The name of the feature with local id, by default LOCAL_LABEL

    Return
    ------
    DataFrame
        Statistics infomations of the pairwise local labels

    Example
    -------
    >>> from pymove.utils.datetime import generate_time_statistics
    >>> df
        local_label   prev_local   time_to_prev   id
    0         house          NaN            NaN    1
    1        market        house          720.0    1
    2        market       market            5.0    1
    3        market       market            1.0    1
    4        school       market          844.0    1
    >>> generate_time_statistics(df)
       local_label   prev_local    mean        std \
               min          max     sum      count
    0        house       market   844.0   0.000000 \
             844.0        844.0   844.0          1
    1       market        house   720.0   0.000000 \
             720.0        720.0   720.0          1
    2       market       market     3.0   2.828427 \
               1.0          5.0     6.0          2
    """
    df_statistics = data.groupby(
        [local_label, PREV_LOCAL]
    ).agg({TIME_TO_PREV: [
        MEAN, STD, MIN, MAX, SUM, COUNT
    ]})
    df_statistics.columns = df_statistics.columns.droplevel(0)
    df_statistics.fillna(0, inplace=True)
    df_statistics.reset_index(inplace=True)

    return df_statistics


def _calc_time_threshold(seg_mean: float, seg_std: float) -> float:
    """
    Auxiliary function for calculating the threshold.

    Based on the mean and standard deviation of the time transitions
    between adjacent places on discrete MoveDataFrame.

    Parameters
    ----------
    seg_mean : float
        The time mean between two local labels (segment).
    seg_std : float
        The time mean between two local labels (segment).

    Return
    ------
    float
        The threshold based on the mean and standard deviation
        of transition time for the segment.

    Examples
    --------
    >>> from pymove.utils.datetime import _calc_time_threshold
    >>> print(_calc_time_threshold(12.3, 2.1))
    14.4
    >>> print(_calc_time_threshold(1, 1.5))
    2.5
    >>> print(_calc_time_threshold(-2, 2))
    0.0
    """
    threshold = seg_std + seg_mean
    threshold = float(f'{threshold:.1f}')
    return threshold


def threshold_time_statistics(
    df_statistics: DataFrame,
    mean_coef: float = 1.0,
    std_coef: float = 1.0,
    inplace: bool = False
) -> DataFrame | None:
    """
    Calculates and creates the threshold column.

    The values are based in the time statistics dataframe for each segment.

    Parameters
    ----------
    df_statistics : DataFrame
        Time Statistics of the pairwise local labels.
    mean_coef : float
        Multiplication coefficient of the mean time for the segment, by default 1.0
    std_coef : float
        Multiplication coefficient of sdt time for the segment, by default 1.0
    inplace : boolean, optional
        wether the operation will be done in the original dataframe,
        by default False

    Return
    ------
    DataFrame
        DataFrame of time statistics with the aditional feature: threshold,
        which indicates the time limit of the trajectory segment, or None

    Example
    -------
    >>> from pymove.utils.datetime import generate_time_statistics
    >>> df
        local_label   prev_local   time_to_prev   id
    0         house          NaN            NaN    1
    1        market        house          720.0    1
    2        market       market            5.0    1
    3        market       market            1.0    1
    4        school       market          844.0    1
    >>> statistics = generate_time_statistics(df)
    >>> statistics
        local_label   prev_local    mean        std     min     max     sum   count
    0         house       market   844.0   0.000000   844.0   844.0   844.0       1
    1        market        house   720.0   0.000000   720.0   720.0   720.0       1
    2        market       market     3.0   2.828427     1.0     5.0     6.0       2
    >>> threshold_time_statistics(statistics)
        local_label   prev_local    mean         std     min \
                max          sum   count   threshold
    0         house       market   844.0    0.000000   844.0 \
              844.0        844.0       1       844.0
    1        market        house   720.0    0.000000   720.0 \
              720.0        720.0       1       720.0
    2        market       market     3.0    2.828427     1.0 \
                5.0          6.0       2         5.8
    """
    if not inplace:
        df_statistics = df_statistics.copy()
    df_statistics[THRESHOLD] = df_statistics.apply(
        lambda x: _calc_time_threshold(x[MEAN] * mean_coef, x[STD] * std_coef), axis=1
    )

    if not inplace:
        return df_statistics
