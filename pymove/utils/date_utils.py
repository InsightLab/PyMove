import datetime


def date_to_str(date):
    """
    Get date, in string's format, from timestamp.

    Parameters
    ----------
    date : pandas._libs.tslibs.timestamps.Timestamp
        Represents a date.

    Returns
    -------
    datestr : String
        Represents the date in string's format.

    Examples
    --------
    >>> from pymove.utils.utils import date_to_str
    >>> date_to_str('2015-12-12 08:00:00.123000')
    '2015-12-12'

    """
    datestr = date.strftime('%Y-%m-%d')
    return datestr


def str_to_datetime(dt_str):
    """
    Converts a datetime in string's format '%Y-%m-%d' or '%Y-%m-%d %H:%M:%S' to datetime's format.

    Parameters
    ----------
    dt_str : String
        Represents a datetime in string's format.

    Returns
    -------
    datetime : datetime.datetime
        Represents a datetime in datetime's format.

    Examples
    --------
    >>> from pymove.utils.utils import str_to_datetime
    >>> str_to_datetime('2015-12-12')
    datetime.datetime(2015, 12, 12, 0, 0)

    """
    if len(dt_str) == 10:
        return datetime.datetime.strptime(dt_str, '%Y-%m-%d')
    else:
        return datetime.datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')


def datetime_to_str(data):
    """
    Converts a date in datetime's format to string's format.

    Parameters
    ----------
    data : datetime.datetime
        Represents a datetime in datetime's format.

    Returns
    -------
    datetime_str : String
        Represents a datetime in string's format '%Y-%m-%d %H:%M:%S'.

    Examples
    --------
    >>> from pymove.utils.utils import datetime_to_str
    >>> datetime_to_str(datetime.datetime(2019, 9, 3, 11, 11, 49, 520512))
    '2019-09-03 11:11:49'

    """
    datetime_str = data.strftime('%Y-%m-%d %H:%M:%S')
    return datetime_str


def datetime_to_min(datetime):
    """
    Converts a datetime to an int representation in minutes.
    To do the reverse use: min_to_datetime.

    Parameters
    ----------
    datetime : datetime.datetime
        Represents a datetime in datetime's format.

    Returns
    -------
    minutes : int
        Represents minutes from.

    Examples
    --------
    >>> from pymove.utils.utils import datetime_to_min
    >>> datetime_to_min(datetime.datetime(2014, 1, 1, 20, 56))
    23143496

    """
    # get an integer time slot from a datetime
    minutes = int((datetime - datetime.utcfromtimestamp(0)).total_seconds() / 60)
    return minutes


# TODO ajeitar isso aqui
def min_to_datetime(min):
    """
    Converts an int representation in minutes to a datetime.
    To do the reverse use: datetime_to_min.

    Parameters
    ----------
    min : int
        Represents minutes.

    Returns
    -------
    min_datetime : datetime.datetime
        Represents minutes in datetime's format.

    Examples
    --------
    >>> from pymove.utils.utils import min_to_datetime
    >>> min_to_datetime(23143496)
    datetime.datetime(2014, 1, 1, 20, 56)

    """
    # get a datetime from an integer time slot
    # utcfromtimestamp (below) is much faster than the line above
    min_datetime = datetime.datetime.utcfromtimestamp(min * 60)
    return min_datetime


#TODO: vê o que são os parametros e tipo dos param
def slot_of_day_to_time(slot_of_day1, time_window_duration=5):
    min1 = slot_of_day1 * time_window_duration
    return datetime.time(min1 // 60, min1 % 60)

#TODO: vê o que são os parametros e tipo dos param
def slot_of_day(dt1, time_window_duration=5):
    return (dt1.hour * 60 + dt1.minute) // time_window_duration

#TODO: vê o que são os parametros e tipo dos param
def datetime_slot(dt1, time_window_duration=5):
    minute = (dt1.minute // time_window_duration) * time_window_duration
    return datetime.datetime(dt1.year, dt1.month, dt1.day, dt1.hour, minute)


# TODO: Finalizar
def datetime_str_to_min_slot(dt_str, time_window_duration=5):
    """
    Converts a datetime string to an int minute time slot (approximated to the time slot).
    Same as datetime_str_to_min_slot, but another implementation.
    To do almost the reverse (consider time slot approximation) use: min_to_datetime.

    Parameters
    ----------
    dt_str : datetime.datetime
        Represents a datetime in datetime's format.

    time_window_duration: int


    Returns
    -------
    dt_slot : int
        Represents minutes from.

    Examples
    --------
    >>> from pymove.utils.utils import datetime_str_to_min_slot
    >>> datetime_str_to_min_slot('2014-01-01 20:56:00)
    23143495

    """
    dt = datetime_to_str(dt_str)
    dt_slot = datetime_slot(dt, time_window_duration)
    return dt_slot


def date_to_day_of_week_int(date):
    """
    Get day of week of a date.
    Monday == 0...Sunday == 6

    Parameters
    ----------
    date : datetime.datetime
        Represents a datetime in datetime's format.

    Returns
    -------
    day_week : int
        Represents day of week.

    Examples
    --------
    >>> from pymove.utils.utils import date_to_day_of_week_int
    >>> date_to_day_of_week_int(datetime.datetime(2014, 1, 1, 20, 56))
    2

    """
    day_week = date.weekday()
    return day_week


def working_day(dt, holidays):
    """Indices if a day specified by the user is a working day.

    Parameters
    ----------
    dt : Datetime
        Specifies the day the user wants to know if it is a business day.

    holidays : Datetime
        Indicates the days that are vacation days and therefore not working days.

    Returns
    -------
        result : boolean
            if true, means that the day informed by the user is a working day.
            if false, means that the day is not a working day.
    """
    result = True

    if type(dt) == str:
        dt = date_to_str(dt)

    if type(dt) == datetime.datetime:
        dt = datetime.date(dt.year, dt.month, dt.day)

    if dt in holidays:
        result = False
    else:
        dow = date_to_day_of_week_int(dt)
        # 5 == saturday, 6 == sunday
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
    date_time : String
        Represents a data.

    Examples
    --------
    >>> from pymove.utils.utils import now_str
    >>> now_str()
    '2019-09-02 13:54:16'

    """
    date_time = datetime_to_str(datetime.datetime.now())
    return date_time
