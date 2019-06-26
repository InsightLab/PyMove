from __future__ import division
import datetime
import time
from pandas._libs.tslibs.timestamps import Timestamp

def timestamp_to_millis(timestamp):
    """
    Converts a local datetime to a POSIX timestamp in milliseconds (like in Java).
    e.g. '2015-12-12 08:00:00.123000' -> 1449907200123 (UTC)
    Java: Sat Dec 12 08:00:00 BRT 2015        -> 1449918000123
    """
    return Timestamp(timestamp).value // 1000000


def millis_to_timestamp(milliseconds):
    """
    Converts a POSIX timestamp in milliseconds (like in Java) to a local datetime.
    e.g.  1449907200123 -> Timestamp('2015-12-12 08:00:00.123000')
    Java: 1449918000123 -> Sat Dec 12 08:00:00 BRT 2015
    """
    return Timestamp(milliseconds, unit='ms')


def date_str(date1):
    return date1.strftime('%Y-%m-%d')


def time_str(date1):
    return date1.strftime('%H:%M:%S')


def str_to_dtime(dt_str):
    if len(dt_str) == 10:
        return datetime.datatime.strptime(dt_str, '%Y-%m-%d')
    else:
        return datetime.datatime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')


def dtime_str(dt1):
    return dt1.strftime('%Y-%m-%d %H:%M:%S')


def now_str():
    return dtime_str(datetime.datetime.now())


def dtime_to_min(dt1):
    """
    Converts a datetime to an int representation in minutes. 
    To do the reverse use: min_to_dtime.
    e.g. in:datetime.datetime(2014, 1, 1, 20, 56) -> out:23143496
    """
    # get an integer time slot from a datetime
    # e.g. in:datetime.datetime(2014, 1, 1, 20, 55) -> out:23143495
    # e.g. in:datetime.datetime(2014, 1, 1, 20, 56) -> out:23143496
    return int((dt1 - datetime.datetime.utcfromtimestamp(0)).total_seconds() / 60)


def min_to_dtime(min1):
    """
    Converts an int representation in minutes to a datetime. 
    To do the reverse use: dtime_to_min.
    e.g. in:23143496 -> out:datetime.datetime(2014, 1, 1, 20, 56)
    """
    # get a datetime from an integer time slot
    # e.g. in:23143495 -> out:datetime.datetime(2014, 1, 1, 20, 55)
    # e.g. in:23143496 -> out:datetime.datetime(2014, 1, 1, 20, 56)
    #return datetime.timedelta(minutes=min1) + datetime.datetime.utcfromtimestamp(0)
    # utcfromtimestamp (below) is much faster than the line above
    return datetime.datetime.utcfromtimestamp(min1 * 60)

def slot_of_day_to_time(slot_of_day1, time_window_duration=5):
    min1 = slot_of_day1 * time_window_duration
    return datetime.time(min1 // 60, min1 % 60)


def slot_of_day(dt1, time_window_duration=5):
    return (dt1.hour * 60 + dt1.minute) // time_window_duration


def datetime_slot(dt1, time_window_duration=5):
    minute = (dt1.minute // time_window_duration) * time_window_duration
    return datetime.datetime(dt1.year, dt1.month, dt1.day, dt1.hour, minute)


def dtime_str_to_min_slot(dt_str, time_window_duration=5):
    """
    Converts a datetime string to an int minute time slot (approximated to the time slot).
    Same as datetime_str_to_min_slot, but another implementation.
    To do almost the reverse (consider time slot approximation) use: min_to_dtime.
    e.g. in:'2014-01-01 20:56:00' -> out:23143495
    """
    dt = dtime(dt_str)
    dt_slot = datetime_slot(dt, time_window_duration)
    return dtime_to_min(dt_slot)


def datetime_str_to_min_slot(dt_str, time_window_duration=5):
    """
    Converts a datetime string to an int minute time slot (approximated to the time slot).
    Same as dtime_str_to_min_slot, but another implementation.
    To do almost the reverse (consider time slot approximation) use: min_to_dtime.
    e.g. in:'2014-01-01 20:56:00' -> out:23143495
    """
    dt = datetime.datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
    dt_min = (dt - datetime.datetime.utcfromtimestamp(0)).total_seconds() / 60
    return int(dt_min // time_window_duration * time_window_duration)
    

def day_of_week(dt1):
    # Monday == 0...Sunday == 6
    return dt1.weekday()


def elapsed_time_dt(start_time):
    return diff_time(start_time, datetime.datetime.now())


def diff_time(start_time, end_time):
    return int((end_time - start_time).total_seconds() * 1000)


def working_day(dt, holidays):
    result = True

    if type(dt) == str:
        dt = dtime(dt)

    if type(dt) == datetime.datetime:
        dt = datetime.date(dt.year, dt.month, dt.day)

    if dt in holidays:
        result = False
    else:
        dow = day_of_week(dt)
        # 5 == saturday, 6 == sunday
        if dow == 5 or dow == 6:
            result = False

    return result


def deltatime_str(deltatime_seconds):
    """
    input: time in seconds. e.g. 1082.7180936336517 -> output: '00:16:48.271'
    output example if more than 24 hours: 25:33:57.123
    https://stackoverflow.com/questions/3620943/measuring-elapsed-time-with-the-time-module 
    """
    time_int = int(deltatime_seconds)
    time_dec = int((deltatime_seconds - time_int) * 1000)
    time_str = '{:02d}:{:02d}:{:02d}.{:03d}'.format(time_int // 3600, time_int % 3600 // 60, time_int % 60, time_dec)
    return time_str

