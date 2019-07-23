from __future__ import division
import numpy as np
#from scipy.ndimage.interpolation import shift
import pandas as pd
import math

#import timeutils
import datetime
import time
from pandas._libs.tslibs.timestamps import Timestamp
from ipywidgets import IntProgress, HTML, VBox
from IPython.display import display

def log_progress(sequence, every=None, size=None, name='Items'):

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = u'{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = "{name}: {index}".format(
            name=name,
            index=str(index or '?')
        )

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
    
def progress_update(size_processed, size_all, start_time, curr_perc_int, step_perc=1):
    """
    update and print current progress.
    e.g.
    curr_perc_int, _ = pu.progress_update(size_processed, size_all, start_time, curr_perc_int)
    returns: curr_perc_int_new, deltatime_str
    """
    curr_perc_new = size_processed*100.0 / size_all
    curr_perc_int_new = int(curr_perc_new)
    if curr_perc_int_new != curr_perc_int and curr_perc_int_new % step_perc == 0:
        deltatime = time.time() - start_time
        deltatime_str_ = deltatime_str(deltatime)
        est_end = deltatime / curr_perc_new * 100
        est_time_str = deltatime_str(est_end - deltatime)
        print('({}/{}) {}% in {} - estimated end in {}'.format(size_processed, size_all, curr_perc_int_new, deltatime_str_, est_time_str))
        return curr_perc_int_new, deltatime_str
    else:
        return curr_perc_int_new, None

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

def date_to_str(date1):
    return date1.strftime('%Y-%m-%d')

def time_to_str(time1):
    return time1.strftime('%H:%M:%S')

def str_to_datatime(dt_str):
    if len(dt_str) == 10:
        return datetime.datetime.strptime(dt_str, '%Y-%m-%d')
    else:
        return datetime.datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')

def str_to_time(dt_str):
    return datetime.datetime.strptime(dt_str, '%H:%M:%S')

def datetime_to_str(dt1):
    return dt1.strftime('%Y-%m-%d %H:%M:%S')

def now_str():
    return datetime_to_str(datetime.datetime.now())

def datetime_to_min(datetime):
    """
    Converts a datetime to an int representation in minutes. 
    To do the reverse use: min_to_dtime.
    e.g. in:datetime.datetime(2014, 1, 1, 20, 56) -> out:23143496
    """
    # get an integer time slot from a datetime
    # e.g. in:datetime.datetime(2014, 1, 1, 20, 55) -> out:23143495
    # e.g. in:datetime.datetime(2014, 1, 1, 20, 56) -> out:23143496
    return int((datetime - datetime.datetime.utcfromtimestamp(0)).total_seconds() / 60)

def min_to_datatime(min1):
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

def datetime_str_to_min_slot(dt_str, time_window_duration=5):
    """
    Converts a datetime string to an int minute time slot (approximated to the time slot).
    Same as datetime_str_to_min_slot, but another implementation.
    To do almost the reverse (consider time slot approximation) use: min_to_dtime.
    e.g. in:'2014-01-01 20:56:00' -> out:23143495
    """
    dt = datetime_to_str(dt_str)
    dt_slot = datetime_slot(dt, time_window_duration)
    return dt_slot

def date_to_day_of_week_int(date):
    # Monday == 0...Sunday == 6
    return date.weekday()

def elapsed_time_dt(start_time):
    return diff_time(start_time, datetime.datetime.now())

def diff_time(start_time, end_time):
    return int((end_time - start_time).total_seconds() * 1000)

def working_day(dt, holidays):
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

def std(sum_sq, size, avg):
    try:
        # squaring with * is over 3 times as fast as with **2
        # http://stackoverflow.com/questions/29046346/comparison-of-power-to-multiplication-in-python
        result = math.sqrt(sum_sq / size - avg * avg)
    except ValueError:
        e = '(size - avg^2) (size=%s, avg=%s, sum_sq=%s) should be non negative, but is %s' % \
            (size, avg, sum_sq, size - avg * avg)
        raise ValueError(e)
    return result

def avg_std(sum1, sum_sq, size):
    avg = sum1 / size
    return avg, std(sum_sq, size, avg)

def std_sample(sum_sq, size, avg):
    return std(sum_sq, size, avg) * math.sqrt(size / (size - 1))

def avg_std_sample(sum1, sum_sq, size):
    avg = sum1 / size
    return avg, std_sample(sum_sq, size, avg)

def arrays_avg(values_array, weights_array=None):
    n = len(values_array)

    if weights_array is None:
        weights_array = np.full(n, 1)
    elif len(weights_array) != n:
        raise ValueError('values_array and qt_array must have the same number of rows')

    n_row = len(values_array[0])
    result = np.full(n_row, 0)
    for i, item in enumerate(values_array):
        for j in range(n_row):
            result[j] += item[j] * weights_array[i]

    sum_qt = array_sum(weights_array)
    for i in range(n_row):
        result[i] /= sum_qt

    return result

def array_sum(values_array):
    sum1 = 0
    for item in values_array:
        sum1 += item

    return sum1

def array_stats(values_array):
    sum1 = 0
    sum_sq = 0
    n = 0
    for item in values_array:
        sum1 += item
        sum_sq += item * item
        n += 1

    return sum1, sum_sq, n

def change_df_feature_values_using_filter(df, id_, feature_name, filter_, values):
    """
    equivalent of: df.at[id_, feature_name][filter_] = values
    e.g. df.at[tid, 'time'][filter_nodes] = intp_result.astype(np.int64)
    dataframe must be indexed by id_: df.set_index(index_name, inplace=True)
    """
    values_feature = df.at[id_, feature_name]
    if filter_.shape == ():
        df.at[id_, feature_name] = values
    else:
        values_feature[filter_] = values
        df.at[id_, feature_name] = values_feature

def change_df_feature_values_using_filter_and_indexes(df, id_, feature_name, filter_, idxs, values):
    """
    equivalent of: df.at[id_, feature_name][filter_][idxs] = values
    e.g. df.at[tid, 'deleted'][filter_][idx_not_in_ascending_order] = True
    dataframe must be indexed by id_: df.set_index(index_name, inplace=True)
    """
    values_feature = df.at[id_, feature_name]
    values_feature_filter = values_feature[filter_]
    values_feature_filter[idxs] = values
    values_feature[filter_] = values_feature_filter
    df.at[id_, feature_name] = values_feature

def list_to_str(input_list, delimiter=','):
    return delimiter.join([x if type(x) == str else repr(x) for x in input_list])  # list comprehension

def list_to_csv_str(input_list):
    return list_to_str(input_list)  # list comprehension

def fill_list_with_new_values(original_list, new_list_values):
    for i in range(len(new_list_values)):
        type1 = type(original_list[i])
        if type1 == int:
            original_list[i] = int(new_list_values[i])
        elif type1 == float:
            original_list[i] = float(new_list_values[i])
        else:
            original_list[i] = new_list_values[i]

def list_to_svm_line(original_list):
    list_size = len(original_list)
    svm_line = '%s ' % original_list[0]
    for i in range(1, list_size):
        #svm_line += '{}:{} '.format(i, repr(original_list[i]))
        svm_line += '{}:{} '.format(i, original_list[i])
    return svm_line.rstrip()

def interpolation(x0, y0, x1, y1, x):
    """
    Used for interpolation and extrapolation.
    interpolation 1: (30, 3, 40, 5, 37) -> 4.4
    interpolation 2: (30, 3, 40, 5, 35) -> 4.0
    extrapolation 1: (30, 3, 40, 5, 25) -> 2.0
    extrapolation 2: (30, 3, 40, 5, 45) -> 6.0
    """
    return y0 + (y1 - y0) * ( (x - x0)/(x1 - x0) )

def shift(arr, num, fill_value=np.nan):
    """
    Similar to pandas shift, but faster.
    See: https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array
    """
    """ Return a new array with the same shape and type as a given array."""
    result = np.empty_like(arr)

    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result = arr
    return result


