import datetime as dt

from numpy import nan
from numpy.testing import assert_equal
from pandas import DataFrame, Timestamp
from pandas.testing import assert_frame_equal

from pymove import MoveDataFrame, datetime
from pymove.utils.constants import (
    COUNT,
    LOCAL_LABEL,
    MAX,
    MEAN,
    MIN,
    PREV_LOCAL,
    STD,
    SUM,
    THRESHOLD,
    TIME_TO_PREV,
)

default_date = dt.datetime.strptime('2018-03-12', '%Y-%m-%d')

default_date_time = dt.datetime.strptime('2018-03-12 12:08:07', '%Y-%m-%d %H:%M:%S')

str_date_default = '2018-03-12'

str_date_time_default = '2018-03-12 12:08:07'

list_data = [
    [39.984094, 116.319236, '2008-10-23 05:44:05', 1],
    [39.984198, 116.319322, '2008-10-23 05:56:06', 1],
    [39.984224, 116.319402, '2008-10-23 05:56:11', 1],
    [39.984224, 116.319402, '2008-10-23 06:10:15', 1],
]


def _default_move_df():
    return MoveDataFrame(
        data=list_data,
    )


def test_date_to_str():

    expected = '2008-10-23'

    time_str = datetime.date_to_str(Timestamp('2008-10-23 05:53:05'))

    assert(time_str == expected)


def test_str_to_datetime():

    expected_date = default_date

    expected_date_time = default_date_time

    converted_date = datetime.str_to_datetime('2018-03-12')

    assert(converted_date == expected_date)

    converted_date_time = datetime.str_to_datetime('2018-03-12 12:08:07')

    assert(converted_date_time == expected_date_time)


def test_to_str():

    expected = str_date_time_default

    data = default_date_time

    str_date_time = datetime.to_str(data)

    assert(str_date_time == expected)


def test_to_min():

    expected = 25347608

    data = default_date_time

    date_to_min = datetime.to_min(data)

    assert(date_to_min == expected)


def test_min_to_datetime():

    expected = dt.datetime.strptime('2018-03-12 12:08:00',
                                    '%Y-%m-%d %H:%M:%S')

    data = 25347608

    min_to_date = datetime.min_to_datetime(data)

    assert(min_to_date == expected)


def test_to_day_of_week_int():

    expected = 0

    data = default_date

    date_to_day_week = datetime.to_day_of_week_int(data)

    assert(date_to_day_week == expected)

    data = default_date_time

    date_to_day_week = datetime.to_day_of_week_int(data)

    assert(date_to_day_week == expected)


def test_working_day():

    data = str_date_default

    working_day = datetime.working_day(data)

    assert(working_day is True)

    data = default_date

    working_day = datetime.working_day(data)

    assert(working_day is True)

    data = '2018-03-17'

    working_day = datetime.working_day(data)

    assert(working_day is False)

    data = dt.datetime.strptime('2018-10-12', '%Y-%m-%d')

    working_day = datetime.working_day(data, country='BR')

    assert(working_day is False)


def test_now_str():

    expected = datetime.to_str(dt.datetime.now())

    time_now = datetime.now_str()

    assert(time_now == expected)


def test_deltatime_str():

    expected = '05.03s'
    actual = datetime.deltatime_str(5.03)
    assert expected == actual

    expected = '18m:35.00s'
    actual = datetime.deltatime_str(1115)
    assert expected == actual

    expected = '03h:05m:15.00s'
    actual = datetime.deltatime_str(11115)
    assert expected == actual


def test_timestamp_to_millis():

    expected = 1520856487000

    data = str_date_time_default

    milliseconds = datetime.timestamp_to_millis(data)

    assert(milliseconds == expected)


def test_millis_to_timestamp():

    expected = default_date_time

    data = 1520856487000

    timestamp = datetime.millis_to_timestamp(data)

    assert(timestamp == expected)


def test_time_to_str():

    expected = '12:08:07'

    data = default_date_time

    time = datetime.time_to_str(data)

    assert(time == expected)


def test_elapsed_time_dt():

    data = default_date_time
    expected = datetime.diff_time(default_date_time,
                                  dt.datetime.now())
    elapsed_time = datetime.elapsed_time_dt(data)

    assert abs(elapsed_time - expected) <= 5


def test_diff_time():

    expected = 388313000

    start_date = default_date_time

    end_date = dt.datetime.strptime('2018-03-17', '%Y-%m-%d')

    diff_time = datetime.diff_time(start_date, end_date)

    assert(diff_time == expected)


def test_create_time_slot_in_minute():
    df = _default_move_df()
    expected = DataFrame({
        'lat': {0: 39.984094, 1: 39.984198, 2: 39.984224, 3: 39.984224},
        'lon': {0: 116.319236, 1: 116.319322, 2: 116.319402, 3: 116.319402},
        'datetime': {
            0: Timestamp('2008-10-23 05:44:05'),
            1: Timestamp('2008-10-23 05:56:06'),
            2: Timestamp('2008-10-23 05:56:11'),
            3: Timestamp('2008-10-23 06:10:15')
        },
        'id': {0: 1, 1: 1, 2: 1, 3: 1},
        'time_slot': {0: 22, 1: 23, 2: 23, 3: 24}
    })
    datetime.create_time_slot_in_minute(df)
    assert_frame_equal(df, expected)


def test_generate_time_statistics():
    df_ = DataFrame(
        data=[
            [261, nan, nan],
            [580, 261, 252],
            [376, 580, 91],
            [386, 376, 17449],
            [644, 386, 21824]
        ],
        columns=[LOCAL_LABEL, PREV_LOCAL, TIME_TO_PREV],
        index=[0, 1, 2, 3, 4]
    )

    expected = DataFrame(
        data=[
            [376, 580.0, 91.0, 0.0, 91.0, 91.0, 91.0, 1],
            [386, 376.0, 17449.0, 0.0, 17449.0, 17449.0, 17449.0, 1],
            [580, 261.0, 252.0, 0.0, 252.0, 252.0, 252.0, 1],
            [644, 386.0, 21824.0, 0.0, 21824.0, 21824.0, 21824.0, 1]
        ],
        columns=[LOCAL_LABEL, PREV_LOCAL, MEAN, STD, MIN, MAX, SUM, COUNT],
        index=[0, 1, 2, 3]
    )

    df_statistics = datetime.generate_time_statistics(df_)
    assert_frame_equal(df_statistics, expected)


def test_calc_time_threshold():
    mean1, std1 = 0.0, 91.0
    mean2, std2 = 0.0, 17449.0
    mean3, std3 = 0.0, 252.0
    mean4, std4 = 0.0, 21824.0

    expected1 = 91.0
    expected2 = 17449.0
    expected3 = 252.0
    expected4 = 21824.0

    threshold1 = datetime._calc_time_threshold(mean1, std1)
    threshold2 = datetime._calc_time_threshold(mean2, std2)
    threshold3 = datetime._calc_time_threshold(mean3, std3)
    threshold4 = datetime._calc_time_threshold(mean4, std4)

    assert_equal(threshold1, expected1)
    assert_equal(threshold2, expected2)
    assert_equal(threshold3, expected3)
    assert_equal(threshold4, expected4)


def test_threshold_time_statistics():
    statistics = DataFrame(
        data=[
            [376, 580.0, 91.0, 0.0, 91.0, 91.0, 91.0, 1],
            [386, 376.0, 17449.0, 0.0, 17449.0, 17449.0, 17449.0, 1],
            [580, 261.0, 252.0, 0.0, 252.0, 252.0, 252.0, 1],
            [644, 386.0, 21824.0, 0.0, 21824.0, 21824.0, 21824.0, 1]
        ],
        columns=[LOCAL_LABEL, PREV_LOCAL, MEAN, STD, MIN, MAX, SUM, COUNT],
        index=[0, 1, 2, 3]
    )

    expected = DataFrame(
        data=[
            [376, 580.0, 91.0, 0.0, 91.0, 91.0, 91.0, 1, 91.0],
            [386, 376.0, 17449.0, 0.0, 17449.0, 17449.0, 17449.0, 1, 17449.0],
            [580, 261.0, 252.0, 0.0, 252.0, 252.0, 252.0, 1, 252.0],
            [644, 386.0, 21824.0, 0.0, 21824.0, 21824.0, 21824.0, 1, 21824.0]
        ],
        columns=[
            LOCAL_LABEL, PREV_LOCAL, MEAN, STD, MIN, MAX, SUM, COUNT, THRESHOLD
        ],
        index=[0, 1, 2, 3]
    )

    datetime.threshold_time_statistics(statistics)
    assert_frame_equal(statistics, expected)
