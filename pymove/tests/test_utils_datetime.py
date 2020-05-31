import datetime as dt

from pandas import Timestamp

from pymove import datetime

default_date = dt.datetime.strptime('2018-03-12', '%Y-%m-%d')

default_date_time = dt.datetime.strptime('2018-03-12 12:08:07', '%Y-%m-%d %H:%M:%S')

str_date_default = '2018-03-12'

str_date_time_default = '2018-03-12 12:08:07'


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

    expected = '00:18:02.718'

    data = 1082.7180936336517

    converted_date = datetime.deltatime_str(data)

    assert(converted_date == expected)


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

    time = datetime.time_to_str(default_date_time)

    assert(time == expected)


def test_elapsed_time_dt():

    expected = datetime.diff_time(default_date_time,
                                  dt.datetime.now())

    data = default_date_time

    elapsed_time = datetime.elapsed_time_dt(data)

    assert(elapsed_time == expected)


def test_diff_time():

    expected = 388313000

    start_date = default_date_time

    end_date = dt.datetime.strptime('2018-03-17', '%Y-%m-%d')

    elapsed_time = datetime.diff_time(start_date, end_date)

    assert(elapsed_time == expected)
