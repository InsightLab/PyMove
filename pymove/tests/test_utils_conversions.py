from numpy import nan
from pandas import DataFrame, Timestamp
from pandas.testing import assert_frame_equal

from pymove import MoveDataFrame, conversions
from pymove.utils.constants import (
    DATETIME,
    DIST_TO_PREV,
    LATITUDE,
    LONGITUDE,
    SPEED_TO_PREV,
    TIME_TO_PREV,
    TRAJ_ID,
)

list_data = [
    [39.984094, 116.319236, '2008-10-23 05:53:05', 1],
    [39.984198, 116.319322, '2008-10-23 05:53:06', 1],
    [39.984224, 116.319402, '2008-10-23 05:53:11', 1],
    [39.984224, 116.319402, '2008-10-23 05:53:11', 1],
]


def _default_move_df():
    return MoveDataFrame(
        data=list_data,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        datetime=DATETIME,
        traj_id=TRAJ_ID,
    )


def test_lat_meters():

    expected = 98224.0229295811

    lat_in_meters = conversions.lat_meters(39.984094)

    assert(lat_in_meters == expected)


def test_list_to_str():

    expected = 'banana,maca,laranja'

    joined_list = conversions.list_to_str(['banana', 'maca', 'laranja'])

    assert(joined_list == expected)


def test_list_to_csv_str():
    expected = 'banana 1:maca 2:laranja'

    joined_list = conversions.list_to_svm_line(['banana', 'maca', 'laranja'])

    assert(joined_list == expected)


def test_lon_to_x_spherical():

    expected = -4285978.172767829

    assert(conversions.lon_to_x_spherical(-38.501597) == expected)


def test_lat_to_y_spherical():

    expected = -423086.2213610324

    assert(conversions.lat_to_y_spherical(-3.797864) == expected)


def test_x_to_lon_spherical():

    expected = -38.50159697513617

    assert(conversions.x_to_lon_spherical(-4285978.17) == expected)


def test_y_to_lat_spherical():

    expected = -35.89350841198311

    assert(conversions.y_to_lat_spherical(-4285978.17) == expected)


def test_ms_to_kmh():

    move_df = _default_move_df()

    expected = DataFrame(
        data=[
            [
                1,
                39.984094,
                116.319236,
                Timestamp('2008-10-23 05:53:05'),
                nan,
                nan,
                nan
            ],
            [
                1,
                39.984198,
                116.319322,
                Timestamp('2008-10-23 05:53:06'),
                13.690153,
                1.0,
                49.284551
            ],
            [
                1,
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                7.403788,
                5.0,
                5.330727
            ],
            [
                1,
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                0.000000,
                0.0,
                nan],
        ],
        columns=[TRAJ_ID,
                 LATITUDE,
                 LONGITUDE,
                 DATETIME,
                 DIST_TO_PREV,
                 TIME_TO_PREV,
                 SPEED_TO_PREV],
        index=[0, 1, 2, 3],
    )

    new_move_df = conversions.ms_to_kmh(move_df, inplace=False)

    assert_frame_equal(new_move_df, expected)

    conversions.ms_to_kmh(move_df, new_label='converted_speed', inplace=True)

    expected.rename(columns={SPEED_TO_PREV: 'converted_speed'}, inplace=True)

    assert_frame_equal(move_df, expected)


def test_kmh_to_ms():
    move_df = _default_move_df()

    expected = DataFrame(
        data=[
            [
                1,
                39.984094,
                116.319236,
                Timestamp('2008-10-23 05:53:05'),
                nan,
                nan,
                nan
            ],
            [
                1,
                39.984198,
                116.319322,
                Timestamp('2008-10-23 05:53:06'),
                13.690153,
                1.0,
                13.690153
            ],
            [
                1,
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                7.403788,
                5.0,
                1.480758
            ],
            [
                1,
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                0.000000,
                0.0,
                nan],
        ],
        columns=[TRAJ_ID,
                 LATITUDE,
                 LONGITUDE,
                 DATETIME,
                 DIST_TO_PREV,
                 TIME_TO_PREV,
                 SPEED_TO_PREV],
        index=[0, 1, 2, 3],
    )

    new_move_df = conversions.kmh_to_ms(move_df, inplace=False)

    assert_frame_equal(new_move_df, expected)

    conversions.kmh_to_ms(move_df, new_label='converted_speed', inplace=True)

    expected.rename(columns={SPEED_TO_PREV: 'converted_speed'}, inplace=True)

    assert_frame_equal(move_df, expected)


def test_meters_to_kilometers():
    move_df = _default_move_df()

    expected = DataFrame(
        data=[
            [
                1,
                39.984094,
                116.319236,
                Timestamp('2008-10-23 05:53:05'),
                nan,
                nan,
                nan
            ],
            [
                1,
                39.984198,
                116.319322,
                Timestamp('2008-10-23 05:53:06'),
                0.013690153134343689,
                1.0,
                13.690153
            ],
            [
                1,
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                0.007403787866531697,
                5.0,
                1.480758
            ],
            [
                1,
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                0.0,
                0.0,
                nan],
        ],
        columns=[TRAJ_ID,
                 LATITUDE,
                 LONGITUDE,
                 DATETIME,
                 DIST_TO_PREV,
                 TIME_TO_PREV,
                 SPEED_TO_PREV],
        index=[0, 1, 2, 3],
    )

    new_move_df = conversions.meters_to_kilometers(move_df, inplace=False)

    assert_frame_equal(new_move_df, expected)

    conversions.meters_to_kilometers(move_df, new_label='converted_distance', inplace=True)

    expected.rename(columns={DIST_TO_PREV: 'converted_distance'}, inplace=True)

    assert_frame_equal(move_df, expected)


def test_kilometers_to_meters():
    move_df = _default_move_df()

    expected = DataFrame(
        data=[
            [
                1,
                39.984094,
                116.319236,
                Timestamp('2008-10-23 05:53:05'),
                nan,
                nan,
                nan
            ],
            [
                1,
                39.984198,
                116.319322,
                Timestamp('2008-10-23 05:53:06'),
                13.690153134343689,
                1.0,
                13.690153
            ],
            [
                1,
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                7.403787866531697,
                5.0,
                1.480758
            ],
            [
                1,
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                0.0,
                0.0,
                nan],
        ],
        columns=[TRAJ_ID,
                 LATITUDE,
                 LONGITUDE,
                 DATETIME,
                 DIST_TO_PREV,
                 TIME_TO_PREV,
                 SPEED_TO_PREV],
        index=[0, 1, 2, 3],
    )

    new_move_df = conversions.kilometers_to_meters(move_df, inplace=False)

    assert_frame_equal(new_move_df, expected)

    conversions.kilometers_to_meters(move_df, new_label='converted_distance', inplace=True)

    expected.rename(columns={DIST_TO_PREV: 'converted_distance'}, inplace=True)

    assert_frame_equal(move_df, expected)


def test_seconds_to_minutes():
    move_df = _default_move_df()

    expected = DataFrame(
        data=[
            [
                1,
                39.984094,
                116.319236,
                Timestamp('2008-10-23 05:53:05'),
                nan,
                nan,
                nan
            ],
            [
                1,
                39.984198,
                116.319322,
                Timestamp('2008-10-23 05:53:06'),
                13.690153134343689,
                0.016666666666666666,
                13.690153
            ],
            [
                1,
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                7.403787866531697,
                0.08333333333333333,
                1.480758
            ],
            [
                1,
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                0.0,
                0.0,
                nan],
        ],
        columns=[TRAJ_ID,
                 LATITUDE,
                 LONGITUDE,
                 DATETIME,
                 DIST_TO_PREV,
                 TIME_TO_PREV,
                 SPEED_TO_PREV],
        index=[0, 1, 2, 3],
    )

    new_move_df = conversions.seconds_to_minutes(move_df, inplace=False)

    assert_frame_equal(new_move_df, expected)

    conversions.seconds_to_minutes(move_df, new_label='converted_time', inplace=True)

    expected.rename(columns={TIME_TO_PREV: 'converted_time'}, inplace=True)

    assert_frame_equal(move_df, expected)


def test_minute_to_seconds():
    move_df = _default_move_df()

    expected = DataFrame(
        data=[
            [
                1,
                39.984094,
                116.319236,
                Timestamp('2008-10-23 05:53:05'),
                nan,
                nan,
                nan
            ],
            [
                1,
                39.984198,
                116.319322,
                Timestamp('2008-10-23 05:53:06'),
                13.690153134343689,
                1.0,
                13.690153
            ],
            [
                1,
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                7.403787866531697,
                5.0,
                1.480758
            ],
            [
                1,
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                0.0,
                0.0,
                nan],
        ],
        columns=[TRAJ_ID,
                 LATITUDE,
                 LONGITUDE,
                 DATETIME,
                 DIST_TO_PREV,
                 TIME_TO_PREV,
                 SPEED_TO_PREV],
        index=[0, 1, 2, 3],
    )

    new_move_df = conversions.minute_to_seconds(move_df, inplace=False)

    assert_frame_equal(new_move_df, expected)

    conversions.minute_to_seconds(move_df, new_label='converted_time', inplace=True)

    expected.rename(columns={TIME_TO_PREV: 'converted_time'}, inplace=True)

    assert_frame_equal(move_df, expected)


def test_minute_to_hours():
    move_df = _default_move_df()

    expected = DataFrame(
        data=[
            [
                1,
                39.984094,
                116.319236,
                Timestamp('2008-10-23 05:53:05'),
                nan,
                nan,
                nan
            ],
            [
                1,
                39.984198,
                116.319322,
                Timestamp('2008-10-23 05:53:06'),
                13.690153134343689,
                0.0002777777777777778,
                13.690153
            ],
            [
                1,
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                7.403787866531697,
                0.0013888888888888887,
                1.480758
            ],
            [
                1,
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                0.0,
                0.0,
                nan],
        ],
        columns=[TRAJ_ID,
                 LATITUDE,
                 LONGITUDE,
                 DATETIME,
                 DIST_TO_PREV,
                 TIME_TO_PREV,
                 SPEED_TO_PREV],
        index=[0, 1, 2, 3],
    )

    new_move_df = conversions.minute_to_hours(move_df, inplace=False)

    assert_frame_equal(new_move_df, expected)

    conversions.minute_to_hours(move_df, new_label='converted_time', inplace=True)

    expected.rename(columns={TIME_TO_PREV: 'converted_time'}, inplace=True)

    assert_frame_equal(move_df, expected)


def test_hours_to_minute():
    move_df = _default_move_df()

    expected = DataFrame(
        data=[
            [
                1,
                39.984094,
                116.319236,
                Timestamp('2008-10-23 05:53:05'),
                nan,
                nan,
                nan
            ],
            [
                1,
                39.984198,
                116.319322,
                Timestamp('2008-10-23 05:53:06'),
                13.690153134343689,
                0.016666666666666666,
                13.690153
            ],
            [
                1,
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                7.403787866531697,
                0.08333333333333334,
                1.480758
            ],
            [
                1,
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                0.0,
                0.0,
                nan],
        ],
        columns=[TRAJ_ID,
                 LATITUDE,
                 LONGITUDE,
                 DATETIME,
                 DIST_TO_PREV,
                 TIME_TO_PREV,
                 SPEED_TO_PREV],
        index=[0, 1, 2, 3],
    )

    new_move_df = conversions.hours_to_minute(move_df, inplace=False)

    assert_frame_equal(new_move_df, expected)

    conversions.hours_to_minute(move_df, new_label='converted_time', inplace=True)

    expected.rename(columns={TIME_TO_PREV: 'converted_time'}, inplace=True)

    assert_frame_equal(move_df, expected)


def test_seconds_to_hours():
    move_df = _default_move_df()

    expected = DataFrame(
        data=[
            [
                1,
                39.984094,
                116.319236,
                Timestamp('2008-10-23 05:53:05'),
                nan,
                nan,
                nan
            ],
            [
                1,
                39.984198,
                116.319322,
                Timestamp('2008-10-23 05:53:06'),
                13.690153134343689,
                0.0002777777777777778,
                13.690153
            ],
            [
                1,
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                7.403787866531697,
                0.001388888888888889,
                1.480758
            ],
            [
                1,
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                0.0,
                0.0,
                nan],
        ],
        columns=[TRAJ_ID,
                 LATITUDE,
                 LONGITUDE,
                 DATETIME,
                 DIST_TO_PREV,
                 TIME_TO_PREV,
                 SPEED_TO_PREV],
        index=[0, 1, 2, 3],
    )

    new_move_df = conversions.seconds_to_hours(move_df, inplace=False)

    assert_frame_equal(new_move_df, expected)

    conversions.seconds_to_hours(move_df, new_label='converted_time', inplace=True)

    expected.rename(columns={TIME_TO_PREV: 'converted_time'}, inplace=True)

    assert_frame_equal(move_df, expected)


def test_hours_to_seconds():
    move_df = _default_move_df()

    expected = DataFrame(
        data=[
            [
                1,
                39.984094,
                116.319236,
                Timestamp('2008-10-23 05:53:05'),
                nan,
                nan,
                nan
            ],
            [
                1,
                39.984198,
                116.319322,
                Timestamp('2008-10-23 05:53:06'),
                13.690153134343689,
                1.0,
                13.690153
            ],
            [
                1,
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                7.403787866531697,
                5.0,
                1.480758
            ],
            [
                1,
                39.984224,
                116.319402,
                Timestamp('2008-10-23 05:53:11'),
                0.0,
                0.0,
                nan],
        ],
        columns=[TRAJ_ID,
                 LATITUDE,
                 LONGITUDE,
                 DATETIME,
                 DIST_TO_PREV,
                 TIME_TO_PREV,
                 SPEED_TO_PREV],
        index=[0, 1, 2, 3],
    )

    new_move_df = conversions.hours_to_seconds(move_df, inplace=False)

    assert_frame_equal(new_move_df, expected)

    conversions.hours_to_seconds(move_df, new_label='converted_time', inplace=True)

    expected.rename(columns={TIME_TO_PREV: 'converted_time'}, inplace=True)

    assert_frame_equal(move_df, expected)
