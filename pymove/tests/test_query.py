from pandas import DataFrame, Timestamp
from pandas.testing import assert_frame_equal

from pymove import MoveDataFrame
from pymove.query import query
from pymove.utils.constants import DATETIME, LATITUDE, LONGITUDE, TRAJ_ID

traj_example = [[16.4, -54.9, Timestamp('2014-10-11 18:00:00'),
                '            GONZALO'],
                [16.4, -55.9, Timestamp('2014-10-12 00:00:00'),
                '            GONZALO'],
                [16.4, -56.9, Timestamp('2014-10-12 06:00:00'),
                '            GONZALO'],
                [16.4, -57.9, Timestamp('2014-10-12 12:00:00'),
                '            GONZALO'],
                [16.4, -58.8, Timestamp('2014-10-12 18:00:00'),
                '            GONZALO'],
                [16.5, -59.7, Timestamp('2014-10-13 00:00:00'),
                '            GONZALO'],
                [16.7, -60.6, Timestamp('2014-10-13 06:00:00'),
                '            GONZALO'],
                [17.0, -61.5, Timestamp('2014-10-13 12:00:00'),
                '            GONZALO'],
                [17.1, -61.8, Timestamp('2014-10-13 14:30:00'),
                '            GONZALO'],
                [17.6, -62.4, Timestamp('2014-10-13 18:00:00'),
                '            GONZALO']]

list_data = [[33.1, -77.0, Timestamp('2012-05-19 00:00:00'),
             '            ALBERTO'],
             [32.8, -77.1, Timestamp('2012-05-19 06:00:00'),
             '            ALBERTO'],
             [32.5, -77.3, Timestamp('2012-05-19 12:00:00'),
             '            ALBERTO'],
             [32.3, -77.6, Timestamp('2012-05-19 18:00:00'),
             '            ALBERTO'],
             [32.1, -78.1, Timestamp('2012-05-20 00:00:00'),
             '            ALBERTO'],
             [31.9, -78.7, Timestamp('2012-05-20 06:00:00'),
             '            ALBERTO'],
             [31.5, -79.5, Timestamp('2012-05-20 12:00:00'),
             '            ALBERTO'],
             [31.1, -79.9, Timestamp('2012-05-20 18:00:00'),
             '            ALBERTO'],
             [30.7, -80.1, Timestamp('2012-05-21 00:00:00'),
             '            ALBERTO'],
             [30.4, -79.8, Timestamp('2012-05-21 06:00:00'),
             '            ALBERTO'],
             [28.8, -68.8, Timestamp('2012-06-17 00:00:00'),
             '              CHRIS'],
             [29.3, -66.9, Timestamp('2012-06-17 06:00:00'),
             '              CHRIS'],
             [30.4, -65.4, Timestamp('2012-06-17 12:00:00'),
             '              CHRIS'],
             [31.8, -64.2, Timestamp('2012-06-17 18:00:00'),
             '              CHRIS'],
             [33.4, -63.5, Timestamp('2012-06-18 00:00:00'),
             '              CHRIS'],
             [35.1, -62.8, Timestamp('2012-06-18 06:00:00'),
             '              CHRIS'],
             [36.7, -62.0, Timestamp('2012-06-18 12:00:00'),
             '              CHRIS'],
             [38.0, -61.3, Timestamp('2012-06-18 18:00:00'),
             '              CHRIS'],
             [38.6, -60.4, Timestamp('2012-06-19 00:00:00'),
             '              CHRIS'],
             [39.1, -59.6, Timestamp('2012-06-19 06:00:00'),
             '              CHRIS'],
             [11.6, -46.7, Timestamp('2012-08-01 12:00:00'),
             '            ERNESTO'],
             [12.0, -48.2, Timestamp('2012-08-01 18:00:00'),
             '            ERNESTO'],
             [12.4, -49.9, Timestamp('2012-08-02 00:00:00'),
             '            ERNESTO'],
             [12.7, -51.7, Timestamp('2012-08-02 06:00:00'),
             '            ERNESTO'],
             [13.0, -53.6, Timestamp('2012-08-02 12:00:00'),
             '            ERNESTO'],
             [13.2, -55.5, Timestamp('2012-08-02 18:00:00'),
             '            ERNESTO'],
             [13.4, -57.5, Timestamp('2012-08-03 00:00:00'),
             '            ERNESTO'],
             [13.6, -59.7, Timestamp('2012-08-03 06:00:00'),
             '            ERNESTO'],
             [13.7, -61.6, Timestamp('2012-08-03 12:00:00'),
             '            ERNESTO'],
             [13.8, -63.3, Timestamp('2012-08-03 18:00:00'),
             '            ERNESTO'],
             [13.6, -44.6, Timestamp('2012-08-10 00:00:00'),
             '             HELENE'],
             [13.5, -46.3, Timestamp('2012-08-10 06:00:00'),
             '             HELENE'],
             [13.4, -48.2, Timestamp('2012-08-10 12:00:00'),
             '             HELENE'],
             [13.4, -50.5, Timestamp('2012-08-10 18:00:00'),
             '             HELENE'],
             [13.4, -52.9, Timestamp('2012-08-11 00:00:00'),
             '             HELENE'],
             [13.4, -55.4, Timestamp('2012-08-11 06:00:00'),
             '             HELENE'],
             [13.3, -57.9, Timestamp('2012-08-11 12:00:00'),
             '             HELENE'],
             [13.3, -59.9, Timestamp('2012-08-11 18:00:00'),
             '             HELENE'],
             [13.5, -61.4, Timestamp('2012-08-12 00:00:00'),
             '             HELENE']]

expected_range_MEDP_data = [[11.6, -46.7, Timestamp('2012-08-01 12:00:00'),
                            '            ERNESTO'],
                            [12.0, -48.2, Timestamp('2012-08-01 18:00:00'),
                            '            ERNESTO'],
                            [12.4, -49.9, Timestamp('2012-08-02 00:00:00'),
                            '            ERNESTO'],
                            [12.7, -51.7, Timestamp('2012-08-02 06:00:00'),
                            '            ERNESTO'],
                            [13.0, -53.6, Timestamp('2012-08-02 12:00:00'),
                            '            ERNESTO'],
                            [13.2, -55.5, Timestamp('2012-08-02 18:00:00'),
                            '            ERNESTO'],
                            [13.4, -57.5, Timestamp('2012-08-03 00:00:00'),
                            '            ERNESTO'],
                            [13.6, -59.7, Timestamp('2012-08-03 06:00:00'),
                            '            ERNESTO'],
                            [13.7, -61.6, Timestamp('2012-08-03 12:00:00'),
                            '            ERNESTO'],
                            [13.8, -63.3, Timestamp('2012-08-03 18:00:00'),
                            '            ERNESTO'],
                            [13.6, -44.6, Timestamp('2012-08-10 00:00:00'),
                            '             HELENE'],
                            [13.5, -46.3, Timestamp('2012-08-10 06:00:00'),
                            '             HELENE'],
                            [13.4, -48.2, Timestamp('2012-08-10 12:00:00'),
                            '             HELENE'],
                            [13.4, -50.5, Timestamp('2012-08-10 18:00:00'),
                            '             HELENE'],
                            [13.4, -52.9, Timestamp('2012-08-11 00:00:00'),
                            '             HELENE'],
                            [13.4, -55.4, Timestamp('2012-08-11 06:00:00'),
                            '             HELENE'],
                            [13.3, -57.9, Timestamp('2012-08-11 12:00:00'),
                            '             HELENE'],
                            [13.3, -59.9, Timestamp('2012-08-11 18:00:00'),
                            '             HELENE'],
                            [13.5, -61.4, Timestamp('2012-08-12 00:00:00'),
                            '             HELENE']]

expected_range_MEDT_data = [[11.6, -46.7, Timestamp('2012-08-01 12:00:00'),
                            '            ERNESTO'],
                            [12.0, -48.2, Timestamp('2012-08-01 18:00:00'),
                            '            ERNESTO'],
                            [12.4, -49.9, Timestamp('2012-08-02 00:00:00'),
                            '            ERNESTO'],
                            [12.7, -51.7, Timestamp('2012-08-02 06:00:00'),
                            '            ERNESTO'],
                            [13.0, -53.6, Timestamp('2012-08-02 12:00:00'),
                            '            ERNESTO'],
                            [13.2, -55.5, Timestamp('2012-08-02 18:00:00'),
                            '            ERNESTO'],
                            [13.4, -57.5, Timestamp('2012-08-03 00:00:00'),
                            '            ERNESTO'],
                            [13.6, -59.7, Timestamp('2012-08-03 06:00:00'),
                            '            ERNESTO'],
                            [13.7, -61.6, Timestamp('2012-08-03 12:00:00'),
                            '            ERNESTO'],
                            [13.8, -63.3, Timestamp('2012-08-03 18:00:00'),
                            '            ERNESTO'],
                            [13.6, -44.6, Timestamp('2012-08-10 00:00:00'),
                            '             HELENE'],
                            [13.5, -46.3, Timestamp('2012-08-10 06:00:00'),
                            '             HELENE'],
                            [13.4, -48.2, Timestamp('2012-08-10 12:00:00'),
                            '             HELENE'],
                            [13.4, -50.5, Timestamp('2012-08-10 18:00:00'),
                            '             HELENE'],
                            [13.4, -52.9, Timestamp('2012-08-11 00:00:00'),
                            '             HELENE'],
                            [13.4, -55.4, Timestamp('2012-08-11 06:00:00'),
                            '             HELENE'],
                            [13.3, -57.9, Timestamp('2012-08-11 12:00:00'),
                            '             HELENE'],
                            [13.3, -59.9, Timestamp('2012-08-11 18:00:00'),
                            '             HELENE'],
                            [13.5, -61.4, Timestamp('2012-08-12 00:00:00'),
                            '             HELENE']]

expected_knn_MEDP_data = [[16.4, -54.9, Timestamp('2014-10-11 18:00:00'),
                          '            GONZALO'],
                          [16.4, -55.9, Timestamp('2014-10-12 00:00:00'),
                          '            GONZALO'],
                          [16.4, -56.9, Timestamp('2014-10-12 06:00:00'),
                          '            GONZALO'],
                          [16.4, -57.9, Timestamp('2014-10-12 12:00:00'),
                          '            GONZALO'],
                          [16.4, -58.8, Timestamp('2014-10-12 18:00:00'),
                          '            GONZALO'],
                          [16.5, -59.7, Timestamp('2014-10-13 00:00:00'),
                          '            GONZALO'],
                          [16.7, -60.6, Timestamp('2014-10-13 06:00:00'),
                          '            GONZALO'],
                          [17.0, -61.5, Timestamp('2014-10-13 12:00:00'),
                          '            GONZALO'],
                          [17.1, -61.8, Timestamp('2014-10-13 14:30:00'),
                          '            GONZALO'],
                          [17.6, -62.4, Timestamp('2014-10-13 18:00:00'),
                          '            GONZALO'],
                          [11.6, -46.7, Timestamp('2012-08-01 12:00:00'),
                          '            ERNESTO'],
                          [12.0, -48.2, Timestamp('2012-08-01 18:00:00'),
                          '            ERNESTO'],
                          [12.4, -49.9, Timestamp('2012-08-02 00:00:00'),
                          '            ERNESTO'],
                          [12.7, -51.7, Timestamp('2012-08-02 06:00:00'),
                          '            ERNESTO'],
                          [13.0, -53.6, Timestamp('2012-08-02 12:00:00'),
                          '            ERNESTO'],
                          [13.2, -55.5, Timestamp('2012-08-02 18:00:00'),
                          '            ERNESTO'],
                          [13.4, -57.5, Timestamp('2012-08-03 00:00:00'),
                          '            ERNESTO'],
                          [13.6, -59.7, Timestamp('2012-08-03 06:00:00'),
                          '            ERNESTO'],
                          [13.7, -61.6, Timestamp('2012-08-03 12:00:00'),
                          '            ERNESTO'],
                          [13.8, -63.3, Timestamp('2012-08-03 18:00:00'),
                          '            ERNESTO'],
                          [13.6, -44.6, Timestamp('2012-08-10 00:00:00'),
                          '             HELENE'],
                          [13.5, -46.3, Timestamp('2012-08-10 06:00:00'),
                          '             HELENE'],
                          [13.4, -48.2, Timestamp('2012-08-10 12:00:00'),
                          '             HELENE'],
                          [13.4, -50.5, Timestamp('2012-08-10 18:00:00'),
                          '             HELENE'],
                          [13.4, -52.9, Timestamp('2012-08-11 00:00:00'),
                          '             HELENE'],
                          [13.4, -55.4, Timestamp('2012-08-11 06:00:00'),
                          '             HELENE'],
                          [13.3, -57.9, Timestamp('2012-08-11 12:00:00'),
                          '             HELENE'],
                          [13.3, -59.9, Timestamp('2012-08-11 18:00:00'),
                          '             HELENE'],
                          [13.5, -61.4, Timestamp('2012-08-12 00:00:00'),
                          '             HELENE']]

expected_knn_MEDT_data = [[16.4, -54.9, Timestamp('2014-10-11 18:00:00'),
                          '            GONZALO'],
                          [16.4, -55.9, Timestamp('2014-10-12 00:00:00'),
                          '            GONZALO'],
                          [16.4, -56.9, Timestamp('2014-10-12 06:00:00'),
                          '            GONZALO'],
                          [16.4, -57.9, Timestamp('2014-10-12 12:00:00'),
                          '            GONZALO'],
                          [16.4, -58.8, Timestamp('2014-10-12 18:00:00'),
                          '            GONZALO'],
                          [16.5, -59.7, Timestamp('2014-10-13 00:00:00'),
                          '            GONZALO'],
                          [16.7, -60.6, Timestamp('2014-10-13 06:00:00'),
                          '            GONZALO'],
                          [17.0, -61.5, Timestamp('2014-10-13 12:00:00'),
                          '            GONZALO'],
                          [17.1, -61.8, Timestamp('2014-10-13 14:30:00'),
                          '            GONZALO'],
                          [17.6, -62.4, Timestamp('2014-10-13 18:00:00'),
                          '            GONZALO'],
                          [13.6, -44.6, Timestamp('2012-08-10 00:00:00'),
                          '             HELENE'],
                          [13.5, -46.3, Timestamp('2012-08-10 06:00:00'),
                          '             HELENE'],
                          [13.4, -48.2, Timestamp('2012-08-10 12:00:00'),
                          '             HELENE'],
                          [13.4, -50.5, Timestamp('2012-08-10 18:00:00'),
                          '             HELENE'],
                          [13.4, -52.9, Timestamp('2012-08-11 00:00:00'),
                          '             HELENE'],
                          [13.4, -55.4, Timestamp('2012-08-11 06:00:00'),
                          '             HELENE'],
                          [13.3, -57.9, Timestamp('2012-08-11 12:00:00'),
                          '             HELENE'],
                          [13.3, -59.9, Timestamp('2012-08-11 18:00:00'),
                          '             HELENE'],
                          [13.5, -61.4, Timestamp('2012-08-12 00:00:00'),
                          '             HELENE']]


def _default_traj_df(data=None):
    if data is None:
        data = traj_example
    return MoveDataFrame(
        data=data,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        datetime=DATETIME,
        traj_id=TRAJ_ID,
    )


def _default_move_df(data=None):
    if data is None:
        data = list_data
    return MoveDataFrame(
        data=data,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        datetime=DATETIME,
        traj_id=TRAJ_ID,
    )


def test_range_query():
    traj_df = _default_traj_df()
    move_df = _default_move_df()
    expected_MEDP = DataFrame(
        data=expected_range_MEDP_data,
        columns=['lat', 'lon', 'datetime', 'id'],
        index=[20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]
    )
    expected_MEDT = DataFrame(
        data=expected_range_MEDT_data,
        columns=['lat', 'lon', 'datetime', 'id'],
        index=[20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]
    )

    medp_move_df = query.range_query(traj_df, move_df, range=100, distance='MEDP')
    assert_frame_equal(medp_move_df, expected_MEDP)

    medt_move_df = query.range_query(traj_df, move_df, range=700, distance='MEDT')
    assert_frame_equal(medt_move_df, expected_MEDT)


def test_knn_query():
    traj_df = _default_traj_df()
    move_df = _default_move_df()
    expected_MEDP = DataFrame(
        data=expected_knn_MEDP_data,
        columns=['lat', 'lon', 'datetime', 'id'],
        index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 20,
               21, 22, 23, 24, 25, 26, 27, 28, 29,
               30, 31, 32, 33, 34, 35, 36, 37, 38]
    )
    expected_MEDT = DataFrame(
        data=expected_knn_MEDT_data,
        columns=['lat', 'lon', 'datetime', 'id'],
        index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 30, 31, 32, 33, 34, 35, 36, 37, 38]
    )

    medp_move_df = query.knn_query(traj_df, move_df, k=2, distance='MEDP')
    assert_frame_equal(medp_move_df, expected_MEDP)

    medt_move_df = query.knn_query(traj_df, move_df, k=2, distance='MEDT')
    assert_frame_equal(medt_move_df, expected_MEDT)