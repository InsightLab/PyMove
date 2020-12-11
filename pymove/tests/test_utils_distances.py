from numpy.testing import assert_almost_equal
from pandas import Timestamp

from pymove import MoveDataFrame, distances

traj_example1 = [[16.4, -54.9, Timestamp('2014-10-11 18:00:00'),
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

traj_example2 = [[33.1, -77.0, Timestamp('2012-05-19 00:00:00'),
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
                 '            ALBERTO']]

traj_example3 = [[13.6, -44.6, Timestamp('2012-08-10 00:00:00'),
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


def test_haversine():
    expected = 9.757976024363016

    dist = distances.haversine(-3.797864, -38.501597, -3.797890, -38.501681)

    assert_almost_equal(dist, expected)


def test_euclidean_distance_in_meters():
    expected = 0.56021344523276

    dist = distances.euclidean_distance_in_meters(
        -3.797864, -38.501597, -3.797890, -38.501681
    )

    assert_almost_equal(dist, expected)


def test_MEDP():
    expected = 241.91923668814994

    move_df1 = MoveDataFrame(data=traj_example1)
    move_df2 = MoveDataFrame(data=traj_example2)

    medp = distances.MEDP(move_df1, move_df2)
    assert_almost_equal(medp, expected)


def test_MEDT():
    expected = 619.9417037397966

    move_df1 = MoveDataFrame(data=traj_example1)
    move_df2 = MoveDataFrame(data=traj_example3)

    medt = distances.MEDT(move_df1, move_df2)
    assert_almost_equal(medt, expected)
