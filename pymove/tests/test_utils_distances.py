from numpy.testing import assert_almost_equal

from pymove import distances


def test_haversine():
    expected = 9.757976024363016

    dist = distances.haversine(-3.797864, -38.501597, -3.797890, -38.501681)

    assert_almost_equal(dist, expected)
