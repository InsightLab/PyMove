from numpy.testing import assert_almost_equal

from pymove import math


def test_std():
    expected = 277.0178494048513

    std = math.std([600, 20, 5])

    assert_almost_equal(std, expected)


def test_avg_std():
    expected = (208.33333333333334, 277.0178494048513)

    avg_std = math.avg_std([600, 20, 5])

    assert_almost_equal(avg_std, expected)


def test_std_sample():
    expected = 339.27619034251916

    avg_std = math.std_sample([600, 20, 5])

    assert_almost_equal(avg_std, expected)


def test_avg_std_sample():
    expected = (208.33333333333334, 339.27619034251916)

    avg_std_sample = math.avg_std_sample([600, 20, 5])

    assert_almost_equal(avg_std_sample, expected)


def test_arrays_avg():
    expected = 208.33333333333334

    array_avg = math.arrays_avg([600, 20, 5])

    assert_almost_equal(array_avg, expected)


def test_array_stats():
    expected = (625, 360425, 3)

    array_stats = math.array_stats([600, 20, 5])

    assert_almost_equal(array_stats, expected)


def test_interpolation():
    expected = 6.799999999999999

    interpolation = math.interpolation(15, 20, 65, 86, 5)

    assert_almost_equal(interpolation, expected)
