from numpy.testing import assert_array_equal, assert_equal

from pymove import visualization


def test_rgb():

    expected = (51, 51, 153)

    rgb = visualization.rgb([0.6, 0.2, 0.2])

    assert_array_equal(rgb, expected)


def test_hex_rgb():

    expected = '#333399'

    hex_rgb = visualization.hex_rgb([0.6, 0.2, 0.2])

    assert_equal(hex_rgb, expected)
