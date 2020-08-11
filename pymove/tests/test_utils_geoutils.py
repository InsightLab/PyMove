import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.testing.compare import compare_images
from numpy.testing import assert_array_equal, assert_equal
from pandas import DataFrame
from pandas.testing import assert_frame_equal
from shapely.geometry import LineString

from pymove.utils import geoutils
from pymove.utils.constants import (
    BIN_GEOHASH,
    GEOHASH,
    LATITUDE,
    LATITUDE_DECODE,
    LONGITUDE,
    LONGITUDE_DECODE,
)


def test_v_color():
    line_1 = LineString([(1, 1), (2, 2), (2, 1), (1, 2)])
    expected_1 = '#ffcc33'

    line_2 = LineString([(1, 1), (2, 1), (2, 2), (1, 2), (1, 1)])
    expected_2 = '#6699cc'

    assert_equal(expected_1, geoutils.v_color(line_1))
    assert_equal(expected_2, geoutils.v_color(line_2))


def test_plot_coords(tmpdir):
    d = tmpdir.mkdir('preprocessing')

    file_write_default = d.join('plot_coords.png')

    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    coords = LineString([(1, 1), (1, 2), (2, 2), (2, 3)])

    fig, ax = plt.subplots(figsize=(21, 9))
    geoutils.plot_coords(ax, coords)
    plt.savefig(filename_write_default, fig=fig, dpi=100)

    test_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(test_dir, 'baseline/plot_coords.png')

    compare_images(
        data_dir, filename_write_default, 0.0001, in_decorator=False
    )


def test_plot_bounds(tmpdir):
    d = tmpdir.mkdir('preprocessing')

    file_write_default = d.join('plot_bounds.png')

    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    bounds = LineString([(1, 1), (1, 2), (2, 2), (2, 3)])

    fig, ax = plt.subplots(figsize=(21, 9))
    geoutils.plot_bounds(ax, bounds)
    plt.savefig(filename_write_default, fig=fig, dpi=100)

    test_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(test_dir, 'baseline/plot_bounds.png')

    compare_images(
        data_dir, filename_write_default, 0.0001, in_decorator=False
    )


def test_plot_line(tmpdir):
    d = tmpdir.mkdir('preprocessing')

    file_write_default = d.join('plot_line.png')

    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    line = LineString([(1, 1), (1, 2), (2, 2), (2, 3)])

    fig, ax = plt.subplots(figsize=(21, 9))
    geoutils.plot_line(ax, line)
    plt.savefig(filename_write_default, fig=fig, dpi=100)

    test_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(test_dir, 'baseline/plot_line.png')

    compare_images(
        data_dir, filename_write_default, 0.0001, in_decorator=False
    )


def test_encode():
    lat1, lon1 = -3.777736, -38.547792
    lat2, lon2 = -3.793388, -38.517722
    lat3, lon3 = -3.783605, -38.521962
    lat4, lon4 = -3.774056, -38.482056
    lat5, lon5 = -3.719155, -38.532494

    assert_equal('7pkddb6356fyzxq', geoutils._encode(lat1, lon1))
    assert_equal('7pkd7t2mbj0z1v7', geoutils._encode(lat2, lon2))
    assert_equal('7pkd7rjnvhzjp90', geoutils._encode(lat3, lon3))
    assert_equal('7pkds2fnx0gr1c0', geoutils._encode(lat4, lon4))
    assert_equal('7pkdg4vqrg6020q', geoutils._encode(lat5, lon5))


def test_decode():
    expected1 = ('-3.777736', '-38.547792')
    expected2 = ('-3.793388', '-38.517722')
    expected3 = ('-3.783605', '-38.521962')
    expected4 = ('-3.774056', '-38.482056')
    expected5 = ('-3.719155', '-38.532494')

    assert_equal(expected1, geoutils._decode('7pkddb6356fyzxq'))
    assert_equal(expected2, geoutils._decode('7pkd7t2mbj0z1v7'))
    assert_equal(expected3, geoutils._decode('7pkd7rjnvhzjp90'))
    assert_equal(expected4, geoutils._decode('7pkds2fnx0gr1c0'))
    assert_equal(expected5, geoutils._decode('7pkdg4vqrg6020q'))


def test_bin_geohash():
    lat1, lon1 = -3.777736, -38.547792
    lat2, lon2 = -3.793388, -38.517722
    lat3, lon3 = -3.783605, -38.521962

    expected1 = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1,
                          0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0,
                          0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1,
                          1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
                          1, 1, 1, 0, 1, 0])

    expected2 = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1,
                          0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1,
                          0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0,
                          0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0,
                          0, 1, 1, 1])

    expected3 = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1,
                          0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1,
                          0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0,
                          1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0,
                          0, 0, 0, 0])

    assert_array_equal(expected1, geoutils._bin_geohash(lat1, lon1))
    assert_array_equal(expected2, geoutils._bin_geohash(lat2, lon2))
    assert_array_equal(expected3, geoutils._bin_geohash(lat3, lon3))


def test_reset_and_create_arrays_none():

    df_ = DataFrame(
        data=[
            [-3.777736, -38.547792],
            [-3.793388, -38.517722],
            [-3.783605, -38.521962],
            [-3.774056, -38.482056],
            [-3.719155, -38.532494],
        ],
        columns=[LATITUDE, LONGITUDE],
        index=[0, 1, 2, 3, 4]
    )

    lat_expected = np.full(shape=5, fill_value=None, dtype=np.float64)
    lon_expected = np.full(shape=5, fill_value=None, dtype=np.float64)
    geohash_expected = np.full(shape=5, fill_value=None, dtype='object_')
    bin_geohash_expected = np.full(shape=5, fill_value=None, dtype=np.ndarray)

    lat, lon, geohash, bin_geohash = geoutils._reset_and_create_arrays_none(df_)

    assert_array_equal(lat, lat_expected)
    assert_array_equal(lon, lon_expected)
    assert_array_equal(geohash, geohash_expected)
    assert_array_equal(bin_geohash, bin_geohash_expected)


def test_create_geohash_df():
    df_ = DataFrame(
        data=[
            [-3.777736, -38.547792],
            [-3.793388, -38.517722],
            [-3.783605, -38.521962],
            [-3.774056, -38.482056],
            [-3.719155, -38.532494],
        ],
        columns=[LATITUDE, LONGITUDE],
        index=[0, 1, 2, 3, 4]
    )

    expected = DataFrame(
        data=[
            [-3.777736, -38.547792, '7pkddb6356fyzxq'],
            [-3.793388, -38.517722, '7pkd7t2mbj0z1v7'],
            [-3.783605, -38.521962, '7pkd7rjnvhzjp90'],
            [-3.774056, -38.482056, '7pkds2fnx0gr1c0'],
            [-3.719155, -38.532494, '7pkdg4vqrg6020q'],
        ],
        columns=[LATITUDE, LONGITUDE, GEOHASH],
        index=[0, 1, 2, 3, 4]
    )

    geoutils.create_geohash_df(df_)

    assert_frame_equal(df_, expected)


def test_create_bin_geohash_df():
    df_ = DataFrame(
        data=[
            [-3.777736, -38.547792],
            [-3.793388, -38.517722],
            [-3.783605, -38.521962],
        ],
        columns=[LATITUDE, LONGITUDE],
        index=[0, 1, 2]
    )

    expected = DataFrame(
        data=[
            [-3.777736, -38.547792, np.array([0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1,
                                              0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1,
                                              1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1,
                                              1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0,
                                              1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1,
                                              1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1,
                                              1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1,
                                              0])],
            [-3.793388, -38.517722, np.array([0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1,
                                              0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0,
                                              1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0,
                                              1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1,
                                              1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0,
                                              1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1,
                                              1, 1, 1, 1, 1, 0, 0, 1, 1, 1])],
            [-3.783605, -38.521962, np.array([0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1,
                                              0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0,
                                              1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0,
                                              1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
                                              1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1,
                                              1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1,
                                              0, 1, 0, 0, 1, 0, 0, 0, 0, 0])],
        ],
        columns=[LATITUDE, LONGITUDE, BIN_GEOHASH],
        index=[0, 1, 2]
    )

    geoutils.create_bin_geohash_df(df_)

    assert_frame_equal(df_, expected)


def test_decode_geohash_to_latlon():
    df_ = DataFrame(
        data=[
            [-3.777736, -38.547792, '7pkddb6356fyzxq'],
            [-3.793388, -38.517722, '7pkd7t2mbj0z1v7'],
            [-3.783605, -38.521962, '7pkd7rjnvhzjp90'],
            [-3.774056, -38.482056, '7pkds2fnx0gr1c0'],
            [-3.719155, -38.532494, '7pkdg4vqrg6020q'],
        ],
        columns=[LATITUDE, LONGITUDE, GEOHASH],
        index=[0, 1, 2, 3, 4]
    )

    expected = DataFrame(
        data=[
            [-3.777736, -38.547792, '7pkddb6356fyzxq', -3.777736, -38.547792],
            [-3.793388, -38.517722, '7pkd7t2mbj0z1v7', -3.793388, -38.517722],
            [-3.783605, -38.521962, '7pkd7rjnvhzjp90', -3.783605, -38.521962],
            [-3.774056, -38.482056, '7pkds2fnx0gr1c0', -3.774056, -38.482056],
            [-3.719155, -38.532494, '7pkdg4vqrg6020q', -3.719155, -38.532494],
        ],
        columns=[LATITUDE, LONGITUDE, GEOHASH, LATITUDE_DECODE, LONGITUDE_DECODE],
        index=[0, 1, 2, 3, 4]
    )

    geoutils.decode_geohash_to_latlon(df_)

    assert_frame_equal(df_, expected)
