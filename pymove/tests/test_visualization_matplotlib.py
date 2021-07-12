import os

import matplotlib.pyplot as plt
from matplotlib.testing.compare import compare_images
from shapely.geometry import LineString

import pymove.visualization.matplotlib as mpl
from pymove import MoveDataFrame
from pymove.utils.constants import (
    DATE,
    DATETIME,
    DAY,
    HOUR,
    LATITUDE,
    LONGITUDE,
    PERIOD,
    TID,
    TRAJ_ID,
)

list_data = [
    [39.984094, 116.319236, '2008-10-23 05:53:05', 1],
    [39.984198, 116.319322, '2008-10-23 05:53:06', 1],
    [39.984224, 116.319402, '2008-10-23 05:53:11', 1],
    [39.984211, 116.319389, '2008-10-23 05:53:16', 1],
    [39.984217, 116.319422, '2008-10-23 05:53:21', 1],
]


def _default_move_df():
    return MoveDataFrame(
        data=list_data,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        datetime=DATETIME,
        traj_id=TRAJ_ID,
    )


def test_show_object_id_by_date(tmpdir):

    move_df = _default_move_df()

    d = tmpdir.mkdir('visualization')

    file_write_default = d.join('shot_points_by_date.png')
    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    mpl.show_object_id_by_date(
        move_data=move_df,
        name=filename_write_default
    )

    test_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(test_dir, 'baseline/shot_points_by_date.png')

    compare_images(
        data_dir,
        filename_write_default,
        0.0001,
        in_decorator=False
    )


def test_plot_traj_by_id(tmpdir):
    move_df = _default_move_df()
    move_df[TID] = ['1', '1', '2', '2', '2']

    d = tmpdir.mkdir('visualization')

    file_write_default = d.join('traj_id.png')
    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    mpl.plot_traj_by_id(move_df, '1', save_fig=True, name=filename_write_default)

    test_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(test_dir, 'baseline/traj_id.png')

    compare_images(
        data_dir,
        filename_write_default,
        0.0001,
        in_decorator=False
    )


def test_plot_all_features(tmpdir):

    move_df = _default_move_df()

    d = tmpdir.mkdir('visualization')

    file_write_default = d.join('features.png')
    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    mpl.plot_all_features(move_df, save_fig=True, name=filename_write_default)

    test_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(test_dir, 'baseline/features.png')

    compare_images(data_dir,
                   filename_write_default,
                   0.0001,
                   in_decorator=False)

    move_df['lat'] = move_df['lat'].astype('str')
    move_df['lon'] = move_df['lon'].astype('str')

    try:
        move_df.plot_all_features(name=filename_write_default)
        raise AssertionError(
            'AttributeError error not raised by MoveDataFrame'
        )
    except AttributeError:
        pass


def test_plot_trajectories(tmpdir):

    move_df = _default_move_df()

    d = tmpdir.mkdir('visualization')

    file_write_default = d.join('trajectories.png')
    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    mpl.plot_trajectories(move_df, save_fig=True, name=filename_write_default)

    test_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(test_dir, 'baseline/trajectories.png')

    compare_images(data_dir,
                   filename_write_default,
                   0.0001,
                   in_decorator=False)


def test_plot_coords(tmpdir):
    d = tmpdir.mkdir('visualization')

    file_write_default = d.join('plot_coords.png')

    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    coords = LineString([(1, 1), (1, 2), (2, 2), (2, 3)])

    _, ax = plt.subplots(figsize=(21, 9))
    mpl.plot_coords(ax, coords)
    plt.savefig(filename_write_default, dpi=100)

    test_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(test_dir, 'baseline/plot_coords.png')

    compare_images(
        data_dir, filename_write_default, 0.0001, in_decorator=False
    )


def test_plot_bounds(tmpdir):
    d = tmpdir.mkdir('visualization')

    file_write_default = d.join('plot_bounds.png')

    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    bounds = LineString([(1, 1), (1, 2), (2, 2), (2, 3)])

    _, ax = plt.subplots(figsize=(21, 9))
    mpl.plot_bounds(ax, bounds)
    plt.savefig(filename_write_default, dpi=100)

    test_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(test_dir, 'baseline/plot_bounds.png')

    compare_images(
        data_dir, filename_write_default, 0.0001, in_decorator=False
    )


def test_plot_line(tmpdir):
    d = tmpdir.mkdir('visualization')

    file_write_default = d.join('plot_line.png')

    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    line = LineString([(1, 1), (1, 2), (2, 2), (2, 3)])

    _, ax = plt.subplots(figsize=(21, 9))
    mpl.plot_line(ax, line)
    plt.savefig(filename_write_default, dpi=100)

    test_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(test_dir, 'baseline/plot_line.png')

    compare_images(
        data_dir, filename_write_default, 0.0001, in_decorator=False
    )
