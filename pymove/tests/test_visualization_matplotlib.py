import os

from matplotlib.testing.compare import compare_images

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
        create_features=False,
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

    assert(HOUR not in move_df)
    assert(DATE not in move_df)
    assert(PERIOD not in move_df)
    assert(DAY not in move_df)

    mpl.show_object_id_by_date(
        move_data=move_df,
        create_features=True,
        name=filename_write_default
    )

    assert(DATE in move_df)
    assert(HOUR in move_df)
    assert(PERIOD in move_df)
    assert(DAY in move_df)


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
