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


def test_show_lat_lon_gps(tmpdir):

    move_df = _default_move_df()

    d = tmpdir.mkdir('visualization')

    file_write_default = d.join('shot_points_by_date.png')
    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    fig = mpl.show_lat_lon_gps(
        move_data=move_df,
        save_fig=True,
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

    assert(fig is not None)

    file_write_default = d.join('shot_points_by_date_line.png')
    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    fig = mpl.show_lat_lon_gps(
        move_data=move_df,
        kind='line',
        plot_start_and_end=False,
        return_fig=False,
        save_fig=True,
        name=file_write_default
    )

    test_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(test_dir, 'baseline/shot_points_by_date_line.png')

    compare_images(
        data_dir,
        filename_write_default,
        0.0001,
        in_decorator=False
    )

    assert(fig is None)
