import codecs
import os

from matplotlib.testing.compare import compare_images
from numpy.testing import assert_array_equal, assert_equal

from pymove import (
    DaskMoveDataFrame,
    MoveDataFrame,
    PandasMoveDataFrame,
    trajectories,
    visualization,
)
from pymove.utils.constants import (
    DATE,
    DATETIME,
    DAY,
    HOUR,
    LATITUDE,
    LONGITUDE,
    PERIOD,
    TRAJ_ID,
    TYPE_DASK,
    TYPE_PANDAS,
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


def test_rgb():

    expected = (51, 51, 153)

    rgb = visualization.rgb([0.6, 0.2, 0.2])

    assert_array_equal(rgb, expected)


def test_hex_rgb():

    expected = '#333399'

    hex_rgb = visualization.hex_rgb([0.6, 0.2, 0.2])

    assert_equal(hex_rgb, expected)


def test_save_map(tmpdir):

    move_df = _default_move_df()

    d = tmpdir.mkdir('prepossessing')

    file_write_default = d.join('test_save_map.html')
    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    visualization.save_map(move_data=move_df, filename=filename_write_default)

    file = codecs.open(file_write_default, 'r')

    map_info = file.read()

    expected = ('[[39.984094, 116.319236], '
                '[39.984198, 116.319322], '
                '[39.984224, 116.319402], '
                '[39.984211, 116.319389], '
                '[39.984217, 116.319422]]')

    assert(expected in map_info)


def test_save_wkt(tmpdir):

    expected = ('id;linestring\n1;'
                'LINESTRING(116.319236 39.984094,'
                '116.319322 39.984198,116.319402 '
                '39.984224,116.319389 39.984211,'
                '116.319422 39.984217)\n')

    move_df = _default_move_df()

    d = tmpdir.mkdir('prepossessing')

    file_write_default = d.join('test_save_map.wkt')
    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    visualization.save_wkt(move_data=move_df, filename=filename_write_default)

    file = codecs.open(file_write_default, 'r')

    map_info = file.read()

    assert_equal(map_info, expected)


def test_show_object_id_by_date(tmpdir):

    move_df = _default_move_df()

    d = tmpdir.mkdir('prepossessing')

    file_write_default = d.join('shot_points_by_date.png')
    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    visualization.show_object_id_by_date(move_data=move_df,
                                         create_features=False,
                                         name=filename_write_default)

    compare_images('./baseline/shot_points_by_date.png',
                   filename_write_default,
                   0.0001,
                   in_decorator=False)

    assert(HOUR not in move_df)
    assert(DATE not in move_df)
    assert(PERIOD not in move_df)
    assert(DAY not in move_df)

    visualization.show_object_id_by_date(move_data=move_df,
                                         create_features=True,
                                         name=filename_write_default)

    assert(DATE in move_df)
    assert(HOUR in move_df)
    assert(PERIOD in move_df)
    assert(DAY in move_df)
