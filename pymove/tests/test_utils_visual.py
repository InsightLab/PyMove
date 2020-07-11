import codecs
import os

from matplotlib.pyplot import cm
from numpy.testing import assert_array_equal, assert_equal

from pymove import MoveDataFrame
from pymove.utils import visual
from pymove.utils.constants import COLORS, DATETIME, LATITUDE, LONGITUDE, TRAJ_ID

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


def test_generate_color():
    color = visual.generate_color()
    assert color in COLORS.values()


def test_rgb():

    expected = (51, 51, 153)

    rgb = visual.rgb([0.6, 0.2, 0.2])

    assert_array_equal(rgb, expected)


def test_hex_rgb():

    expected = '#333399'

    hex_rgb = visual.hex_rgb([0.6, 0.2, 0.2])

    assert_equal(hex_rgb, expected)


def test_cmap_hex_color():
    cm_hex = visual.cmap_hex_color(cm.jet, 0)
    assert cm_hex == '#000080'


def test_get_cmap():
    cmap = visual.get_cmap('tab20')
    assert cmap.N == 20


def test_save_wkt(tmpdir):

    expected = ('id;linestring\n1;'
                'LINESTRING(116.319236 39.984094,'
                '116.319322 39.984198,116.319402 '
                '39.984224,116.319389 39.984211,'
                '116.319422 39.984217)\n')

    move_df = _default_move_df()

    d = tmpdir.mkdir('utils')

    file_write_default = d.join('test_save_map.wkt')
    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    visual.save_wkt(move_data=move_df, filename=filename_write_default)

    file = codecs.open(file_write_default, 'r')

    map_info = file.read()

    assert_equal(map_info, expected)
