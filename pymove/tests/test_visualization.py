import codecs
import os

from matplotlib.testing.compare import compare_images
from numpy.testing import assert_array_equal, assert_equal
from pandas import DataFrame, Timestamp
from pandas.testing import assert_frame_equal

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
    TILES,
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


def _assert_plot(map_info):

    map_info = map_info.replace(' ', '')

    count_l_map = map_info.count('L.map')

    count_l_tileLayer = map_info.count('L.tileLayer')

    count_l_marker = map_info.count('L.marker')

    count_l_popup = map_info.count('L.popup')

    count_head = map_info.count('head')

    count_body = map_info.count('body')

    count_script = map_info.count('script')

    assert(count_l_map == 1
           and count_l_tileLayer == 1
           and count_l_marker == 2
           and count_l_popup == 2
           and count_head == 4
           and count_body == 5
           and count_script == 18)

    assert('L.marker(\n[39.984094,116.319236]' in map_info
           and 'L.marker(\n[39.984217,116.319422]' in map_info
           and 'center:[39.984094,116.319236]' in map_info
           and ('L.polyline(\n[[39.984094,116.319236],'
                '[39.984198,116.319322],[39.984224,116.319402],'
                '[39.984211,116.319389],[39.984217,116.319422]]') in map_info)


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

    # print(file_write_default)

    test_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(test_dir, 'baseline/shot_points_by_date.png')

    compare_images(data_dir,
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


def test_show_lat_lon_gps(tmpdir):

    move_df = _default_move_df()

    d = tmpdir.mkdir('prepossessing')

    file_write_default = d.join('shot_points_by_date.png')
    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    fig = visualization.show_lat_lon_gps(move_data=move_df,
                                         save_fig=True,
                                         name=filename_write_default)

    test_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(test_dir, 'baseline/shot_points_by_date.png')

    compare_images(data_dir,
                   filename_write_default,
                   0.0001,
                   in_decorator=False)

    assert(fig is not None)

    file_write_default = d.join('shot_points_by_date_line.png')
    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    fig = visualization.show_lat_lon_gps(move_data=move_df,
                                         kind='line',
                                         plot_start_and_end=False,
                                         return_fig=False,
                                         save_fig=True,
                                         name=file_write_default)

    test_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(test_dir, 'baseline/shot_points_by_date_line.png')

    compare_images(data_dir,
                   filename_write_default,
                   0.0001,
                   in_decorator=False)

    assert(fig is None)


def test_create_base_map():

    move_df = _default_move_df()

    base_map = visualization.create_base_map(move_data=move_df)

    assert_array_equal(base_map.location, [39.984094, 116.319236])

    assert(base_map.control_scale is True)


def test_plot_markers(tmpdir):

    move_df = _default_move_df()

    base_map = visualization.plot_markers(move_df, n_rows=3)

    assert_array_equal(base_map.location, [39.984094, 116.319236])

    d = tmpdir.mkdir('prepossessing')

    file_write_default = d.join('plot_markers.html')
    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    visualization.plot_markers(move_df,
                               save_as_html=True,
                               filename=filename_write_default)

    file = codecs.open(filename_write_default, 'r')

    map_info = file.read()

    map_info = map_info.replace(' ', '')

    count_l_map = map_info.count('L.map')

    count_l_tileLayer = map_info.count('L.tileLayer')

    count_l_marker = map_info.count('L.marker')

    count_l_popup = map_info.count('L.popup')

    count_head = map_info.count('head')

    count_body = map_info.count('body')

    count_script = map_info.count('script')

    assert(count_l_map == 1
           and count_l_tileLayer == 1
           and count_l_marker == 5
           and count_l_popup == 5
           and count_head == 2
           and count_body == 3
           and count_script == 12)

    assert('L.marker(\n[39.984094,116.319236]' in map_info
           and 'L.marker(\n[39.984217,116.319422]' in map_info
           and 'L.marker(\n[39.984198,116.319322]' in map_info
           and 'L.marker(\n[39.984224,116.319402]' in map_info
           and 'L.marker(\n[39.984211,116.319389]' in map_info)


def test_plot_trajectories_with_folium(tmpdir):

    move_df = _default_move_df()

    d = tmpdir.mkdir('prepossessing')

    file_write_default = d.join('plot_trajectories_with_folium.html')
    filename = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    base_map = visualization.plot_trajectories_with_folium(move_df,
                                                           n_rows=3,
                                                           save_as_html=True,
                                                           filename=filename)

    assert_array_equal(base_map.location, [39.984094, 116.319236])

    file = codecs.open(filename, 'r')

    map_info = file.read()

    map_info = map_info.replace(' ', '')

    count_l_map = map_info.count('L.map')

    count_l_tileLayer = map_info.count('L.tileLayer')

    count_l_marker = map_info.count('L.marker')

    count_l_popup = map_info.count('L.popup')

    count_head = map_info.count('head')

    count_body = map_info.count('body')

    count_script = map_info.count('script')

    assert(count_l_map == 1
           and count_l_tileLayer == 1
           and count_l_marker == 2
           and count_l_popup == 2
           and count_head == 4
           and count_body == 5
           and count_script == 18)

    assert('L.marker(\n[39.984094,116.319236]' in map_info
           and 'L.marker(\n[39.984224,116.319402]' in map_info
           and 'center:[39.984094,116.319236]' in map_info
           and ('L.polyline(\n[[39.984094,116.319236],'
                '[39.984198,116.319322],[39.984224,116.319402]]') in map_info)


def test_plot_trajectory_by_id_folium(tmpdir):

    move_df = _default_move_df()

    d = tmpdir.mkdir('prepossessing')

    file_write_default = d.join('plot_trajectory_by_id_folium.html')
    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    base_map = visualization.plot_trajectory_by_id_folium(move_df,
                                                          id_=1,
                                                          save_as_html=True,
                                                          filename=filename_write_default)

    assert_array_equal(base_map.location, [39.984094, 116.319236])

    file = codecs.open(filename_write_default, 'r')

    map_info = file.read()

    _assert_plot(map_info)


def test_plot_trajectory_by_period(tmpdir):

    move_df = _default_move_df()

    d = tmpdir.mkdir('prepossessing')

    file_write_default = d.join('plot_trajectory_by_period_with_folium.html')
    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    base_map = visualization.plot_trajectory_by_period(move_df,
                                                       period='Early morning',
                                                       save_as_html=True,
                                                       filename=filename_write_default)

    assert_array_equal(base_map.location, [39.984094, 116.319236])

    file = codecs.open(filename_write_default, 'r')

    map_info = file.read()

    _assert_plot(map_info)


def test_plot_trajectory_by_day_week(tmpdir):

    move_df = _default_move_df()

    d = tmpdir.mkdir('prepossessing')

    file_write_default = d.join('plot_trajectory_by_day_week.html')
    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    base_map = visualization.plot_trajectory_by_day_week(move_df,
                                                         day_week='Thursday',
                                                         save_as_html=True,
                                                         filename=filename_write_default)

    assert_array_equal(base_map.location, [39.984094, 116.319236])

    file = codecs.open(filename_write_default, 'r')

    map_info = file.read()

    _assert_plot(map_info)


def test_plot_trajectory_by_date(tmpdir):

    move_df = _default_move_df()

    d = tmpdir.mkdir('prepossessing')

    file_write_default = d.join('plot_trajectory_by_date.html')
    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    base_map = visualization.plot_trajectory_by_date(move_df,
                                                     start_date='2008-10-23',
                                                     end_date='2008-10-23',
                                                     save_as_html=True,
                                                     filename=filename_write_default)

    assert_array_equal(base_map.location, [39.984094, 116.319236])

    file = codecs.open(filename_write_default, 'r')

    map_info = file.read()

    _assert_plot(map_info)


def test_plot_trajectory_by_hour(tmpdir):

    move_df = _default_move_df()

    d = tmpdir.mkdir('prepossessing')

    file_write_default = d.join('plot_trajectory_by_hour.html')
    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    base_map = visualization.plot_trajectory_by_hour(move_df,
                                                     start_hour=5,
                                                     end_hour=5,
                                                     save_as_html=True,
                                                     filename=filename_write_default)

    assert_array_equal(base_map.location, [39.984094, 116.319236])

    file = codecs.open(filename_write_default, 'r')

    map_info = file.read()

    _assert_plot(map_info)


def test_plot_stops(tmpdir):

    move_df = _default_move_df()

    d = tmpdir.mkdir('prepossessing')

    file_write_default = d.join('plot_stops.html')
    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    base_map = visualization.plot_stops(move_df, radius=10)

    assert_array_equal(base_map.location, [39.984094, 116.319236])

    visualization.plot_stops(move_df,
                             radius=10,
                             save_as_html=True,
                             filename=filename_write_default)

    file = codecs.open(filename_write_default, 'r')

    map_info = file.read()

    map_info = map_info.replace(' ', '')

    count_l_map = map_info.count('L.map')

    count_l_tileLayer = map_info.count('L.tileLayer')

    count_l_marker = map_info.count('L.circle')

    count_l_popup = map_info.count('L.popup')

    count_head = map_info.count('head')

    count_body = map_info.count('body')

    count_script = map_info.count('script')

    assert(count_l_map == 1
           and count_l_tileLayer == 1
           and count_l_marker == 3
           and count_l_popup == 3
           and count_head == 4
           and count_body == 5
           and count_script == 18)

    assert('L.circle(\n[39.984224,116.319402]' in map_info
           and 'L.circle(\n[39.984211,116.319389]' in map_info
           and 'L.circle(\n[39.984217,116.319422]' in map_info
           and 'center:[39.984094,116.319236]' in map_info)


def test_faster_cluster(tmpdir):

    move_df = _default_move_df()

    d = tmpdir.mkdir('prepossessing')

    file_write_default = d.join('faster_cluster.html')
    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    base_map = visualization.faster_cluster(move_df)

    assert_array_equal(base_map.location, [39.984094, 116.319236])

    visualization.faster_cluster(move_df,
                                 save_as_html=True,
                                 filename=filename_write_default)

    file = codecs.open(filename_write_default, 'r')

    map_info = file.read()

    map_info = map_info.replace(' ', '')

    count_l_map = map_info.count('L.map')

    count_l_tileLayer = map_info.count('L.tileLayer')

    count_l_marker = map_info.count('L.circle')

    count_head = map_info.count('head')

    count_body = map_info.count('body')

    count_script = map_info.count('script')

    assert(count_l_map == 1
           and count_l_tileLayer == 1
           and count_l_marker == 1
           and count_head == 2
           and count_body == 3
           and count_script == 14)

    assert(('data=[[39.984094,116.319236],'
            '[39.984198,116.319322],'
            '[39.984224,116.319402],'
            '[39.984211,116.319389],'
            '[39.984217,116.319422]]') in map_info)


def test_cluster(tmpdir):

    move_df = _default_move_df()

    d = tmpdir.mkdir('prepossessing')

    file_write_default = d.join('cluster.html')
    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    base_map = visualization.cluster(move_df)

    assert_array_equal(base_map.location, [39.984094, 116.319236])

    visualization.cluster(move_df,
                          save_as_html=True,
                          filename=filename_write_default)

    file = codecs.open(filename_write_default, 'r')

    map_info = file.read()

    map_info = map_info.replace(' ', '')

    count_l_map = map_info.count('L.map')

    count_l_tileLayer = map_info.count('L.tileLayer')

    count_l_marker = map_info.count('L.marker')

    count_l_popup = map_info.count('L.popup')

    count_head = map_info.count('head')

    count_body = map_info.count('body')

    count_script = map_info.count('script')

    assert(count_l_map == 1
           and count_l_tileLayer == 1
           and count_l_marker == 6
           and count_l_popup == 5
           and count_head == 2
           and count_body == 3
           and count_script == 14)

    assert('L.marker(\n[39.984094,116.319236]' in map_info
           and 'L.marker(\n[39.984198,116.319322]' in map_info
           and 'L.marker(\n[39.984224,116.319402]' in map_info
           and 'L.marker(\n[39.984211,116.319389]' in map_info
           and 'L.marker(\n[39.984217,116.319422]' in map_info)


def test_heatmap(tmpdir):

    move_df = _default_move_df()

    d = tmpdir.mkdir('prepossessing')

    file_write_default = d.join('heatmap.html')
    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    base_map = visualization.heatmap(move_df)

    assert_array_equal(base_map.location, [39.984094, 116.319236])

    visualization.heatmap(move_df,
                          save_as_html=True,
                          filename=filename_write_default)

    file = codecs.open(filename_write_default, 'r')

    map_info = file.read()

    map_info = map_info.replace(' ', '')

    count_l_map = map_info.count('L.map')

    count_l_tileLayer = map_info.count('L.tileLayer')

    count_l_marker = map_info.count('L.heatLayer')

    count_head = map_info.count('head')

    count_body = map_info.count('body')

    count_script = map_info.count('script')

    assert(count_l_map == 1
           and count_l_tileLayer == 1
           and count_l_marker == 1
           and count_head == 2
           and count_body == 3
           and count_script == 14)

    assert(('L.heatLayer(\n[[39.984094,116.319236,1.0],'
            '[39.984198,116.319322,1.0],'
            '[39.984211,116.319389,1.0],'
            '[39.984217,116.319422,1.0],'
            '[39.984224,116.319402,1.0]]')in map_info)


def test_add_trajectories_to_folium_map(tmpdir):
    move_df = _default_move_df()

    d = tmpdir.mkdir('prepossessing')

    file_write_default = d.join('map.html')
    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    base_map = visualization.create_base_map(
        move_data=move_df,
        lat_origin=None,
        lon_origin=None,
        tile=TILES[0],
        default_zoom_start=12)

    visualization.heatmap(move_df,
                          save_as_html=True,
                          filename=filename_write_default)

    file = codecs.open(filename_write_default, 'r')

    map_info = file.read()

    map_info = map_info.replace(' ', '')

    count_l_map = map_info.count('L.map')

    count_l_tileLayer = map_info.count('L.tileLayer')

    count_l_marker = map_info.count('L.heatLayer')

    count_head = map_info.count('head')

    count_body = map_info.count('body')

    count_script = map_info.count('script')

    assert(count_l_map == 1
           and count_l_tileLayer == 1
           and count_l_marker == 1
           and count_head == 2
           and count_body == 3
           and count_script == 14)

    assert(('L.heatLayer(\n[[39.984094,116.319236,1.0],'
            '[39.984198,116.319322,1.0],'
            '[39.984211,116.319389,1.0],'
            '[39.984217,116.319422,1.0],'
            '[39.984224,116.319402,1.0]]')in map_info)


def test_filter_generated_feature():

    expected_one_value = DataFrame(
        data=[
            [39.984198, 116.319322, Timestamp('2008-10-23 05:53:06'), 1],
        ],
        columns=['lat', 'lon', 'datetime', 'id'],
        index=[1],
    )

    expected_multiples_value = DataFrame(
        data=[
            [39.984198, 116.319322, Timestamp('2008-10-23 05:53:06'), 1],
            [39.984211, 116.319389, Timestamp('2008-10-23 05:53:16'), 1],
        ],
        columns=['lat', 'lon', 'datetime', 'id'],
        index=[1, 3],
    )

    move_df = _default_move_df()

    filtered_df = visualization._filter_generated_feature(move_df, 'lat', [39.984198])

    assert_frame_equal(filtered_df, expected_one_value)

    filtered_df = visualization._filter_generated_feature(move_df, 'lat',
                                                          [39.984198, 39.984211])

    assert_frame_equal(filtered_df, expected_multiples_value)

    try:
        visualization._filter_generated_feature(move_df, 'lat', [33.5659])
        raise AssertionError(
            'KeyError error not raised by MoveDataFrame'
        )
    except KeyError:
        pass


def test_filter_and_generate_colors():

    move_df = _default_move_df()

    expected_df = DataFrame(
        data=[
            [39.984094, 116.319236, Timestamp('2008-10-23 05:53:05'), 1],
            [39.984198, 116.319322, Timestamp('2008-10-23 05:53:06'), 1],
            [39.984224, 116.319402, Timestamp('2008-10-23 05:53:11'), 1]
        ],
        columns=['lat', 'lon', 'datetime', 'id'],
        index=[0, 1, 2],
    )

    expected_items = [(1, 'black')]

    mv_df, items = visualization._filter_and_generate_colors(move_df, 1, 3)

    assert_frame_equal(mv_df, expected_df)

    assert_array_equal(items, expected_items)


def test_add_begin_end_markers_to_folium_map(tmpdir):

    move_df = _default_move_df()

    base_map = visualization.create_base_map(
        move_data=move_df,
        lat_origin=None,
        lon_origin=None,
        tile=TILES[0],
        default_zoom_start=12)

    visualization._add_begin_end_markers_to_folium_map(move_df, base_map)

    d = tmpdir.mkdir('prepossessing')

    file_write_default = d.join('base_map_color.html')
    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    base_map.save(filename_write_default)

    file = codecs.open(filename_write_default, 'r')

    map_info = file.read()

    map_info = map_info.replace(' ', '')

    count_l_map = map_info.count('L.map')

    count_l_tileLayer = map_info.count('L.tileLayer')

    count_l_marker = map_info.count('L.marker')

    count_l_popup = map_info.count('L.popup')

    count_head = map_info.count('head')

    count_body = map_info.count('body')

    count_script = map_info.count('script')

    assert(count_l_map == 1
           and count_l_tileLayer == 1
           and count_l_marker == 2
           and count_l_popup == 2
           and count_head == 2
           and count_body == 3
           and count_script == 12)

    assert(('L.marker(\n[39.984094,116.319236],'
            '\n{"clusteredMarker":true,"color":"green"}')in map_info
           and ('L.marker(\n[39.984217,116.319422],'
                '\n{"clusteredMarker":true,"color":"red"}')in map_info)
