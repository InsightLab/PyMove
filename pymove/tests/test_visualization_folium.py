import codecs
import os

from numpy.testing import assert_array_equal
from pandas import DataFrame, Timestamp
from pandas.testing import assert_frame_equal

from pymove import MoveDataFrame
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
    USER_POINT,
    LINE_COLOR,
    POI_POINT,
    EVENT_POINT,
    EVENT_ID,
    UID
)
from pymove.visualization import folium

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


def test_save_map(tmpdir):

    move_df = _default_move_df()

    d = tmpdir.mkdir('visualization')

    file_write_default = d.join('test_save_map.html')
    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    folium.save_map(move_data=move_df, filename=filename_write_default)

    file = codecs.open(file_write_default, 'r')

    map_info = file.read()

    expected = ('[[39.984094, 116.319236], '
                '[39.984198, 116.319322], '
                '[39.984224, 116.319402], '
                '[39.984211, 116.319389], '
                '[39.984217, 116.319422]]')

    assert(expected in map_info)


def test_create_base_map():

    move_df = _default_move_df()

    base_map = folium.create_base_map(move_data=move_df)

    assert_array_equal(base_map.location, [39.984094, 116.319236])

    assert(base_map.control_scale is True)


def test_plot_markers(tmpdir):

    move_df = _default_move_df()

    base_map = folium.plot_markers(move_df, n_rows=3)

    assert_array_equal(base_map.location, [39.984094, 116.319236])

    d = tmpdir.mkdir('visualization')

    file_write_default = d.join('plot_markers.html')
    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    folium.plot_markers(
        move_df,
        save_as_html=True,
        filename=filename_write_default
    )

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

    d = tmpdir.mkdir('visualization')

    file_write_default = d.join('plot_trajectories_with_folium.html')
    filename = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    base_map = folium.plot_trajectories_with_folium(
        move_df,
        n_rows=3,
        save_as_html=True,
        filename=filename
    )

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

    d = tmpdir.mkdir('visualization')

    file_write_default = d.join('plot_trajectory_by_id_folium.html')
    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    base_map = folium.plot_trajectory_by_id_folium(
        move_df,
        id_=1,
        save_as_html=True,
        filename=filename_write_default
    )

    assert_array_equal(base_map.location, [39.984094, 116.319236])

    file = codecs.open(filename_write_default, 'r')

    map_info = file.read()

    _assert_plot(map_info)


def test_plot_trajectory_by_period(tmpdir):

    move_df = _default_move_df()

    d = tmpdir.mkdir('visualization')

    file_write_default = d.join('plot_trajectory_by_period_with_folium.html')
    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    base_map = folium.plot_trajectory_by_period(
        move_df,
        period='Early morning',
        save_as_html=True,
        filename=filename_write_default
    )

    assert_array_equal(base_map.location, [39.984094, 116.319236])

    file = codecs.open(filename_write_default, 'r')

    map_info = file.read()

    _assert_plot(map_info)


def test_plot_trajectory_by_day_week(tmpdir):

    move_df = _default_move_df()

    d = tmpdir.mkdir('visualization')

    file_write_default = d.join('plot_trajectory_by_day_week.html')
    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    base_map = folium.plot_trajectory_by_day_week(
        move_df,
        day_week='Thursday',
        save_as_html=True,
        filename=filename_write_default
    )

    assert_array_equal(base_map.location, [39.984094, 116.319236])

    file = codecs.open(filename_write_default, 'r')

    map_info = file.read()

    _assert_plot(map_info)


def test_plot_trajectory_by_date(tmpdir):

    move_df = _default_move_df()

    d = tmpdir.mkdir('visualization')

    file_write_default = d.join('plot_trajectory_by_date.html')
    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    base_map = folium.plot_trajectory_by_date(
        move_df,
        start_date='2008-10-23',
        end_date='2008-10-23',
        save_as_html=True,
        filename=filename_write_default
    )

    assert_array_equal(base_map.location, [39.984094, 116.319236])

    file = codecs.open(filename_write_default, 'r')

    map_info = file.read()

    _assert_plot(map_info)


def test_plot_trajectory_by_hour(tmpdir):

    move_df = _default_move_df()

    d = tmpdir.mkdir('visualization')

    file_write_default = d.join('plot_trajectory_by_hour.html')
    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    base_map = folium.plot_trajectory_by_hour(
        move_df,
        start_hour=5,
        end_hour=5,
        save_as_html=True,
        filename=filename_write_default
    )

    assert_array_equal(base_map.location, [39.984094, 116.319236])

    file = codecs.open(filename_write_default, 'r')

    map_info = file.read()

    _assert_plot(map_info)


def test_plot_stops(tmpdir):

    move_df = _default_move_df()

    d = tmpdir.mkdir('visualization')

    file_write_default = d.join('plot_stops.html')
    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    base_map = folium.plot_stops(move_df, radius=10)

    assert_array_equal(base_map.location, [39.984094, 116.319236])

    folium.plot_stops(
        move_df,
        radius=10,
        save_as_html=True,
        filename=filename_write_default
    )

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

    d = tmpdir.mkdir('visualization')

    file_write_default = d.join('faster_cluster.html')
    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    base_map = folium.faster_cluster(move_df)

    assert_array_equal(base_map.location, [39.984094, 116.319236])

    folium.faster_cluster(
        move_df,
        save_as_html=True,
        filename=filename_write_default
    )

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

    d = tmpdir.mkdir('visualization')

    file_write_default = d.join('cluster.html')
    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    base_map = folium.cluster(move_df)

    assert_array_equal(base_map.location, [39.984094, 116.319236])

    folium.cluster(
        move_df,
        save_as_html=True,
        filename=filename_write_default
    )

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

    d = tmpdir.mkdir('visualization')

    file_write_default = d.join('heatmap.html')
    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    base_map = folium.heatmap(move_df)

    assert_array_equal(base_map.location, [39.984094, 116.319236])

    folium.heatmap(
        move_df,
        save_as_html=True,
        filename=filename_write_default
    )

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

    d = tmpdir.mkdir('visualization')

    file_write_default = d.join('map.html')
    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    base_map = folium.create_base_map(
        move_data=move_df,
        lat_origin=None,
        lon_origin=None,
        tile=TILES[0],
        default_zoom_start=12
    )

    folium.heatmap(
        move_df,
        save_as_html=True,
        filename=filename_write_default
    )

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

    filtered_df = folium._filter_generated_feature(move_df, 'lat', [39.984198])

    assert_frame_equal(filtered_df, expected_one_value)

    filtered_df = folium._filter_generated_feature(
        move_df, 'lat', [39.984198, 39.984211]
    )

    assert_frame_equal(filtered_df, expected_multiples_value)

    try:
        folium._filter_generated_feature(move_df, 'lat', [33.5659])
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

    mv_df, items = folium._filter_and_generate_colors(move_df, 1, 3)
    print(items, mv_df)
    assert_frame_equal(mv_df, expected_df)

    assert_array_equal(items, expected_items)


def test_add_begin_end_markers_to_folium_map(tmpdir):

    move_df = _default_move_df()

    base_map = folium.create_base_map(
        move_data=move_df,
        lat_origin=None,
        lon_origin=None,
        tile=TILES[0],
        default_zoom_start=12)

    folium._add_begin_end_markers_to_folium_map(move_df, base_map)

    d = tmpdir.mkdir('visualization')

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
    
def test_plot_incial_end_points(tmpdir):
    move_df = _default_move_df()

    base_map = folium.create_base_map(
        move_data=move_df,
        lat_origin=None,
        lon_origin=None,
        tile=TILES[0],
        default_zoom_start=12
    )  

    slice_tags = move_df.columns

    folium.plot_incial_end_points(
        list_rows=list(move_df.iterrows()),
        user_lat='lat',
        user_lon='lon',
        slice_tags=slice_tags,
        base_map=base_map,
        map_=base_map
    )

    d = tmpdir.mkdir('visualization')

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

    assert(('L.marker(\n[39.984094,116.319236]')in map_info
           and ('L.marker(\n[39.984217,116.319422],')in map_info)

def test_add_traj_folium(tmpdir):

    move_df = _default_move_df()

    d = tmpdir.mkdir('visualization')

    file_write_default = d.join('add_traj_folium.html')
    filename = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    base_map = folium.add_traj_folium(
        move_data=move_df,
        user_lat=LATITUDE,
        user_lon=LONGITUDE,
        user_point=USER_POINT,
        line_color=LINE_COLOR,
        user_datetime=DATETIME,
        sort=False,
        base_map=None,
        slice_tags=None,
        tiles=TILES[0]
    )

    assert_array_equal(base_map.location, [move_df['lat'].mean(), move_df['lon'].mean()])

    base_map.save(filename)

    file = codecs.open(filename, 'r')

    map_info = file.read()

    map_info = map_info.replace(' ', '')

    count_l_map = map_info.count('L.map')

    count_l_tileLayer = map_info.count('L.tileLayer')

    count_l_marker = map_info.count('L.marker')

    count_l_popup = map_info.count('L.popup')

    count_l_polyline = map_info.count('L.polyline')

    count_l_circle = map_info.count('L.circle')

    count_head = map_info.count('head')

    count_body = map_info.count('body')

    count_script = map_info.count('script')



    assert(count_l_map == 1
           and count_l_tileLayer == 1
           and count_l_marker == 2
           and count_l_popup == 7
           and count_l_polyline == 1
           and count_l_circle == 5
           and count_head == 2
           and count_body == 3
           and count_script == 12)

    assert('L.marker(\n[39.984094,116.319236]' in map_info 
            and 'L.marker(\n[39.984217,116.319422]' in map_info
            and 'L.polyline(\n[[39.984094,116.319236],'
            '[39.984198,116.319322],[39.984224,116.319402],'
            '[39.984211,116.319389],[39.984217,116.319422]]' in map_info
    )


def test_add_point_folium(tmpdir):
    
    move_df = _default_move_df()

    d = tmpdir.mkdir('visualization')

    file_write_default = d.join('add_point_folium')
    filename = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )   

    base_map = folium.add_point_folium(
        move_data=move_df,
        user_lat=LATITUDE,
        user_lon=LONGITUDE,
        user_point=USER_POINT,
        poi_point=POI_POINT,
        base_map=None,
        slice_tags=None,
        tiles=TILES[0]
    )

    assert_array_equal(base_map.location, [move_df['lat'].mean(), move_df['lon'].mean()])

    base_map.save(filename)

    file = codecs.open(filename, 'r')

    map_info = file.read()

    map_info = map_info.replace(' ','')

    count_l_map = map_info.count('L.map')

    count_l_tileLayer = map_info.count('L.tileLayer')

    count_l_marker = map_info.count('L.marker')

    count_l_popup = map_info.count('L.popup')

    count_l_polyline = map_info.count('L.polyline')

    count_l_circle = map_info.count('L.circle')

    count_head = map_info.count('head')

    count_body = map_info.count('body')

    count_script = map_info.count('script')



    assert(count_l_map == 1
        and count_l_tileLayer == 1
        and count_l_marker == 0
        and count_l_popup == 5
        and count_l_polyline == 0
        and count_l_circle == 5
        and count_head == 2
        and count_body == 3
        and count_script == 12)

    assert('L.circle(\n[39.984094,116.319236]' in map_info
        and 'L.circle(\n[39.984198,116.319322]' in map_info
        and 'L.circle(\n[39.984224,116.319402]' in map_info
        and 'L.circle(\n[39.984211,116.319389]' in map_info
        and 'L.circle(\n[39.984217,116.319422]' in map_info)

def test_add_poi_folium(tmpdir):
    
    move_df = _default_move_df()

    d = tmpdir.mkdir('visualization')

    file_write_default = d.join('add_point_folium')
    filename = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )   

    base_map = folium.add_poi_folium(
        move_data=move_df,
        poi_lat=LATITUDE,
        poi_lon=LONGITUDE,
        poi_point=POI_POINT,
        base_map=None,
        slice_tags=None,
    )

    assert_array_equal(base_map.location, [move_df['lat'].mean(), move_df['lon'].mean()])

    base_map.save(filename)

    file = codecs.open(filename, 'r')

    map_info = file.read()

    map_info = map_info.replace(' ','')

    count_l_map = map_info.count('L.map')

    count_l_tileLayer = map_info.count('L.tileLayer')

    count_l_marker = map_info.count('L.marker')

    count_l_popup = map_info.count('L.popup')

    count_l_polyline = map_info.count('L.polyline')

    count_l_circle = map_info.count('L.circle')

    count_head = map_info.count('head')

    count_body = map_info.count('body')

    count_script = map_info.count('script')



    assert(count_l_map == 1
        and count_l_tileLayer == 1
        and count_l_marker == 0
        and count_l_popup == 5
        and count_l_polyline == 0
        and count_l_circle == 5
        and count_head == 2
        and count_body == 3
        and count_script == 12)

    assert('L.circle(\n[39.984094,116.319236]' in map_info
        and 'L.circle(\n[39.984198,116.319322]' in map_info
        and 'L.circle(\n[39.984224,116.319402]' in map_info
        and 'L.circle(\n[39.984211,116.319389]' in map_info
        and 'L.circle(\n[39.984217,116.319422]' in map_info)

def test_add_event_folium(tmpdir):
    
    move_df = _default_move_df()

    d = tmpdir.mkdir('visualization')

    file_write_default = d.join('add_event_folium')
    filename = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )   

    base_map = folium.add_event_folium(
        move_data=move_df,
        event_lat=LATITUDE,
        event_lon=LONGITUDE,
        event_point=EVENT_POINT,
        radius=15,
        base_map=None,
        slice_tags=None,
        tiles=TILES[0]
    )

    assert_array_equal(base_map.location, [move_df['lat'].mean(), move_df['lon'].mean()])

    base_map.save(filename)

    file = codecs.open(filename, 'r')

    map_info = file.read()

    map_info = map_info.replace(' ','')

    count_l_map = map_info.count('L.map')

    count_l_tileLayer = map_info.count('L.tileLayer')

    count_l_marker = map_info.count('L.marker')

    count_l_popup = map_info.count('L.popup')

    count_l_polyline = map_info.count('L.polyline')

    count_l_circle = map_info.count('L.circle')

    count_head = map_info.count('head')

    count_body = map_info.count('body')

    count_script = map_info.count('script')



    assert(count_l_map == 1
        and count_l_tileLayer == 1
        and count_l_marker == 0
        and count_l_popup == 5
        and count_l_polyline == 0
        and count_l_circle == 5
        and count_head == 2
        and count_body == 3
        and count_script == 12)

    assert('L.circle(\n[39.984094,116.319236]' in map_info
        and 'L.circle(\n[39.984198,116.319322]' in map_info
        and 'L.circle(\n[39.984224,116.319402]' in map_info
        and 'L.circle(\n[39.984211,116.319389]' in map_info
        and 'L.circle(\n[39.984217,116.319422]' in map_info)

def test_show_trajs_with_event(tmpdir):
    
    move_df = _default_move_df()

    df_event = move_df.iloc[0:3,:]

    d = tmpdir.mkdir('visualization')

    file_write_default = d.join('show_trajs_with_event')
    filename = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )   

    list_ = folium.show_trajs_with_event(
        move_data=move_df,
        window_time_subject=4,
        df_event=df_event,
        window_time_event=4,
        radius=150,
        event_lat_=LATITUDE,
        event_lon_=LONGITUDE,
        event_datetime_=DATETIME,
        user_lat=LATITUDE,
        user_lon=LONGITUDE,
        user_datetime=DATETIME,
        event_id_='id',
        event_point=EVENT_POINT,
        user_id='id',
        user_point=USER_POINT,
        line_color=LINE_COLOR,
        slice_event_show=None,
        slice_subject_show=None,
    )

    assert len(list_) == 3, "list with wrong number of elements"
    for i in list_:
        base_map = i[0]
        assert(base_map.control_scale is True)

def test_show_traj_id_with_event(tmpdir):
    
    move_df = _default_move_df()

    df_event = move_df.iloc[0:3,:]

    d = tmpdir.mkdir('visualization')

    file_write_default = d.join('show_traj_with_id_event')
    filename = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )   

    list_ = folium.show_traj_id_with_event(
        move_data=move_df,
        window_time_subject=4,
        subject_id = 1,
        df_event=df_event,
        window_time_event=4,
        radius=150,
        event_lat_=LATITUDE,
        event_lon_=LONGITUDE,
        event_datetime_=DATETIME,
        user_lat=LATITUDE,
        user_lon=LONGITUDE,
        user_datetime=DATETIME,
        event_id_='id',
        event_point=EVENT_POINT,
        user_id='id',
        user_point=USER_POINT,
        line_color=LINE_COLOR,
        slice_event_show=None,
        slice_subject_show=None,
    )

    assert type(list_) == tuple, "Wrong type"
    assert len(list_) == 2, "list with wrong number of elements"
    assert len(list_[1]) == 2
    assert list_[0].control_scale is True
    
def test_create_geojson_features_line(tmpdir):
    
    move_df = _default_move_df()


    d = tmpdir.mkdir('visualization')

    file_write_default = d.join('create_geojson_features_line')
    filename = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )   

    features = folium._create_geojson_features_line(move_df)

    assert len(move_df)-1 == len(features)
    assert [[116.319236, 39.984094], [116.319322, 39.984198]] == features[0]['geometry']['coordinates']
    assert [[116.319322, 39.984198], [116.319402, 39.984224]] == features[1]['geometry']['coordinates']
    assert [[116.319402, 39.984224], [116.319389, 39.984211]] == features[2]['geometry']['coordinates']
    assert [[116.319389, 39.984211], [116.319422, 39.984217]] == features[3]['geometry']['coordinates']

def test_plot_traj_timestamp_geo_json(tmpdir):
    
    move_df = _default_move_df()

    d = tmpdir.mkdir('visualization')

    file_write_default = d.join('test_traj_timestamp_geo_json')
    filename = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )   

    base_map = folium.plot_traj_timestamp_geo_json(move_df)

    base_map.save('map2.html')

    file = codecs.open('map2.html', 'r')

    map_info = file.read()

    map_info = map_info.replace(' ', '')

    count_l_map = map_info.count('L.map')

    count_l_tileLayer = map_info.count('L.tileLayer')

    count_l_marker = map_info.count('L.heatLayer')

    count_l_polyline = map_info.count('L.polyline')

    count_l_popup = map_info.count('popup')

    count_l_circle = map_info.count('L.circle')

    count_head = map_info.count('head')

    count_body = map_info.count('body')

    count_script = map_info.count('script')

    assert(count_l_map == 1
        and count_l_tileLayer == 1
        and count_l_marker == 0
        and count_l_polyline == 0
        and count_l_circle == 2
        and count_l_popup == 6
        and count_head == 2
        and count_body == 3
        and count_script == 22)

def test_heatmap(tmpdir):

    move_df = _default_move_df()

    d = tmpdir.mkdir('visualization')

    file_write_default = d.join('heatmap.html')
    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    base_map = folium.heatmap(move_df)

    assert_array_equal(base_map.location, [39.984094, 116.319236])

    folium.heatmap(
        move_df,
        save_as_html=True,
        filename=filename_write_default
    )

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

def test_heatmap_with_time(tmpdir):

    move_df = _default_move_df()

    d = tmpdir.mkdir('visualization')

    file_write_default = d.join('heatmap_with_time.html')
    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    base_map = folium.heatmap_with_time(move_df)

    assert_array_equal(base_map.location, [39.984094, 116.319236])

    folium.heatmap(
        move_df,
        save_as_html=True,
        filename=filename_write_default
    )

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