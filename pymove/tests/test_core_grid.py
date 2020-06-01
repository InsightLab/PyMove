import os

import joblib
from numpy import array
from numpy.testing import assert_array_equal, assert_equal
from pandas import DataFrame, Timestamp
from pandas.testing import assert_frame_equal
from shapely.geometry import Polygon

from pymove import MoveDataFrame
from pymove.core.grid import Grid
from pymove.utils.constants import DATETIME, LATITUDE, LONGITUDE, TRAJ_ID

list_data = [
    [39.984094, 116.319236, '2008-10-23 05:53:05', 1],
    [39.984198, 116.319322, '2008-10-23 05:53:06', 1],
    [39.984224, 116.319402, '2008-10-23 05:53:11', 1],
    [39.984211, 116.319389, '2008-10-23 05:53:16', 1],
    [39.984217, 116.319422, '2008-10-23 05:53:21', 1],
    [39.984710, 116.319865, '2008-10-23 05:53:23', 1],
    [39.984674, 116.319810, '2008-10-23 05:53:28', 1],
    [39.984623, 116.319773, '2008-10-23 05:53:33', 1],
    [39.984606, 116.319732, '2008-10-23 05:53:38', 1],
    [39.984555, 116.319728, '2008-10-23 05:53:43', 1]
]


unsorted_list_data = [
    [39.984198, 116.319322, '2008-10-23 05:53:06', 1],
    [39.984094, 116.319236, '2008-10-23 05:53:05', 1],
    [39.984224, 116.319402, '2008-10-23 05:53:11', 1],
    [39.984710, 116.319865, '2008-10-23 05:53:23', 1],
    [39.984674, 116.319810, '2008-10-23 05:53:28', 1],
    [39.984623, 116.319773, '2008-10-23 05:53:33', 1],
    [39.984606, 116.319732, '2008-10-23 05:53:38', 1],
    [39.984555, 116.319728, '2008-10-23 05:53:43', 1],
    [39.984217, 116.319422, '2008-10-23 05:53:21', 1],
    [39.984211, 116.319389, '2008-10-23 05:53:16', 1]
]


def _default_move_df(data=None):
    if data is None:
        data = list_data
    return MoveDataFrame(
        data=data,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        datetime=DATETIME,
        traj_id=TRAJ_ID,
    )


def _default_grid():
    return Grid(data=_default_move_df(), cell_size=15)


def test_get_grid():

    grid = _default_grid().get_grid()

    expected = {'lon_min_x': 116.319236,
                'lat_min_y': 39.984094,
                'grid_size_lat_y': 5,
                'grid_size_lon_x': 5,
                'cell_size_by_degree': 0.0001353464801860623
                }
    assert_equal(grid, expected)


def test_create_update_index_grid_feature():

    move_df = _default_move_df()
    grid = Grid(move_df, 15)
    grid.create_update_index_grid_feature(data=move_df)

    # Testing function with sorted data
    expected = DataFrame(
        data=[
            [39.984094, 116.319236, Timestamp('2008-10-23 05:53:05'), 1, 0, 0],
            [39.984198, 116.319322, Timestamp('2008-10-23 05:53:06'), 1, 0, 0],
            [39.984224, 116.319402, Timestamp('2008-10-23 05:53:11'), 1, 0, 1],
            [39.984211, 116.319389, Timestamp('2008-10-23 05:53:16'), 1, 0, 1],
            [39.984217, 116.319422, Timestamp('2008-10-23 05:53:21'), 1, 0, 1],
            [39.984710, 116.319865, Timestamp('2008-10-23 05:53:23'), 1, 4, 4],
            [39.984674, 116.319810, Timestamp('2008-10-23 05:53:28'), 1, 4, 4],
            [39.984623, 116.319773, Timestamp('2008-10-23 05:53:33'), 1, 3, 3],
            [39.984606, 116.319732, Timestamp('2008-10-23 05:53:38'), 1, 3, 3],
            [39.984555, 116.319728, Timestamp('2008-10-23 05:53:43'), 1, 3, 3]
        ],
        columns=['lat', 'lon', 'datetime', 'id', 'index_grid_lat', 'index_grid_lon'],
        index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    )

    assert_frame_equal(move_df, expected)

    # Testing function with unsorted data
    unsorted_move_df = _default_move_df(unsorted_list_data)

    grid = Grid(move_df, 15)
    grid.create_update_index_grid_feature(data=unsorted_move_df)

    expected = DataFrame(
        data=[
            [39.984094, 116.319236, Timestamp('2008-10-23 05:53:05'), 1, 0, 0],
            [39.984198, 116.319322, Timestamp('2008-10-23 05:53:06'), 1, 0, 0],
            [39.984224, 116.319402, Timestamp('2008-10-23 05:53:11'), 1, 0, 1],
            [39.984211, 116.319389, Timestamp('2008-10-23 05:53:16'), 1, 0, 1],
            [39.984217, 116.319422, Timestamp('2008-10-23 05:53:21'), 1, 0, 1],
            [39.984710, 116.319865, Timestamp('2008-10-23 05:53:23'), 1, 4, 4],
            [39.984674, 116.319810, Timestamp('2008-10-23 05:53:28'), 1, 4, 4],
            [39.984623, 116.319773, Timestamp('2008-10-23 05:53:33'), 1, 3, 3],
            [39.984606, 116.319732, Timestamp('2008-10-23 05:53:38'), 1, 3, 3],
            [39.984555, 116.319728, Timestamp('2008-10-23 05:53:43'), 1, 3, 3]
        ],
        columns=['lat', 'lon', 'datetime', 'id', 'index_grid_lat', 'index_grid_lon'],
        index=[1, 0, 2, 9, 8, 3, 4, 5, 6, 7],
    )
    assert_frame_equal(unsorted_move_df, expected)


def test_create_one_polygon_to_point_on_grid():
    expected = [[39.984094, 116.3193713464802],
                [39.984229346480184, 116.3193713464802],
                [39.984229346480184, 116.31950669296039],
                [39.984094, 116.31950669296039],
                [39.984094, 116.3193713464802]]

    grid = _default_grid()

    polygon = grid.create_one_polygon_to_point_on_grid(index_grid_lat=0, index_grid_lon=1)

    polygon_coordinates = array(polygon.exterior.coords)

    assert_array_equal(polygon_coordinates, expected)


def test_create_all_polygons_to_all_point_on_grid():
    expected = DataFrame(
        data=[
            [1, 0, 0, Polygon(((39.984094, 116.319236),
                               (39.984229346480184, 116.319236),
                               (39.984229346480184, 116.3193713464802),
                               (39.984094, 116.3193713464802),
                               (39.984094, 116.319236)))],
            [1, 0, 1, Polygon(((39.984094, 116.319236),
                               (39.984229346480184, 116.319236),
                               (39.984229346480184, 116.3193713464802),
                               (39.984094, 116.3193713464802),
                               (39.984094, 116.319236)))],
            [1, 4, 4, Polygon(((39.984094, 116.3193713464802),
                               (39.984229346480184, 116.3193713464802),
                               (39.984229346480184, 116.31950669296039),
                               (39.984094, 116.31950669296039),
                               (39.984094, 116.3193713464802)))],
            [1, 3, 3, Polygon(((39.984094, 116.3193713464802),
                               (39.984229346480184, 116.3193713464802),
                               (39.984229346480184, 116.31950669296039),
                               (39.984094, 116.31950669296039),
                               (39.984094, 116.3193713464802)))],
        ],
        columns=['id', 'index_grid_lat', 'index_grid_lon', 'polygon'],
        index=[0, 2, 5, 7],
    )
    move_df = _default_move_df()
    grid = Grid(move_df, 15)

    all_polygon = grid.create_all_polygons_to_all_point_on_grid(move_df)

    assert_frame_equal(all_polygon, expected)


def test_point_to_index_grid():
    grid = _default_grid()

    index_lat, index_lon = grid.point_to_index_grid(event_lat=39.984217,
                                                    event_lon=116.319422)

    assert(index_lat == 0.0)

    assert(index_lon == 1.0)


def test_save_grid_pkl(tmpdir):
    expected = {'lon_min_x': 116.319236,
                'lat_min_y': 39.984094,
                'grid_size_lat_y': 5,
                'grid_size_lon_x': 5,
                'cell_size_by_degree': 0.0001353464801860623
                }
    d = tmpdir.mkdir('prepossessing')

    file_write_default = d.join('test_save_grid.pkl')
    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    grid = _default_grid()

    grid.save_grid_pkl(filename_write_default)

    saved_grid = grid.read_grid_pkl(filename_write_default)

    assert_equal(saved_grid, expected)


def read_grid_pkl(tmpdir):
    expected = {'lon_min_x': 116.319236,
                'lat_min_y': 39.984094,
                'grid_size_lat_y': 5,
                'grid_size_lon_x': 5,
                'cell_size_by_degree': 0.0001353464801860623
                }
    d = tmpdir.mkdir('prepossessing')

    file_write_default = d.join('test_read_grid.pkl')
    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    grid = _default_grid()

    with open(filename_write_default, 'wb') as f:
        joblib.dump(grid.get_grid(), f)

    saved_grid = grid.read_grid_pkl(filename_write_default)

    assert_equal(saved_grid, expected)
