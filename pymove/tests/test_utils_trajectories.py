import os
from collections import defaultdict

import numpy as np
from numpy.testing import assert_array_equal, assert_equal
from pandas.testing import assert_frame_equal

from pymove import DaskMoveDataFrame, MoveDataFrame, PandasMoveDataFrame, trajectories
from pymove.utils.constants import (
    DATETIME,
    LATITUDE,
    LONGITUDE,
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

str_data_default = """
lat,lon,datetime,id
39.984094, 116.319236, 2008-10-23 05:53:05, 1
39.984198, 116.319322, 2008-10-23 05:53:06, 1
39.984224, 116.319402, 2008-10-23 05:53:11, 1
39.984211, 116.319389, 2008-10-23 05:53:16, 1
39.984217, 116.319422, 2008-10-23 05:53:21, 1
"""


def _default_move_df():
    return MoveDataFrame(
        data=list_data,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        datetime=DATETIME,
        traj_id=TRAJ_ID,
    )


def test_read_csv(tmpdir):

    expected = _default_move_df()

    d = tmpdir.mkdir('prepossessing')

    file_default_columns = d.join('test_read_default.csv')
    file_default_columns.write(str_data_default)
    filename_default = os.path.join(
        file_default_columns.dirname, file_default_columns.basename
    )

    pandas_move_df = trajectories.read_csv(filename_default, type_=TYPE_PANDAS)

    assert isinstance(pandas_move_df, PandasMoveDataFrame)

    assert_frame_equal(pandas_move_df, expected)

    dask_move_df = trajectories.read_csv(filename_default, type_=TYPE_DASK)

    assert isinstance(dask_move_df, DaskMoveDataFrame)


def test_format_labels():

    expected = {'col1': 'id',
                'col3': 'lon',
                'col2': 'lat',
                'col4': 'datetime'}
    labels = trajectories.format_labels('col1', 'col2', 'col3', 'col4')

    assert_equal(labels, expected)


def test_shift():

    expected = [np.nan, np.nan, np.nan, 1, 2]
    array_ = [1.0, 2.0, 3.0, 4.0, 5.0]
    shifted_array = trajectories.shift(arr=array_, num=3)
    assert_array_equal(shifted_array, expected)

    expected = [4, 5, 0, 0, 0]
    array_ = [1, 2, 3, 4, 5]
    shifted_array = trajectories.shift(arr=array_, num=-3, fill_value=0)
    assert_array_equal(shifted_array, expected)


def test_fill_list_with_new_values():

    exected = [2, 3, 4]
    original_list = [2, 3, 4]
    new_list = []
    trajectories.fill_list_with_new_values(original_list=original_list,
                                           new_list_values=new_list)
    assert_array_equal(original_list, exected)

    exected = [2, 6]
    original_list = []
    new_list = [2, 6]
    trajectories.fill_list_with_new_values(original_list=original_list,
                                           new_list_values=new_list)
    assert_array_equal(original_list, exected)

    exected = [5, 6, 7]
    original_list = [2, 3]
    new_list = [5, 6, 7]
    trajectories.fill_list_with_new_values(original_list=original_list,
                                           new_list_values=new_list)
    assert_array_equal(original_list, exected)


def test_save_bbox(tmpdir):
    d = tmpdir.mkdir('utils')
    file_write_default = d.join('bbox.html')
    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    bbox = (22.147577, 113.54884299999999, 41.132062, 121.156224)
    expected = {
        'cartodbpositron': 1,
        'fit_bounds': 1,
        'poly_line': 1
    }
    m = trajectories.save_bbox(bbox, file=filename_write_default, return_map=True)
    to_dict = m.to_dict(ordered=False)['children']
    actual = defaultdict(int)
    for key in to_dict.keys():
        value = key.split('_')
        if len(value) > 1:
            value = '_'.join(key.split('_')[:-1])
        else:
            value = value[0]
        actual[value] += 1
    actual = dict(actual)
    assert_equal(expected, actual)
