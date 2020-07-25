import os
from collections import defaultdict

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_equal
from pandas import DataFrame
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


def test_flatten_dict():
    d = {'a': 1, 'b': {'c': 2, 'd': 3}}
    expected = {'a': 1, 'b_c': 2, 'b_d': 3}
    actual = trajectories.flatten_dict(d)
    assert_equal(actual, expected)


def test_flatten_columns():
    d = {'a': 1, 'b': {'c': 2, 'd': 3}}
    df = DataFrame({'col1': [0], 'col2': [d]})
    expected = DataFrame({
        'col1': [0],
        'col2_a': [1],
        'col2_b_c': [2],
        'col2_b_d': [3],
    })
    actual = trajectories.flatten_columns(df, ['col2'])
    actual = actual[sorted(actual.columns)]
    assert_frame_equal(actual, expected)


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


def test_object_for_array():
    data_1 = '[1, 2, 3]'
    data_2 = '[1.5, 2.5, 3.5]'
    data_3 = '[event, event]'

    expected_data_1 = np.array([1., 2., 3.], dtype=np.float32)
    expected_data_2 = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    expected_data_3 = np.array(['event', 'event'], dtype='object_')

    assert_array_almost_equal(trajectories.object_for_array(data_1), expected_data_1)
    assert_array_almost_equal(trajectories.object_for_array(data_2), expected_data_2)
    assert_array_equal(trajectories.object_for_array(data_3), expected_data_3)


def test_column_to_array():
    list_data_1 = [
        '[1, 2, 3]',
        '[5, 8]',
        '[13, 21, 34, 55]',
        '[89, 144]'
    ]

    list_data_2 = [
        '[event]',
        '[missa, culto]',
        '[festa da cidade]'
    ]

    df_1 = DataFrame(list_data_1, columns=['label'])
    df_2 = DataFrame(list_data_2, columns=['label'])

    expected_data_1 = DataFrame(
        data={'label': [[1, 2, 3],
                        [5, 8],
                        [13, 21, 34, 55],
                        [89, 144]]},
        index=[0, 1, 2, 3]
    )

    expected_data_2 = DataFrame(
        data={'label': [['event'],
                        ['missa', 'culto'],
                        ['festa da cidade']]},
        index=[0, 1, 2]
    )

    trajectories.column_to_array(df_1, 'label')
    trajectories.column_to_array(df_2, 'label')

    assert_frame_equal(df_1, expected_data_1)
    assert_frame_equal(df_2, expected_data_2)


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
    m = trajectories.plot_bbox(bbox, file=filename_write_default, save_map=True)
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
