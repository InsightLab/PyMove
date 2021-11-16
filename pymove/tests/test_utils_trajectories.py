import os

import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_equal
from pandas import DataFrame
from pandas.testing import assert_frame_equal

from pymove import DaskMoveDataFrame, MoveDataFrame, PandasMoveDataFrame, trajectories
from pymove.utils.constants import (
    DATETIME,
    LATITUDE,
    LOCAL_LABEL,
    LONGITUDE,
    PREV_LOCAL,
    TID,
    TID_STAT,
    TRAJ_ID,
    TYPE_DASK,
    TYPE_PANDAS,
)
from pymove.utils.networkx import build_transition_graph_from_df
from pymove.utils.trajectories import (
    append_trajectory,
    columns_to_array,
    object_for_array,
    split_trajectory,
)

list_data = [
    [39.984094, 116.319236, '2008-10-23 05:53:05', 1],
    [39.984198, 116.319322, '2008-10-23 05:53:06', 1],
    [39.984224, 116.319402, '2008-10-23 05:53:11', 1],
    [39.984211, 116.319389, '2008-10-23 05:53:16', 1],
    [39.984217, 116.319422, '2008-10-23 05:53:21', 1],
]

list_data2 = {
    TRAJ_ID: [[1, 1, 1, 1, 1, 1]],
    DATETIME: [[pd.Timestamp('2017-09-02 22:00:27'),
                pd.Timestamp('2017-09-02 22:01:36'),
                pd.Timestamp('2017-09-02 22:03:08'),
                pd.Timestamp('2017-09-02 22:03:46'),
                pd.Timestamp('2017-09-02 22:07:19'),
                pd.Timestamp('2017-09-02 22:07:40')]],
    LOCAL_LABEL: [[85, 673, 394, 263, 224, 623]],
    LATITUDE: [[-3.8347478, -3.8235834, -3.813889,
                -3.9067654, -3.8857223, -3.8828723]],
    LONGITUDE: [[-38.592189, -38.590389, -38.5904445,
                 -38.5907723, -38.5928892, -38.5929789]],
    TID: [['12017090222', '12017090222', '12017090222',
           '12017090222', '12017090222', '12017090222']]
}

list_data4 = {
    TRAJ_ID: [[1, 1, 1], [2, 2, 2, 2]],
    DATETIME: [[pd.Timestamp('2017-09-02 22:00:27'),
                pd.Timestamp('2017-09-02 22:01:36'),
                pd.Timestamp('2017-09-02 22:03:08')],
               [pd.Timestamp('2017-09-02 23:03:46'),
                pd.Timestamp('2017-09-02 23:07:19'),
                pd.Timestamp('2017-09-02 23:07:40'),
                pd.Timestamp('2017-09-02 23:09:10')]],
    LOCAL_LABEL: [[85, 673, 394], [263, 224, 623, 394]],
    LATITUDE: [[-3.8347478, -3.8235834, -3.813889],
               [-3.9067654, -3.8857223, -3.8828723, -3.9939834]],
    LONGITUDE: [[-38.592189, -38.590389, -38.5904445],
                [-38.5907723, -38.5928892, -38.5929789, -38.7040900]],
}

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

    d = tmpdir.mkdir('utils')

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
    shifted_array = trajectories.shift(arr=array_, num=-3)
    assert_array_equal(shifted_array, expected)

    expected = [False, False, False, True, True]
    array_ = [True, True, True, True, True]
    shifted_array = trajectories.shift(arr=array_, num=3)
    assert_array_equal(shifted_array, expected)

    expected = ['dewberry', 'eggplant', 'nan', 'nan', 'nan']
    array_ = ['apple', 'banana', 'coconut', 'dewberry', 'eggplant']
    shifted_array = trajectories.shift(arr=array_, num=-3, fill_value=np.nan)
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


def test_append_trajectory():
    traj_df = pd.DataFrame(list_data4)
    graph = build_transition_graph_from_df(traj_df)

    expected = pd.DataFrame({
        TRAJ_ID: [[1, 1, 1], [2, 2, 2, 2], [3, 3, 3]],
        DATETIME: [[pd.Timestamp('2017-09-02 22:00:27'), pd.Timestamp('2017-09-02 22:01:36'),
                    pd.Timestamp('2017-09-02 22:03:08')],
                   [pd.Timestamp('2017-09-02 23:03:46'), pd.Timestamp('2017-09-02 23:07:19'),
                    pd.Timestamp('2017-09-02 23:07:40'), pd.Timestamp('2017-09-02 23:09:10')],
                   [pd.Timestamp('2017-09-02 23:07:19'), pd.Timestamp('2017-09-02 23:07:40'),
                    pd.Timestamp('2017-09-02 23:09:10')]],
        LOCAL_LABEL: [[85, 673, 394], [263, 224, 623, 394], [224.0, 623.0, 394.0]],
        LATITUDE: [[-3.8347478, -3.8235834, -3.813889],
                   [-3.9067654, -3.8857223, -3.8828723, -3.9939834],
                   [-3.8857223, -3.8828723, -3.9939834]],
        LONGITUDE: [[-38.592189, -38.590389, -38.5904445],
                    [-38.5907723, -38.5928892, -38.5929789, -38.70409],
                    [-38.5928892, -38.5929789, -38.70409]],
    })

    trajectory = [224, 623, 394]
    append_trajectory(traj_df, trajectory, graph)
    assert_frame_equal(traj_df, expected)


def test_split_trajectory():
    trajectory = pd.DataFrame(list_data2).loc[0]

    expected = pd.DataFrame({
        'id': [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
        'datetime': [[pd.Timestamp('2017-09-02 22:00:27'), pd.Timestamp('2017-09-02 22:01:36'),
                      pd.Timestamp('2017-09-02 22:03:08'), pd.Timestamp('2017-09-02 22:03:46')],
                     [pd.Timestamp('2017-09-02 22:01:36'), pd.Timestamp('2017-09-02 22:03:08'),
                      pd.Timestamp('2017-09-02 22:03:46'), pd.Timestamp('2017-09-02 22:07:19')],
                     [pd.Timestamp('2017-09-02 22:03:08'), pd.Timestamp('2017-09-02 22:03:46'),
                      pd.Timestamp('2017-09-02 22:07:19'), pd.Timestamp('2017-09-02 22:07:40')]],
        'local_label': [[85, 673, 394, 263], [673, 394, 263, 224], [394, 263, 224, 623]],
        'lat': [[-3.8347478, -3.8235834, -3.813889, -3.9067654],
                [-3.8235834, -3.813889, -3.9067654, -3.8857223],
                [-3.813889, -3.9067654, -3.8857223, -3.8828723]],
        'lon': [[-38.592189, -38.590389, -38.5904445, -38.5907723],
                [-38.590389, -38.5904445, -38.5907723, -38.5928892],
                [-38.5904445, -38.5907723, -38.5928892, -38.5929789]],
        'tid': [['12017090222', '12017090222', '12017090222', '12017090222'],
                ['12017090222', '12017090222', '12017090222', '12017090222'],
                ['12017090222', '12017090222', '12017090222', '12017090222']]
    })

    split = split_trajectory(trajectory, size_window=4, size_jump=1)
    assert_frame_equal(split, expected)


def test_object_for_array():
    data_1 = '[1, 2, 3]'
    data_2 = '[1.5, 2.5, 3.5]'
    data_3 = "['event', 'event']"

    expected_data_1 = np.array([1., 2., 3.], dtype=np.float32)
    expected_data_2 = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    expected_data_3 = np.array(['event', 'event'], dtype='object_')

    assert_array_almost_equal(object_for_array(data_1), expected_data_1)
    assert_array_almost_equal(object_for_array(data_2), expected_data_2)
    assert_array_equal(object_for_array(data_3), expected_data_3)


def test_columns_to_array():
    df = DataFrame({
        'ids': ['[1, 1, 1]', '[2, 2, 2]', '[3, 3, 3, 3]', '[4, 4]'],
        'descritions': ["['event', 'event', 'event']", "['bike', 'bike', 'bike']",
                        "['car', 'car', 'car', 'car']", "['house', 'house']"],
        'price': ['[10.5, 20.5, 13.5]', '[50.2, 33.4, 90.0]',
                  '[1.0, 2.9, 3.4, 8.4]', '[100.4, 150.5]']
    })

    expected = DataFrame({
        'ids': [[1, 1, 1], [2, 2, 2], [3, 3, 3, 3], [4, 4]],
        'descritions': [['event', 'event', 'event'], ['bike', 'bike', 'bike'],
                        ['car', 'car', 'car', 'car'], ['house', 'house']],
        'price': [[10.5, 20.5, 13.5], [50.2, 33.4, 90.0],
                  [1.0, 2.9, 3.4, 8.4], [100.4, 150.5]]
    })

    columns_to_array(df)
    assert_frame_equal(df, expected)
