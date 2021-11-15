import pandas as pd
from numpy.testing import assert_array_almost_equal
from pandas.testing import assert_frame_equal

from pymove.utils.constants import (
    DATETIME,
    DESTINY,
    LABEL,
    LATITUDE,
    LOCAL_LABEL,
    LONGITUDE,
    PREV_LOCAL,
    START,
    TID,
    TID_STAT,
    TRAJ_ID,
)
from pymove.utils.data_augmentation import (
    _augmentation,
    append_row,
    flatten_trajectories_dataframe,
    generate_trajectories_df,
    get_all_paths,
    instance_crossover_augmentation,
    sliding_window,
    split_crossover,
    transition_graph_augmentation_all_vertex,
)
from pymove.utils.networkx import build_transition_graph_from_df

list_data1 = [[1, pd.Timestamp('2017-09-02 21:59:34'), 162, -3.8431323, -38.5933142, '12017090221'],
              [1, pd.Timestamp('2017-09-02 22:00:27'),  85, -3.8347478, -38.5921890, '12017090222'],
              [1, pd.Timestamp('2017-09-02 22:01:36'), 673, -3.8235834, -38.5903890, '12017090222'],
              [1, pd.Timestamp('2017-09-02 22:03:08'), 394, -3.8138890, -38.5904445, '12017090222'],
              [1, pd.Timestamp('2017-09-02 22:03:46'), 263, -3.9067654, -38.5907723, '12017090222'],
              [1, pd.Timestamp('2017-09-02 22:07:19'), 224, -3.8857223, -38.5928892, '12017090222'],
              [1, pd.Timestamp('2017-09-02 22:07:40'), 623, -3.8828723, -38.5929789, '12017090222']]

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

list_data3 = {
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
    TID: [['12017090222', '12017090222', '12017090222'],
          ['22017090223', '22017090223', '22017090223', '22017090223']]
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
                [-38.5907723, -38.5928892, -38.5929789, -38.7040900]]
}


def test_append_row():
    df = pd.DataFrame(list_data2)

    expected = pd.DataFrame(
        data={
            TRAJ_ID: [[1, 1, 1, 1, 1, 1], [2, 2, 2]],
            DATETIME: [[pd.Timestamp('2017-09-02 22:00:27'),
                        pd.Timestamp('2017-09-02 22:01:36'),
                        pd.Timestamp('2017-09-02 22:03:08'),
                        pd.Timestamp('2017-09-02 22:03:46'),
                        pd.Timestamp('2017-09-02 22:07:19'),
                        pd.Timestamp('2017-09-02 22:07:40')],
                       [pd.Timestamp('2017-09-03 14:10:15'),
                        pd.Timestamp('2017-09-03 14:20:30'),
                        pd.Timestamp('2017-09-03 14:30:45')]],
            LOCAL_LABEL: [[85, 673, 394, 263, 224, 623],
                          [673, 263, 623]],
            LATITUDE: [[-3.8347478, -3.8235834, -3.813889,
                        -3.9067654, -3.8857223, -3.8828723],
                       [-3.8235834, -3.9067654, -3.8828723]],
            LONGITUDE: [[-38.592189, -38.590389, -38.5904445,
                         -38.5907723, -38.5928892, -38.5929789],
                        [-38.590389, -38.5907723, -38.5929789]],
            TID: [['12017090222', '12017090222', '12017090222',
                   '12017090222', '12017090222', '12017090222'],
                  ['22017090314', '22017090314', '22017090314']]})

    row = pd.Series(
        data={
            TRAJ_ID: [2, 2, 2],
            LOCAL_LABEL: [673, 263, 623],
            DATETIME: [pd.Timestamp('2017-09-03 14:10:15'),
                       pd.Timestamp('2017-09-03 14:20:30'),
                       pd.Timestamp('2017-09-03 14:30:45')],
            LATITUDE: [-3.8235834, -3.9067654, -3.8828723],
            LONGITUDE: [-38.590389, -38.5907723, -38.5929789],
            TID: ['22017090314', '22017090314', '22017090314']})

    append_row(df, row)
    assert_frame_equal(df, expected)


def test_generate_trajectories_df():
    df = pd.DataFrame(
        list_data1,
        columns=[TRAJ_ID, DATETIME, LOCAL_LABEL, LATITUDE, LONGITUDE, TID]
    )

    expected = pd.DataFrame({
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
    })

    traj_df = generate_trajectories_df(df)
    assert_frame_equal(traj_df, expected)


def test_split_crossover():
    s1 = [0, 2, 4, 6, 8]
    s2 = [1, 3, 5, 7, 9]

    expected1 = [0, 2, 5, 7, 9]
    expected2 = [1, 3, 4, 6, 8]

    s1, s2 = split_crossover(s1, s2)

    assert_array_almost_equal(expected1, s1)
    assert_array_almost_equal(expected2, s2)


def test_augmentation():
    df = pd.DataFrame(list_data3)

    expected = pd.DataFrame(
        data = {
            TRAJ_ID: [[1, 1, 1], [2, 2, 2, 2], [1, 2, 2], [2, 2, 1, 1]],
            DATETIME: [[pd.Timestamp('2017-09-02 22:00:27'), pd.Timestamp('2017-09-02 22:01:36'),
                        pd.Timestamp('2017-09-02 22:03:08')],
                       [pd.Timestamp('2017-09-02 23:03:46'), pd.Timestamp('2017-09-02 23:07:19'),
                        pd.Timestamp('2017-09-02 23:07:40'), pd.Timestamp('2017-09-02 23:09:10')],
                       [pd.Timestamp('2017-09-02 22:00:27'), pd.Timestamp('2017-09-02 23:07:40'),
                        pd.Timestamp('2017-09-02 23:09:10')],
                       [pd.Timestamp('2017-09-02 23:03:46'), pd.Timestamp('2017-09-02 23:07:19'),
                        pd.Timestamp('2017-09-02 22:01:36'), pd.Timestamp('2017-09-02 22:03:08')]],
            LOCAL_LABEL: [[85, 673, 394], [263, 224, 623, 394],
                          [ 85, 623, 394], [263, 224, 673, 394]],
            LATITUDE: [[-3.8347478, -3.8235834, -3.813889],
                       [-3.9067654, -3.8857223, -3.8828723, -3.9939834],
                       [-3.8347478, -3.8828723, -3.9939834],
                       [-3.9067654, -3.8857223, -3.8235834, -3.8138890]],
            LONGITUDE: [[-38.592189, -38.590389, -38.5904445],
                        [-38.5907723, -38.5928892, -38.5929789, -38.70409],
                        [-38.592189 , -38.5929789, -38.70409  ],
                        [-38.5907723, -38.5928892, -38.590389 , -38.5904445]],
            TID: [['12017090222', '12017090222', '12017090222'],
                  ['22017090223', '22017090223', '22017090223', '22017090223'],
                  ['12017090222', '22017090223', '22017090223'],
                  ['22017090223', '22017090223', '12017090222', '12017090222']]
        }
    )

    aug_df = _augmentation(df, 0.5)
    assert_frame_equal(aug_df, expected)


def test_flatten_trajectories_dataframe():
    traj_df = pd.DataFrame(list_data3)

    expected = pd.DataFrame(
        data={
            TRAJ_ID: [1, 1, 1, 2, 2, 2, 2],
            DATETIME: [pd.Timestamp('2017-09-02 22:00:27'), pd.Timestamp('2017-09-02 22:01:36'),
                       pd.Timestamp('2017-09-02 22:03:08'), pd.Timestamp('2017-09-02 23:03:46'),
                       pd.Timestamp('2017-09-02 23:07:19'), pd.Timestamp('2017-09-02 23:07:40'),
                       pd.Timestamp('2017-09-02 23:09:10')],
            LOCAL_LABEL: [85, 673, 394, 263, 224, 623, 394],
            LATITUDE: [-3.8347478, -3.8235834, -3.813889, -3.9067654, -3.8857223,
                       -3.8828723, -3.9939834],
            LONGITUDE: [-38.592189, -38.590389, -38.5904445, -38.5907723, -38.5928892,
                        -38.5929789, -38.70409],
            TID: ['12017090222', '12017090222', '12017090222', '22017090223', '22017090223',
                  '22017090223', '22017090223']
        }
    )

    df = flatten_trajectories_dataframe(traj_df)
    assert_frame_equal(df, expected)


def test_instance_crossover_augmentation():
    traj_df = pd.DataFrame(list_data3)

    expected = pd.DataFrame(
        data={
            TRAJ_ID: [[1, 1, 1], [2, 2, 2, 2], [1, 2, 2], [2, 2, 1, 1]],
            DATETIME: [[pd.Timestamp('2017-09-02 22:00:27'), pd.Timestamp('2017-09-02 22:01:36'),
                        pd.Timestamp('2017-09-02 22:03:08')],
                       [pd.Timestamp('2017-09-02 23:03:46'), pd.Timestamp('2017-09-02 23:07:19'),
                        pd.Timestamp('2017-09-02 23:07:40'), pd.Timestamp('2017-09-02 23:09:10')],
                       [pd.Timestamp('2017-09-02 22:00:27'), pd.Timestamp('2017-09-02 23:07:40'),
                        pd.Timestamp('2017-09-02 23:09:10')],
                       [pd.Timestamp('2017-09-02 23:03:46'), pd.Timestamp('2017-09-02 23:07:19'),
                        pd.Timestamp('2017-09-02 22:01:36'), pd.Timestamp('2017-09-02 22:03:08')]],
            LOCAL_LABEL: [[85, 673, 394], [263, 224, 623, 394], [ 85, 623, 394], [263, 224, 673, 394]],
            LATITUDE: [[-3.8347478, -3.8235834, -3.813889],
                       [-3.9067654, -3.8857223, -3.8828723, -3.9939834],
                       [-3.8347478, -3.8828723, -3.9939834],
                       [-3.9067654, -3.8857223, -3.8235834, -3.813889 ]],
            LONGITUDE: [[-38.592189, -38.590389, -38.5904445],
                        [-38.5907723, -38.5928892, -38.5929789, -38.70409],
                        [-38.592189 , -38.5929789, -38.70409  ],
                        [-38.5907723, -38.5928892, -38.590389 , -38.5904445]],
            TID: [['12017090222', '12017090222', '12017090222'],
                  ['22017090223', '22017090223', '22017090223', '22017090223'],
                  ['12017090222', '22017090223', '22017090223'],
                  ['22017090223', '22017090223', '12017090222', '12017090222']]
        })

    aug_df = instance_crossover_augmentation(traj_df)
    assert_frame_equal(aug_df, expected)


def test_sliding_window():
    traj_df = pd.DataFrame(list_data2)

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

    sw_df = sliding_window(traj_df, size_window=4, size_jump=1)
    assert_frame_equal(sw_df, expected)


def test_get_all_paths():
    traj_df = pd.DataFrame(list_data4)
    graph = build_transition_graph_from_df(traj_df)

    expected = pd.DataFrame(
        data={
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
                        [-38.5928892, -38.5929789, -38.70409]]
        })

    get_all_paths(traj_df, graph, '224', '394')
    assert_frame_equal(traj_df, expected)


def test_transition_graph_augmentation_all_vertex():
    traj_df = pd.DataFrame(list_data4)

    expected = pd.DataFrame({
        TRAJ_ID: [[1, 1, 1], [2, 2, 2, 2], [3, 3, 3], [4, 4, 4]],
        DATETIME: [[pd.Timestamp('2017-09-02 22:00:27'), pd.Timestamp('2017-09-02 22:01:36'),
                    pd.Timestamp('2017-09-02 22:03:08')],
                   [pd.Timestamp('2017-09-02 23:03:46'), pd.Timestamp('2017-09-02 23:07:19'),
                    pd.Timestamp('2017-09-02 23:07:40'), pd.Timestamp('2017-09-02 23:09:10')],
                   [pd.Timestamp('2017-09-02 23:03:46'), pd.Timestamp('2017-09-02 23:07:19'),
                    pd.Timestamp('2017-09-02 23:07:40')],
                   [pd.Timestamp('2017-09-02 23:07:19'), pd.Timestamp('2017-09-02 23:07:40'),
                    pd.Timestamp('2017-09-02 23:09:10')]],
        LOCAL_LABEL: [[85, 673, 394], [263, 224, 623, 394],
                      [263.0, 224.0, 623.0], [224.0, 623.0, 394.0]],
        LATITUDE: [[-3.8347478, -3.8235834, -3.813889],
                   [-3.9067654, -3.8857223, -3.8828723, -3.9939834],
                   [-3.9067654, -3.8857223, -3.8828723],
                   [-3.8857223, -3.8828723, -3.9939834]],
        LONGITUDE: [[-38.592189, -38.590389, -38.5904445],
                    [-38.5907723, -38.5928892, -38.5929789, -38.70409],
                    [-38.5907723, -38.5928892, -38.5929789],
                    [-38.5928892, -38.5929789, -38.70409]]
    })

    transition_graph_augmentation_all_vertex(traj_df)
    assert_frame_equal(traj_df, expected)
