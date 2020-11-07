import numpy as np
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
    START,
    TID,
    TRAJ_ID,
)
from pymove.utils.data_augmentation import (
    _augmentation,
    append_row,
    augmentation_trajectories_df,
    generate_destiny_feature,
    generate_start_feature,
    insert_points_in_df,
    instance_crossover_augmentation,
    split_crossover,
)

list_data1 = [['abc-0000', 1, 3.1234567, 38.1234567,
               pd.Timestamp('2020-01-01 06:08:15'), 'abc-00002020010106'],
              ['abc-0000', 2, 3.1234567, 38.1234567,
               pd.Timestamp('2020-01-01 06:16:51'), 'abc-00002020010106'],
              ['abc-0000', 3, 3.1234567, 38.1234567,
               pd.Timestamp('2020-01-01 06:31:41'), 'abc-00002020010106'],
              ['abc-0000', 4, 3.1234567, 38.1234567,
               pd.Timestamp('2020-01-01 06:45:25'), 'abc-00002020010106'],
              ['abc-0000', 9, 3.1234567, 38.1234567,
               pd.Timestamp('2020-01-01 06:49:18'), 'abc-00002020010106'],
              ['def-1111', 5, 3.1234567, 38.1234567,
               pd.Timestamp('2020-01-01 09:10:15'), 'def-11112020010109'],
              ['def-1111', 6, 3.1234567, 38.1234567,
               pd.Timestamp('2020-01-01 09:15:45'), 'def-11112020010109'],
              ['def-1111', 7, 3.1234567, 38.1234567,
               pd.Timestamp('2020-01-01 09:25:34'), 'def-11112020010109'],
              ['def-1111', 8, 3.1234567, 38.1234567,
               pd.Timestamp('2020-01-01 09:40:25'), 'def-11112020010109'],
              ['def-1111', 9, 3.1234567, 38.1234567,
               pd.Timestamp('2020-01-01 09:52:53'), 'def-11112020010109']]

list_data2 = {
    TRAJ_ID: ['abc-0000', 'def-1111'],
    LOCAL_LABEL: [[5, 7, 9], [2, 4, 6, 8, 9]],
    DATETIME: [[pd.Timestamp('2020-01-01 06:08:15'),
                pd.Timestamp('2020-01-01 06:16:51'),
                pd.Timestamp('2020-01-01 06:20:40')],
               [pd.Timestamp('2020-01-01 09:10:15'),
                pd.Timestamp('2020-01-01 09:15:45'),
                pd.Timestamp('2020-01-01 09:25:30'),
                pd.Timestamp('2020-01-01 09:30:17'),
                pd.Timestamp('2020-01-01 09:45:16')]],
    LATITUDE: [[3.1234567165374756, 3.1234567165374756,
                3.1234567165374756],
               [3.1234567165374756, 3.1234567165374756,
                3.1234567165374756, 3.1234567165374756,
                3.1234567165374756]],
    LONGITUDE: [[38.12345504760742, 38.12345504760742,
                 38.12345504760742],
                [38.12345504760742, 38.12345504760742,
                 38.12345504760742, 38.12345504760742,
                 38.12345504760742]],
    TID: [['abc-00002020010106', 'abc-00002020010106',
           'abc-00002020010106'],
          ['def-11112020010109', 'def-11112020010109',
           'abc-00002020010106', 'abc-00002020010106',
           'abc-00002020010106']]}


def test_append_row():
    traj_df = pd.DataFrame(
        data={
            TRAJ_ID: ['abc-0000'],
            LOCAL_LABEL: [[1, 2]],
            DATETIME: [[pd.Timestamp('2020-01-01 06:08:15')]],
            LATITUDE: [[3.1234567165374756]],
            LONGITUDE: [[38.12345504760742]],
            TID: [['abc-00002020010106']]})

    expected = pd.DataFrame(
        data={
            TRAJ_ID: ['abc-0000', 'def-1111'],
            LOCAL_LABEL: [[1, 2], [5, 6]],
            DATETIME: [[pd.Timestamp('2020-01-01 06:08:15')],
                       [pd.Timestamp('2020-01-01 07:10:15')]],
            LATITUDE: [[3.1234567165374756], [3.623471260070801]],
            LONGITUDE: [[38.12345504760742], [38.397525787353516]],
            TID: [['abc-00002020010106'], ['def-1111202001010']]})

    row = pd.Series(
        data={TRAJ_ID: 'def-1111',
              LOCAL_LABEL: [5, 6],
              DATETIME: [pd.Timestamp('2020-01-01 7:10:15')],
              LATITUDE: [3.6234712461],
              LONGITUDE: [38.39752597257],
              TID: ['def-1111202001010']})

    append_row(traj_df, row=row)
    assert_frame_equal(expected, traj_df)


def test__augmentation():
    traj_df = pd.DataFrame(list_data2)

    expected = pd.DataFrame(
        data={
            TRAJ_ID: ['abc-0000_def-1111', 'def-1111_abc-0000'],
            LOCAL_LABEL: [[5, 6, 8, 9], [2, 4, 7, 9]],
            DATETIME: [[pd.Timestamp('2020-01-01 06:08:15'),
                        pd.Timestamp('2020-01-01 09:25:30'),
                        pd.Timestamp('2020-01-01 09:30:17'),
                        pd.Timestamp('2020-01-01 09:45:16')],
                       [pd.Timestamp('2020-01-01 09:10:15'),
                        pd.Timestamp('2020-01-01 09:15:45'),
                        pd.Timestamp('2020-01-01 06:16:51'),
                        pd.Timestamp('2020-01-01 06:20:40')]],
            LATITUDE: [[3.12345672, 3.12345672, 3.12345672, 3.12345672],
                       [3.12345672, 3.12345672, 3.12345672, 3.12345672]],
            LONGITUDE: [[38.12345505, 38.12345505, 38.12345505, 38.12345505],
                        [38.12345505, 38.12345505, 38.12345505, 38.12345505]],
            TID: [['abc-00002020010106', 'abc-00002020010106',
                   'abc-00002020010106', 'abc-00002020010106'],
                  ['def-11112020010109', 'def-11112020010109',
                   'abc-00002020010106', 'abc-00002020010106']]
        }
    )

    aug_df = pd.DataFrame(columns=traj_df.columns)

    _augmentation(traj_df, aug_df)
    assert_frame_equal(expected, aug_df)


def test_split_crossover():
    s1 = [0, 2, 4, 6, 8]
    s2 = [1, 3, 5, 7, 9]

    expected1 = [0, 2, 5, 7, 9]
    expected2 = [1, 3, 4, 6, 8]

    s1, s2 = split_crossover(s1, s2)

    assert_array_almost_equal(expected1, s1)
    assert_array_almost_equal(expected2, s2)


def test_insert_points_in_df():
    move_df = pd.DataFrame(
        data=np.array(list_data1, dtype=object),
        columns=[TRAJ_ID, LOCAL_LABEL, LATITUDE, LONGITUDE, DATETIME, TID]
    )

    aug_df = pd.DataFrame(
        data={
            TRAJ_ID: ['abc-0000_def-1111'],
            LOCAL_LABEL: [[5, 6, 7, 9]],
            LATITUDE: [[3.1234567165374756, 3.1234567165374756,
                        3.1234567165374756, 3.1234567165374756]],
            LONGITUDE: [[38.12345504760742, 38.12345504760742,
                         38.12345504760742, 38.12345504760742]],
            DATETIME: [[pd.Timestamp('2020-01-01 06:08:15'),
                        pd.Timestamp('2020-01-01 09:15:45'),
                        pd.Timestamp('2020-01-01 09:30:45'),
                        pd.Timestamp('2020-01-01 09:30:45')]],
            TID: [['abc-0000202001010', 'abc-0000202001010',
                   'abc-0000202001010', 'abc-0000202001010']],
            DESTINY: [9]})

    expected = pd.DataFrame(
        data=np.array(
            [['abc-0000', 1, 3.1234567, 38.1234567,
              pd.Timestamp('2020-01-01 06:08:15'), 'abc-00002020010106'],
             ['abc-0000', 2, 3.1234567, 38.1234567,
              pd.Timestamp('2020-01-01 06:16:51'), 'abc-00002020010106'],
             ['abc-0000', 3, 3.1234567, 38.1234567,
              pd.Timestamp('2020-01-01 06:31:41'), 'abc-00002020010106'],
             ['abc-0000', 4, 3.1234567, 38.1234567,
              pd.Timestamp('2020-01-01 06:45:25'), 'abc-00002020010106'],
             ['abc-0000', 9, 3.1234567, 38.1234567,
              pd.Timestamp('2020-01-01 06:49:18'), 'abc-00002020010106'],
             ['def-1111', 5, 3.1234567, 38.1234567,
              pd.Timestamp('2020-01-01 09:10:15'), 'def-11112020010109'],
             ['def-1111', 6, 3.1234567, 38.1234567,
              pd.Timestamp('2020-01-01 09:15:45'), 'def-11112020010109'],
             ['def-1111', 7, 3.1234567, 38.1234567,
              pd.Timestamp('2020-01-01 09:25:34'), 'def-11112020010109'],
             ['def-1111', 8, 3.1234567, 38.1234567,
              pd.Timestamp('2020-01-01 09:40:25'), 'def-11112020010109'],
             ['def-1111', 9, 3.1234567, 38.1234567,
              pd.Timestamp('2020-01-01 09:52:53'), 'def-11112020010109'],
             ['abc-0000_def-1111', 5, 3.1234567165374756, 38.12345504760742,
              pd.Timestamp('2020-01-01 06:08:15'), 'abc-0000202001010'],
             ['abc-0000_def-1111', 6, 3.1234567165374756, 38.12345504760742,
              pd.Timestamp('2020-01-01 09:15:45'), 'abc-0000202001010'],
             ['abc-0000_def-1111', 7, 3.1234567165374756, 38.12345504760742,
              pd.Timestamp('2020-01-01 09:30:45'), 'abc-0000202001010'],
             ['abc-0000_def-1111', 9, 3.1234567165374756, 38.12345504760742,
              pd.Timestamp('2020-01-01 09:30:45'), 'abc-0000202001010']],
            dtype=object),
        columns=[TRAJ_ID, LOCAL_LABEL, LATITUDE, LONGITUDE, DATETIME, TID])

    insert_points_in_df(move_df, aug_df)
    assert_frame_equal(expected, move_df)


def test_generate_start_feature():
    move_df = pd.DataFrame(list_data2)

    expected = pd.DataFrame(
        data={
            TRAJ_ID: ['abc-0000', 'def-1111'],
            LOCAL_LABEL: [[5, 7, 9], [2, 4, 6, 8, 9]],
            DATETIME: [[pd.Timestamp('2020-01-01 06:08:15'),
                        pd.Timestamp('2020-01-01 06:16:51'),
                        pd.Timestamp('2020-01-01 06:20:40')],
                       [pd.Timestamp('2020-01-01 09:10:15'),
                        pd.Timestamp('2020-01-01 09:15:45'),
                        pd.Timestamp('2020-01-01 09:25:30'),
                        pd.Timestamp('2020-01-01 09:30:17'),
                        pd.Timestamp('2020-01-01 09:45:16')]],
            LATITUDE: [[3.1234567165374756, 3.1234567165374756,
                        3.1234567165374756],
                       [3.1234567165374756, 3.1234567165374756,
                        3.1234567165374756, 3.1234567165374756,
                        3.1234567165374756]],
            LONGITUDE: [[38.12345504760742, 38.12345504760742,
                         38.12345504760742],
                        [38.12345504760742, 38.12345504760742,
                         38.12345504760742, 38.12345504760742,
                         38.12345504760742]],
            TID: [['abc-00002020010106', 'abc-00002020010106',
                   'abc-00002020010106'],
                  ['def-11112020010109', 'def-11112020010109',
                   'abc-00002020010106', 'abc-00002020010106',
                   'abc-00002020010106']],
            START: [5, 2]
        })

    generate_start_feature(move_df, label_trajectory=LOCAL_LABEL)
    assert_frame_equal(move_df, expected)


def test_generate_destiny_feature():
    move_df = pd.DataFrame(list_data2)

    expected = pd.DataFrame(
        data={
            TRAJ_ID: ['abc-0000', 'def-1111'],
            LOCAL_LABEL: [[5, 7, 9], [2, 4, 6, 8, 9]],
            DATETIME: [[pd.Timestamp('2020-01-01 06:08:15'),
                        pd.Timestamp('2020-01-01 06:16:51'),
                        pd.Timestamp('2020-01-01 06:20:40')],
                       [pd.Timestamp('2020-01-01 09:10:15'),
                        pd.Timestamp('2020-01-01 09:15:45'),
                        pd.Timestamp('2020-01-01 09:25:30'),
                        pd.Timestamp('2020-01-01 09:30:17'),
                        pd.Timestamp('2020-01-01 09:45:16')]],
            LATITUDE: [[3.1234567165374756, 3.1234567165374756,
                        3.1234567165374756],
                       [3.1234567165374756, 3.1234567165374756,
                        3.1234567165374756, 3.1234567165374756,
                        3.1234567165374756]],
            LONGITUDE: [[38.12345504760742, 38.12345504760742,
                         38.12345504760742],
                        [38.12345504760742, 38.12345504760742,
                         38.12345504760742, 38.12345504760742,
                         38.12345504760742]],
            TID: [['abc-00002020010106', 'abc-00002020010106',
                   'abc-00002020010106'],
                  ['def-11112020010109', 'def-11112020010109',
                   'abc-00002020010106', 'abc-00002020010106',
                   'abc-00002020010106']],
            DESTINY: [9, 9]
        })

    generate_destiny_feature(move_df, label_trajectory=LOCAL_LABEL)
    assert_frame_equal(move_df, expected)


def test_augmentation_trajectories_df():
    move_df = pd.DataFrame(list_data2)

    expected = pd.DataFrame(
        data={
            TRAJ_ID: ['abc-0000_def-1111', 'def-1111_abc-0000'],
            LOCAL_LABEL: [[5, 6, 8, 9], [2, 4, 7, 9]],
            DATETIME: [[pd.Timestamp('2020-01-01 06:08:15'),
                        pd.Timestamp('2020-01-01 09:25:30'),
                        pd.Timestamp('2020-01-01 09:30:17'),
                        pd.Timestamp('2020-01-01 09:45:16')],
                       [pd.Timestamp('2020-01-01 09:10:15'),
                        pd.Timestamp('2020-01-01 09:15:45'),
                        pd.Timestamp('2020-01-01 06:16:51'),
                        pd.Timestamp('2020-01-01 06:20:40')]],
            LATITUDE: [[3.12345672, 3.12345672, 3.12345672, 3.12345672],
                       [3.12345672, 3.12345672, 3.12345672, 3.12345672]],
            LONGITUDE: [[38.12345505, 38.12345505, 38.12345505, 38.12345505],
                        [38.12345505, 38.12345505, 38.12345505, 38.12345505]],
            TID: [['abc-00002020010106', 'abc-00002020010106',
                   'abc-00002020010106', 'abc-00002020010106'],
                  ['def-11112020010109', 'def-11112020010109',
                   'abc-00002020010106', 'abc-00002020010106']],
            DESTINY: [[9], [9]]})

    aug_df = augmentation_trajectories_df(move_df, label_trajectory=LOCAL_LABEL)
    assert_frame_equal(expected, aug_df)


def test_instance_crossover_augmentation():
    move_df = pd.DataFrame(
        list_data1,
        columns=[TRAJ_ID, LOCAL_LABEL, LATITUDE, LONGITUDE, DATETIME, TID]
    )

    expected = pd.DataFrame(
        data={
            TRAJ_ID: ['abc-0000', 'abc-0000', 'abc-0000', 'abc-0000', 'abc-0000',
                      'def-1111', 'def-1111', 'def-1111', 'def-1111', 'def-1111',
                      'abc-0000_def-1111', 'abc-0000_def-1111', 'abc-0000_def-1111',
                      'abc-0000_def-1111', 'abc-0000_def-1111', 'def-1111_abc-0000',
                      'def-1111_abc-0000', 'def-1111_abc-0000', 'def-1111_abc-0000',
                      'def-1111_abc-0000'],
            LOCAL_LABEL: [1.0, 2.0, 3.0, 4.0, 9.0, 5.0, 6.0, 7.0, 8.0, 9.0,
                          1.0, 2.0, 7.0, 8.0, 9.0, 5.0, 6.0, 3.0, 4.0, 9.0],
            LATITUDE: [3.1234567, 3.1234567, 3.1234567, 3.1234567, 3.1234567,
                       3.1234567, 3.1234567, 3.1234567, 3.1234567, 3.1234567,
                       3.1234567, 3.1234567, 3.1234567, 3.1234567, 3.1234567,
                       3.1234567, 3.1234567, 3.1234567, 3.1234567, 3.1234567],
            LONGITUDE: [38.1234567, 38.1234567, 38.1234567, 38.1234567, 38.1234567,
                        38.1234567, 38.1234567, 38.1234567, 38.1234567, 38.1234567,
                        38.1234567, 38.1234567, 38.1234567, 38.1234567, 38.1234567,
                        38.1234567, 38.1234567, 38.1234567, 38.1234567, 38.1234567],
            DATETIME: [pd.Timestamp('2020-01-01 06:08:15'),
                       pd.Timestamp('2020-01-01 06:16:51'),
                       pd.Timestamp('2020-01-01 06:31:41'),
                       pd.Timestamp('2020-01-01 06:45:25'),
                       pd.Timestamp('2020-01-01 06:49:18'),
                       pd.Timestamp('2020-01-01 09:10:15'),
                       pd.Timestamp('2020-01-01 09:15:45'),
                       pd.Timestamp('2020-01-01 09:25:34'),
                       pd.Timestamp('2020-01-01 09:40:25'),
                       pd.Timestamp('2020-01-01 09:52:53'),
                       pd.Timestamp('2020-01-01 06:08:15'),
                       pd.Timestamp('2020-01-01 06:16:51'),
                       pd.Timestamp('2020-01-01 09:25:34'),
                       pd.Timestamp('2020-01-01 09:40:25'),
                       pd.Timestamp('2020-01-01 09:52:53'),
                       pd.Timestamp('2020-01-01 09:10:15'),
                       pd.Timestamp('2020-01-01 09:15:45'),
                       pd.Timestamp('2020-01-01 06:31:41'),
                       pd.Timestamp('2020-01-01 06:45:25'),
                       pd.Timestamp('2020-01-01 06:49:18')],
            TID: ['abc-00002020010106', 'abc-00002020010106',
                  'abc-00002020010106', 'abc-00002020010106',
                  'abc-00002020010106', 'def-11112020010109',
                  'def-11112020010109', 'def-11112020010109',
                  'def-11112020010109', 'def-11112020010109',
                  'abc-00002020010106_def-11112020010109',
                  'abc-00002020010106_def-11112020010109',
                  'abc-00002020010106_def-11112020010109',
                  'abc-00002020010106_def-11112020010109',
                  'abc-00002020010106_def-11112020010109',
                  'def-11112020010109_abc-00002020010106',
                  'def-11112020010109_abc-00002020010106',
                  'def-11112020010109_abc-00002020010106',
                  'def-11112020010109_abc-00002020010106',
                  'def-11112020010109_abc-00002020010106']
        }
    )

    instance_crossover_augmentation(move_df, label_trajectory=LOCAL_LABEL)
    assert_frame_equal(move_df, expected)
