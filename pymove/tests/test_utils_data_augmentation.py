import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal
from pandas.testing import assert_frame_equal

from pymove.utils.constants import (
    DATETIME,
    LABEL,
    LATITUDE,
    LOCAL_LABEL,
    LONGITUDE,
    TID,
    TRAJ_ID,
    TRAJECTORY,
)
from pymove.utils.data_augmentation import (
    append_row,
    augmentation_trajectories_df,
    generate_target_feature,
    generate_trajectories_df,
    insert_points_in_df,
    instance_crossover,
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
    TRAJECTORY: [[5, 7], [2, 4, 6, 8]],
    DATETIME: [[pd.Timestamp('2020-01-01 06:08:15'),
                pd.Timestamp('2020-01-01 06:16:51')],
               [pd.Timestamp('2020-01-01 09:10:15'),
                pd.Timestamp('2020-01-01 09:15:45')]],
    LATITUDE: [[3.1234567165374756, 3.1234567165374756],
               [3.1234567165374756, 3.1234567165374756]],
    LONGITUDE: [[38.12345504760742, 38.12345504760742],
                [38.12345504760742, 38.12345504760742]],
    TID: [['abc-00002020010106', 'abc-00002020010106'],
          ['def-11112020010109', 'def-11112020010109']],
    LABEL: [9, 9]
}


def test_generate_trajectories_df():

    move_df = pd.DataFrame(
        data=np.array(list_data1, dtype=object),
        columns=[TRAJ_ID, LOCAL_LABEL, LATITUDE, LONGITUDE, DATETIME, TID]
    )

    expected = pd.DataFrame(
        data={
            TRAJ_ID: ['abc-0000', 'def-1111'],

            TRAJECTORY: [[1, 2, 3, 4, 9], [5, 6, 7, 8, 9]],

            DATETIME: [[pd.Timestamp('2020-01-01 06:08:15'),
                        pd.Timestamp('2020-01-01 06:16:51'),
                        pd.Timestamp('2020-01-01 06:31:41'),
                        pd.Timestamp('2020-01-01 06:45:25'),
                        pd.Timestamp('2020-01-01 06:49:18')],
                       [pd.Timestamp('2020-01-01 09:10:15'),
                        pd.Timestamp('2020-01-01 09:15:45'),
                        pd.Timestamp('2020-01-01 09:25:34'),
                        pd.Timestamp('2020-01-01 09:40:25'),
                        pd.Timestamp('2020-01-01 09:52:53')]],

            LATITUDE: [[3.1234567165374756,
                        3.1234567165374756, 3.1234567165374756,
                        3.1234567165374756, 3.1234567165374756],
                       [3.1234567165374756,
                        3.1234567165374756, 3.1234567165374756,
                        3.1234567165374756, 3.1234567165374756]],

            LONGITUDE: [[38.12345504760742,
                         38.12345504760742, 38.12345504760742,
                         38.12345504760742, 38.12345504760742],
                        [38.12345504760742,
                         38.12345504760742, 38.12345504760742,
                         38.12345504760742, 38.12345504760742]],

            TID: [['abc-00002020010106',
                   'abc-00002020010106', 'abc-00002020010106',
                   'abc-00002020010106', 'abc-00002020010106'],
                  ['def-11112020010109',
                   'def-11112020010109', 'def-11112020010109',
                   'def-11112020010109', 'def-11112020010109']]})

    traj_df = generate_trajectories_df(move_df)
    assert_frame_equal(expected, traj_df)


def test_generate_target_feature():
    traj_df = pd.DataFrame(
        data={
            TRAJ_ID: ['abc-0000'],
            TRAJECTORY: [[1, 2]],
            DATETIME: [[pd.Timestamp('2020-01-01 06:08:15'),
                        pd.Timestamp('2020-01-01 06:16:51')]],
            LATITUDE: [[3.1234567165374756, 3.1234567165374756]],
            LONGITUDE: [[38.12345504760742, 38.12345504760742]],
            TID: [['abc-00002020010106', 'abc-00002020010106']]})

    expected = pd.DataFrame(
        data={
            TRAJ_ID: ['abc-0000'],
            TRAJECTORY: [[1]],
            DATETIME: [[pd.Timestamp('2020-01-01 06:08:15'),
                        pd.Timestamp('2020-01-01 06:16:51')]],
            LATITUDE: [[3.1234567165374756, 3.1234567165374756]],
            LONGITUDE: [[38.12345504760742, 38.12345504760742]],
            TID: [['abc-00002020010106', 'abc-00002020010106']],
            LABEL: [2]})

    generate_target_feature(traj_df)
    assert_frame_equal(expected, traj_df)


def test_split_crossover():
    s1 = [0, 2, 4, 6, 8]
    s2 = [1, 3, 5, 7, 9]

    expected1 = [0, 2, 5, 7, 9]
    expected2 = [1, 3, 4, 6, 8]

    s1, s2 = split_crossover(s1, s2)

    assert_array_almost_equal(expected1, s1)
    assert_array_almost_equal(expected2, s2)


def test_append_row():
    traj_df = pd.DataFrame(
        data={
            TRAJ_ID: ['abc-0000'],
            TRAJECTORY: [[1]],
            DATETIME: [[pd.Timestamp('2020-01-01 06:08:15')]],
            LATITUDE: [[3.1234567165374756]],
            LONGITUDE: [[38.12345504760742]],
            TID: [['abc-00002020010106']],
            LABEL: 2})

    expected = pd.DataFrame(
        data={
            TRAJ_ID: ['abc-0000', 'def-1111'],
            TRAJECTORY: [[1], [5]],
            DATETIME: [[pd.Timestamp('2020-01-01 06:08:15')],
                       [pd.Timestamp('2020-01-01 07:10:15')]],
            LATITUDE: [[3.1234567165374756], [3.623471260070801]],
            LONGITUDE: [[38.12345504760742], [38.397525787353516]],
            TID: [['abc-00002020010106'], ['def-1111202001010']],
            LABEL: [2.0, 6.0]})

    row = pd.Series(
        data={TRAJ_ID: 'def-1111',
              TRAJECTORY: [5],
              DATETIME: [pd.Timestamp('2020-01-01 7:10:15')],
              LATITUDE: [3.6234712461],
              LONGITUDE: [38.39752597257],
              TID: ['def-1111202001010'],
              LABEL: 6})

    append_row(traj_df, row=row)
    assert_frame_equal(expected, traj_df)


def test_augmentation_trajectories_df():
    traj_df = pd.DataFrame(
        data=list_data2
    )

    expected = pd.DataFrame(
        data={
            TRAJ_ID: ['abc-0000_def-1111', 'def-1111_abc-0000',
                      'def-1111_abc-0000', 'abc-0000_def-1111'],
            TRAJECTORY: [[5, 6, 8], [2, 4, 7], [2, 4, 7], [5, 6, 8]],
            DATETIME: [[pd.Timestamp('2020-01-01 06:08:15'),
                        pd.Timestamp('2020-01-01 09:15:45')],
                       [pd.Timestamp('2020-01-01 09:10:15'),
                        pd.Timestamp('2020-01-01 06:16:51')],
                       [pd.Timestamp('2020-01-01 09:10:15'),
                        pd.Timestamp('2020-01-01 06:16:51')],
                       [pd.Timestamp('2020-01-01 06:08:15'),
                        pd.Timestamp('2020-01-01 09:15:45')]],
            LATITUDE: [[3.1234567165374756, 3.1234567165374756],
                       [3.1234567165374756, 3.1234567165374756],
                       [3.1234567165374756, 3.1234567165374756],
                       [3.1234567165374756, 3.1234567165374756]],
            LONGITUDE: [[38.12345504760742, 38.12345504760742],
                        [38.12345504760742, 38.12345504760742],
                        [38.12345504760742, 38.12345504760742],
                        [38.12345504760742, 38.12345504760742]],
            TID: [['abc-0000202001010', 'abc-0000202001010'],
                  ['def-1111202001011', 'def-1111202001011'],
                  ['def-1111202001012', 'def-1111202001012'],
                  ['abc-0000202001013', 'abc-0000202001013']],
            LABEL: [9, 9, 9, 9]})

    aug_df = augmentation_trajectories_df(traj_df)
    assert_frame_equal(expected, aug_df)


def test_insert_points_in_df():
    move_df = pd.DataFrame(
        data=np.array(list_data1, dtype=object),
        columns=[TRAJ_ID, LOCAL_LABEL, LATITUDE, LONGITUDE, DATETIME, TID]
    )

    aug_df = pd.DataFrame(
        data={
            TRAJ_ID: ['abc-0000_def-1111'],
            TRAJECTORY: [[5, 6, 7]],
            DATETIME: [[pd.Timestamp('2020-01-01 06:08:15'),
                        pd.Timestamp('2020-01-01 09:15:45'),
                        pd.Timestamp('2020-01-01 09:30:45'),
                        pd.Timestamp('2020-01-01 09:30:45')]],
            LATITUDE: [[3.1234567165374756, 3.1234567165374756,
                        3.1234567165374756, 3.1234567165374756]],
            LONGITUDE: [[38.12345504760742, 38.12345504760742,
                         38.12345504760742, 38.12345504760742]],
            TID: [['abc-0000202001010', 'abc-0000202001010',
                   'abc-0000202001010', 'abc-0000202001010']],
            LABEL: [9]})

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


def test_instance_crossover():
    move_df = pd.DataFrame(
        data=np.array(list_data1, dtype=object),
        columns=[TRAJ_ID, LOCAL_LABEL, LATITUDE, LONGITUDE, DATETIME, TID]
    )

    expected = pd.DataFrame(
        data=np.array([['abc-0000', 1, 3.1234567, 38.1234567,
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
                       ['abc-0000_def-1111', 1, 3.1234567165374756, 38.12345504760742,
                        pd.Timestamp('2020-01-01 06:08:15'), 'abc-0000202001010'],
                       ['abc-0000_def-1111', 2, 3.1234567165374756, 38.12345504760742,
                        pd.Timestamp('2020-01-01 06:16:51'), 'abc-0000202001010'],
                       ['abc-0000_def-1111', 7, 3.1234567165374756, 38.12345504760742,
                        pd.Timestamp('2020-01-01 09:25:34'), 'abc-0000202001010'],
                       ['abc-0000_def-1111', 8, 3.1234567165374756, 38.12345504760742,
                        pd.Timestamp('2020-01-01 09:40:25'), 'abc-0000202001010'],
                       ['abc-0000_def-1111', 9, 3.1234567165374756, 38.12345504760742,
                        pd.Timestamp('2020-01-01 09:52:53'), 'abc-0000202001010'],
                       ['def-1111_abc-0000', 5, 3.1234567165374756, 38.12345504760742,
                        pd.Timestamp('2020-01-01 09:10:15'), 'def-1111202001011'],
                       ['def-1111_abc-0000', 6, 3.1234567165374756, 38.12345504760742,
                        pd.Timestamp('2020-01-01 09:15:45'), 'def-1111202001011'],
                       ['def-1111_abc-0000', 3, 3.1234567165374756, 38.12345504760742,
                        pd.Timestamp('2020-01-01 06:31:41'), 'def-1111202001011'],
                       ['def-1111_abc-0000', 4, 3.1234567165374756, 38.12345504760742,
                        pd.Timestamp('2020-01-01 06:45:25'), 'def-1111202001011'],
                       ['def-1111_abc-0000', 9, 3.1234567165374756, 38.12345504760742,
                        pd.Timestamp('2020-01-01 06:49:18'), 'def-1111202001011'],
                       ['def-1111_abc-0000', 5, 3.1234567165374756, 38.12345504760742,
                        pd.Timestamp('2020-01-01 09:10:15'), 'def-1111202001012'],
                       ['def-1111_abc-0000', 6, 3.1234567165374756, 38.12345504760742,
                        pd.Timestamp('2020-01-01 09:15:45'), 'def-1111202001012'],
                       ['def-1111_abc-0000', 3, 3.1234567165374756, 38.12345504760742,
                        pd.Timestamp('2020-01-01 06:31:41'), 'def-1111202001012'],
                       ['def-1111_abc-0000', 4, 3.1234567165374756, 38.12345504760742,
                        pd.Timestamp('2020-01-01 06:45:25'), 'def-1111202001012'],
                       ['def-1111_abc-0000', 9, 3.1234567165374756, 38.12345504760742,
                        pd.Timestamp('2020-01-01 06:49:18'), 'def-1111202001012'],
                       ['abc-0000_def-1111', 1, 3.1234567165374756, 38.12345504760742,
                        pd.Timestamp('2020-01-01 06:08:15'), 'abc-0000202001013'],
                       ['abc-0000_def-1111', 2, 3.1234567165374756, 38.12345504760742,
                        pd.Timestamp('2020-01-01 06:16:51'), 'abc-0000202001013'],
                       ['abc-0000_def-1111', 7, 3.1234567165374756, 38.12345504760742,
                        pd.Timestamp('2020-01-01 09:25:34'), 'abc-0000202001013'],
                       ['abc-0000_def-1111', 8, 3.1234567165374756, 38.12345504760742,
                        pd.Timestamp('2020-01-01 09:40:25'), 'abc-0000202001013'],
                       ['abc-0000_def-1111', 9, 3.1234567165374756, 38.12345504760742,
                        pd.Timestamp('2020-01-01 09:52:53'), 'abc-0000202001013']],
                      dtype=object),
        columns=[TRAJ_ID, LOCAL_LABEL, LATITUDE, LONGITUDE, DATETIME, TID]
    )

    instance_crossover(move_df)
    assert_frame_equal(expected, move_df)
