# from pymove import trajutils
from pymove import segmentation

from pymove.core import dataframe
from pymove import filters
from tqdm import tqdm_notebook as tqdm
from pymove import utils as ut
import time
import numpy as np
import pandas as pd


def create_or_update_out_of_the_bbox(df_, bbox, new_label='out_Bbox'):
    """
    Create or update a boolean feature to detect points out of the bbox.

    Pareameters
    ___________
    move_data: dataframe
        The input trajectories data.
    bbox : tuple
        Tuple of 4 elements, containg the minimum and maximum values of latitude and longitude of the bounding box.
    new_label: string, optional('out_Bbox', by default)
        The name of the new boolean feature with detected points out of the bbox.

    Returns
    _______
    df_: dataframe
        Returns dataframe with a boolean feature with detected points out of the bbox.
    """
    try:
        print('\nCreate or update a boolean feature to detect points out of the bbox')
        start_time = time.time()
        df_out_bbox = filters.by_bbox(df_, bbox, filter_out=True)

        print('...Creating a new label named as {}'.format(new_label))
        df_[new_label] = False
        if df_out_bbox.shape[0] > 0:
            print('...Setting {} as True\n'.format(new_label))
            df_.at[df_out_bbox.index, new_label] = True

        print(df_[new_label].value_counts())
        print('\nTotal Time: {:.2f} seconds'.format((time.time() - start_time)))
        print('-----------------------------------------------------\n')
    except Exception as e:
        raise e


def create_or_update_deactivate_signal(df_, max_time_between_adj_points=7200, label_time='time_to_prev'):
    """
    Create or update a feature deactivate_signal if the max time between  adjacent points is equal or less than max_time_between_adj_points.

    Parameters
    __________
    df_: dataframe
        The input trajectories data.
    max_time_between_adj_points: float, int, optional, defualt 7200.
        The max time between adjacent points.
    label_time: string, optional, defualt 'time_to_prev'.

    Returns
    _______
        Returns a dataframe with 4 new features, dist_to_prev, time_to_prev, speed_to_prev, and deactivate_signal.
    """
    try:
        print('Create or update deactivate signal if time max > {} seconds\n'.format(max_time_between_adj_points))
        start_time = time.time()
        if df_.index.name is not None:
            print('...reseting index')
            df_.reset_index(inplace=True)

        # trajutils.create_update_dist_time_speed_features(df_)
        df_.generate_dist_time_speed_features()
        if label_time in df_:
            print('...creating or update columns deactivate_signal')
            df_['deactivate_signal'] = False
            idx_start = df_[df_[label_time] >= max_time_between_adj_points].index
            idx_end = idx_start - np.full(len(idx_start), 1, dtype=np.int32)
            idx = np.concatenate([idx_start, idx_end], axis=0)
            df_.at[idx, 'deactivate_signal'] = True
        else:
            print('{} not in Dataframe'.format(label_time))

        print('\nTotal Time: {:.2f} seconds'.format((time.time() - start_time)))
        print('-----------------------------------------------------\n')
    except Exception as e:
        raise e


def create_or_update_jump_by_dist_max(df_, max_dist_between_adj_points=3000, label_dist='dist_to_prev'):
    """
    Create or update Jump if the maximum distance between adjacent points is greater than max_dist_between_adj_points.

    Parameters
    __________
    df_: dataframe
        The input trajectories data.
    max_dist_between_adj_points: float, int, optional, defualt 3000.
        The maximum distance between adjacent points.
    label_dist: string, optional, defualt 'dist_to_prev'.

    Returns
    _______
        The dataframe with 4 new features dist_to_prev, time_to_prev, speed_to_prev, and jump.
    """
    try:
        print('Create or update Jump if dist max > {} meters\n'.format(max_dist_between_adj_points))
        start_time = time.time()

        if df_.index.name is not None:
            print('...reseting index')
            df_.reset_index(inplace=True)

        # trajutils.create_update_dist_time_speed_features(df_)
        df_.generate_dist_time_speed_features()

        if label_dist in df_:
            print('...creating or update columns Jump')
            df_['jump'] = False
            idx_start = df_[df_[label_dist] >= max_dist_between_adj_points].index
            idx_end = idx_start - np.full(len(idx_start), 1, dtype=np.int32)
            idx = np.concatenate([idx_start, idx_end], axis=0)
            df_.at[idx, 'jump'] = True
        else:
            print('{} not in Dataframe'.format(label_dist))

        print('\nTotal Time: {:.2f} seconds'.format((time.time() - start_time)))
        print('-----------------------------------------------------\n')
    except Exception as e:
        raise e


def create_or_update_block_signal_by_time_max(df_, label_id='id', time_max_stop=7200.00):
    """
    Create a new feature that inform poits with speed = 0

    Parameters
    __________
    df_: dataFrame
        The input trajectories data.
    label_id: string, optional, defualt 'id'.

    time_max_stop: float, int, optional, defualt 7200.00.

    Returns
    _______

    """
    try:
        start_time = time.time()
        """ set information"""
        label_segment = 'segment_block'

        # trajutils.create_update_dist_time_speed_features(df_)
        df_.generate_dist_time_speed_features()

        """ Segment trajectory by 0.0 distance """
        # trajutils.segment_traj_by_max_speed(df_, label_id=label_id, max_speed_between_adj_points=0.0,
        # label_segment=label_segment)

        # trajutils.segment_traj_by_max_dist(df_, label_id=label_id, max_dist_between_adj_points=0.0, label_segment=label_segment)
        segmentation.by_max_dist(df_, label_id=label_id, max_dist_between_adj_points=0.0, label_segment=label_segment)

        if label_segment in df_:

            # trajutils.create_update_dist_time_speed_features(df_, label_id=label_segment)
            df_.generate_dist_time_speed_features(label_id=label_segment)

            print('Create or update block signal feature as True or False')
            label_block = 'block_signal'

            # set label_block as False
            df_[label_block] = False
            print('... new label named as {} was created in Dataframe'.format(label_block))

            # SUM the segment block to dectect the id that has or more time stopped
            df_agg_tid = df_.groupby(by=label_segment).agg({'time_to_prev': 'sum'})

            # filter the ids by a time_max_to_stop
            idx = df_agg_tid[df_agg_tid['time_to_prev'] > time_max_stop].index

            df_.loc[df_[label_segment].isin(idx), label_block] = True
            print('... block signal was set to true using {} seconds to time_max_stop'.format(time_max_stop))

        else:
            print('{} is not in Dataframe'.format(label_segment))
        print('\nTotal Time: {:.2f} seconds'.format((time.time() - start_time)))
        print('-----------------------------------------------------\n')
    except Exception as e:
        raise e


def create_or_update_short_traj_detect(df_, label_tid='id', max_dist_between_adj_points=3000,
                                       max_time_between_adj_points=7200,
                                       max_speed_between_adj_points=50.0, k_segment_max=50):
    try:
        print('\nCreate or update short_traj as True or False')
        start_time = time.time()
        """ Create or update features"""
        # trajutils.create_update_dist_time_speed_features(df_, label_tid)

        label_segment = 'segment_traj'

        """ Segment trajectory by dist, time and speed"""
        # trajutils.segment_traj_by_max_dist_time_speed(df_, label_tid, max_dist_between_adj_points,
        # max_time_between_adj_points, max_speed_between_adj_points, label_segment=label_segment)
        segmentation.by_dist_time_speed(df_, label_tid, max_dist_between_adj_points, max_time_between_adj_points,
                                        max_speed_between_adj_points, label_segment=label_segment)

        if label_segment in df_:
            label_short_traj = 'short_traj'
            df_[label_short_traj] = False
            print('... new lab named as {} was created in Dataframe'.format(label_short_traj))

            df_count_tid = df_.groupby(by=label_segment).size()
            idx = df_count_tid[df_count_tid < k_segment_max].index

            df_.loc[df_[label_segment].isin(idx), label_short_traj] = True
            print('... outlier was set to true in column {}'.format(label_short_traj))
        else:
            print('{} is not in Dataframe'.format(label_segment))
        print('\nTotal Time: {:.2f} seconds'.format((time.time() - start_time)))
        print('-----------------------------------------------------\n')
    except Exception as e:
        raise e


def filter_block_signal_by_repeated_amount_of_points(df_, amount_max_of_points_stop=30.0, filter_out=False):
    try:
        # time_max_stop = 30 points if gap between two points is 1 minute
        label_block = 'block_signal'
        if (label_block in df_):
            # get id amount of stop points
            df_count_tid = df_.groupby(by=[label_block]).size()

            if filter_out:
                idx = df_count_tid[df_count_tid > amount_max_of_points_stop].index
            else:
                idx = df_count_tid[df_count_tid <= amount_max_of_points_stop].index
            return df_[df_[label_block].isin(idx)]
        else:
            print('\n...{} does not exist'.format(label_block))
    except Exception as e:
        raise e


def filter_block_signal_by_time(df_, time_max_stop=7200.00, filter_out=False):
    try:
        # time_max_stop = 1800s --> 30 minutos
        label_block = 'block_signal'
        if (label_block in df_):
            df_agg_tid = df_.groupby(by=label_block).agg({'time_to_prev': 'sum'})
            if filter_out:
                idx = df_agg_tid[(df_agg_tid['time_to_prev'] > time_max_stop)].index
            else:
                idx = df_agg_tid[(df_agg_tid['time_to_prev'] <= time_max_stop)].index

            return df_[df_[label_block].isin(idx)]
        else:
            print('\n...{} does not exist'.format(label_block))
    except Exception as e:
        raise e


def filter_longer_time_to_stop_segment_by_id(df_, label_id='id', label_segment_stop='segment_stop'):
    try:
        if (label_id in df_) & (label_segment_stop in df_):
            # group segment by id and stop
            df_agg_id_stop = df_.groupby([label_id, label_segment_stop], as_index=False).agg({'time_to_prev': 'sum'})
            # get time max to each id
            segments = df_agg_id_stop.loc[df_agg_id_stop.groupby([label_id], as_index=False)["time_to_prev"].idxmax()][
                label_segment_stop]

            return df_[df_[label_segment_stop].isin(segments)]
    except Exception as e:
        raise e
