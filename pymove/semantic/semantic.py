import time

import numpy as np

from pymove.preprocessing import filters, segmentation, stay_point_detection
from pymove.utils.constants import (
    BLOCK,
    DEACTIVATED,
    DIST_TO_PREV,
    JUMP,
    OUT_BBOX,
    SEGMENT_STOP,
    SHORT,
    TID_PART,
    TIME_TO_PREV,
    TRAJ_ID,
)


def _end_create_operation(move_data, new_label, start_time, inplace):
    """
    Returns the dataframe after create operation

    move_data: dataframe
        The input trajectories data.
    new_label: string
        The name of the new feature with detected deactivated signals.
    start_time: float
        Time when the operation started
    inplace : boolean
        if set to true the original dataframe will be altered to contain
        the result of the filtering, otherwise a copy will be returned.

    Returns
    _______
    dataframe or None
        DataFrame with the additional features

    """

    print(move_data[new_label].value_counts())
    print('\nTotal Time: %.2f seconds' % (time.time() - start_time))
    print('-----------------------------------------------------\n')

    if not inplace:
        return move_data


def _process_simple_filter(
        move_data, new_label, feature, value, start_time, inplace
):
    """
    Processes create operation with simple filter

    move_data: dataframe
        The input trajectories data.
    new_label: string
        The name of the new feature with detected deactivated signals.
    feature: string
        Feature column to compare
    value: float
        Value to compare feature
    start_time: float
        Time when the operation started
    inplace : boolean
        if set to true the original dataframe will be altered to contain
        the result of the filtering, otherwise a copy will be returned.

    Returns
    _______
    dataframe or None
        DataFrame with the additional features

    """

    move_data[new_label] = False
    filter_ = move_data[feature] >= value
    idx_start = move_data[filter_].index
    idx_end = idx_start - np.full(len(idx_start), 1, dtype=np.int32)
    idx = np.concatenate([idx_start, idx_end], axis=0)
    move_data.at[idx, new_label] = True

    return _end_create_operation(
        move_data, new_label, start_time, inplace
    )


def create_or_update_out_of_the_bbox(
        move_data, bbox, new_label=OUT_BBOX, inplace=True
):
    """
    Create or update a boolean feature to detect points out of the bbox.

    Parameters
    __________
    move_data: dataframe
        The input trajectories data.
    bbox : tuple
        Tuple of 4 elements, containing the minimum and maximum values
        of latitude and longitude of the bounding box.
    new_label: string, optional, default 'out_Bbox'
        The name of the new feature with detected points out of the bbox.
    inplace : boolean, optional, default True
        if set to true the original dataframe will be altered to contain
        the result of the filtering, otherwise a copy will be returned.

    Returns
    _______
    dataframe
        Returns dataframe with a boolean feature with detected
        points out of the bbox.

    """

    try:
        if not inplace:
            move_data = move_data[:]

        print('\nCreate or update boolean feature to detect points out of the bbox')
        start_time = time.time()
        filtered_ = filters.by_bbox(move_data, bbox, filter_out=True)

        print('...Creating a new label named as %s' % new_label)
        move_data[new_label] = False
        if filtered_.shape[0] > 0:
            print('...Setting % as True\n' % new_label)
            move_data.at[filtered_.index, new_label] = True

        return _end_create_operation(
            move_data, new_label, start_time, inplace
        )
    except Exception as e:
        raise e


def create_or_update_gps_deactivated_signal(
        move_data,
        max_time_between_adj_points=7200,
        new_label=DEACTIVATED,
        inplace=True
):
    """
    Create or update a feature deactivate_signal if the max time between
    adjacent points is equal or less than max_time_between_adj_points.

    Parameters
    __________
    move_data: dataframe
        The input trajectories data.
    max_time_between_adj_points: float, int, optional, defualt 7200.
        The max time between adjacent points.
    new_label: string, optional, default 'deactivate_signal'.
        The name of the new feature with detected deactivated signals.
    inplace : boolean, optional, default True
        if set to true the original dataframe will be altered to contain
        the result of the filtering, otherwise a copy will be returned.

    Returns
    _______
    dataframe or None
        DataFrame with the additional features
        'time_to_prev', 'time_to_next', 'time_prev_to_next', 'deactivate_signal'

    """

    try:
        if not inplace:
            move_data = move_data[:]

        start_time = time.time()
        message = 'Create or update deactivated signal if time max > %s seconds\n'
        print(message % max_time_between_adj_points)
        move_data.generate_time_features()

        return _process_simple_filter(
            move_data,
            new_label,
            TIME_TO_PREV,
            max_time_between_adj_points,
            start_time,
            inplace
        )
    except Exception as e:
        raise e


def create_or_update_gps_jump(
        move_data,
        max_dist_between_adj_points=3000,
        new_label=JUMP,
        inplace=True
):
    """
    Create or update Jump if the maximum distance between adjacent
    points is greater than max_dist_between_adj_points.

    Parameters
    __________
    move_data: dataframe
        The input trajectories data.
    max_dist_between_adj_points: float, int, optional, default 3000.
        The maximum distance between adjacent points.
    new_label: string, optional, default 'gps_jump'.
        The name of the new feature with detected deactivated signals.
    inplace : boolean, optional, default True
        if set to true the original dataframe will be altered to contain
        the result of the filtering, otherwise a copy will be returned.

    Returns
    _______
    dataframe or None
        DataFrame with the additional features:
        'dist_to_prev', 'dist_to_next', 'dist_prev_to_next', 'jump'

    """
    try:
        if not inplace:
            move_data = move_data[:]

        start_time = time.time()
        message = 'Create or update jump if dist max > %s meters\n'
        print(message % max_dist_between_adj_points)
        move_data.generate_dist_features()

        return _process_simple_filter(
            move_data,
            new_label,
            DIST_TO_PREV,
            max_dist_between_adj_points,
            start_time,
            inplace
        )
    except Exception as e:
        raise e


def create_or_update_short_trajectory(
        move_data,
        max_dist_between_adj_points=3000,
        max_time_between_adj_points=7200,
        max_speed_between_adj_points=50,
        k_segment_max=50,
        label_tid=TID_PART,
        new_label=SHORT,
        inplace=True
):
    """

    Parameters
    ----------
    move_data : dataframe
       The input trajectory data
    max_dist_between_adj_points : float, optional, default 3000
        Specify the maximum distance a point should have from
        the previous point, in order not to be dropped
    max_time_between_adj_points : float, optional, default 7200
        Specify the maximum travel time between two adjacent points
    max_speed_between_adj_points : float, optional, default 50
        Specify the maximum speed of travel between two adjacent points
    k_segment_max: int, optional
        Specify the maximum number of segments in the trajectory
    label_tid:  str, optional, default 'tid_part'
        ?
    new_label: str, optional, default 'short_traj'
        The name of the new feature with short trajectories.
    inplace : boolean, optional, default True
        if set to true the original dataframe will be altered to
        contain the result of the filtering, otherwise a copy will be returned.

    Returns
    -------
    dataframe or None
        DataFrame with the aditional features:
       'dist_to_prev', 'time_to_prev', 'speed_to_prev', 'tid_part', 'short_traj'

    """

    try:
        if not inplace:
            move_data = move_data[:]

        print('\nCreate or update short trajectories...')
        start_time = time.time()

        segmentation.by_dist_time_speed(
            move_data,
            max_dist_between_adj_points=max_dist_between_adj_points,
            max_time_between_adj_points=max_time_between_adj_points,
            max_speed_between_adj_points=max_speed_between_adj_points,
            label_new_tid=label_tid
        )
        move_data[new_label] = False

        df_count_tid = move_data.groupby(by=label_tid).size()
        filter_ = df_count_tid <= k_segment_max
        idx = df_count_tid[filter_].index
        move_data.loc[move_data[label_tid].isin(idx), new_label] = True

        return _end_create_operation(
            move_data, new_label, start_time, inplace
        )
    except Exception as e:
        raise e


def create_or_update_gps_block_signal(
        move_data,
        max_time_stop=7200,
        new_label=BLOCK,
        label_tid=TRAJ_ID,
        inplace=True
):
    """
    Create a new feature that inform points with speed = 0

    Parameters
    __________
    move_data: dataFrame
        The input trajectories data.
    max_time_stop: float, optional, default 7200
        Maximum time allowed with speed 0
    new_label: string, optional, default 'block_signal_time'.
        The name of the new feature with detected deactivated signals.
    label_tid : str, optional, default 'tid_dist'
        The label of the column containing the ids of the formed segments.
        Is the new slitted id.
    inplace : boolean, optional, default True
        if set to true the original dataframe will be altered to contain
        the result of the filtering, otherwise a copy will be returned.

    Returns
    _______
    dataframe or None
        DataFrame with the additional features:
        'dist_to_prev', 'time_to_prev', 'speed_to_prev',
        'tid_dist', 'block_signal'

    """

    try:
        if not inplace:
            move_data = move_data[:]

        message = 'Create or update block_signal if max time stop > %s seconds\n'
        print(message % max_time_stop)
        start_time = time.time()
        segmentation.by_max_dist(
            move_data, max_dist_between_adj_points=0.0, label_new_tid=label_tid
        )

        print('Updating dist time speed values')
        move_data.generate_dist_time_speed_features(label_id=label_tid)

        move_data[new_label] = False

        # SUM the segment block to dectect the id that has or more time stopped
        df_agg_tid = move_data.groupby(by=label_tid).agg({TIME_TO_PREV: 'sum'})
        filter_ = df_agg_tid[TIME_TO_PREV] >= max_time_stop
        idx = df_agg_tid[filter_].index
        move_data.loc[move_data[label_tid].isin(idx), new_label] = True

        return _end_create_operation(
            move_data, new_label, start_time, inplace
        )
    except Exception as e:
        raise e


def filter_block_signal_by_repeated_amount_of_points(
        move_data,
        amount_max_of_points_stop=30.0,
        max_time_stop=7200,
        filter_out=False,
        inplace=False
):
    """
    Filters from dataframe points with blocked signal by ammount of points.

    Parameters
    __________
    move_data: dataFrame
        The input trajectories data.
    amount_max_of_points_stop: float, optional, default 30
        Maximum number of stopped points
    max_time_stop: float, optional, default 7200
        Maximum time allowed with speed 0
    filter_out: boolean, optional, default True
        Whether to keep or discard points with blocked signal
    inplace : boolean, optional, default True
        if set to true the original dataframe will be altered to contain
        the result of the filtering, otherwise a copy will be returned.

    Returns
    _______
    dataframe or None
        Filtered DataFrame with the additional features
        'dist_to_prev', 'time_to_prev', 'speed_to_prev',
        'tid_dist', 'block_signal'

    """

    try:
        if not inplace:
            move_data = move_data[:]

        if BLOCK not in move_data:
            create_or_update_gps_block_signal(move_data, max_time_stop)

        df_count_tid = move_data.groupby(by=[BLOCK]).count()
        filter_ = df_count_tid > amount_max_of_points_stop
        if filter_out:
            idx = df_count_tid[filter_].index
        else:
            idx = df_count_tid[~filter_].index

        move_data = move_data[move_data[BLOCK].isin(idx)]
        if not inplace:
            return move_data

    except Exception as e:
        raise e


def filter_block_signal_by_time(
        move_data,
        max_time_stop=7200,
        filter_out=False,
        inplace=False
):
    """
    Filters from dataframe points with blocked signal by time.

    Parameters
    __________
    move_data: dataFrame
        The input trajectories data.
    max_time_stop: float, optional, default 7200
        Maximum time allowed with speed 0
    filter_out: boolean, optional, default True
        Whether to keep or discard points with blocked signal
    inplace : boolean, optional, default True
        if set to true the original dataframe will be altered to contain
        the result of the filtering, otherwise a copy will be returned.

    Returns
    _______
    dataframe or None
        Filtered DataFrame with the additional features
        'dist_to_prev', 'time_to_prev', 'speed_to_prev',
        'tid_dist', 'block_signal'

    """

    try:
        if not inplace:
            move_data = move_data[:]

        if BLOCK not in move_data:
            create_or_update_gps_block_signal(move_data, max_time_stop)

        df_agg_tid = move_data.groupby(by=BLOCK).agg({TIME_TO_PREV: 'sum'})
        filter_ = df_agg_tid[TIME_TO_PREV] > max_time_stop
        if filter_out:
            idx = df_agg_tid[filter_].index
        else:
            idx = df_agg_tid[filter_].index

        move_data = move_data[move_data[BLOCK].isin(idx)]
        if not inplace:
            return move_data

    except Exception as e:
        raise e


def filter_longer_time_to_stop_segment_by_id(
        move_data,
        label_id=TRAJ_ID,
        label_segment_stop=SEGMENT_STOP,
        filter_out=False,
        inplace=False
):
    """
    Filters from dataframe segment with longest stop time.

    Parameters
    __________
    move_data: dataFrame
        The input trajectories data.
    label_tid : str, optional, default 'id'
        The label of the column containing the ids of the formed segments.
        Is the new slitted id.
    label_segment_stop: str, optional, default 'segment_stop'
    filter_out: boolean, optional, default True
        Whether to keep or discard points with blocked signal
    inplace : boolean, optional, default True
        if set to true the original dataframe will be altered to contain
        the result of the filtering, otherwise a copy will be returned.

    Returns
    _______
    dataframe or None
        Filtered DataFrame with the additional features
        'dist_to_prev', 'time_to_prev', 'speed_to_prev',
        'tid_dist', 'block_signal'

    """
    try:
        if not inplace:
            move_data = move_data[:]

        if label_segment_stop not in move_data:
            stay_point_detection.create_or_update_move_stop_by_dist_time(
                move_data
            )

        df_agg_id_stop = move_data.groupby(
            [label_id, label_segment_stop], as_index=False
        ).agg({TIME_TO_PREV: 'sum'})

        filter_ = df_agg_id_stop.groupby(
            [label_id], as_index=False
        )[TIME_TO_PREV].idxmax()

        if filter_out:
            segments = df_agg_id_stop.loc[filter_]
        else:
            segments = df_agg_id_stop.loc[~df_agg_id_stop.index.isin(filter_)]
        segments = segments[label_segment_stop]

        move_data = move_data[move_data[label_segment_stop].isin(segments)]
        if not inplace:
            return move_data

    except Exception as e:
        raise e
