from typing import TYPE_CHECKING, Optional, Text, Tuple, Union

import numpy as np
from pandas import DataFrame

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
from pymove.utils.log import timer_decorator

if TYPE_CHECKING:
    from pymove.core.pandas import PandasMoveDataFrame
    from pymove.core.dask import DaskMoveDataFrame


def _end_create_operation(
    move_data: DataFrame, new_label: Text, inplace: bool
) -> Optional[DataFrame]:
    """
    Returns the dataframe after create operation

    move_data: dataframe
        The input trajectories data.
    new_label: string
        The name of the new feature with detected deactivated signals.
    inplace : boolean
        if set to true the original dataframe will be altered to contain
        the result of the filtering, otherwise a copy will be returned.

    Returns
    _______
    DataFrame
        DataFrame with the additional features or None

    """

    print(move_data[new_label].value_counts())
    if not inplace:
        return move_data


def _process_simple_filter(
    move_data: DataFrame, new_label: Text, feature: Text, value: float, inplace: bool
) -> Optional[DataFrame]:
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
    inplace : boolean
        if set to true the original dataframe will be altered to contain
        the result of the filtering, otherwise a copy will be returned.

    Returns
    _______
    DataFrame
        DataFrame with the additional features or None

    """

    move_data[new_label] = False
    filter_ = move_data[feature] >= value
    idx_start = move_data[filter_].index
    idx_end = idx_start - np.full(len(idx_start), 1, dtype=np.int32)
    idx = np.concatenate([idx_start, idx_end], axis=0)
    move_data.at[idx, new_label] = True

    return _end_create_operation(
        move_data, new_label, inplace
    )


@timer_decorator
def create_or_update_out_of_the_bbox(
    move_data: DataFrame,
    bbox: Tuple[int, int, int, int],
    new_label: Optional[Text] = OUT_BBOX,
    inplace: Optional[bool] = True
) -> Optional[DataFrame]:
    """
    Create or update a boolean feature to detect points out of the bbox.

    Parameters
    __________
    move_data: dataframe
        The input trajectories data.
    bbox : tuple
        Tuple of 4 elements, containing the minimum and maximum values
        of latitude and longitude of the bounding box.
    new_label: string, optional
        The name of the new feature with detected points out of the bbox,
        by default OUT_BBOX
    inplace : boolean, optional
        if set to true the original dataframe will be altered to contain
        the result of the filtering, otherwise a copy will be returned,
        by default True

    Returns
    _______
    DataFrame
        Returns dataframe with a boolean feature with detected
        points out of the bbox, or None

    """

    if not inplace:
        move_data = move_data[:]

    print('\nCreate or update boolean feature to detect points out of the bbox')
    filtered_ = filters.by_bbox(move_data, bbox, filter_out=True)

    print('...Creating a new label named as %s' % new_label)
    move_data[new_label] = False
    if filtered_.shape[0] > 0:
        print('...Setting % as True\n' % new_label)
        move_data.at[filtered_.index, new_label] = True

    return _end_create_operation(
        move_data, new_label, inplace
    )


@timer_decorator
def create_or_update_gps_deactivated_signal(
    move_data: Union['PandasMoveDataFrame', 'DaskMoveDataFrame'],
    max_time_between_adj_points: Optional[float] = 7200,
    new_label: Optional[Text] = DEACTIVATED,
    inplace: Optional[bool] = True
) -> Optional[Union['PandasMoveDataFrame', 'DaskMoveDataFrame']]:
    """
    Create or update a feature deactivate_signal if the max time between
    adjacent points is equal or less than max_time_between_adj_points.

    Parameters
    __________
    move_data: dataframe
        The input trajectories data.
    max_time_between_adj_points: float, optional
        The max time between adjacent points, by default 7200
    new_label: string, optional
        The name of the new feature with detected deactivated signals,
        by default DEACTIVATED
    inplace : boolean, optional
        if set to true the original dataframe will be altered to contain
        the result of the filtering, otherwise a copy will be returned,
        by default True

    Returns
    _______
    DataFrame
        DataFrame with the additional features or None
        'time_to_prev', 'time_to_next', 'time_prev_to_next', 'deactivate_signal'

    """

    if not inplace:
        move_data = move_data[:]

    message = 'Create or update deactivated signal if time max > %srs seconds\n'
    print(message % max_time_between_adj_points)
    move_data.generate_time_features()

    return _process_simple_filter(
        move_data,
        new_label,
        TIME_TO_PREV,
        max_time_between_adj_points,
        inplace
    )


@timer_decorator
def create_or_update_gps_jump(
    move_data: Union['PandasMoveDataFrame', 'DaskMoveDataFrame'],
    max_dist_between_adj_points: Optional[float] = 3000,
    new_label: Optional[Text] = JUMP,
    inplace: Optional[bool] = True
) -> Optional[Union['PandasMoveDataFrame', 'DaskMoveDataFrame']]:
    """
    Create or update Jump if the maximum distance between adjacent
    points is greater than max_dist_between_adj_points.

    Parameters
    __________
    move_data: dataframe
        The input trajectories data.
    max_dist_between_adj_points: float, optional
        The maximum distance between adjacent points, by default 3000
    new_label: string, optional
        The name of the new feature with detected deactivated signals, by default GPS_JUMP
    inplace : boolean, optional
        if set to true the original dataframe will be altered to contain
        the result of the filtering, otherwise a copy will be returned,
        by default True

    Returns
    _______
    DataFrame
        DataFrame with the additional features or None
        'dist_to_prev', 'dist_to_next', 'dist_prev_to_next', 'jump'

    """

    if not inplace:
        move_data = move_data[:]

    message = 'Create or update jump if dist max > %srs meters\n'
    print(message % max_dist_between_adj_points)
    move_data.generate_dist_features()

    return _process_simple_filter(
        move_data,
        new_label,
        DIST_TO_PREV,
        max_dist_between_adj_points,
        inplace
    )


@timer_decorator
def create_or_update_short_trajectory(
    move_data: Union['PandasMoveDataFrame', 'DaskMoveDataFrame'],
    max_dist_between_adj_points: Optional[float] = 3000,
    max_time_between_adj_points: Optional[float] = 7200,
    max_speed_between_adj_points: Optional[float] = 50,
    k_segment_max: Optional[int] = 50,
    label_tid: Optional[Text] = TID_PART,
    new_label: Optional[Text] = SHORT,
    inplace: Optional[bool] = True
) -> Optional[Union['PandasMoveDataFrame', 'DaskMoveDataFrame']]:
    """

    Parameters
    ----------
    move_data : dataframe
       The input trajectory data
    max_dist_between_adj_points : float, optional
        Specify the maximum distance a point should have from
        the previous point, in order not to be dropped, by default 3000
    max_time_between_adj_points : float, optional
        Specify the maximum travel time between two adjacent points, by default 7200
    max_speed_between_adj_points : float, optional
        Specify the maximum speed of travel between two adjacent points, by default 50
    k_segment_max: int, optional
        Specify the maximum number of segments in the trajectory, by default 50
    label_tid:  str, optional
        The label of the column containing the ids of the formed segments,
        by default TID_PART
    new_label: str, optional
        The name of the new feature with short trajectories, by default SHORT
    inplace : boolean, optional
        if set to true the original dataframe will be altered to contain
        the result of the filtering, otherwise a copy will be returned,
        by default True

    Returns
    -------
    DataFrame
        DataFrame with the aditional features or None
       'dist_to_prev', 'time_to_prev', 'speed_to_prev', 'tid_part', 'short_traj'

    """

    if not inplace:
        move_data = move_data[:]

    print('\nCreate or update short trajectories...')

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
        move_data, new_label, inplace
    )


@timer_decorator
def create_or_update_gps_block_signal(
    move_data: Union['PandasMoveDataFrame', 'DaskMoveDataFrame'],
    max_time_stop: Optional[float] = 7200,
    new_label: Optional[Text] = BLOCK,
    label_tid: Optional[Text] = TID_PART,
    inplace: Optional[bool] = True
) -> Optional[Union['PandasMoveDataFrame', 'DaskMoveDataFrame']]:
    """
    Create a new feature that inform points with speed = 0

    Parameters
    __________
    move_data: dataFrame
        The input trajectories data.
    max_time_stop: float, optional
        Maximum time allowed with speed 0, by default 7200
    new_label: string, optional
        The name of the new feature with detected deactivated signals, by default BLOCK
    label_tid : str, optional
        The label of the column containing the ids of the formed segments,
        by default TID_PART
        Is the new slitted id.
    inplace : boolean, optional
        if set to true the original dataframe will be altered to contain
        the result of the filtering, otherwise a copy will be returned,
        by default True

    Returns
    _______
    DataFrame
        DataFrame with the additional features or None
        'dist_to_prev', 'time_to_prev', 'speed_to_prev',
        'tid_dist', 'block_signal'

    """

    if not inplace:
        move_data = move_data[:]

    message = 'Create or update block_signal if max time stop > %srs seconds\n'
    print(message % max_time_stop)
    segmentation.by_max_dist(
        move_data, max_dist_between_adj_points=0.0, label_new_tid=label_tid
    )

    print('Updating dist time speed values')
    move_data.generate_dist_time_speed_features(label_id=label_tid)

    move_data[new_label] = False

    # SUM the segment block to detect the id that has or more time stopped
    df_agg_tid = move_data.groupby(by=label_tid).agg({TIME_TO_PREV: 'sum'})
    filter_ = df_agg_tid[TIME_TO_PREV] >= max_time_stop
    idx = df_agg_tid[filter_].index
    move_data.loc[move_data[label_tid].isin(idx), new_label] = True

    return _end_create_operation(
        move_data, new_label, inplace
    )


@timer_decorator
def filter_block_signal_by_repeated_amount_of_points(
    move_data: Union['PandasMoveDataFrame', 'DaskMoveDataFrame'],
    amount_max_of_points_stop: Optional[float] = 30.0,
    max_time_stop: Optional[float] = 7200,
    filter_out: Optional[bool] = False,
    label_tid: Optional[Text] = TID_PART,
    inplace: Optional[bool] = False
) -> Optional[Union['PandasMoveDataFrame', 'DaskMoveDataFrame']]:
    """
    Filters from dataframe points with blocked signal by amount of points.

    Parameters
    __________
    move_data: dataFrame
        The input trajectories data.
    amount_max_of_points_stop: float, optional
        Maximum number of stopped points, by default 30
    max_time_stop: float, optional
        Maximum time allowed with speed 0, by default 7200
    filter_out: boolean, optional
        If set to True, it will return trajectory points with blocked signal,
        by default False
    label_tid : str, optional
        The label of the column containing the ids of the formed segments,
        by default TID_PART
    inplace : boolean, optional
        if set to true the original dataframe will be altered to contain
        the result of the filtering, otherwise a copy will be returned,
        by default True

    Returns
    _______
    DataFrame
        Filtered DataFrame with the additional features or None
        'dist_to_prev', 'time_to_prev', 'speed_to_prev',
        'tid_dist', 'block_signal'

    """

    if not inplace:
        move_data = move_data[:]

    if BLOCK not in move_data:
        create_or_update_gps_block_signal(
            move_data, max_time_stop, label_tid=label_tid
        )

    df_count_tid = move_data.groupby(by=[label_tid]).sum()
    filter_ = df_count_tid[BLOCK] > amount_max_of_points_stop

    if filter_out:
        idx = df_count_tid[~filter_].index
    else:
        idx = df_count_tid[filter_].index

    filter_ = move_data[move_data[label_tid].isin(idx)].index
    move_data.drop(index=filter_, inplace=True)
    if not inplace:
        return move_data


@timer_decorator
def filter_block_signal_by_time(
    move_data: Union['PandasMoveDataFrame', 'DaskMoveDataFrame'],
    max_time_stop: Optional[float] = 7200,
    filter_out: Optional[bool] = False,
    label_tid: Optional[Text] = TID_PART,
    inplace: Optional[bool] = False
) -> Optional[Union['PandasMoveDataFrame', 'DaskMoveDataFrame']]:
    """
    Filters from dataframe points with blocked signal by time.

    Parameters
    __________
    move_data: dataFrame
        The input trajectories data.
    max_time_stop: float, optional
        Maximum time allowed with speed 0, by default 7200
    filter_out: boolean, optional
        If set to True, it will return trajectory points with blocked signal,
        by default False
    label_tid : str, optional
        The label of the column containing the ids of the formed segments,
        by default TID_PART
    inplace : boolean, optional
        if set to true the original dataframe will be altered to contain
        the result of the filtering, otherwise a copy will be returned,
        by default True

    Returns
    _______
    DataFrame
        Filtered DataFrame with the additional features or None
        'dist_to_prev', 'time_to_prev', 'speed_to_prev',
        'tid_dist', 'block_signal'

    """

    if not inplace:
        move_data = move_data.copy()

    if BLOCK not in move_data:
        create_or_update_gps_block_signal(
            move_data, max_time_stop, label_tid=label_tid
        )

    df_agg_tid = move_data.groupby(by=label_tid).agg(
        {TIME_TO_PREV: 'sum', BLOCK: 'sum'}
    )
    filter_ = (df_agg_tid[TIME_TO_PREV] > max_time_stop) & (df_agg_tid[BLOCK] > 0)

    if filter_out:
        idx = df_agg_tid[~filter_].index
    else:
        idx = df_agg_tid[filter_].index

    filter_ = move_data[move_data[label_tid].isin(idx)].index
    move_data.drop(index=filter_, inplace=True)

    if not inplace:
        return move_data


@timer_decorator
def filter_longer_time_to_stop_segment_by_id(
    move_data: Union['PandasMoveDataFrame', 'DaskMoveDataFrame'],
    dist_radius: Optional[float] = 30,
    time_radius: Optional[float] = 900,
    label_id: Optional[Text] = TRAJ_ID,
    label_segment_stop: Optional[Text] = SEGMENT_STOP,
    filter_out: Optional[bool] = False,
    inplace: Optional[bool] = False
) -> Optional[Union['PandasMoveDataFrame', 'DaskMoveDataFrame']]:
    """
    Filters from dataframe segment with longest stop time.

    Parameters
    __________
    move_data: dataFrame
        The input trajectories data.
    dist_radius : float, optional
        The dist_radius defines the distance used in the segmentation, by default 30
    time_radius :  float, optional
        The time_radius used to determine if a segment is a stop, by default 30
        If the user stayed in the segment for a time
        greater than time_radius, than the segment is a stop.
    label_tid : str, optional
        The label of the column containing the ids of the formed segments,
        by default TRAJ_ID
    label_segment_stop: str, optional
        by default 'segment_stop'
    filter_out: boolean, optional
        If set to True, it will return trajectory points with longer time, by default True
    inplace : boolean, optional
        if set to true the original dataframe will be altered to contain
        the result of the filtering, otherwise a copy will be returned,
        by default True

    Returns
    _______
    DataFrame
        Filtered DataFrame with the additional features or None
        'dist_to_prev', 'time_to_prev', 'speed_to_prev',
        'tid_dist', 'block_signal'

    """

    if not inplace:
        move_data = move_data.copy()

    if label_segment_stop not in move_data:
        stay_point_detection.create_or_update_move_stop_by_dist_time(
            move_data, dist_radius, time_radius
        )

    df_agg_id_stop = move_data.groupby(
        [label_id, label_segment_stop], as_index=False
    ).agg({TIME_TO_PREV: 'sum'})

    filter_ = df_agg_id_stop.groupby(
        [label_id], as_index=False
    ).idxmax()[TIME_TO_PREV]

    if filter_out:
        segments = df_agg_id_stop.loc[~df_agg_id_stop.index.isin(filter_)]
    else:
        segments = df_agg_id_stop.loc[df_agg_id_stop.index.isin(filter_)]
    segments = segments[label_segment_stop]

    filter_ = move_data[move_data[label_segment_stop].isin(segments)].index
    move_data.drop(index=filter_, inplace=True)
    if not inplace:
        return move_data
