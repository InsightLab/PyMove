import numpy as np
import pandas as pd

from pymove.utils.constants import (
    DIST_TO_PREV,
    SPEED_TO_PREV,
    TID_DIST,
    TID_PART,
    TID_SPEED,
    TID_TIME,
    TIME_TO_PREV,
    TRAJ_ID,
)
from pymove.utils.log import progress_bar


def bbox_split(bbox, number_grids):
    """
    splits the bounding box in N grids of the same size.

    Parameters
    ----------
    bbox: tuple
        Tuple of 4 elements, containing the minimum and maximum values
        of latitude and longitude of the bounding box.
    number_grids: Integer
        Determines the number of grids to split the bounding box.

    Returns
    -------
    dataframe
        Returns the latitude and longitude coordinates of
        the grids after the split.

    """

    lat_min = bbox[0]
    lon_min = bbox[1]
    lat_max = bbox[2]
    lon_max = bbox[3]

    const_lat = abs(abs(lat_max) - abs(lat_min)) / number_grids
    const_lon = abs(abs(lon_max) - abs(lon_min)) / number_grids
    print('const_lat: %s\nconst_lon: %s' % (const_lat, const_lon))

    move_data = pd.DataFrame(
        columns=['lat_min', 'lon_min', 'lat_max', 'lon_max']
    )
    for i in range(number_grids):
        move_data = move_data.append(
            {
                'lat_min': lat_min,
                'lon_min': lon_min + (const_lon * i),
                'lat_max': lat_max,
                'lon_max': lon_min + (const_lon * (i + 1)),
            },
            ignore_index=True,
        )

    return move_data


def _drop_single_point(move_data, label_new_tid, label_id):
    """
    Removes trajectory with single point.

    Parameters
    ----------
    move_data: dataframe
        dataframe with trajectories
    label_new_tid : String
        The label of the column containing the ids of the formed segments.
        Is the new splitted id.
    label_id : String
         Indicates the label of the id column in the user"srs dataframe.

    """

    shape_before_drop = move_data.shape
    idx = move_data[move_data[label_new_tid] == -1].index
    if idx.shape[0] > 0:
        print('...Drop Trajectory with a unique GPS point\n')
        ids_before_drop = move_data[label_id].unique().shape[0]
        move_data.drop(index=idx, inplace=True)
        print(
            '...Object - before drop: %s - after drop: %s'
            % (ids_before_drop, move_data[label_id].unique().shape[0])
        )
        print(
            '...Shape - before drop: %s - after drop: %s'
            % (shape_before_drop, move_data.shape)
        )
    else:
        print('...No trajs with only one point.', move_data.shape)


def _filter_and_dist_time_speed(move_data, idx, max_dist, max_time, max_speed):
    """
    Filters the dataframe considering thresholds for time, dist and speed

    Parameters
    ----------
    move_data : dataframe
        Dataframe to be filtered
    idx : int
        row to compare
    max_dist : float
        maximum dist diference
    max_time : float
        maximum time diference
    max_speed : float
        maximum speed diference

    Returns
    -------
    numpy.ndarray of booleans
        filtered indexes from the dataframe

    """

    return (
        (np.nan_to_num(move_data.at[idx, DIST_TO_PREV]) > max_dist)
        | (np.nan_to_num(move_data.at[idx, TIME_TO_PREV]) > max_time)
        | (np.nan_to_num(move_data.at[idx, SPEED_TO_PREV]) > max_speed)
    )


def _filter_or_dist_time_speed(move_data, idx, feature, max_between_adj_points):
    """
    Filters the dataframe considering thresholds for time, dist and speed

    Parameters
    ----------
    move_data : dataframe
        Dataframe to be filtered
    idx : int
        row to compare
    feature : str
        feature to compare
    max_between_adj_points : float
        maximum points diference

    Returns
    -------
    numpy.ndarray
        filtered indexes from the dataframe

    """

    return np.nan_to_num(move_data.at[idx, feature]) > max_between_adj_points


def prepare_segmentation(move_data, label_id, label_new_tid):
    """
    Resets the dataframe index, collects unique ids and
    initiates curr_id and count

    Parameters
    ----------
    move_data : dataframe
        Dataframe to be filtered
    label_id : str
        label of the feature
    label_new_tid : str
        label of the new feature

    Returns
    -------
    int
        initial curr_tid
    numpy.ndarray
        unique ids
    int
        initial count

    """

    if move_data.index.name is None:
        print('...setting %s as index' % label_id, flush=True)
        move_data.set_index(label_id, inplace=True)
    curr_tid = 0
    if label_new_tid not in move_data:
        move_data[label_new_tid] = curr_tid

    ids = move_data.index.unique()
    count = 0
    return curr_tid, ids, count


def _update_curr_tid_count(
        filter_, move_data, idx, label_new_tid, curr_tid, count
):
    """
    Updates the tid

    Parameters
    ----------
    filter_ : numpy.ndarray
        Filtered indexes
    move_data : dataframe
        Dataframe to be filtered
    idx : int
        row to compare
    label_new_tid : str
        label of the new feature
    curr_tid : int
        current tid
    count : int
        count of

    Returns
    -------
    int
        updated current tid
    int
        updated count ids

    """

    curr_tid += 1
    if filter_.shape == ():
        print('id: %s has no point to split' % idx)
        move_data.at[idx, label_new_tid] = curr_tid
        count += 1
    else:
        tids = np.empty(filter_.shape[0], dtype=np.int64)
        tids.fill(curr_tid)
        for i, has_problem in enumerate(filter_):
            if has_problem:
                curr_tid += 1
                tids[i:] = curr_tid
        count += tids.shape[0]
        move_data.at[idx, label_new_tid] = tids
    return curr_tid, count


def _filter_by(
        move_data, label_id, label_new_tid, drop_single_points, **kwargs
):
    """
    Splits the trajectories into segments.

    Parameters
    ----------
    move_data : dataframe
       The input trajectory data
    label_id : String, optional(dic_labels["id"] by default)
         Indicates the label of the id column in the user"srs dataframe.
    label_new_tid : String, optional(TID_PART by default)
        The label of the column containing the ids of the formed segments.
        Is the new splitted id.
    drop_single_points : boolean, optional(True by default)
        If set to True, drops the trajectories with only one point.
    **kwargs : arguments
        depends on the type of segmentation
        - all : if is a segmentation by all features
        - max_dist : maximum dist between adjacent points
        - max_time : maximum time between adjacent points
        - max_speed : maximum speed between adjacent points
        - feature : feature to use for segmentation
        - max_between_adj_points : maximum value for feature

    Returns
    -------
    dataframe
        DataFrame with the aditional features: label_new_tid,
        that indicates the trajectory segment to which the point belongs to.

    Note
    ----
    Time, distance and speeed features must be updated after split.

    """

    curr_tid, ids, count = prepare_segmentation(
        move_data, label_id, label_new_tid
    )

    for idx in progress_bar(ids, desc='Generating %s' % label_new_tid):
        if kwargs['all']:
            filter_ = _filter_and_dist_time_speed(
                move_data,
                idx,
                kwargs['max_dist'],
                kwargs['max_time'],
                kwargs['max_speed']
            )
        else:
            filter_ = _filter_or_dist_time_speed(
                move_data,
                idx,
                kwargs['feature'],
                kwargs['max_between_adj_points']
            )

        curr_tid, count = _update_curr_tid_count(
            filter_, move_data, idx, label_new_tid, curr_tid, count
        )

    if label_id == label_new_tid:
        move_data.reset_index(drop=True, inplace=True)
        print('... label_tid = label_new_id, then reseting and drop index')
    else:
        move_data.reset_index(inplace=True)
        print('... Reseting index\n')

    if drop_single_points:
        _drop_single_point(move_data, label_new_tid, label_id)
        move_data.generate_dist_time_speed_features()

    print('------------------------------------------\n')

    return move_data


def by_dist_time_speed(
        move_data,
        label_id=TRAJ_ID,
        max_dist_between_adj_points=3000,
        max_time_between_adj_points=7200,
        max_speed_between_adj_points=50.0,
        drop_single_points=True,
        label_new_tid=TID_PART,
        inplace=True,
):
    """
    Splits the trajectories into segments based on distance, time and speed.

    Parameters
    ----------
    move_data : dataframe
       The input trajectory data
    label_id : String, optional(dic_labels["id"] by default)
         Indicates the label of the id column in the user"srs dataframe.
    max_dist_between_adj_points : Float, optional(3000 by default)
        Specify the maximum distance a point should have from
        the previous point, in order not to be dropped
    max_time_between_adj_points : Float, optional(7200 by default)
        Specify the maximum travel time between two adjacent points
    max_speed_between_adj_points : Float, optional(50.0 by default)
        Specify the maximum speed of travel between two adjacent points
    drop_single_points : boolean, optional(True by default)
        If set to True, drops the trajectories with only one point.
    label_new_tid : String, optional(TID_PART by default)
        The label of the column containing the ids of the formed segments.
        Is the new splitted id.
    inplace : boolean, optional(True by default)
        if set to true the original dataframe will be altered to
        contain the result of the filtering, otherwise a copy will be returned.

    Returns
    -------
    dataframe
        DataFrame with the aditional features: label_new_tid,
        that indicates the trajectory segment to which the point belongs to.

    Note
    ----
    Time, distance and speeed features must be updated after split.

    """

    if not inplace:
        move_data = move_data[:]

    print('\nSplit trajectories')
    print('...max_dist_between_adj_points:', max_dist_between_adj_points)
    print('...max_time_between_adj_points:', max_time_between_adj_points)
    print('...max_speed_between_adj_points:', max_speed_between_adj_points)

    if TIME_TO_PREV not in move_data:
        move_data.generate_dist_time_speed_features()

    try:
        move_data = _filter_by(
            move_data,
            label_id,
            label_new_tid,
            drop_single_points,
            max_dist=max_dist_between_adj_points,
            max_time=max_time_between_adj_points,
            max_speed=max_speed_between_adj_points,
            all=True
        )
        if not inplace:
            return move_data

    except Exception as e:
        raise e


def by_max_dist(
        move_data,
        label_id=TRAJ_ID,
        max_dist_between_adj_points=3000,
        drop_single_points=True,
        label_new_tid=TID_DIST,
        inplace=True,
):
    """
    Segments the trajectories based on distance.

    Parameters
    ----------
    move_data : dataframe
       The input trajectory data
    label_id : String, optional(dic_labels["id"] by default)
         Indicates the label of the id column in the user"srs dataframe.
    max_dist_between_adj_points : Float, optional(50.0 by default)
        Specify the maximum dist between two adjacent points
    drop_single_points : boolean, optional(True by default)
        If set to True, drops the trajectories with only one point.
    label_new_tid : String, optional(TID_DIST by default)
        The label of the column containing the ids of the formed segments.
        Is the new splitted id.
    inplace : boolean, optional(True by default)
        if set to true the original dataframe will be altered to
        contain the result of the filtering, otherwise a copy will be returned.

    Returns
    -------
    dataframe
        DataFrame with the aditional features: label_segment,
        that indicates the trajectory segment to which the point belongs to.

    Note
    ----
    Speed features must be updated after split.

    """

    if not inplace:
        move_data = move_data[:]

    print(
        'Split trajectories by max distance between adjacent points:',
        max_dist_between_adj_points,
    )

    if DIST_TO_PREV not in move_data:
        move_data.generate_dist_time_speed_features()

    try:
        move_data = _filter_by(
            move_data,
            label_id,
            label_new_tid,
            drop_single_points,
            feature=DIST_TO_PREV,
            max_between_adj_points=max_dist_between_adj_points,
            all=False
        )
        if not inplace:
            return move_data

    except Exception as e:
        raise e


def by_max_time(
        move_data,
        label_id=TRAJ_ID,
        max_time_between_adj_points=900.0,
        drop_single_points=True,
        label_new_tid=TID_TIME,
        inplace=True,
):
    """
    Splits the trajectories into segments based on a maximum.

    Parameters
    ----------
    move_data : dataframe
       The input trajectory data
    label_id : String, optional(id by default)
         Indicates the label of the id column in the users dataframe.
    max_time_between_adj_points : Float, optional(50.0 by default)
        Specify the maximum time between two adjacent points
    drop_single_points : boolean, optional(True by default)
        If set to True, drops the trajectories with only one point.
    label_new_tid : String, optional(TID_TIME by default)
        The label of the column containing the ids of the formed segments.
        Is the new splitted id.
    inplace : boolean, optional(True by default)
        if set to true the original dataframe will be altered to contain
        the result of the filtering, otherwise a copy will be returned.


    Returns
    -------
    dataframe
        DataFrame with the additional features: label_segment,
        that indicates the trajectory segment to which the point belongs to.

    Note
    ----
    Speed features must be updated after split.

    """

    if not inplace:
        move_data = move_data[:]

    print(
        'Split trajectories by max_time_between_adj_points:',
        max_time_between_adj_points,
    )

    if TIME_TO_PREV not in move_data:
        move_data.generate_dist_time_speed_features()

    try:
        move_data = _filter_by(
            move_data,
            label_id,
            label_new_tid,
            drop_single_points,
            feature=TIME_TO_PREV,
            max_between_adj_points=max_time_between_adj_points,
            all=False
        )
        if not inplace:
            return move_data

    except Exception as e:
        raise e


def by_max_speed(
        move_data,
        label_id=TRAJ_ID,
        max_speed_between_adj_points=50.0,
        drop_single_points=True,
        label_new_tid=TID_SPEED,
        inplace=True,
):
    """
    Splits the trajectories into segments based on a maximum speed.

    Parameters
    ----------
    move_data : dataframe.
       The input trajectory data.
    label_id : String, optional(id by default).
         Indicates the label of the id column in the users dataframe.
    max_speed_between_adj_points : Float, optional(50.0 by default).
        Specify the maximum speed between two adjacent points.
    drop_single_points : boolean, optional(True by default)
        If set to True, drops the trajectories with only one point.
    label_new_tid : String, optional(TID_SPEED by default)
        The label of the column containing the ids of the formed segments.
        Is the new splitted id.
    inplace : boolean, optional(True by default)
        if set to true the original dataframe will be altered to
        contain the result of the filtering, otherwise a copy will be returned.

    Returns
    -------
    dataframe
        DataFrame with the aditional features: label_segment,
        that indicates the trajectory segment to which the point belongs to.

    Note
    ----
    Speed features must be updated after split.

    """

    if not inplace:
        move_data = move_data[:]

    print(
        'Split trajectories by max_speed_between_adj_points:',
        max_speed_between_adj_points,
    )

    if SPEED_TO_PREV not in move_data:
        move_data.generate_dist_time_speed_features()

    try:
        move_data = _filter_by(
            move_data,
            label_id,
            label_new_tid,
            drop_single_points,
            feature=SPEED_TO_PREV,
            max_between_adj_points=max_speed_between_adj_points,
            all=False
        )
        if not inplace:
            return move_data

    except Exception as e:
        raise e
