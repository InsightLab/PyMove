import numpy as np
import pandas as pd
from tqdm import tqdm

from pymove.core.dataframe import PandasMoveDataFrame
from pymove.utils.constants import (
    DIST_TO_PREV,
    SPEED_TO_PREV,
    TIME_TO_PREV,
    TRAJ_ID,
    TID_PART,
    TID_DIST,
    TID_TIME,
    TID_SPEED
)


def bbox_split(bbox, number_grids):
    """
    splits the bounding box in N grids of the same size.

    Parameters
    ----------
    bbox: tuple
        Tuple of 4 elements, containg the minimum and maximum values of latitude and longitude of the bounding box.
    number_grids: Integer
        Determines the number of grids to split the bounding box.

    Returns
    -------
    move_data : dataframe
        Returns the latittude and longitude coordenates of the grids after the split.
    """

    lat_min = bbox[0]
    lon_min = bbox[1]
    lat_max = bbox[2]
    lon_max = bbox[3]

    const_lat = abs(abs(lat_max) - abs(lat_min)) / number_grids
    const_lon = abs(abs(lon_max) - abs(lon_min)) / number_grids
    print("const_lat: {}\nconst_lon: {}".format(const_lat, const_lon))

    move_data = pd.DataFrame(
        columns=["lat_min", "lon_min", "lat_max", "lon_max"]
    )
    for i in range(number_grids):
        move_data = move_data.append(
            {
                "lat_min": lat_min,
                "lon_min": lon_min + (const_lon * i),
                "lat_max": lat_max,
                "lon_max": lon_min + (const_lon * (i + 1)),
            },
            ignore_index=True,
        )

    return move_data


def _drop_single_point(move_data, label_new_tid, label_id):
    """
    Removes trajectory with single point

    Parameters
    ----------
    move_data: dataframe
        dataframe with trajectories
    label_new_tid : String
        The label of the column containing the ids of the formed segments. Is the new splitted id.
    label_id : String
         Indicates the label of the id column in the user"s dataframe.
    """
    shape_before_drop = move_data.shape
    idx = move_data[move_data[label_new_tid] == -1].index
    if idx.shape[0] > 0:
        print("...Drop Trajectory with a unique GPS point\n")
        ids_before_drop = move_data[label_id].unique().shape[0]
        move_data.drop(index=idx, inplace=True)
        print(
            "...Object - before drop: {} - after drop: {}".format(
                ids_before_drop, move_data[label_id].unique().shape[0]
            )
        )
        print(
            "...Shape - before drop: {} - after drop: {}".format(
                shape_before_drop, move_data.shape
            )
        )
    else:
        print("...No trajs with only one point.", move_data.shape)


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
         Indicates the label of the id column in the user"s dataframe.
    max_dist_between_adj_points : Float, optional(3000 by default)
        Specify the maximun distance a point should have from the previous point, in order not to be dropped
    max_time_between_adj_points : Float, optional(7200 by default)
        Specify the maximun travel time between two adjacent points
    max_speed_between_adj_points : Float, optional(50.0 by default)
        Specify the maximun speed of travel between two adjacent points
    drop_single_points : boolean, optional(True by default)
        If set to True, drops the trajectories with only one point.
    label_new_tid : String, optional(TID_PART by default)
        The label of the column containing the ids of the formed segments. Is the new splitted id.
    inplace : boolean, optional(True by default)
        if set to true the original dataframe will be altered to contain the result of the filtering,
        otherwise a copy will be returned.

    Returns
    ------
    Returns the dataFrame with the aditional features: label_new_tid, that indicates the trajectory segment
        to which the point belongs to.

    Note
    -----
    Time, distance and speeed features must be updated after split.
    """

    if not inplace:
        move_data = PandasMoveDataFrame(data=move_data.to_DataFrame())

    print("\nSplit trajectories")
    print("...max_time_between_adj_points:", max_time_between_adj_points)
    print("...max_dist_between_adj_points:", max_dist_between_adj_points)
    print("...max_speed:", max_speed_between_adj_points)

    try:

        if TIME_TO_PREV not in move_data:
            move_data.generate_dist_time_speed_features()

        if move_data.index.name is None:
            print("...setting {} as index".format(label_id), flush=True)
            move_data.set_index(label_id, inplace=True)

        curr_tid = 0
        if label_new_tid not in move_data:
            move_data[label_new_tid] = curr_tid

        ids = move_data.index.unique()
        count = 0

        for idx in tqdm(ids, desc=f"Generating {label_new_tid}"):
            curr_tid += 1

            filter_ = (
                    (np.nan_to_num(move_data.at[idx, TIME_TO_PREV]) > max_time_between_adj_points)
                    | (
                            np.nan_to_num(move_data.at[idx, DIST_TO_PREV])
                            > max_dist_between_adj_points
                    )
                    | (
                            np.nan_to_num(move_data.at[idx, SPEED_TO_PREV])
                            > max_speed_between_adj_points
                    )
            )

            # check if object have only one point to be removed
            if filter_.shape == ():
                # trajectories with only one point is useless for interpolation and so they must be removed.
                count += 1
                move_data.at[idx, label_new_tid] = -1
                curr_tid += -1
            else:
                tids = np.empty(filter_.shape[0], dtype=np.int64)
                tids.fill(curr_tid)

                for i, has_problem in enumerate(filter_):
                    if has_problem:
                        curr_tid += 1
                        tids[i:] = curr_tid
                count += tids.shape[0]
                move_data.at[idx, label_new_tid] = tids

        if label_id == label_new_tid:
            move_data.reset_index(drop=True, inplace=True)
            print("... label_id = label_new_id, then reseting and drop index")
        else:
            move_data.reset_index(inplace=True)
            print("... Reseting index\n")

        if drop_single_points:
            _drop_single_point(move_data, label_new_tid, label_id)
            move_data.generate_dist_time_speed_features()

        print("------------------------------------------\n")
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
    """ Segments the trajectories based on distance.
    Parameters
    ----------
    move_data : dataframe
       The input trajectory data
    label_id : String, optional(dic_labels["id"] by default)
         Indicates the label of the id column in the user"s dataframe.
    max_dist_between_adj_points : Float, optional(50.0 by default)
        Specify the maximun dist between two adjacent points
    drop_single_points : boolean, optional(True by default)
        If set to True, drops the trajectories with only one point.
    label_new_tid : String, optional(TID_DIST by default)
        The label of the column containing the ids of the formed segments. Is the new splitted id.
    inplace : boolean, optional(True by default)
        if set to true the original dataframe will be altered to contain the result of the filtering,
        otherwise a copy will be returned.

    Returns
    ------
    Returns the dataFrame with the aditional features: label_segment, that indicates the trajectory segment
        to which the point belongs to.

    Note
    -----
    Speed features must be updated after split.
    """

    if not inplace:
        move_data = PandasMoveDataFrame(data=move_data.to_DataFrame())

    print(
        "Split trajectories by max distance between adjacent points:",
        max_dist_between_adj_points,
    )
    try:

        if DIST_TO_PREV not in move_data:
            move_data.generate_dist_features()

        if move_data.index.name is None:
            print("...setting {} as index".format(label_id), flush=True)
            move_data.set_index(label_id, inplace=True)

        curr_tid = 0
        if label_new_tid not in move_data:
            move_data[label_new_tid] = curr_tid

        ids = move_data.index.unique()
        count = 0

        for idx in tqdm(ids, desc=f"Generating {label_new_tid}"):
            # increment index to trajectory
            curr_tid += 1

            # filter dist max
            dist = (
                np.nan_to_num(move_data.at[idx, DIST_TO_PREV]) > max_dist_between_adj_points
            )
            # check if object have more than one point to split
            if dist.shape == ():
                print("id: {} has not point to split".format(idx))
                move_data.at[idx, label_new_tid] = curr_tid
                count += 1
            else:
                tids = np.empty(dist.shape[0], dtype=np.int64)
                tids.fill(curr_tid)
                for i, has_problem in enumerate(dist):
                    if has_problem:
                        curr_tid += 1
                        tids[i:] = curr_tid
                count += tids.shape[0]
                move_data.at[idx, label_new_tid] = tids

        if label_id == label_new_tid:
            move_data.reset_index(drop=True, inplace=True)
            print("... label_id = label_new_id, then reseting and drop index")
        else:
            move_data.reset_index(inplace=True)
            print("... Reseting index")

        if drop_single_points:
            _drop_single_point(move_data, label_new_tid, label_id)
            move_data.generate_dist_features()

        print("------------------------------------------\n")
        if not inplace:
            return move_data
    except Exception as e:
        print("label_id:{}\nidx:{}\n".format(label_id, idx))
        raise e


def by_max_time(
    move_data,
    label_id=TRAJ_ID,
    max_time_between_adj_points=900.0,
    drop_single_points=True,
    label_new_tid=TID_TIME,
    inplace=True,
):
    """ Splits the trajectories into segments based on a maximum time set by the user.
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
        The label of the column containing the ids of the formed segments. Is the new splitted id.
    inplace : boolean, optional(True by default)
        if set to true the original dataframe will be altered to contain the result of the filtering,
        otherwise a copy will be returned.


    ------
    Returns the dataFrame with the aditional features: label_segment, that indicates the trajectory segment
        to which the point belongs to.

    Note
    -----
    Speed features must be updated after split.
    """

    if not inplace:
        move_data = PandasMoveDataFrame(data=move_data.to_DataFrame())


    print(
        "Split trajectories by max_time_between_adj_points:",
        max_time_between_adj_points,
    )
    try:

        if TIME_TO_PREV not in move_data:
            move_data.generate_dist_time_speed_features()

        if move_data.index.name is None:
            print("...setting {} as index".format(label_id), flush=True)
            move_data.set_index(label_id, inplace=True)

        curr_tid = 0
        if label_new_tid not in move_data:
            move_data[label_new_tid] = curr_tid

        ids = move_data.index.unique()
        count = 0

        for idx in tqdm(ids, desc=f"Generating {label_new_tid}"):
            # increment index to trajectory
            curr_tid += 1

            # filter time max
            times = (
                np.nan_to_num(move_data.at[idx, TIME_TO_PREV]) > max_time_between_adj_points
            )

            # check if object have only one point to be removed
            if times.shape == ():
                print("id: {} has not point to split".format(id))
                move_data.at[idx, label_new_tid] = curr_tid
                count += 1
            else:
                tids = np.empty(times.shape[0], dtype=np.int64)
                tids.fill(curr_tid)
                for i, has_problem in enumerate(times):
                    if has_problem:
                        curr_tid += 1
                        tids[i:] = curr_tid
                count += tids.shape[0]
                move_data.at[idx, label_new_tid] = tids

        if label_id == label_new_tid:
            move_data.reset_index(drop=True, inplace=True)
            print("... label_id = label_new_id, then reseting and drop index")
        else:
            move_data.reset_index(inplace=True)
            print("... Reseting index")

        if drop_single_points:
            _drop_single_point(move_data, label_new_tid, label_id)
            move_data.generate_dist_time_speed_features()

        print("------------------------------------------\n")
        if not inplace:
            return move_data
    except Exception as e:
        print("label_id:{}\nidx:{}\n".format(label_id, idx))
        raise e


def by_max_speed(
    move_data,
    label_id=TRAJ_ID,
    max_speed_between_adj_points=50.0,
    drop_single_points=True,
    label_new_tid=TID_SPEED,
    inplace=True,
):
    """ Splits the trajectories into segments based on a maximum speed set by the user.
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
        The label of the column containing the ids of the formed segments. Is the new splitted id.
    inplace : boolean, optional(True by default)
        if set to true the original dataframe will be altered to contain the result of the filtering,
        otherwise a copy will be returned.


    Returns
    ------
    Returns the dataFrame with the aditional features: label_segment, that indicates the trajectory segment
        to which the point belongs to.

    Note
    -----
    Speed features must be updated after split.
    """
    if not inplace:
        move_data = PandasMoveDataFrame(data=move_data.to_DataFrame())

    print(
        "Split trajectories by max_speed_between_adj_points:",
        max_speed_between_adj_points,
    )
    try:

        if SPEED_TO_PREV not in move_data:
            move_data.generate_dist_time_speed_features()

        if move_data.index.name is None:
            print("...setting {} as index".format(label_id), flush=True)
            move_data.set_index(label_id, inplace=True)

        curr_tid = 0
        if label_new_tid not in move_data:
            move_data[label_new_tid] = curr_tid

        ids = move_data.index.unique()
        count = 0

        for idx in tqdm(ids, desc=f"Generating {label_new_tid}"):
            # increment index to trajectory
            curr_tid += 1

            # filter speed max
            speed = (
                np.nan_to_num(move_data.at[idx, SPEED_TO_PREV]) > max_speed_between_adj_points
            )
            # check if object have only one point to be removed
            if speed.shape == ():
                print("id: {} has not point to split".format(id))
                move_data.at[idx, label_new_tid] = curr_tid
                count += 1
            else:
                tids = np.empty(speed.shape[0], dtype=np.int64)
                tids.fill(curr_tid)
                for i, has_problem in enumerate(speed):
                    if has_problem:
                        curr_tid += 1
                        tids[i:] = curr_tid
                count += tids.shape[0]
                move_data.at[idx, label_new_tid] = tids

        if label_id == label_new_tid:
            move_data.reset_index(drop=True, inplace=True)
            print("... label_id = label_new_id, then reseting and drop index")
        else:
            move_data.reset_index(inplace=True)
            print("... Reseting index")

        if drop_single_points:
            _drop_single_point(move_data, label_new_tid, label_id)
            move_data.generate_dist_time_speed_features()

        print("------------------------------------------\n")
        if not inplace:
            return move_data
    except Exception as e:
        print("label_id:{}\nidx:{}\n".format(label_id, idx))
        raise e
