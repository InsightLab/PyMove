import numpy as np
import pandas as pd
import time

from pymove.core.dataframe import PandasMoveDataFrame

from pymove.utils.constants import (
    TIME_TO_PREV,
    DIST_TO_PREV,
    SPEED_TO_PREV,
    TRAJ_ID,
    TID_DIST)
from pymove.utils.trajectories import progress_update


def bbox_split(bbox, number_grids):
    """splits the bounding box in N grids of the same size.

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

    move_data = pd.DataFrame(columns=["lat_min", "lon_min", "lat_max", "lon_max"])
    for i in range(number_grids):
        move_data = move_data.append({"lat_min": lat_min,
                                      "lon_min": lon_min + (const_lon * i),
                                      "lat_max": lat_max,
                                      "lon_max": lon_min + (const_lon * (i + 1))},
                                     ignore_index=True)

    return move_data


def by_dist_time_speed(
        move_data,
        label_id=TRAJ_ID,
        max_dist_between_adj_points=3000,
        max_time_between_adj_points=7200,
        max_speed_between_adj_points=50.0,
        drop_single_points=True,
        label_new_tid="tid_part",
        inplace=True
):
    """Splits the trajectories into segments based on distance, time and speed.

    Parameters
    ----------
    move_data : dataframe
       The input trajectory data
    label_id : String, optional(dic_labels["id"] by default)
         Indicates the label of the id column in the user"s dataframe.
    max_dist_between_adj_points : Float, optinal(3000 by default)
        Specify the maximun distance a point should have from the previous point, in order not to be dropped
    max_time_between_adj_points : Float, optinal(7200 by default)
        Specify the maximun travel time between two adjacent points
    max_speed_between_adj_points : Float, optinal(50.0 by default)
        Specify the maximun speed of travel between two adjacent points
    drop_single_points : boolean, optional(True by default)
        If set to True, drops the trajectories with only one point.
    label_new_tid : String, optional("tid_part" by default)
        The label of the column containing the ids of the formed segments. Is the new splitted id.

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
            print("...setting {} as index".format(label_id))
            move_data.set_index(label_id, inplace=True)

        curr_tid = 0
        if label_new_tid not in move_data:
            move_data[label_new_tid] = curr_tid

        ids = move_data.index.unique()
        count = 0
        move_datasize = move_data.shape[0]
        curr_perc_int = -1
        start_time = time.time()

        for idx in ids:
            curr_tid += 1

            filter_ = (move_data.at[idx, TIME_TO_PREV] > max_time_between_adj_points) | \
                      (move_data.at[idx, DIST_TO_PREV] > max_dist_between_adj_points) | \
                      (move_data.at[idx, SPEED_TO_PREV] > max_speed_between_adj_points)

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

            curr_perc_int, est_time_str = progress_update(count, move_datasize, start_time, curr_perc_int, step_perc=20)

        if label_id == label_new_tid:
            move_data.reset_index(drop=True, inplace=True)
            print("... label_id = label_new_id, then reseting and drop index")
        else:
            move_data.reset_index(inplace=True)
            print("... Reseting index\n")

        if drop_single_points:
            shape_before_drop = move_data.shape
            idx = move_data[move_data[label_new_tid] == -1].index
            if idx.shape[0] > 0:
                print("...Drop Trajectory with a unique GPS point\n")
                ids_before_drop = move_data[label_id].unique().shape[0]
                move_data.drop(index=idx, inplace=True)
                print("...Object - before drop: {} - after drop: {}".format(ids_before_drop,
                                                                            move_data[label_id].unique().shape[0]))
                print("...Shape - before drop: {} - after drop: {}".format(shape_before_drop, move_data.shape))
                move_data.generate_dist_time_speed_features()
            else:
                print("...No trajs with only one point.", move_data.shape)

        if not inplace:
            return move_data
    except Exception as e:
        raise e


def by_speed(
        move_data,
        label_id=TRAJ_ID,
        max_speed_between_adj_points=50.0,
        drop_single_points=True,
        label_new_tid="tid_speed",
        inplace=True
):
    """Segments the trajectories based on speed.

    Parameters
    ----------
    move_data : dataframe
       The input trajectory data
    label_id : String, optional(dic_labels["id"] by default)
         Indicates the label of the id column in the user"s dataframe.
    max_speed_between_adj_points : Float, optinal(50.0 by default)
        Specify the maximun speed of travel between two adjacent points
    drop_single_points : boolean, optional(True by default)
        If set to True, drops the trajectories with only one point.
    label_new_tid : String, optional("tid_speed" by default)
        The label of the column containing the ids of the formed segments. Is the new splitted id.

    Returns
    ------
    Returns the dataFrame with the aditional features: label_new_tid, that indicates the trajectory segment
        to which the point belongs to.

    Note
    -----
    Speed features must be updated after split.
    """
    if not inplace:
        move_data = PandasMoveDataFrame(data=move_data.to_DataFrame())

    print("\nSplit trajectories by max_speed_between_adj_points:", max_speed_between_adj_points)
    try:
        if SPEED_TO_PREV not in move_data:
            move_data.generate_dist_time_speed_features()

        if move_data.index.name is None:
            print("...setting {} as index".format(label_id))
            move_data.set_index(label_id, inplace=True)

        curr_tid = 0
        if label_new_tid not in move_data:
            move_data[label_new_tid] = curr_tid

        ids = move_data.index.unique()
        count = 0
        move_datasize = move_data.shape[0]
        curr_perc_int = -1
        start_time = time.time()

        for idx in ids:
            """ increment index to trajectory"""
            curr_tid += 1

            """ filter speed max"""
            speed = (move_data.at[idx, SPEED_TO_PREV] > max_speed_between_adj_points)

            """ check if object have only one point to be removed """
            if speed.shape == ():
                count += 1
                # set object  = -1 to remove ahead
                move_data.at[idx, label_new_tid] = -1
                curr_tid += -1
            else:
                tids = np.empty(speed.shape[0], dtype=np.int64)
                tids.fill(curr_tid)
                for i, has_problem in enumerate(speed):
                    if has_problem:
                        curr_tid += 1
                        tids[i:] = curr_tid
                count += tids.shape[0]
                move_data.at[idx, label_new_tid] = tids

            curr_perc_int, est_time_str = progress_update(count, move_datasize, start_time, curr_perc_int, step_perc=20)

        if label_id == label_new_tid:
            move_data.reset_index(drop=True, inplace=True)
            print("... label_id = label_new_id, then reseting and drop index")
        else:
            move_data.reset_index(inplace=True)
            print("... Reseting index\n")

        if drop_single_points:
            shape_before_drop = move_data.shape
            idx = move_data[move_data[label_new_tid] == -1].index
            if idx.shape[0] > 0:
                print("...Drop Trajectory with a unique GPS point\n")
                ids_before_drop = move_data[label_id].unique().shape[0]
                move_data.drop(index=idx, inplace=True)
                print("...Object - before drop: {} - after drop: {}".format(ids_before_drop,
                                                                            move_data[label_id].unique().shape[0]))
                print("...Shape - before drop: {} - after drop: {}".format(shape_before_drop, move_data.shape))
                move_data.generate_dist_time_speed_features()
            else:
                print("...No trajs with only one point.", move_data.shape)

        if not inplace:
            return move_data
    except Exception as e:
        raise e


def by_time(
        move_data,
        label_id=TRAJ_ID,
        max_time_between_adj_points=900.0,
        drop_single_points=True,
        label_new_tid="tid_time",
        inplace=True
):
    """Segments the trajectories into segments based on time.

    Parameters
    ----------
    move_data : dataframe
       The input trajectory data
    label_id : String, optional(dic_labels["id"] by default)
         Indicates the label of the id column in the user"s dataframe.
    max_time_between_adj_points : Float, optinal(50.0 by default)
        Specify the maximun time of travel between two adjacent points
    drop_single_points : boolean, optional(True by default)
        If set to True, drops the trajectories with only one point.
    label_new_tid : String, optional("tid_time" by default)
        The label of the column containing the ids of the formed segments. Is the new splitted id.

    Returns
    ------
    Returns the dataFrame with the aditional features: label_new_tid, that indicates the trajectory segment
        to which the point belongs to.
    """
    if not inplace:
        move_data = PandasMoveDataFrame(data=move_data.to_DataFrame())

    print("\nSplit trajectories by max_time_between_adj_points:", max_time_between_adj_points)
    try:

        if TIME_TO_PREV not in move_data:
            move_data.generate_dist_time_speed_features()

        if move_data.index.name is None:
            print("...setting {} as index".format(label_id))
            move_data.set_index(label_id, inplace=True)

        curr_tid = 0
        if label_new_tid not in move_data:
            move_data[label_new_tid] = curr_tid

        ids = move_data.index.unique()
        count = 0
        move_datasize = move_data.shape[0]
        curr_perc_int = -1
        start_time = time.time()

        for idx in ids:
            """ increment index to trajectory"""
            curr_tid += 1

            """ filter time max"""
            times = (move_data.at[idx, TIME_TO_PREV] > max_time_between_adj_points)

            """ check if object have only one point to be removed """
            if times.shape == ():
                count += 1
                # set object  = -1 to remove ahead
                move_data.at[idx, label_new_tid] = -1
                curr_tid += -1
            else:
                tids = np.empty(times.shape[0], dtype=np.int64)
                tids.fill(curr_tid)
                for i, has_problem in enumerate(times):
                    if has_problem:
                        curr_tid += 1
                        tids[i:] = curr_tid
                count += tids.shape[0]
                move_data.at[idx, label_new_tid] = tids

            curr_perc_int, est_time_str = progress_update(count, move_datasize, start_time, curr_perc_int, step_perc=20)

        if label_id == label_new_tid:
            move_data.reset_index(drop=True, inplace=True)
            print("... label_id = label_new_id, then reseting and drop index")
        else:
            move_data.reset_index(inplace=True)
            print("... Reseting index\n")

        if drop_single_points:
            shape_before_drop = move_data.shape
            idx = move_data[move_data[label_new_tid] == -1].index
            if idx.shape[0] > 0:
                print("...Drop Trajectory with a unique GPS point\n")
                ids_before_drop = move_data[label_id].unique().shape[0]
                move_data.drop(index=idx, inplace=True)
                print("...Object - before drop: {} - after drop: {}".format(ids_before_drop,
                                                                            move_data[label_id].unique().shape[0]))
                print("...Shape - before drop: {} - after drop: {}".format(shape_before_drop, move_data.shape))
                move_data.generate_dist_time_speed_features()
            else:
                print("...No trajs with only one point.", move_data.shape)

        if not inplace:
            return move_data
    except Exception as e:
        raise e

def by_max_dist(move_data, label_id=TRAJ_ID,  max_dist_between_adj_points=3000, label_segment=TID_DIST):
    """ Segments the trajectories based on distance.
    Parameters
    ----------
    move_data : dataframe
       The input trajectory data
    label_id : String, optional(dic_labels["id"] by default)
         Indicates the label of the id column in the user"s dataframe.
    max_dist_between_adj_points : Float, optinal(50.0 by default)
        Specify the maximun dist between two adjacent points
    label_segment : String, optional("tid_dist" by default)
        The label of the column containing the ids of the formed segments. Is the new splitted id.

    Returns
    ------
    Returns the dataFrame with the aditional features: label_segment, that indicates the trajectory segment
        to which the point belongs to.

    Note
    -----
    Speed features must be updated after split.
    """
    print('Split trajectories by max distance between adjacent points:', max_dist_between_adj_points)
    try:

        if DIST_TO_PREV not in move_data:
            move_data.generate_dist_features()

        if move_data.index.name is None:
            print('...setting {} as index'.format(label_id))
            move_data.set_index(label_id, inplace=True)

        curr_tid = 0
        if label_segment not in move_data:
            move_data[label_segment] = curr_tid

        ids = move_data.index.unique()
        count = 0
        df_size = move_data.shape[0]
        curr_perc_int = -1
        start_time = time.time()


        for idx in ids:
            """ increment index to trajectory"""
            curr_tid += 1

            """ filter dist max"""
            dist = (move_data.at[idx, DIST_TO_PREV] > max_dist_between_adj_points)
            """ check if object have more than one point to split"""
            if dist.shape == ():
                print('id: {} has not point to split'.format(idx))
                move_data.at[idx, label_segment] = curr_tid
                count+=1
            else:
                tids = np.empty(dist.shape[0], dtype=np.int64)
                tids.fill(curr_tid)
                for i, has_problem in enumerate(dist):
                    if has_problem:
                        curr_tid += 1
                        tids[i:] = curr_tid
                count += tids.shape[0]
                move_data.at[idx, label_segment] = tids

            curr_perc_int, est_time_str = progress_update(count, df_size, start_time, curr_perc_int, step_perc=20)

        if label_id == label_segment:
            move_data.reset_index(drop=True, inplace=True)
            print('... label_id = label_new_id, then reseting and drop index')
        else:
            move_data.reset_index(inplace=True)
            print('... Reseting index')
        print('\nTotal Time: {:.2f} seconds'.format((time.time() - start_time)))
        print('------------------------------------------\n')
    except Exception as e:
        print('label_id:{}\nidx:{}\n'.format(label_id, idx))
        raise e

def by_max_time(move_data, label_id=TRAJ_ID, max_time_between_adj_points=900.0, label_segment='tid_time'):
    """ Splits the trajectories into segments based on a maximum time set by the user.
    Parameters
    ----------
    move_data : dataframe
       The input trajectory data
    label_id : String, optional(id by default)
         Indicates the label of the id column in the users dataframe.
    max_time_between_adj_points : Float, optinal(50.0 by default)
        Specify the maximum time between two adjacent points
    label_segment : String, optional("tid_time" by default)
        The label of the column containing the ids of the formed segments. Is the new splitted id.
    Returns
    ------
    Returns the dataFrame with the aditional features: label_segment, that indicates the trajectory segment
        to which the point belongs to.

    Note
    -----
    Speed features must be updated after split.
    """

    print('Split trajectories by max_time_between_adj_points:', max_time_between_adj_points)
    try:
        if move_data.index.name is None:
            print('...setting {} as index'.format(label_id))
            move_data.set_index(label_id, inplace=True)

        curr_tid = 0
        if label_segment not in move_data:
            move_data[label_segment] = curr_tid

        ids = move_data.index.unique()
        count = 0
        df_size = move_data.shape[0]
        curr_perc_int = -1
        start_time = time.time()

        for idx in ids:
            """ increment index to trajectory"""
            curr_tid += 1

            """ filter time max"""
            times = (move_data.at[idx, TIME_TO_PREV] > max_time_between_adj_points)

            """ check if object have only one point to be removed """
            if times.shape == ():
                print('id: {} has not point to split'.format(id))
                move_data.at[idx, label_segment] = curr_tid
                count+=1
            else:
                tids = np.empty(times.shape[0], dtype=np.int64)
                tids.fill(curr_tid)
                for i, has_problem in enumerate(times):
                    if has_problem:
                        curr_tid += 1
                        tids[i:] = curr_tid
                count += tids.shape[0]
                move_data.at[idx, label_segment] = tids

            curr_perc_int, est_time_str = progress_update(count, df_size, start_time, curr_perc_int, step_perc=20)

        if label_id == label_segment:
            move_data.reset_index(drop=True, inplace=True)
            print('... label_id = label_new_id, then reseting and drop index')
        else:
            move_data.reset_index(inplace=True)
            print('... Reseting index')
        print('\nTotal Time: {:.2f} seconds'.format((time.time() - start_time)))
        print('------------------------------------------\n')
        #if drop_single_points:
         #   shape_before_drop = move_data.shape
          #  idx = move_data[ move_data[label_segment] == -1 ].index
           # if idx.shape[0] > 0:
            #    print('...Drop Trajectory with a unique GPS point\n')
             #   ids_before_drop = move_data[label_id].unique().shape[0]
              #  move_data.drop(index=idx, inplace=True)
               # print('...Object - before drop: {} - after drop: {}'.format(ids_before_drop, move_data[label_id].unique().shape[0]))
               # print('...Shape - before drop: {} - after drop: {}'.format(shape_before_drop, move_data.shape))
            #else:
             #   print('...No trajs with only one point.', move_data.shape)

    except Exception as e:
        print('label_id:{}\nidx:{}\n'.format(label_id, idx))
        raise e

def by_max_speed(move_data, label_id=TRAJ_ID, max_speed_between_adj_points=50.0, label_segment='tid_speed'):
    """ Splits the trajectories into segments based on a maximum speed set by the user.
    Parameters
    ----------
    move_data : dataframe.
       The input trajectory data.
    label_id : String, optional(id by default).
         Indicates the label of the id column in the users dataframe.
    max_speed_between_adj_points : Float, optinal(50.0 by default).
        Specify the maximum speed between two adjacent points.
    label_segment : String, optional("tid_time" by default).
        The label of the column containing the ids of the formed segments. Is the new splitted id.

    Returns
    ------
    Returns the dataFrame with the aditional features: label_segment, that indicates the trajectory segment
        to which the point belongs to.

    Note
    -----
    Speed features must be updated after split.
    """
    print('Split trajectories by max_speed_between_adj_points:', max_speed_between_adj_points)
    try:
        if move_data.index.name is None:
            print('...setting {} as index'.format(label_id))
            move_data.set_index(label_id, inplace=True)

        curr_tid = 0
        if label_segment not in move_data:
            move_data[label_segment] = curr_tid

        ids = move_data.index.unique()
        count = 0
        df_size = move_data.shape[0]
        curr_perc_int = -1
        start_time = time.time()

        for idx in ids:
            """ increment index to trajectory"""
            curr_tid += 1

            """ filter speed max"""
            speed = (move_data.at[idx, SPEED_TO_PREV] > max_speed_between_adj_points)
            """ check if object have only one point to be removed """
            if speed.shape == ():
                print('id: {} has not point to split'.format(id))
                move_data.at[idx, label_segment] = curr_tid
                count+=1
            else:
                tids = np.empty(speed.shape[0], dtype=np.int64)
                tids.fill(curr_tid)
                for i, has_problem in enumerate(speed):
                    if has_problem:
                        curr_tid += 1
                        tids[i:] = curr_tid
                count += tids.shape[0]
                move_data.at[idx, label_segment] = tids

            curr_perc_int, est_time_str = progress_update(count, df_size, start_time, curr_perc_int, step_perc=20)

        if label_id == label_segment:
            move_data.reset_index(drop=True, inplace=True)
            print('... label_id = label_new_id, then reseting and drop index')
        else:
            move_data.reset_index(inplace=True)
            print('... Reseting index')
        print('\nTotal Time: {:.2f} seconds'.format((time.time() - start_time)))
        print('------------------------------------------\n')

        #if drop_single_points:
         #   shape_before_drop = move_data.shape
          #  idx = move_data[move_data[label_segment] == -1].index
           # if idx.shape[0] > 0:
            #    print('...Drop Trajectory with a unique GPS point\n')
             #   ids_before_drop = move_data[label_id].unique().shape[0]
              #  move_data.drop(index=idx, inplace=True)
               # print('...Object - before drop: {} - after drop: {}'.format(ids_before_drop, move_data[label_id].unique().shape[0]))
               # print('...Shape - before drop: {} - after drop: {}'.format(shape_before_drop, move_data.shape))
                #create_update_dist_time_speed_features(move_data, label_segment, dic_labels)
            #else:
                #print('...No trajs with only one point.', move_data.shape)

    except Exception as e:
        print('label_id:{}\nidx:{}\n'.format(label_id, idx))
        raise e
