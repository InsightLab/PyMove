# import time

# import numpy as np
# from scipy.interpolate import interp1d

# from pymove.core.dataframe import PandasMoveDataFrame
# from pymove.utils.constants import TID
# from pymove.utils.log import progress_bar
# from pymove.utils.trajectories import shift
# from pymove.utils.transformations import feature_values_using_filter

# # Fuction to solve problems after Map-matching


# def check_time_dist(
#     move_data,
#     index_name="tid",
#     tids=None,
#     max_dist_between_adj_points=5000,
#     max_time_between_adj_points=900,
#     max_speed=30,
#     inplace=True,
# ):
#     """
#     Used to verify that the trajectories points are in the correct order after
#     map matching, considering time and distance.

#     Parameters
#     ----------
#     move_data : dataframe
#        The input trajectories data
#     index_name: String, optional("tid" by default)
#         The name of the column to set as the new index during function execution. Indicates the tid column.
#     tids: array, optional(None by default)
#         The list of the unique keys of the index_name column.
#     max_dist_between_adj_points: double, optional(5000 by default)
#         The maximum distance between two adjacent points.
#     max_time_between_adj_points: double, optional(900 by default)
#         The maximum time interval between two adjacent points.
#     max_speed: double, optional(30 by default)
#         The maximum speed between two adjacent points.
#     inplace: boolean, optional(True by default)
#         if set to true the original dataframe will be altered,
#         otherwise the alteration will be made in a copy, that will be returned.

#     Returns
#     -------
#         move_data : dataframe
#             A copy of the original dataframe, with the alterations done by the function. (When inplace is False)
#         None
#             When inplace is True
#     """

#     if not inplace:
#         move_data = PandasMoveDataFrame(data=move_data.to_data_frame())

#     try:
#         if TID not in move_data:
#             move_data.generate_tid_based_on_id_datetime()

#         if move_data.index.name is not None:
#             print("reseting index...")
#             move_data.reset_index(inplace=True)

#         if tids is None:
#             tids = move_data[index_name].unique()

#         if move_data.index.name is None:
#             print("creating index...")
#             move_data.set_index(index_name, inplace=True)

#         for tid in progress_bar(
#             tids, desc="checking ascending distance and time"
#         ):
#             filter_ = move_data.at[tid, "isNode"] != 1

#             # be sure that distances are in ascending order
#             dists = move_data.at[tid, "distFromTrajStartToCurrPoint"][filter_]
#             assert np.all(
#                 dists[:-1] < dists[1:]
#             ), "distance feature is not in ascending order"

#             # be sure that times are in ascending order
#             times = move_data.at[tid, "time"][filter_].astype(np.float64)
#             assert np.all(
#                 times[:-1] < times[1:]
#             ), "time feature is not in ascending order"

#         count = 0

#         for tid in progress_bar(
#             tids, desc="checking delta_times, delta_dists and speeds"
#         ):
#             filter_ = move_data.at[tid, "isNode"] != 1

#             dists = move_data.at[tid, "distFromTrajStartToCurrPoint"][filter_]
#             delta_dists = (shift(dists, -1) - dists)[
#                 :-1
#             ]  # do not use last element (np.nan)

#             assert np.all(
#                 delta_dists <= max_dist_between_adj_points
#             ), "delta_dists must be <= {}".format(max_dist_between_adj_points)

#             times = move_data.at[tid, "time"][filter_].astype(np.float64)
#             delta_times = ((shift(times, -1) - times) / 1000.0)[
#                 :-1
#             ]  # do not use last element (np.nan)

#             assert np.all(
#                 delta_times <= max_time_between_adj_points
#             ), "delta_times must be <= {}".format(max_time_between_adj_points)

#             assert np.all(delta_times > 0), "delta_times must be > 0"

#             assert np.all(delta_dists > 0), "delta_dists must be > 0"

#             speeds = delta_dists / delta_times
#             assert np.all(speeds <= max_speed), "speeds > {}".format(max_speed)

#             size_id = 1 if filter_.shape == () else filter_.shape[0]
#             count += size_id

#         move_data.reset_index(inplace=True)
#         if not inplace:
#             return move_data

#     except Exception as e:
#         raise e


# def fix_time_not_in_ascending_order_id(
#     move_data, tid, index_name="tid", inplace=True
# ):
#     """
#     Used to correct time order between points of a  trajectory, after map
#     matching operations.

#     Parameters
#     ----------
#     move_data : dataframe
#        The input trajectories data
#     tid : String
#         The tid of the trajectory the user want to correct.
#     index_name: String, optional("tid" by default)
#         The name of the column to set as the new index during function execution. Indicates the tid column.
#     inplace: boolean, optional(True by default)
#         if set to true the original dataframe will be altered,
#         otherwise the alteration will be made in a copy, that will be returned.

#     Returns
#     -------
#         move_data : dataframe
#             A copy of the original dataframe, with the alterations done by the function. (When inplace is False)
#         size_id

#     Notes
#     -----
#     Do not use trajectories with only 1 point.
#     """

#     if not inplace:
#         move_data = PandasMoveDataFrame(data=move_data.to_data_frame())

#     if TID not in move_data:
#         move_data.generate_tid_based_on_id_datetime()

#     if "deleted" not in move_data:
#         move_data["deleted"] = False

#     if move_data.index.name is None:
#         print("creating index...")
#         move_data.set_index(index_name, inplace=True)

#     filter_ = (move_data.at[tid, "isNode"] != 1) & (
#         ~move_data.at[tid, "deleted"]
#     )

#     # be sure that distances are in ascending order
#     dists = move_data.at[tid, "distFromTrajStartToCurrPoint"][filter_]
#     assert np.all(
#         dists[:-1] <= dists[1:]
#     ), "distance feature is not in ascending order"

#     if filter_.shape == ():
#         size_id = 1
#         move_data.at[tid, "deleted"] = True
#     else:
#         size_id = filter_.shape[0]
#         times = move_data.at[tid, "time"][filter_]
#         idx_not_in_ascending_order = np.where(times[:-1] >= times[1:])[0] + 1

#         if idx_not_in_ascending_order.shape[0] > 0:
#             # print(tid, "idx_not_in_ascending_order:", idx_not_in_ascending_order, "times.shape", times.shape)

#             move_data.feature_values_using_filter_and_indexes(
#                 move_data,
#                 tid,
#                 "deleted",
#                 filter_,
#                 idx_not_in_ascending_order,
#                 True,
#             )
#             # equivalent of: move_data.at[tid, "deleted"][filter_][idx_not_in_ascending_order] = True

#             fix_time_not_in_ascending_order_id(
#                 move_data, tid, index_name=index_name
#             )

#     if inplace:
#         return size_id
#     else:
#         return move_data, size_id


# def fix_time_not_in_ascending_order_all(
#     move_data, index_name="tid", drop_marked_to_delete=False, inplace=True
# ):
#     """
#     Used to correct time order between points of the trajectories, after map
#     matching operations.

#     Parameters
#     ----------
#     move_data : dataframe
#        The input trajectories data
#     index_name: String, optional("tid" by default)
#         The name of the column to set as the new index during function execution.
#     drop_marked_to_delete: boolean, optional (False by default)
#         Indicates if rows marked as deleted should be dropped.
#     inplace: boolean, optional(True by default)
#         if set to true the original dataframe will be altered,
#         otherwise the alteration will be made in a copy, that will be returned.

#     Returns
#     -------
#         move_data : dataframe
#             A copy of the original dataframe, with the alterations done by the function. (When inplace is False)
#         None
#             When inplace is True
#     """

#     if not inplace:
#         move_data = PandasMoveDataFrame(data=move_data.to_data_frame())

#     try:

#         if TID not in move_data:
#             move_data.generate_tid_based_on_id_datetime()

#         if move_data.index.name is not None:
#             print("reseting index...")
#             move_data.reset_index(inplace=True)

#         print("dropping duplicate distances... shape before:", move_data.shape)
#         move_data.drop_duplicates(
#             subset=[index_name, "isNode", "distFromTrajStartToCurrPoint"],
#             keep="first",
#             inplace=True,
#         )
#         print("shape after:", move_data.shape)

#         print("sorting by id and distance...")
#         move_data.sort_values(
#             by=[index_name, "distFromTrajStartToCurrPoint"], inplace=True
#         )
#         print("sorting done")

#         tids = move_data[index_name].unique()
#         move_data["deleted"] = False

#         print("starting fix...")
#         time.time()
#         for tid in progress_bar(tids):
#             fix_time_not_in_ascending_order_id(move_data, tid, index_name)

#         move_data.reset_index(inplace=True)
#         idxs = move_data[move_data["deleted"]].index
#         print("{} rows marked for deletion.".format(idxs.shape[0]))

#         if idxs.shape[0] > 0 and drop_marked_to_delete:
#             print("shape before dropping: {}".format(move_data.shape))
#             move_data.drop(index=idxs, inplace=True)
#             move_data.drop(labels="deleted", axis=1, inplace=True)
#             print("shape after dropping: {}".format(move_data.shape))

#         if not inplace:
#             return move_data
#     except Exception as e:
#         raise e


# def interpolate_add_deltatime_speed_features(
#     move_data,
#     label_tid="tid",
#     max_time_between_adj_points=900,
#     max_dist_between_adj_points=5000,
#     max_speed=30,
#     inplace=True,
# ):
#     """Use to interpolate distances (x) to find times (y).
#      Parameters
#     ----------
#     move_data : dataframe
#        The input trajectories data
#     label_tid: String, optional("tid" by default)
#         The name of the column to set as the new index during function execution. Indicates the tid column.
#     max_dist_between_adj_points: double, optional(5000 by default)
#         The maximum distance between two adjacent points. Used only for verification.
#     max_time_between_adj_points: double, optional(900 by default)
#         The maximum time interval between two adjacent points. Used only for verification.
#     max_speed: double, optional(30 by default)
#         The maximum speed between two adjacent points. Used only for verification.
#     inplace: boolean, optional(True by default)
#         if set to true the original dataframe will be altered,
#         otherwise the alteration will be made in a copy, that will be returned.

#     Returns
#     -------
#         move_data : dataframe
#             A copy of the original dataframe, with the alterations done by the function. (When inplace is False)
#         None
#             When inplace is True
#     """

#     if not inplace:
#         move_data = PandasMoveDataFrame(data=move_data.to_data_frame())

#     if TID not in move_data:
#         move_data.generate_tid_based_on_id_datetime()

#     if move_data.index.name is not None:
#         print("reseting index...")
#         move_data.reset_index(inplace=True)

#     tids = move_data[label_tid].unique()
#     # tids = [2]

#     if move_data.index.name is None:
#         print("creating index...")
#         move_data.set_index(label_tid, inplace=True)

#     drop_trajectories = []
#     size = move_data.shape[0]
#     count = 0
#     time.time()

#     move_data["delta_time"] = np.nan
#     move_data["speed"] = np.nan

#     try:
#         for tid in progress_bar(tids):
#             filter_nodes = move_data.at[tid, "isNode"] == 1
#             size_id = 1 if filter_nodes.shape == () else filter_nodes.shape[0]
#             count += size_id

#             # y - time of snapped points
#             y_ = move_data.at[tid, "time"][~filter_nodes]
#             if y_.shape[0] < 2:
#                 # print("traj: {} - insuficient points ({}) for interpolation.
#                 # adding to drop list...".format(tid,  y_.shape[0]))
#                 drop_trajectories.append(tid)
#                 continue

#             assert np.all(
#                 y_[1:] >= y_[:-1]
#             ), "time feature is not in ascending order"

#             # x - distance from traj start to snapped points
#             x_ = move_data.at[tid, "distFromTrajStartToCurrPoint"][
#                 ~filter_nodes
#             ]

#             assert np.all(
#                 x_[1:] >= x_[:-1]
#             ), "distance feature is not in ascending order"

#             # remove duplicates in distances to avoid np.inf in future interpolation results
#             idx_duplicates = np.where(x_[1:] == x_[:-1])[0]
#             if idx_duplicates.shape[0] > 0:
#                 x_ = np.delete(x_, idx_duplicates)
#                 y_ = np.delete(y_, idx_duplicates)

#             if y_.shape[0] < 2:
#                 # print("traj: {} - insuficient points ({}) for interpolation.
#                 # adding to drop list...".format(tid,  y_.shape[0]))
#                 drop_trajectories.append(tid)
#                 continue

#             # compute delta_time and distance between points
#             # values = (ut.shift(move_data.at[tid, "time"][filter_nodes].astype(np.float64), -1)
#             # - move_data.at[tid, "time"][filter_nodes]) / 1000
#             # ut.change_move_datafeature_values_using_filter(move_data, tid, "delta_time", filter_nodes, values)
#             delta_time = ((shift(y_.astype(np.float64), -1) - y_) / 1000.0)[
#                 :-1
#             ]
#             dist_curr_to_next = (shift(x_, -1) - x_)[:-1]
#             speed = (dist_curr_to_next / delta_time)[:-1]

#             assert np.all(
#                 delta_time <= max_time_between_adj_points
#             ), "delta_time between points cannot be more than {}".format(
#                 max_time_between_adj_points
#             )
#             assert np.all(
#                 dist_curr_to_next <= max_dist_between_adj_points
#             ), "distance between points cannot be more than {}".format(
#                 max_dist_between_adj_points
#             )
#             assert np.all(
#                 speed <= max_speed
#             ), "speed between points cannot be more than {}".format(max_speed)

#             assert np.all(
#                 x_[1:] >= x_[:-1]
#             ), "distance feature is not in ascending order"

#             f_intp = interp1d(x_, y_, fill_value="extrapolate")

#             x2_ = move_data.at[tid, "distFromTrajStartToCurrPoint"][
#                 filter_nodes
#             ]
#             assert np.all(
#                 x2_[1:] >= x2_[:-1]
#             ), "distances in nodes are not in ascending order"

#             intp_result = f_intp(x2_)  # .astype(np.int64)
#             assert np.all(
#                 intp_result[1:] >= intp_result[:-1]
#             ), "resulting times are not in ascending order"

#             assert ~np.isin(
#                 np.inf, intp_result
#             ), "interpolation results with np.inf value(srs)"

#             # update time features for nodes. initially they are empty.
#             values = intp_result.astype(np.int64)
#             feature_values_using_filter(
#                 move_data, tid, "time", filter_nodes, values
#             )

#             # create delta_time feature
#             values = (
#                 shift(
#                     move_data.at[tid, "time"][filter_nodes].astype(np.float64),
#                     -1,
#                 )
#                 - move_data.at[tid, "time"][filter_nodes]
#             ) / 1000
#             feature_values_using_filter(
#                 move_data, tid, "delta_time", filter_nodes, values
#             )

#             # create speed feature
#             values = (
#                 move_data.at[tid, "edgeDistance"][filter_nodes]
#                 / move_data.at[tid, "delta_time"][filter_nodes]
#             )
#             feature_values_using_filter(
#                 move_data, tid, "speed", filter_nodes, values
#             )

#     except Exception as e:
#         raise e

#     print(count, size)
#     print(
#         "we still need to drop {} trajectories with only 1 gps point".format(
#             len(drop_trajectories)
#         )
#     )
#     move_data.reset_index(inplace=True)
#     idxs_drop = move_data[
#         move_data[label_tid].isin(drop_trajectories)
#     ].index.values
#     print(
#         "dropping {} rows in {} trajectories with only 1 gps point".format(
#             idxs_drop.shape[0], len(drop_trajectories)
#         )
#     )
#     if idxs_drop.shape[0] > 0:
#         print("shape before dropping: {}".format(move_data.shape))
#         move_data.drop(index=idxs_drop, inplace=True)
#         print("shape after dropping: {}".format(move_data.shape))

#     if not inplace:
#         return move_data
