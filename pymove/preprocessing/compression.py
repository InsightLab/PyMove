import numpy as np

from pymove.preprocessing.stay_point_detection import (
    create_or_update_move_stop_by_dist_time,
)
from pymove.utils.constants import (
    LAT_MEAN,
    LATITUDE,
    LON_MEAN,
    LONGITUDE,
    SEGMENT_STOP,
    STOP,
    TRAJ_ID,
)
from pymove.utils.log import progress_bar, timer_decorator


@timer_decorator
def compress_segment_stop_to_point(
    move_data,
    label_segment=SEGMENT_STOP,
    label_stop=STOP,
    point_mean='default',
    drop_moves=False,
    label_id=TRAJ_ID,
    dist_radius=30,
    time_radius=900,
    inplace=False,
):
    """
    Compress the trajectories using the stop points in the dataframe.
    Compress a segment to point setting lat_mean e lon_mean to each segment.

    Parameters
    ----------
    move_data : dataframe
       The input trajectory data
    label_segment : String, optional("segment_stop" by default)
        The label of the column containing the ids of the formed segments.
        Is the new splitted id.
    label_stop : String, optional(stop by default)
        Is the name of the column that indicates if a point is a stop.
    point_mean : String, optional(default by default)
        Indicates whether the mean points should be calculated using
        centroids or the point that repeat the most.
    drop_moves : Boolean, optional(False by default)
        If set to true, the moving points will be dropped from the dataframe.
    label_id : String, optional(id by default)
         Used to create the stay points used in the compression.
         If the dataset already has the stop move, this
         parameter should be ignored.
         Indicates the label of the id column in the user"srs dataframe.
    dist_radius : Double, optional(30 by default)
        Used to create the stay points used in the compression.
        If the dataset already has the stop move, this
        parameter should be ignored.
        The first step in this function is segmenting the trajectory.
        The segments are used to find the stop points.
        The dist_radius defines the distance used in the segmentation.
    time_radius :  Double, optional(900 by default)
        Used to create the stay points used in the compression.
        If the dataset already has the stop move, this
         parameter should be ignored.
        The time_radius used to determine if a segment is a stop.
        If the user stayed in the segment for a time
        greater than time_radius, than the segment is a stop.
    inplace : boolean, optional(True by default)
        if set to true the original dataframe will be altered to contain
        the result of the filtering, otherwise a copy will be returned.

    Returns
    -------
    DataFrame with 3 additional features: segment_stop, lat_mean and lon_mean
        segment_stop indicates the trajectory segment to which the point belongs
        lat_mean and lon_mean:
            if the default option is used, lat_mean and lon_mean are defined
            based on point that repeats most within the segment
            On the other hand, if centroid option is used,
            lat_mean and lon_mean are defined by centroid of
            the all points into segment

    """

    try:
        if not inplace:
            move_data = move_data[:]

        if (label_segment not in move_data) & (label_stop not in move_data):
            create_or_update_move_stop_by_dist_time(
                move_data, dist_radius, time_radius, label_id
            )

        print('...setting mean to lat and lon...')
        lat_mean = np.full(move_data.shape[0], -1.0, dtype=np.float64)
        lon_mean = np.full(move_data.shape[0], -1.0, dtype=np.float64)

        if drop_moves is False:
            lat_mean[move_data[~move_data[label_stop]].index] = np.NaN
            lon_mean[move_data[~move_data[label_stop]].index] = np.NaN
        else:
            print('...move segments will be dropped...')

        print('...get only segments stop...', flush=True)
        segments = move_data[move_data[label_stop]][label_segment].unique()

        for idx in progress_bar(
            segments, desc=f'Generating {label_segment} and {label_stop}'
        ):
            filter_ = move_data[label_segment] == idx

            size_id = move_data[filter_].shape[0]
            # verify if filter is None
            if size_id > 1:
                # get first and last point of each stop segment
                ind_start = move_data[filter_].iloc[[0]].index
                ind_end = move_data[filter_].iloc[[-1]].index

                if point_mean == 'default':
                    # print('...Lat and lon are defined based on point
                    # that repeats most within the segment')
                    p = (
                        move_data[filter_]
                        .groupby([LATITUDE, LONGITUDE], as_index=False)
                        .agg({'id': 'count'})
                        .sort_values(['id'])
                        .tail(1)
                    )
                    lat_mean[ind_start] = p.iloc[0, 0]
                    lon_mean[ind_start] = p.iloc[0, 1]
                    lat_mean[ind_end] = p.iloc[0, 0]
                    lon_mean[ind_end] = p.iloc[0, 1]

                elif point_mean == 'centroid':
                    # set lat and lon mean to first_point
                    # and last points to each segment
                    lat_mean[ind_start] = move_data.loc[filter_][LATITUDE].mean()
                    lon_mean[ind_start] = move_data.loc[filter_][LONGITUDE].mean()
                    lat_mean[ind_end] = move_data.loc[filter_][LATITUDE].mean()
                    lon_mean[ind_end] = move_data.loc[filter_][LONGITUDE].mean()
            else:
                print('There are segments with only one point: {}'.format(idx))

        move_data[LAT_MEAN] = lat_mean
        move_data[LON_MEAN] = lon_mean
        del lat_mean
        del lon_mean

        shape_before = move_data.shape[0]
        # filter points to drop
        filter_drop = (
            (move_data[LAT_MEAN] == -1.0)
            & (move_data[LON_MEAN] == -1.0)
        )
        shape_drop = move_data[filter_drop].shape[0]

        if shape_drop > 0:
            print('...Dropping %s points...' % shape_drop)
            move_data.drop(move_data[filter_drop].index, inplace=True)

        print(
            '...Shape_before: %s\n...Current shape: %s'
            % (shape_before, move_data.shape[0])
        )

        if not inplace:
            return move_data
    except Exception as e:
        raise e
