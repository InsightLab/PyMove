import time
from pymove.utils.trajectories import progress_update

def compress_segment_stop_to_point(move_data, label_segment='segment_stop', label_stop='stop', point_mean='default',
                                   drop_moves=True):
    """ compreess a segment to point setting lat_mean e lon_mean to each segment"""
    try:

        if (label_segment in move_data) & (label_stop in move_data):
            # start_time = time.time()

            print("...setting mean to lat and lon...")
            move_data['lat_mean'] = -1.0
            move_data['lon_mean'] = -1.0

            if drop_moves is False:
                move_data.at[move_data[move_data[label_stop] == False].index, 'lat_mean'] = np.NaN
                move_data.at[move_data[move_data[label_stop] == False].index, 'lon_mean'] = np.NaN
            else:
                print('...move segments will be dropped...')

            print("...get only segments stop...")
            segments = move_data[move_data[label_stop] == True][label_segment].unique()

            sum_size_id = 0
            move_datasize = move_data[move_data[label_stop] == True].shape[0]
            curr_perc_int = -1
            start_time = time.time()

            for idx in tqdm(segments):
                filter_ = (move_data[label_segment] == idx)

                size_id = move_data[filter_].shape[0]
                # veirify se o filter is None
                if (size_id > 1):
                    # get first and last point of each stop segment
                    ind_start = move_data[filter_].iloc[[0]].index
                    ind_end = move_data[filter_].iloc[[-1]].index

                    # default: point
                    if point_mean == 'default':
                        # print('...Lat and lon are defined based on point that repeats most within the segment')
                        p = move_data[filter_].groupby(['lat', 'lon'], as_index=False).agg({'id': 'count'}).sort_values(
                            ['id']).tail(1)
                        move_data.at[ind_start, 'lat_mean'] = p.iloc[0, 0]
                        move_data.at[ind_start, 'lon_mean'] = p.iloc[0, 1]
                        move_data.at[ind_end, 'lat_mean'] = p.iloc[0, 0]
                        move_data.at[ind_end, 'lon_mean'] = p.iloc[0, 1]

                    elif point_mean == 'centroid':
                        # print('...Lat and lon are defined by centroid of the all points into segment')
                        # set lat and lon mean to first_point and last points to each segment
                        move_data.at[ind_start, 'lat_mean'] = move_data.loc[filter_]['lat'].mean()
                        move_data.at[ind_start, 'lon_mean'] = move_data.loc[filter_]['lon'].mean()
                        move_data.at[ind_end, 'lat_mean'] = move_data.loc[filter_]['lat'].mean()
                        move_data.at[ind_end, 'lon_mean'] = move_data.loc[filter_]['lon'].mean()
                else:
                    print('There are segments with only one point: {}'.format(idx))

                sum_size_id += size_id
                curr_perc_int, est_time_str = progress_update(sum_size_id, move_datasize, start_time, curr_perc_int,
                                                                 step_perc=5)

            shape_before = move_data.shape[0]

            # filter points to drop
            filter_drop = (move_data['lat_mean'] == -1.0) & (move_data['lon_mean'] == -1.0)
            shape_drop = move_data[filter_drop].shape[0]

            if shape_drop > 0:
                print("...Dropping {} points...".format(shape_drop))
                move_data.drop(move_data[filter_drop].index, inplace=True)

            print("...Shape_before: {}\n...Current shape: {}".format(shape_before, move_data.shape[0]))
            print('...Compression time: {:.3f} seconds'.format((time.time() - start_time)))
            print('-----------------------------------------------------\n')
        else:
            print('{} or {} is not in dataframe'.format(label_stop, label_segment))
    except Exception as e:
        raise e


