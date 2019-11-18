from pymove.utils.constants import (
	DATETIME,
    SITUATION,
    TRAJ_ID,
    DIST_TO_PREV,
    MOVE,
    STOP)
import time
from pymove.preprocessing.segmentation import segment_traj_by_max_dist
import numpy as np


def create_update_datetime_in_format_cyclical(move_data, label_datetime=DATETIME):
    try:
        # https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/
        # https://www.avanwyk.com/encoding-cyclical-features-for-deep-learning/
        print('Encoding cyclical continuous features - 24-hour time')
        if label_datetime in move_data:
            hours = move_data[label_datetime].dt.hour
            move_data['hour_sin'] = np.sin(2 * np.pi * hours / 23.0)
            move_data['hour_cos'] = np.cos(2 * np.pi * hours / 23.0)
            print('...hour_sin and  hour_cos features were created...\n')
    except Exception as e:
        raise e


def create_or_update_move_stop_by_dist_time(move_data, label_id=TRAJ_ID , dist_radius=30, time_radius=900):
    try:
        start_time = time.time()
        label_segment_stop = 'segment_stop'
        segment_traj_by_max_dist(move_data, label_id=label_id, max_dist_between_adj_points=dist_radius,
                                           label_segment=label_segment_stop)

        if (label_segment_stop in move_data):
            # update dist, time and speed using segment_stop

            move_data.generate_dist_time_speed_features(label_id=label_segment_stop)
            print(move_data)

            print('Create or update stop as True or False')
            print('...Creating stop features as True or False using {} to time in seconds'.format(time_radius))
            move_data[STOP] = False
            move_dataagg_tid = move_data.groupby(by=label_segment_stop).agg({'time_to_prev': 'sum'}).query(
                'time_to_prev > ' + str(time_radius)).index
            idx = move_data[move_data[label_segment_stop].isin(move_dataagg_tid)].index
            move_data.at[idx, STOP] = True
            print(move_data[STOP].value_counts())
            print('\nTotal Time: {:.2f} seconds'.format((time.time() - start_time)))
            print('-----------------------------------------------------\n')
    except Exception as e:
        raise e


def create_update_move_and_stop_by_radius(move_data, radius=0, target_label=DIST_TO_PREV,
                                          new_label=SITUATION):
    try:
        print('\nCreating or updating features MOVE and STOPS...\n')

        if DIST_TO_PREV not in move_data:
            move_data.generate_dist_features()

        conditions = (move_data[target_label] > radius), (move_data[target_label] <= radius)
        choices = [MOVE, STOP]

        move_data[new_label] = np.select(conditions, choices, np.nan)
        print('\n....There are {} stops to this parameters\n'.format(move_data[move_data[new_label] == STOP].shape[0]))
    except Exception as e:
        raise e
