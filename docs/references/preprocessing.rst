===================
Preprocessing
===================

.. currentmodule:: pymove.preprocessing

.. autosummary::
	compression.compress_segment_stop_to_point

	.. autofunction:: pymove.preprocessing.compression.compress_segment_stop_to_point


.. autosummary::
	filters.by_bbox
	filters.by_datetime
	filters.by_label
	filters.by_id
	filters.by_tid
	filters.outliers
	filters.clean_duplicates
	filters.clean_consecutive_duplicates
	filters.clean_nan_values
	filters._filter_single_by_max
	filters._filter_single_by_max
	filters._filter_data
	filters._clean_gps
	filters.clean_gps_jumps_by_distance
	filters.clean_gps_nearby_points_by_distances
	filters.clean_gps_nearby_points_by_speed
	filters.clean_gps_speed_max_radius
	filters.clean_trajectories_with_few_points
	filters.clean_trajectories_short_and_few_points
	filters.clean_id_by_time_max

	.. autofunction:: pymove.preprocessing.filters.by_bbox
	.. autofunction:: pymove.preprocessing.filters.by_datetime
	.. autofunction:: pymove.preprocessing.filters.by_label
	.. autofunction:: pymove.preprocessing.filters.by_id
	.. autofunction:: pymove.preprocessing.filters.by_tid
	.. autofunction:: pymove.preprocessing.filters.outliers
	.. autofunction:: pymove.preprocessing.filters.clean_duplicates
	.. autofunction:: pymove.preprocessing.filters.clean_consecutive_duplicates
	.. autofunction:: pymove.preprocessing.filters.clean_nan_values
	.. autofunction:: pymove.preprocessing.filters._filter_single_by_max
	.. autofunction:: pymove.preprocessing.filters._filter_single_by_max
	.. autofunction:: pymove.preprocessing.filters._filter_data
	.. autofunction:: pymove.preprocessing.filters._clean_gps
	.. autofunction:: pymove.preprocessing.filters.clean_gps_jumps_by_distance
	.. autofunction:: pymove.preprocessing.filters.clean_gps_nearby_points_by_distances
	.. autofunction:: pymove.preprocessing.filters.clean_gps_nearby_points_by_speed
	.. autofunction:: pymove.preprocessing.filters.clean_gps_speed_max_radius
	.. autofunction:: pymove.preprocessing.filters.clean_trajectories_with_few_points
	.. autofunction:: pymove.preprocessing.filters.clean_trajectories_short_and_few_points
	.. autofunction:: pymove.preprocessing.filters.clean_id_by_time_max


.. autosummary::
	map_matching.


.. autosummary::
	segmentation.bbox_split
	segmentation._drop_single_point
	segmentation._filter_and_dist_time_speed
	segmentation._filter_or_dist_time_speed
	segmentation.prepare_segmentation
	segmentation._update_curr_tid_count
	segmentation._filter_by
	segmentation.by_dist_time_speed
	segmentation.by_max_dist
	segmentation.by_max_time
	segmentation.by_max_speed


	.. autofunction:: pymove.preprocessing.segmentation.bbox_split
	.. autofunction:: pymove.preprocessing.segmentation._drop_single_point
	.. autofunction:: pymove.preprocessing.segmentation._filter_and_dist_time_speed
	.. autofunction:: pymove.preprocessing.segmentation._filter_or_dist_time_speed
	.. autofunction:: pymove.preprocessing.segmentation.prepare_segmentation
	.. autofunction:: pymove.preprocessing.segmentation._update_curr_tid_count
	.. autofunction:: pymove.preprocessing.segmentation._filter_by
	.. autofunction:: pymove.preprocessing.segmentation.by_dist_time_speed
	.. autofunction:: pymove.preprocessing.segmentation.by_max_dist
	.. autofunction:: pymove.preprocessing.segmentation.by_max_time
	.. autofunction:: pymove.preprocessing.segmentation.by_max_speed


.. autosummary::
	stay_point_detection.create_update_datetime_in_format_cyclical
	stay_point_detection.create_or_update_move_stop_by_dist_time
	stay_point_detection.create_update_move_and_stop_by_radius



	.. autofunction:: pymove.preprocessing.stay_point_detection.create_update_datetime_in_format_cyclical
	.. autofunction:: pymove.preprocessing.stay_point_detection.create_or_update_move_stop_by_dist_time
	.. autofunction:: pymove.preprocessing.stay_point_detection.create_update_move_and_stop_by_radius






