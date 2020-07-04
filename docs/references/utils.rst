===================
Utils
===================

.. currentmodule:: pymove.utils

.. autosummary::
	conversions.lat_meters
    conversions.list_to_str
    conversions.list_to_csv_str
    conversions.list_to_svm_line
    conversions.lon_to_x_spherical
    conversions.lat_to_y_spherical
    conversions.x_to_lon_spherical
    conversions.y_to_lat_spherical
    conversions.ms_to_kmh
    conversions.kmh_to_ms
    conversions.meters_to_kilometers
    conversions.kilometers_to_meters
    conversions.seconds_to_minutes
    conversions.minute_to_seconds
    conversions.minute_to_hours
    conversions.hours_to_minute
    conversions.seconds_to_hours
    conversions.hours_to_seconds

    .. autofunction:: pymove.utils.conversions.lat_meters
    .. autofunction:: pymove.utils.conversions.list_to_str
    .. autofunction:: pymove.utils.conversions.list_to_csv_str
    .. autofunction:: pymove.utils.conversions.list_to_svm_line
    .. autofunction:: pymove.utils.conversions.lon_to_x_spherical
    .. autofunction:: pymove.utils.conversions.lat_to_y_spherical
    .. autofunction:: pymove.utils.conversions.x_to_lon_spherical
    .. autofunction:: pymove.utils.conversions.y_to_lat_spherical
    .. autofunction:: pymove.utils.conversions.ms_to_kmh
    .. autofunction:: pymove.utils.conversions.kmh_to_ms
    .. autofunction:: pymove.utils.conversions.meters_to_kilometers
    .. autofunction:: pymove.utils.conversions.kilometers_to_meters
    .. autofunction:: pymove.utils.conversions.seconds_to_minutes
    .. autofunction:: pymove.utils.conversions.minute_to_seconds
    .. autofunction:: pymove.utils.conversions.minute_to_hours
    .. autofunction:: pymove.utils.conversions.hours_to_minute
    .. autofunction:: pymove.utils.conversions.seconds_to_hours
    .. autofunction:: pymove.utils.conversions.hours_to_seconds


.. autosummary::
	datetime.date_to_str
    datetime.str_to_datetime
    datetime.to_str
    datetime.to_min
    datetime.min_to_datetime
    datetime.to_day_of_week_int
    datetime.working_day
    datetime.now_str
    datetime.deltatime_str
    datetime.timestamp_to_millis
    datetime.millis_to_timestamp
    datetime.time_to_str
    datetime.str_to_time
    datetime.elapsed_time_dt
    datetime.diff_time

    .. autofunction:: pymove.utils.datetime.date_to_str
    .. autofunction:: pymove.utils.datetime.str_to_datetime
    .. autofunction:: pymove.utils.datetime.to_str
    .. autofunction:: pymove.utils.datetime.to_min
    .. autofunction:: pymove.utils.datetime.min_to_datetime
    .. autofunction:: pymove.utils.datetime.to_day_of_week_int
    .. autofunction:: pymove.utils.datetime.working_day
    .. autofunction:: pymove.utils.datetime.now_str
    .. autofunction:: pymove.utils.datetime.deltatime_str
    .. autofunction:: pymove.utils.datetime.timestamp_to_millis
    .. autofunction:: pymove.utils.datetime.millis_to_timestamp
    .. autofunction:: pymove.utils.datetime.time_to_str
    .. autofunction:: pymove.utils.datetime.str_to_time
    .. autofunction:: pymove.utils.datetime.elapsed_time_dt
    .. autofunction:: pymove.utils.datetime.diff_time


.. autosummary::
	db.connect_postgres
    db.write_postgres
    db.read_postgres
    db.read_sql_inmem_uncompressed
    db.read_sql_tmpfile
    db.connect_mongo
    db.get_mongo_collection
    db.write_mongo
    db.read_mongo

    .. autofunction:: pymove.utils.db.connect_postgres
    .. autofunction:: pymove.utils.db.write_postgres
    .. autofunction:: pymove.utils.db.read_postgres
    .. autofunction:: pymove.utils.db.read_sql_inmem_uncompressed
    .. autofunction:: pymove.utils.db.read_sql_tmpfile
    .. autofunction:: pymove.utils.db.connect_mongo
    .. autofunction:: pymove.utils.db.get_mongo_collection
    .. autofunction:: pymove.utils.db.write_mongo
    .. autofunction:: pymove.utils.db.read_mongo


.. autosummary::
	distances.haversine

    .. autofunction:: pymove.utils.distances.haversine


.. autosummary::
	log.timer_decorator
    log._log_progress

    .. autofunction:: pymove.utils.log.timer_decorator
    .. autofunction:: pymove.utils.log._log_progress


.. autosummary::
	mapfolium.add_map_legend

    .. autofunction:: pymove.utils.mapfolium.add_map_legend


.. autosummary::
	math.std
    math.avg_std
    math.std_sample
    math.avg_std_sample
    math.arrays_avg
    math.array_stats
    math.interpolation

    .. autofunction:: pymove.utils.math.std
    .. autofunction:: pymove.utils.math.avg_std
    .. autofunction:: pymove.utils.math.std_sample
    .. autofunction:: pymove.utils.math.avg_std_sample
    .. autofunction:: pymove.utils.math.arrays_avg
    .. autofunction:: pymove.utils.math.array_stats
    .. autofunction:: pymove.utils.math.interpolation


.. autosummary::
	mem.reduce_mem_usage_automatic
    mem.total_size
    mem.begin_operation
    mem.end_operation
    mem.sizeof_fmt
    mem.top_mem_vars

    .. autofunction:: pymove.utils.mem.reduce_mem_usage_automatic
    .. autofunction:: pymove.utils.mem.total_size
    .. autofunction:: pymove.utils.mem.begin_operation
    .. autofunction:: pymove.utils.mem.end_operation
    .. autofunction:: pymove.utils.mem.sizeof_fmt
    .. autofunction:: pymove.utils.mem.top_mem_vars


.. autosummary::
	trajectories.read_csv
    trajectories.format_labels
    trajectories.flatten_dict
    trajectories.flatten_columns
    trajectories.shift
    trajectories.fill_list_with_new_values
    trajectories.save_bbox

    .. autofunction:: pymove.utils.trajectories.read_csv
    .. autofunction:: pymove.utils.trajectories.format_labels
    .. autofunction:: pymove.utils.trajectories.flatten_dict
    .. autofunction:: pymove.utils.trajectories.flatten_columns
    .. autofunction:: pymove.utils.trajectories.shift
    .. autofunction:: pymove.utils.trajectories.fill_list_with_new_values
    .. autofunction:: pymove.utils.trajectories.save_bbox


.. autosummary::
	transformations.feature_values_using_filter
    transformations.feature_values_using_filter_and_indexes

    .. autofunction:: pymove.utils.transformations.feature_values_using_filter
    .. autofunction:: pymove.utils.transformations.feature_values_using_filter_and_indexes
