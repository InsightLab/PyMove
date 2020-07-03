===================
Utils
===================

.. currentmodule:: pymove.utils

.. autosummary::
	conversions.

.. autosummary::
	datetime.

    .. autofunction:: pymove.utils.datetime.date_to_str
    .. autofunction:: pymove.utils.datetime.str_to_datetime

.. autosummary::
	db.

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
	distances.

    .. autofunction:: pymove.utils.distances.haversine


.. autosummary::
	log.

    .. autofunction:: pymove.utils.log.timer_decorator
    .. autofunction:: pymove.utils.log._log_progress


.. autosummary::
	mapfolium.

    .. autofunction:: pymove.utils.mapfolium.add_map_legend


.. autosummary::
	math.

    .. autofunction:: pymove.utils.math.std
    .. autofunction:: pymove.utils.math.avg_std
    .. autofunction:: pymove.utils.math.std_sample
    .. autofunction:: pymove.utils.math.avg_std_sample
    .. autofunction:: pymove.utils.math.arrays_avg
    .. autofunction:: pymove.utils.math.array_stats
    .. autofunction:: pymove.utils.math.interpolation


.. autosummary::
	mem.

    .. autofunction:: pymove.utils.mem.reduce_mem_usage_automatic
    .. autofunction:: pymove.utils.mem.total_size
    .. autofunction:: pymove.utils.mem.begin_operation
    .. autofunction:: pymove.utils.mem.end_operation
    .. autofunction:: pymove.utils.mem.sizeof_fmt
    .. autofunction:: pymove.utils.mem.top_mem_vars


.. autosummary::
	trajectories.

    .. autofunction:: pymove.utils.trajectories.read_csv
    .. autofunction:: pymove.utils.trajectories.format_labels
    .. autofunction:: pymove.utils.trajectories.flatten_dict
    .. autofunction:: pymove.utils.trajectories.flatten_columns
    .. autofunction:: pymove.utils.trajectories.shift
    .. autofunction:: pymove.utils.trajectories.fill_list_with_new_values
    .. autofunction:: pymove.utils.trajectories.save_bbox


.. autosummary::
	transformations.

    .. autofunction:: pymove.utils.transformations.feature_values_using_filter
    .. autofunction:: pymove.utils.transformations.feature_values_using_filter_and_indexes
