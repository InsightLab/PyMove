import numpy as np

from pymove.utils.constants import (
    DATETIME,
    DIST_PREV_TO_NEXT,
    DIST_TO_NEXT,
    DIST_TO_PREV,
    LATITUDE,
    LONGITUDE,
    SPEED_TO_PREV,
    TID,
    TIME_TO_PREV,
    TRAJ_ID,
)


def get_bbox_by_radius(coordinates, radius=1000):
    """
    Defines minimum and maximum coordinates,
    given a distance radius from a point.

    Parameters
    ----------
    coords : tupla (lat, lon)
        The coordinates of point

    radius: Float, optional (1000 by default)

    Returns
    -------
    array
        coordinates min and max of the bbox

    References
    ----------
        https://mathmesquita.me/2017/01/16/filtrando-localizacao-em-um-raio.html
    """
    try:
        earth_radius = 6371000
        r = radius / earth_radius

        lat, lon = np.radians(coordinates)

        latmin = lat - r
        latmax = lat + r

        delta_lon = np.arcsin(np.sin(r) / np.cos(lat))

        lonmin = lon - delta_lon
        lonmax = lon + delta_lon

        return np.rad2deg([latmin, lonmin, latmax, lonmax])

    except Exception as e:
        raise e


def by_bbox(move_data, bbox, filter_out=False, inplace=False):
    """
    Filters points of the trajectories according to specified bounding box.

    Parameters
    ----------
    move_data : dataframe
       The input trajectories data
    bbox : tuple
        Tuple of 4 elements, containing the minimum and maximum values
        of latitude and longitude of the bounding box.
    filter_out : boolean, optional(false by default)
        If set to false the function will return the trajectories points
        within the bounding box, and the points outside otherwise
    inplace : boolean, optional(false by default)
        if set to true the original dataframe will be altered to contain
        the result of the filtering, otherwise a copy will be returned.

    Returns
    -------
    dataframe or None
        Returns dataframe with trajectories points filtered by bounding box.

    """

    filter_ = (
        (move_data[LATITUDE] >= bbox[0])
        & (move_data[LONGITUDE] >= bbox[1])
        & (move_data[LATITUDE] <= bbox[2])
        & (move_data[LONGITUDE] <= bbox[3])
    )
    try:
        if filter_out:
            filter_ = ~filter_

        return move_data.drop(index=move_data[~filter_].index, inplace=inplace)
    except Exception as e:
        raise e


def by_datetime(
    move_data,
    start_datetime=None,
    end_datetime=None,
    filter_out=False,
    inplace=False,
):
    """
    Filters trajectories points according to specified time range.

    Parameters
    ----------
    move_data : dataframe
        The input trajectory data
    start_datetime : String
        The start date and time (Datetime format) of the time range
    end_datetime : String
        The end date and time (Datetime format) of the time range
    filter_out : boolean, optional, default False
        If set to true, the function will return the points of
        the trajectories with timestamp outside the time range.
        The points whithin the time range will be return if filter_out is False.
    inplace : boolean, optional, default False
        if set to true the original dataframe will be altered to contain
        the result of the filtering, otherwise a copy will be returned.

    Returns
    -------
    dataframe or None
        Returns dataframe with trajectories points filtered by time range.

    """

    try:
        if start_datetime is not None and end_datetime is not None:
            filter_ = (
                (move_data[DATETIME] >= start_datetime)
                & (move_data[DATETIME] <= end_datetime)
            )
        elif end_datetime is not None:
            filter_ = move_data[DATETIME] <= end_datetime
        else:
            filter_ = move_data[DATETIME] >= start_datetime

        if filter_out:
            filter_ = ~filter_

        return move_data.drop(index=move_data[~filter_].index, inplace=inplace)
    except Exception as e:
        raise e


def by_label(move_data, value, label_name, filter_out=False, inplace=False):
    """
    Filters trajectories points according to specified value and column label.

    Parameters
    ----------
    move_data : dataframe
        The input trajectory data
    value : The type_ of the feature values to be use to filter the trajectories
        Specifies the value used to filter the trajectories points
    label_name : String
        Specifies the label of the column used in the filtering
    filter_out : boolean, optional(false by default)
        If set to True, it will return trajectory points with feature
        value different from the value specified in the parameters
        The trajectories points with the same feature value as
        the one specifed in the parameters.
    inplace : boolean, optional(false by default)
        if set to true the original dataframe will be altered to contain
        the result of the filtering, otherwise a copy will be returned.

    Returns
    -------
    dataframe or None
        Returns dataframe with trajectories points filtered by label.

    """

    try:
        filter_ = move_data[label_name] == value
        if filter_out:
            filter_ = ~filter_

        return move_data.drop(index=move_data[~filter_].index, inplace=inplace)
    except Exception as e:
        raise e


def by_id(
    move_data, id_=None, label_id=TRAJ_ID, filter_out=False, inplace=False
):
    """
    Filters trajectories points according to specified trajectory id.

    Parameters
    ----------
    move_data : dataframe
        The input trajectory data
    id_ : Integer
        Specifies the number of the id used to filter the trajectories points
    label_id : String, optional, default 'id'
        The label of the column which contains the id of the trajectories
    filter_out : boolean, optional(false by default)
        If set to true, the function will return the points of the trajectories
        with the same id as the one specified by the parameter value.
        If set to false it will return the points of the trajectories
        with a different id from the one specified in the parameters.
    inplace : boolean, optional(false by default)
        if set to true the original dataframe will be altered to contain
        the result of the filtering, otherwise a copy will be returned.

    Returns
    -------
    dataframe or None
        Returns dataframe with trajectories points filtered by id.

    """

    return by_label(move_data, id_, label_id, filter_out, inplace)


def by_tid(
    move_data, tid_=None, filter_out=False, inplace=False
):
    """
    Filters trajectories points according to a specified  trajectory tid.

    Parameters
    ----------
    move_data : dataframe
        The input trajectory data
    tid_ : String
        Specifies the number of the tid used to filter the trajectories points
    label_tid : String, optional, default 'tid'
        The label of the column in the user"srs dataframe which contains
        the tid of the trajectories
    filter_out : boolean, optional(false by default)
        If set to true, the function will return the points of the
        trajectories with the same tid as the one specified.
        If set to false it will return the points of the trajectories
        with a different tid from the one specified in the parameters.
    inplace : boolean, optional(false by default)
        if set to true the original dataframe will be altered to contain
        the result of the filtering, otherwise a copy will be returned.

    Returns
    -------
    dataframe or None
        Returns a dataframe with trajectories points filtered.

    """

    return by_label(move_data, tid_, TID, filter_out, inplace)


def outliers(
    move_data,
    jump_coefficient=3.0,
    threshold=1,
    filter_out=False,
    inplace=False,
):
    """
    Filters trajectories points that are outliers.

    Parameters
    ----------
    move_data : dataframe
        The input trajectory data
    jump_coefficient : Float, optional(3.0 by default)
    threshold : Float, optional(1 by default)
        Minimum value that the distance features must have
        in order to be considered outliers
    filter_out : boolean, optional(false by default)
        If set to true, the function will return the points of the
        trajectories that are not outliers.
        If set to false it will return the points of the
        trajectories are outliers.
    inplace : boolean, optional(false by default)
        if set to true the original dataframe will be altered to contain
        the result of the filtering, otherwise a copy will be returned.


    Returns
    -------
    dataframe or None
        Returns a dataframe with the trajectories outliers.

    """

    if DIST_TO_PREV not in move_data:
        move_data.generate_dist_features()

    if move_data.index.name is not None:
        print('...Reset index for filtering\n')
        move_data.reset_index(inplace=True)

    if (
        DIST_TO_PREV in move_data
        and DIST_TO_NEXT
        and DIST_PREV_TO_NEXT in move_data
    ):
        jump = jump_coefficient * move_data[DIST_PREV_TO_NEXT]
        filter_ = (
            (move_data[DIST_TO_NEXT] > threshold)
            & (move_data[DIST_TO_PREV] > threshold)
            & (move_data[DIST_PREV_TO_NEXT] > threshold)
            & (jump < move_data[DIST_TO_NEXT])
            & (jump < move_data[DIST_TO_PREV])
        )

        if filter_out:
            filter_ = ~filter_

        print('...Filtering jumps \n')
        return move_data.drop(index=move_data[~filter_].index, inplace=inplace)

    else:
        print('...Distances features were not created')
        return move_data


def clean_duplicates(
    move_data, subset=None, keep='first', inplace=False, sort=True
):
    """
    Removes the duplicate rows of the Dataframe, optionally only certain
    columns can be consider.

    Parameters
    ----------
    move_data : dataframe
        The input trajectory data
    subset : String or Array of Strings, optional, default None
        Specify  Column label or sequence of labels, considered
        for identifying duplicates. By default all columns are used.
    keep : 'first', 'last', optional, default 'first'
        If keep is set as first, all the duplicates except for
        the first occurrence will be dropped.
        On the other hand if set to last, all duplicates except for
        the last occurrence will be dropped.
        If set to False, all duplicates are dropped.
    inplace : boolean, optional, default False
        if set to true the original dataframe will be altered,
        the duplicates will be dropped in place,
        otherwise a copy will be returned.
    sort : boolean, optional, default True
        If set to True the data will be sorted by id and datetime.
        If set to False the data won"t be sorted.

    Returns
    -------
    dataframe or None
        Returns a dataframe without the trajectories duplicates.

    """

    print('\nRemove rows duplicates by subset')

    if sort is True:
        print(
            '...Sorting by %s and %s to increase performance\n'
            % (TRAJ_ID, DATETIME)
        )
        move_data.sort_values([TRAJ_ID, DATETIME], inplace=True)

    idx = move_data.duplicated(subset=subset)
    tam_drop = move_data[idx].shape[0]
    if tam_drop > 0:
        print('...There are %s GPS points duplicated' % tam_drop)
        return move_data.drop_duplicates(subset, keep, inplace)
    else:
        print('...There are no GPS points duplicated')


def clean_consecutive_duplicates(
    move_data, subset=None, keep='first', inplace=False
):
    """
    Removes consecutive duplicate rows of the Dataframe, optionally only
    certain columns can be consider.

    Parameters
    ----------
    move_data : dataframe
        The input trajectory data
    subset : Array of Strings, optional, default None(None by default)
        Specifies  Column label or sequence of labels, considered for
        identifying duplicates. By default all columns are used.
    keep : 'first', 'last', optional, default 'first'
        If keep is set as first, all the duplicates except for
        the first occurrence will be dropped.
        On the other hand if set to last, all duplicates except for
        the last occurrence will be dropped.
        If set to False, all duplicates are dropped.
    inplace : boolean, optional, default False
        if set to true the original dataframe will be altered,
        the duplicates will be dropped in place,
        otherwise a copy will be returned.

    Returns
    -------
    dataframe or None
        The filtered trajectories points without consecutive duplicates.

    """

    if keep == 'first':
        n = 1
    else:
        n = -1
    if subset is None:
        filter_ = (move_data.shift(n) != move_data).any(axis=1)
    else:
        filter_ = (move_data[subset].shift(n) != move_data[subset]).any(axis=1)

    return move_data.drop(index=move_data[~filter_].index, inplace=inplace)


def clean_nan_values(
    move_data, axis=0, how='any', thresh=None, subset=None, inplace=False
):
    """
    Removes missing values from the dataframe.

    Parameters
    ----------
    move_data : dataframe
        The input trajectory data
    axis : Integer or String (default 0)
        Determines if rows or columns that contain missing values are removed.
        If set to 0 or "index", the function drops the rows
        containing the missing value.
        If set to 1 or "columns", drops the columns
        containing the missing value.
    how : String, optional (default "any")
        Determines if a row or column is dropped for having at least
        one NA value or all value NA.
        If set to "any", the rows or columns will be dropped,
        if it has any NA values.
        If set to "all", the rows or columns will be dropped,
        if all of it"srs values are NA.
    thresh : Integer, optional (None by default)
        Minimum non-NA required value to avoid dropping
    subset : array of String
        Indicates the labels along the other axis to consider.
        E.g. if you want to drop columns,
        subset would indicate a list of rows to be included.
    inplace : boolean, default(False by default)
        if set to true the operation is done in place, the original dataframe
        will be altered and None is returned.

    Returns
    -------
    dataframe or None
        The filtered trajectories without nan values.

    """

    return move_data.dropna(
        axis=axis, how=how, thresh=thresh, subset=subset, inplace=inplace
    )


def _filter_single_by_max(move_data, **kwargs):
    """
    Filters from a dataframe rows with features below value.

    Parameters
    ----------
    move_data : dataframe
        Dataframe to be filtered.
    **kwargs : arguments
        - arg1 : feature
        - arg2 : value

    Returns
    -------
    dataframe
        Filtered dataframe.

    """

    return move_data[move_data[kwargs['arg1']] <= kwargs['arg2']]


def _filter_speed_max_radius(move_data, **kwargs):
    """
    Filters from a dataframe rows with current or previous row features
    exceeding value.

    Parameters
    ----------
    move_data : dataframe
        Dataframe to be filtered.
    **kwargs : arguments
        - arg1 : feature
        - arg2 : value

    Returns
    -------
    dataframe
        Filtered dataframe.

    """
    filter_ = (
        (np.nan_to_num(move_data[kwargs['arg1']].shift(1)) > kwargs['arg2'])
        | (np.nan_to_num(move_data[kwargs['arg1']]) > kwargs['arg2'])
    )
    return move_data[filter_]


def _filter_data(move_data, f, kwargs):
    """
    Filter the dataframe using condition from given function

    Parameters
    ----------
    move_data : dataframe
        Dataframe to be filtered.
    f : function
        Filtering function
    **kwargs : arguments
        - arg1 : feature
        - arg2 : value
        - outliers : special behaviour if cleaning by outliers

    Returns
    -------
    dataframe
        Filtered dataframe.
    int
        Number of rows to be dropped

    """

    if kwargs['outliers']:
        filter_data_points = f(
            move_data,
            jump_coefficient=kwargs['arg1'],
            threshold=kwargs['arg2'],
            inplace=False
        )
    else:
        filter_data_points = f(
            move_data,
            arg1=kwargs['arg1'],
            arg2=kwargs['arg2'],
            inplace=False
        )
    rows_to_drop = filter_data_points.shape[0]
    return filter_data_points, rows_to_drop


def _clean_gps(move_data, f, **kwargs):
    """
    Cleans gps points from a dataframe using condition from given function

    Parameters
    ----------
    move_data : dataframe
        Dataframe to be filtered.
    f : function
        Filtering function
    **kwargs : arguments
        - arg1 : feature
        - arg2 : value
        - outliers : special behaviour if cleaning by outliers

    Returns
    -------
    dataframe
        Filtered dataframe.

    """

    if move_data.index.name is not None:
        print('...Reset index for filtering\n')
        move_data.reset_index(inplace=True)

    filter_data_points, rows_to_drop = _filter_data(move_data, f, kwargs)

    sum_drop = 0
    while rows_to_drop > 0:
        print('...Dropping %s rows of gps points\n' % rows_to_drop)
        shape_before = move_data.shape[0]
        move_data.drop(index=filter_data_points.index, inplace=True)
        sum_drop = sum_drop + rows_to_drop
        print(
            '...Rows before: %s, Rows after:%s, Sum drop:%s\n'
            % (shape_before, move_data.shape[0], sum_drop)
        )

        filter_data_points, rows_to_drop = _filter_data(move_data, f, kwargs)

    print('%s GPS points were dropped' % sum_drop)

    return move_data


def clean_gps_jumps_by_distance(
    move_data,
    label_id=TRAJ_ID,
    jump_coefficient=3.0,
    threshold=1,
    label_dtype=np.float64,
    inplace=False,
):
    """
    Removes the trajectories points that are outliers from the dataframe.

    Parameters
    ----------
    move_data : dataframe
        The input trajectory data
    label_id : String, optional(the class dic_labels["id"] by default)
         Indicates the label of the id column in the user"srs dataframe.
    jump_coefficient : Float, optional(3.0 by default)
    threshold : Float, optional(1 by default)
        Minimum value that the distance features must have
        in order to be considered outliers
    label_dtype : type_, optional(np.float64 by default)
        Represents column id type_. By default it"srs np.float64.
    inplace : boolean, default(False by default)
        if set to true the operation is done in place, the original
        dataframe will be altered and None is returned.

    Returns
    -------
    dataframe or None
        The filtered trajectories without the gps jumps.

    """

    if not inplace:
        move_df = move_data[:]
    else:
        move_df = move_data
    # sum_drop = 0

    if DIST_TO_PREV not in move_df:
        move_df.generate_dist_features(
            label_id=label_id, label_dtype=label_dtype
        )

    try:
        print(
            '\nCleaning gps jumps by distance to jump_coefficient %s...\n'
            % jump_coefficient
        )

        move_df = _clean_gps(
            move_df,
            outliers,
            arg1=jump_coefficient,
            arg2=threshold,
            outliers=True
        )
        if not inplace:
            return move_df

    except Exception as e:
        raise e


def clean_gps_nearby_points_by_distances(
    move_data,
    label_id=TRAJ_ID,
    radius_area=10.0,
    label_dtype=np.float64,
    inplace=False,
):
    """
    Removes points from the trajectories when the distance between them and the
    point before is smaller than the value set by the user.

    Parameters
    ----------
    move_data : dataframe
        The input trajectory data
    label_id : String, optional(the class dic_labels["id"] by default)
         Indicates the label of the id column in the user"srs dataframe.
    radius_area : Float, optional(10.0 by default)
        Species the minimum distance a point must have to it"srs previous point,
        in order not to be dropped.
    label_dtype : type_, optional(np.float64 by default)
        Represents column id type_. By default it"srs np.float64.
    inplace : boolean, default(False by default)
        if set to true the operation is done in place, the original
        dataframe will be altered and None is returned.

    Returns
    -------
    dataframe or None
        The filtered trajectories without the gps nearby points by distance.

    """
    if not inplace:
        move_df = move_data[:]
    else:
        move_df = move_data
    # sum_drop = 0

    if DIST_TO_PREV not in move_df:
        move_df.generate_dist_features(
            label_id=label_id, label_dtype=label_dtype
        )

    try:
        print(
            '\nCleaning gps points from radius of %s meters\n'
            % radius_area
        )

        move_df = _clean_gps(
            move_df,
            _filter_single_by_max,
            arg1=DIST_TO_PREV,
            arg2=radius_area,
            outliers=False
        )
        if not inplace:
            return move_df

    except Exception as e:
        raise e


def clean_gps_nearby_points_by_speed(
    move_data,
    label_id=TRAJ_ID,
    speed_radius=0.0,
    label_dtype=np.float64,
    inplace=False,
):
    """
    Removes points from the trajectories when the speed of travel between them
    and the point before is smaller than the value set by the user.

    Parameters
    ----------
    move_data : dataframe
        The input trajectory data
    label_id : String, optional(the class dic_labels["id"] by default)
         Indicates the label of the id column in the user"srs dataframe.
    speed_radius : Float, optional(0.0 by default)
        Species the minimum speed a point must have from it"srs previous point,
        in order not to be dropped.
    label_dtype : type_, optional(np.float64 by default)
        Represents column id type_. By default it"srs np.float64.
    inplace : boolean, default(False by default)
        if set to true the operation is done in place, the original
        dataframe will be altered and None is returned.

    Returns
    -------
    dataframe or None
        The filtered trajectories without the gps nearby points by speed.

    """

    if not inplace:
        move_df = move_data[:]
    else:
        move_df = move_data
    # sum_drop = 0

    if SPEED_TO_PREV not in move_df:
        move_df.generate_dist_time_speed_features(
            label_id=label_id, label_dtype=label_dtype
        )

    try:

        print(
            '\nCleaning gps points using %s speed radius\n'
            % speed_radius
        )

        move_df = _clean_gps(
            move_df,
            _filter_single_by_max,
            arg1=SPEED_TO_PREV,
            arg2=speed_radius,
            outliers=False
        )
        if not inplace:
            return move_df

    except Exception as e:
        raise e


def clean_gps_speed_max_radius(
    move_data,
    label_id=TRAJ_ID,
    speed_max=50.0,
    label_dtype=np.float64,
    inplace=False,
):
    """
    Recursively removes trajectories points with speed higher than the value
    specified by the user. Given any point p of the trajectory, the point will
    be removed if one of the following happens: if the travel speed from the
    point before p to p is greater than the  max value of speed between adjacent
    points set by the user. Or the travel speed between point p and the next
    point is greater than the value set by the user. When the cleaning is done,
    the function will update the time and distance features in the dataframe and
    will call itself again. The function will finish processing when it can no
    longer find points disrespecting the limit of speed.

    Parameters
    ----------
    move_data : dataframe
        The input trajectory data
    label_id : String, optional(the class dic_labels["id"] by default)
        Indicates the label of the id column in the user"srs dataframe.
    speed_max : Float. Optional(50.0 by default)
        Indicates the maximum value a point"srs speed_to_prev and speed_to_next
        should have, in order not to be dropped.
    label_dtype : type_, optional(np.float64 by default)
        Represents column id type_. By default it"srs np.float64.
    inplace : boolean, default(False by default)
        if set to true the operation is done in place, the original
        dataframe will be altered and None is returned.

    Returns
    -------
    dataframe or None
        The filtered trajectories without the gps nearby points.

    """

    if not inplace:
        move_df = move_data[:]
    else:
        move_df = move_data
    # sum_drop = 0

    if SPEED_TO_PREV not in move_df:
        move_df.generate_dist_time_speed_features(
            label_id=label_id, label_dtype=label_dtype
        )
    print(move_df)

    try:
        print(
            '\nClean gps points with speed max > %s meters by seconds'
            % speed_max
        )

        move_df = _clean_gps(
            move_df,
            _filter_speed_max_radius,
            arg1=SPEED_TO_PREV,
            arg2=speed_max,
            outliers=False
        )
        if not inplace:
            return move_df

    except Exception as e:
        raise e


def clean_trajectories_with_few_points(
    move_data, label_tid=TID, min_points_per_trajectory=2, inplace=False
):
    """
    Removes from the given dataframe, trajectories with fewer points than was
    specified by the user.

    Parameters
    ----------
    move_data : dataframe
        The input trajectory data
    label_tid : String, optional(dic_features_label["tid"] by default)
        The label of the column which contains the tid of the trajectories
    min_points_per_trajectory: Integer, optional(2 by default)
        Specifies the minimum number of points a trajectory must have
        in order not to be dropped
    inplace : boolean, default(False by default)
        if set to true the operation is done in place, the original
        dataframe will be altered and None is returned.

    Returns
    -------
    dataframe or None
        The filtered trajectories without the minimum number of gps points.

    Raises
    ------
    KeyError
        If the label feature is not in the dataframe

    """

    if not inplace:
        move_df = move_data[:]
    else:
        move_df = move_data

    if label_tid not in move_df:
        raise KeyError('%s not in dataframe' % label_tid)

    try:
        print(
            '\nCleaning gps points from trajectories of fewer than %s points\n'
            % min_points_per_trajectory
        )

        if move_df.index.name is not None:
            print('\n...Reset index for filtering\n')
            move_df.reset_index(inplace=True)

        move_datacount_tid = move_df.groupby(by=label_tid).size()
        filter_ = move_datacount_tid < min_points_per_trajectory
        tids_with_few_points = move_datacount_tid[filter_].index
        shape_before_drop = move_df.shape
        idx = move_df[move_df[label_tid].isin(tids_with_few_points)].index

        if idx.shape[0] > 0:
            print(
                '\n...There are %s ids with few points'
                % tids_with_few_points.shape[0]
            )
            print(
                '\n...Tids before drop: %s'
                % move_df[label_tid].unique().shape[0]
            )
            move_df.drop(index=idx, inplace=True)
            print(
                '\n...Tids after drop: %s'
                % move_df[label_tid].unique().shape[0]
            )
            print(
                '\n...Shape - before drop: %s - after drop: %s'
                % (shape_before_drop, move_df.shape)
            )

        if not inplace:
            return move_df

    except Exception as e:
        raise e


def clean_trajectories_short_and_few_points(
    move_data,
    label_id=TID,
    min_trajectory_distance=100,
    min_points_per_trajectory=2,
    label_dtype=np.float64,
    inplace=False,
):
    """
    Eliminates from the given dataframe trajectories with fewer points and
    shorter length than specified values by the user.

    Parameters
    ----------
    move_data : dataframe
        The input trajectory data
    label_id : String, optional( tid by default)
        The label of the column which contains the tid of the trajectories
    min_trajectory_distance: Integer, optional(100 by default)
        Specifies the minimun lenght a trajectory must have
        in order not to be dropped
    min_points_per_trajectory: Integer, optional(2 by default)
        Specifies the minimun number of points a trajectory must have
        in order not to be dropped
    label_dtype : type_, optional(np.float64 by default)
        Represents column id type_. By default it"srs np.float64.
    inplace : boolean, default(False by default)
        if set to true the operation is done in place, the original
        dataframe will be altered and None is returned.

    Returns
    -------
    dataframe or None
        The filtered trajectories with the minimum gps points and distance.

    Notes
    -----
        remove_tids_with_few_points must be performed before updating features.

    """

    if not inplace:
        move_df = move_data[:]
    else:
        move_df = move_data

    print('\nRemove short trajectories...')
    clean_trajectories_with_few_points(
        move_df, label_id, min_points_per_trajectory, inplace=True
    )

    if DIST_TO_PREV not in move_df:
        move_df.generate_dist_features(
            label_id=label_id, label_dtype=label_dtype
        )

    try:
        print('\n...Dropping unnecessary trajectories...')

        if move_df.index.name is not None:
            print('reseting index')
            move_df.reset_index(inplace=True)

        move_dataagg_tid = move_df.groupby(by=label_id).agg(
            {DIST_TO_PREV: 'sum'}
        )
        filter_ = move_dataagg_tid[DIST_TO_PREV] < min_trajectory_distance
        tid_selection = move_dataagg_tid[filter_].index

        print(
            '\n...short trajectories and trajectories with a minimum distance (%s): %s'
            % (move_dataagg_tid.shape[0], min_trajectory_distance)
        )
        print('\n...There are %s tid do drop' % tid_selection.shape[0])
        shape_before_drop = move_df.shape

        idx = move_df[move_df[label_id].isin(tid_selection)].index
        if idx.shape[0] > 0:
            tids_before_drop = move_df[label_id].unique().shape[0]
            print(
                '\n...Tids - before drop: %s - after drop: %s'
                % (tids_before_drop, move_df[label_id].unique().shape[0])
            )
            move_df.drop(index=idx, inplace=True)
            print(
                '\n...Shape - before drop: %s - after drop: %s'
                % (shape_before_drop, move_df.shape)
            )

        if not inplace:
            return move_df

    except Exception as e:
        raise e


def clean_id_by_time_max(
    move_data,
    label_id=TRAJ_ID,
    label_dtype=np.float64,
    time_max=3600,
    inplace=False,
):
    """
    Clears GPS points with time by ID greater than a user-defined limit.

    Parameters
    ----------
    move_data: dataframe.
        The input data.
    label_id: string, optional( id by default).
        The label of the column which contains the id of the trajectories.
    label_dtype : type_, optional(np.float64 by default)
        Represents column id type_. By default it"srs np.float64.
    time_max: float. optional(3600 by default).
        Indicates the maximum value time a set of points with the
        same id should have in order not to be dropped.
    inplace : boolean, default(False by default)
        if set to true the operation is done in place, the original
        dataframe will be altered and None is returned.

    Returns
    -------
    dataframe or None
        The filtered trajectories with the maximum time.

    """

    if not inplace:
        move_df = move_data[:]
    else:
        move_df = move_data

    if TIME_TO_PREV not in move_df:
        move_df.generate_dist_time_speed_features(
            label_id=label_id, label_dtype=label_dtype
        )

    try:
        print(
            '\nClean gps points with time max by id < %s seconds'
            % time_max
        )
        move_dataid_drop = (
            move_df.groupby([label_id], as_index=False)
            .agg({TIME_TO_PREV: 'sum'})
            .query('%s < %s' % (TIME_TO_PREV, time_max))
        )
        print(
            '...Ids total: %s\nIds to drop:%s'
            % (
                move_df[label_id].nunique(),
                move_dataid_drop[label_id].nunique()
            )
        )
        if move_dataid_drop.shape[0] > 0:
            before_drop = move_df.shape[0]
            filter_ = move_df[label_id].isin(move_dataid_drop[label_id])
            idx = move_df[filter_].index
            move_df.drop(idx, inplace=True)
            print(
                '...Rows before drop: %s\n Rows after drop: %s'
                % (before_drop, move_df.shape[0])
            )

        if not inplace:
            return move_df

    except Exception as e:
        raise e
