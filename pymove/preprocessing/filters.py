import numpy as np
from pymove.utils import constants
from pymove.utils.constants import (
	LATITUDE,
	LONGITUDE,
	DATETIME,
	TRAJ_ID,
	TID,
	UID,
	TIME_TO_PREV,
	SPEED_TO_PREV,
	DIST_TO_PREV,
	DIST_PREV_TO_NEXT,
	DIST_TO_NEXT,
	DAY,
	PERIOD,
	TYPE_PANDAS,
	TB,
	GB,
	MB,
	KB,
	B)


def by_bbox(df_, bbox, filter_out=False, inplace=False):
    """Filters points of the trajectories according to especified bounding box.

    Parameters
    ----------
    df_ : dataframe
       The input trajectory data

    bbox : tuple
        Tuple of 4 elements, containg the minimum and maximum values of latitude and longitude of the bounding box.

    filter_out : boolean, optional(false by default)
        If set to false the function will return the trajectories points within the bounding box, and the points outside otherwise

    dic_labels : dict, optional(the classe's dic_labels by default)
        Dictionary mapping the user's dataframe labels to the pattern of the PyRoad's lib

    inplace : boolean, optional(false by default)
        if set to true the original dataframe will be altered to contain the result of the filtering, otherwise a copy will be returned.

    Returns
    -------
    df : dataframe
        Returns dataframe with trajectories points filtered by bounding box.

    Example
    -------
        filter_bbox(df_, [-3.90, -38.67, -3.68, -38.38]) -> Fortaleza
            lat_down =  bbox[0], lon_left =  bbox[1], lat_up = bbox[2], lon_right = bbox[3]
    """
    try:
        filter_ = (df_[constants.LATITUDE] >= bbox[0]) & (df_[constants.LATITUDE] <= bbox[2]) & (
                    df_[constants.LONGITUDE] >= bbox[1]) & (df_[constants.LONGITUDE] <= bbox[3])
        if filter_out:
            filter_ = ~filter_

        if inplace:
            df_.drop(index=df_[~filter_].index, inplace=True)
            return df_
        else:
            return df_.loc[filter_]
    except Exception as e:
        raise e


def by_datetime(df_, startDatetime=None, endDatetime=None, filter_out=False):
    """Filters trajectories points according to especified time range

    Parameters
    ----------
    df_ : dataframe
        The input trajectory data

    startDatetime : String
        The start date and time (Datetime format) of the time range used in the filtering

    endDatetime : String
        The end date and time (Datetime format) of the time range used in the filtering

    dic_labels : dict, optional(the classe's dic_labels by default)
        Dictionary mapping the user dataframe labels to the pattern of the PyRoad's lib

    filter_out : boolean, optional(false by default) (CONFIRMAR )
        If set to true, the function will return the points of the trajectories with timestamp outside the time range.
        The points whitin the time range will be return if filter_out is set to false.

    Returns
    -------
    df : dataframe
        Returns dataframe with trajectories points filtered by especified time range.
    """

    try:
        if startDatetime is not None and endDatetime is not None:
            filter_ = (df_[constants.DATETIME] > startDatetime) & (df_[constants.DATETIME] <= endDatetime)
        elif endDatetime is not None:
            filter_ = (df_[constants.DATETIME] <= endDatetime)
        else:
            filter_ = (df_[constants.DATETIME] > startDatetime)

        if filter_out:
            filter_ = ~filter_

        return df_[filter_]

    except Exception as e:
        raise e


def by_label(df_, value, label_name, filter_out=False):
    """Filters trajectories points according to especified value and collum label

    Parameters
    ----------
    df_ : dataframe
        The input trajectory data

    value : The type of the feature values to be use to filter the trajectories
        Specifies the value used to filter the trajectories points

    label_name : String
        Specifes the label of the colum used in the filtering

    filter_out : boolean, optional(false by default)
        If set to True, it will return trajectory points with feature value different from the value specified in the parameters
        The trajectories points with the same feature value as the one especifed in the parameters.

    Returns
    -------
    df : dataframe
        Returns dataframe with trajectories points filtered by label.
    """

    try:
        filter_ = (df_[label_name] == value)

        if filter_out:
            filter_ = ~filter_

        return df_[filter_]

    except Exception as e:
        raise e


def by_id(df_, id_=None, label_id=constants.TRAJ_ID, filter_out=False):
    """Filters trajectories points according to especified trajectory id

    Parameters
    ----------
    df_ : dataframe
        The input trajectory data

    id_ : Integer
        Specifies the number of the id used to filter the trajectories points

    label_id : String, optional(dic_labels['id'] by default)
        The label of the colum which contains the id of the trajectories

    filter_out : boolean, optional(false by default)
        If set to true, the function will return the points of the trajectories with the same id as the one especified by the parameter value.
        If set to false it will return the points of the trajectories with a different id from the one especified in the parameters.

    Returns
    -------
    df : dataframe
        Returns dataframe with trajectories points filtered by id.
    """
    return by_label(df_, id_, label_id, filter_out)


def by_tid(df_, tid_=None, label_tid=constants.TID, filter_out=False):
    """Filters trajectories points according to especified trajectory tid

    Parameters
    ----------
    df_ : dataframe
        The input trajectory data

    tid_ : String
        Specifies the number of the tid used to filter the trajectories points

    label_tid : String, optional(dic_features_label['tid'] by default)
        The label of the colum in the user's dataframe which contains the tid of the trajectories

    filter_out : boolean, optional(false by default)
        If set to true, the function will return the points of the trajectories with the same tid as the one especified by the parameter value.
        If set to false it will return the points of the trajectories with a different tid from the one especified in the parameters.

    Returns
    -------
    df : dataframe
        Returns a dataframe with trajectories points filtered.
    """
    if TID not in df_:
        df_.generate_tid_based_on_id_datatime()
    return by_label(df_, tid_, label_tid, filter_out)


# nome antigo era jumps, mas outliers fica mais claro
def outliers(df_, jump_coefficient=3.0, threshold=1, filter_out=False):
    """Filters trajectories points that are outliers.

        Parameters
        ----------
        df_ : dataframe
            The input trajectory data

        jump_coefficient : Float, optional(3.0 by default)
            #Nao sei exatamente como explicar

        threshold : Float, optional(1 by default)
            Minimum value that the distance features("dist_to_next", 'dist_to_prev','dist_prev_to_next') must have
            in order to be considered outliers

        filter_out : boolean, optional(false by default)
            If set to true, the function will return the points of the trajectories that are not outiliers.
            If set to false it will return the points of the trajectories are outiliers.

        Returns
        -------
        df : dataframe
            Returns a dataframe with the trajectories outiliers.
        """
    if DIST_TO_PREV not in df_:
        df_.generate_dist_features()

    if df_.index.name is not None:
        print('...Reset index for filtering\n')
        df_.reset_index(inplace=True)

    if DIST_TO_PREV in df_ and DIST_TO_NEXT and DIST_PREV_TO_NEXT in df_:
        filter_ = (df_[DIST_TO_NEXT] > threshold) & (
                    df_[DIST_TO_PREV] > threshold) & (
                              df_[DIST_PREV_TO_NEXT] > threshold) & \
                  (jump_coefficient * df_[DIST_PREV_TO_NEXT] < df_[
                      constants.DIST_TO_NEXT]) & \
                  (jump_coefficient * df_[DIST_PREV_TO_NEXT] < df_[
                      constants.DIST_TO_PREV])

        if filter_out:
            filter_ = ~filter_

        print('...Filtring jumps \n')
        return df_[filter_]

    else:
        print('...Distances features were not created')
        return df_


"""----------------------  FUCTIONS TO DATA CLEANING   ----------------------------------- """


# DUVIDA SOBRE RETORNO
def clean_duplicates(df_, subset=None, keep='first', inplace=False, sort=True, return_idx=False):
    """Removes the duplicate rows of the Dataframe, optionally only certaind columns can be consider.

    Parameters
    ----------
    df_ : dataframe
        The input trajectory data

    subset : String or Array of Strings, optional(None by default)
        Specify  Column label or sequence of labels, considered for identifying duplicates. By default all columns are used.

    keep : String if the option are 'first' or 'last and boolean if False. Optional(first by default)
        if keep is set as first, all the duplicates except for the first occurrence will be droped. On the other hand if seted to last, all duplicates except for the last occurrence will be droped. If set to False, all duplicates are droped.

    inplace : boolean, optional(False by default)
        if set to true the original dataframe will be altered, the duplicates will be droped in place, otherwise a copy will be returned.

    sort : boolean, optional(True by default)
        If set to True the data will be sorted by id and datetime, to increase performace. If set to False the data won't be sorted.

    return_idx : boolean

    Returns
    -------

    """

    print('\nRemove rows duplicates by subset')
    if sort is True:
        print('...Sorting by {} and {} to increase performance\n'.format(TRAJ_ID, DATETIME))
        df_.sort_values([TRAJ_ID, DATETIME], inplace=True)

    idx = df_.duplicated(subset=subset)
    tam_drop = df_[idx].shape[0]

    if tam_drop > 0:
        df_.drop_duplicates(subset, keep, inplace)
        print('...There are {} GPS points duplicated'.format(tam_drop))
    else:
        print('...There are no GPS points duplicated')

    if return_idx:
        return return_idx


def clean_consecutive_duplicates(df, subset=None, keep='first', inplace=False):
    """Removes consecutives duplicate rows of the Dataframe, optionally only certaind columns can be consider.

    Parameters
    ----------
    df_ : dataframe
        The input trajectory data

    subset : Array of Strings, optional(None by default)
        Specifies  Column label or sequence of labels, considered for identifying duplicates. By default all columns are used.

    keep : String. Optional(first by default)
        Determine wich duplicate will be removed.
        if keep is set as first, all the duplicates except for the first occurrence will be droped. Otherwise, all duplicates except for the last occurrence will be droped.

    inplace : boolean, optional(False by default)
        if set to true the original dataframe will be altered, the duplicates will be droped in place, otherwise a copy will be returned.

    Returns
    -------
    df : dataframe
        The filtered trajectories points without duplicates.
    """

    if keep == 'first':
        n = 1
    else:
        n = -1

    if subset is None:
        filter_ = (df.shift(n) != df).any(axis=1)
    else:
        filter_ = (df[subset].shift(n) != df[subset]).any(axis=1)

    if inplace:
        df.drop(index=df[~filter_].index, inplace=True)
        return df
    else:
        return df.loc[filter_]


def clean_NaN_values(df_, axis=0, how='any', thresh=None, subset=None, inplace=True):
    # df.isna().sum()
    """Removes missing values from the dataframe.

    Parameters
    ----------
    axis : Integer or String (default 0)
        Determines if rows or columns that contain missing values are removed. If set to 0 or 'index', the function drops the rows containing the missing value.
        If set to 1 or 'columns', drops the columns containing the missing value.

    how : String, optional (default 'any')
        Especifies if a row or column is droped for having at least one NA value or all value NA.
        If set to 'any', the rows or columns will be droped, if it has any NA values.
        If set to 'all', the rows or columns will be droped, if all of it's values are NA.

    thresh : Integer, optional (None by default)
        Minimum non-NA required value to avoid dropping

    subset : array of String
        Indicates the labels along the other axis to consider. E.g. if you want to drop columns, subset would indicate a list of rows to be included.

    inplace : boolean, default(True by default)
        if set to true the operation is done in place, the original dataframe will be altered and None is returned.

    """
    df_.dropna(axis=axis, how=how, thresh=thresh, subset=None, inplace=inplace)


# Duvida sobre label_dtype
def clean_gps_jumps_by_distance(df_, label_id=constants.TRAJ_ID, jump_coefficient=3.0, threshold=1, label_dtype=np.float64, sum_drop=0):
    """Removes the trajectories points that are outliers from the dataframe.

    Parameters
    ----------
    df_ : dataframe
        The input trajectory data

    label_id : String, optional(the classe's dic_labels['id'] by default)
         Indicates the label of the id column in the user's dataframe.

    jump_coefficient : Float, optional(3.0 by default)
        #Nao sei exatamente como explicar

    threshold : Float, optional(1 by default)
        Minimum value that the distance features("dist_to_next", 'dist_to_prev','dist_prev_to_next') must have
        in order to be considered outliers

    dic_labels : dict, optional(the classe's dic_labels by default)
        Dictionary mapping the user's dataframe labels to the pattern of the PyRoad's lib

    label_dtype :

    sum_drop:Integer, optional(0 by default)
        Specifies the number of colums that have been droped.

    """
    if DIST_TO_PREV not in df_:
        df_.generate_dist_features(label_id = label_id,  label_dtype=label_dtype)
    try:
        print('\nCleaning gps jumps by distance to jump_coefficient {}...\n'.format(jump_coefficient))
        df_jumps = outliers(df_, jump_coefficient, threshold)
        rows_to_drop = df_jumps.shape[0]

        if rows_to_drop > 0:
            print('...Dropping {} rows of gps points\n'.format(rows_to_drop))
            shape_before = df_.shape[0]
            df_.drop(index=df_jumps.index, inplace=True)
            sum_drop = sum_drop + rows_to_drop
            print('...Rows before: {}, Rows after:{}, Sum drop:{}\n'.format(shape_before, df_.shape[0], sum_drop))
            clean_gps_jumps_by_distance(df_, label_id, jump_coefficient, threshold, label_dtype, sum_drop)
        else:
            print('{} GPS points were dropped'.format(sum_drop))

    except Exception as e:
        raise e


# Lsbel_dtype
def clean_gps_nearby_points_by_distances(df_, label_id=constants.TRAJ_ID, radius_area=10.0,
                                         label_dtype=np.float64):
    """Removes points from the trajectories when the distance between them and the point before is smaller than the
    value set by the user.

    Parameters
    ----------
    df_ : dataframe
        The input trajectory data

    label_id : String, optional(the classe's dic_labels['id'] by default)
         Indicates the label of the id column in the user's dataframe.

    dic_labels : dict, optional(the classe's dic_labels by default)
        Dictionary mapping the user's dataframe labels to the pattern of the PyRoad's lib

    radius_area : Float, optional(10.0 by default)
        Species the minimum distance a point must have to it's previous point, in order not to be droped.

    label_dtype :

    """
    if DIST_TO_PREV not in df_:
        df_.generate_dist_features(label_id = label_id,  label_dtype=label_dtype)

    try:
        print('\nCleaning gps points from radius of {} meters\n'.format(radius_area))
        if df_.index.name is not None:
            print('...Reset index for filtering\n')
            df_.reset_index(inplace=True)

        if constants.DIST_TO_PREV in df_:
            filter_nearby_points = (df_[constants.DIST_TO_PREV] <= radius_area)

            idx = df_[filter_nearby_points].index
            print('...There are {} gps points to drop\n'.format(idx.shape[0]))
            if idx.shape[0] > 0:
                print('...Dropping {} gps points\n'.format(idx.shape[0]))
                shape_before = df_.shape[0]
                df_.drop(index=idx, inplace=True)
                print('...Rows before: {}, Rows after:{}\n'.format(shape_before, df_.shape[0]))
                clean_gps_nearby_points_by_distances(df_, label_id, radius_area, label_dtype)
        else:
            print('...{} is not in the dataframe'.format(constants.DIST_TO_PREV))
    except Exception as e:
        raise e


# Lsbel_dtype
def clean_gps_nearby_points_by_speed(df_, label_id=constants.TRAJ_ID, speed_radius=0.0, label_dtype=np.float64):
    """Removes points from the trajectories when the speed of travel between them and the point before is smaller than the
           value set by the user.

    Parameters
    ----------
    df_ : dataframe
        The input trajectory data

    label_id : String, optional(the classe's dic_labels['id'] by default)
         Indicates the label of the id column in the user's dataframe.

    dic_labels : dict, optional(the classe's dic_labels by default)
        Dictionary mapping the user's dataframe labels to the pattern of the PyRoad's lib

    speed_radius : Float, optional(0.0 by default)
        Species the minimum speed a point must have from it's previous point, in order not to be droped.

    label_dtype :

    """
    if SPEED_TO_PREV not in df_:
        df_.generate_dist_time_speed_features(label_id = label_id, label_dtype = label_dtype)
    try:
        print('\nCleaning gps points using {} speed radius\n'.format(speed_radius))
        if df_.index.name is not None:
            print('...Reset index for filtering\n')
            df_.reset_index(inplace=True)

        if constants.SPEED_TO_PREV in df_:
            filter_nearby_points = (df_[constants.SPEED_TO_PREV] <= speed_radius)

            idx = df_[filter_nearby_points].index
            print('...There are {} gps points to drop\n'.format(idx.shape[0]))
            if idx.shape[0] > 0:
                print('...Dropping {} gps points\n'.format(idx.shape[0]))
                shape_before = df_.shape[0]
                df_.drop(index=idx, inplace=True)
                print('...Rows before: {}, Rows after:{}\n'.format(shape_before, df_.shape[0]))
                clean_gps_nearby_points_by_speed(df_ = df_, label_id = label_id, label_dtype = label_dtype)
        else:
            print('...{} is not in the dataframe'.format(constants.DIST_TO_PREV))
    except Exception as e:
        raise e


# LABEL_DTYPE
def clean_gps_speed_max_radius(df_, label_id=constants.TRAJ_ID, speed_max=50.0, label_dtype=np.float64):

    """Recursively removes trajectories points with speed higher than the value especifeid by the user.
    Given any point p of the trajectory, the point will be removed if one of the following happens:
    if the travel speed from the point before p to p is greater than the  max value of speed between adjacent
    points set by the user. Or the travel speed between point p and the next point is greater than the value set by
    the user. When the clening is done, the function will update the time and distance features in the dataframe and
    will call itself again.
    The function will finish processing when it can no longer find points disrespecting the limit of speed.

    Parameters
    ----------
    df_ : dataframe
        The input trajectory data

    label_id : String, optional(the classe's dic_labels['id'] by default)
        Indicates the label of the id column in the user's dataframe.

    dic_labels : dict, optional(the classe's dic_labels by default)
        Dictionary mapping the user's dataframe labels to the pattern of the PyRoad's lib

    speed_max : Float. Optional(50.0 by default)
        Indicates the maximun value a point's speed_to_prev and speed_to_next should have, in order not to be dropped.

    label_dtype :

    """
    if SPEED_TO_PREV not in df_:
        df_.generate_dist_time_speed_features(label_id = label_id, label_dtype = label_dtype)

    print('\nClean gps points with speed max > {} meters by seconds'.format(speed_max))

    if constants.SPEED_TO_PREV in df_:
        filter_ = (df_[constants.SPEED_TO_PREV] > speed_max) | (
                    df_[constants.SPEED_TO_PREV] > speed_max)

        idx = df_[filter_].index

        print('...There {} gps points with speed_max > {}\n'.format(idx.shape[0], speed_max))
        if idx.shape[0] > 0:
            print('...Dropping {} rows of jumps by speed max\n'.format(idx.shape[0]))
            shape_before = df_.shape[0]
            df_.drop(index=idx, inplace=True)
            print('...Rows before: {}, Rows after:{}\n'.format(shape_before, df_.shape[0]))
            clean_gps_speed_max_radius(df_, label_id, speed_max, label_dtype)


# DESCUBRIR O QUE EH LABEL_DTYPE
def clean_trajectories_with_few_points(df_, label_tid=constants.TID,
                                       min_points_per_trajectory=2, label_dtype=np.float64):
    """Eliminates from the given dataframe, trajectories with fewer points than was especified by the user

    Parameters
    ----------
    df_ : dataframe
        The input trajectory data

    label_tid : String, optional(dic_features_label['tid'] by default)
        The label of the colum which contains the tid of the trajectories

    dic_labels : dict, optional(the classe's dic_labels by default)
        Dictionary mapping the user's dataframe labels to the pattern of the PyRoad's lib

    min_points_per_trajectory: Integer, optional(2 by default)
        Specifies the minimun number of points a trajectory must have in order not to be dropped

    label_dtype:

    """
    if TID not in df_:
        df_.generate_tid_based_on_id_datatime()

    if df_.index.name is not None:
        print('\n...Reset index for filtering\n')
        df_.reset_index(inplace=True)

    df_count_tid = df_.groupby(by=label_tid).size()
    tids_with_few_points = df_count_tid[df_count_tid < min_points_per_trajectory].index

    print('\n...There are {} ids with few points'.format(tids_with_few_points.shape[0]))
    shape_before_drop = df_.shape
    idx = df_[df_[label_tid].isin(tids_with_few_points)].index
    if idx.shape[0] > 0:
        print('\n...Tids before drop: {}'.format(df_[label_tid].unique().shape[0]))
        df_.drop(index=idx, inplace=True)
        print('\n...Tids after drop: {}'.format(df_[label_tid].unique().shape[0]))
        print('\n...Shape - before drop: {} - after drop: {}'.format(shape_before_drop, df_.shape))
        df_.generate_dist_time_speed_features(label_id = label_tid, label_dtype = label_dtype)



# confirmar se remove_tids_with_few_points eh o clean_trajectories_with_few_points
# updating features no notes, nao se refere a propria funcao?
# DESCUBRIR O QUE EH LABEL_DTYPE
def clean_trajectories_short_and_few_points_(df_, label_id=constants.TID, min_trajectory_distance=100,
                                             min_points_per_trajectory=2,
                                             label_dtype=np.float64):
    """Eliminates from the given dataframe trajectories with fewer points and shorter length than specified values
       by the user.

    Parameters
    ----------
    df_ : dataframe
        The input trajectory data

    label_id : String, optional(dic_features_label['tid'] by default)
        The label of the colum which contains the tid of the trajectories

    dic_labels : dict, optional(the classe's dic_labels by default)
        Dictionary mapping the user's dataframe labels to the pattern of the PyRoad's lib

    min_trajectory_distance: Integer, optional(100 by default)
        Specifies the minimun lenght a trajectory must have in order not to be dropped

    min_points_per_trajectory: Integer, optional(2 by default)
        Specifies the minimun number of points a trajectory must have in order not to be dropped

    label_dtype:

    Notes
    -----
        remove_tids_with_few_points must be performed before updating features, because
        those features can only be computed with at least 2 points per trajactories

    """
    print('\nRemove short trajectories...')
    clean_trajectories_with_few_points(df_, label_id, min_points_per_trajectory, label_dtype)

    if DIST_TO_PREV not in df_:
        df_.generate_dist_features(label_id = label_id,  label_dtype=label_dtype)

    if df_.index.name is not None:
        print('reseting index')
        df_.reset_index(inplace=True)

    print('\n...Dropping unnecessary trajectories...')
    df_agg_tid = df_.groupby(by=label_id).agg({constants.DIST_TO_PREV: 'sum'})

    filter_ = (df_agg_tid[constants.DIST_TO_PREV] < min_trajectory_distance)
    tid_selection = df_agg_tid[filter_].index
    print('\n...short trajectories and trajectories with a minimum distance ({}): {}'.format(df_agg_tid.shape[0],
                                                                                             min_trajectory_distance))
    print('\n...There are {} tid do drop'.format(tid_selection.shape[0]))
    shape_before_drop = df_.shape
    idx = df_[df_[label_id].isin(tid_selection)].index
    if idx.shape[0] > 0:
        tids_before_drop = df_[label_id].unique().shape[0]
        df_.drop(index=idx, inplace=True)
        print('\n...Tids - before drop: {} - after drop: {}'.format(tids_before_drop, df_[label_id].unique().shape[0]))
        print('\n...Shape - before drop: {} - after drop: {}'.format(shape_before_drop, df_.shape))
        clean_trajectories_short_and_few_points_(df_, min_trajectory_distance, min_points_per_trajectory,
                                                 label_dtype)
