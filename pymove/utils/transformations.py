from pymove.core.pandas import PandasMoveDataFrame


def feature_values_using_filter(
    move_data, id_, feature_name, filter_, values, inplace=True
):
    """
    Changes the values of the feature defined by the user.

    Parameters
    ----------
    move_data : dataframe
       The input trajectories data.
    id_ : String
        Indicates the index to be changed.
    feature_name : String
        The name of the column that the user wants to change values for.
    filter_ : Array
        Indicates the rows with the index "id_" of the "feature_name"
        that must be changed.
    values : ?
        THe new values to be set to the selected feature.
    inplace: boolean, optional(True by default)
        if set to true the original dataframe will be altered,
        otherwise the alteration will be made in a copy, that will be returned.

    Returns
    -------
    dataframe or None
        A copy of the original dataframe, with the alterations done
        by the function. (When inplace is False)

    Notes
    -----
    equivalent to: move_data.at[id_, feature_name][filter_] = values
    e.g. move_data.at[tid, "time"][filter_nodes] = intp_result.astype(np.int64)
    dataframe must be indexed by id_:
    move_data.set_index(index_name, inplace=True)

    """

    if not inplace:
        move_data = PandasMoveDataFrame(data=move_data.to_data_frame())

    values_feature = move_data.at[id_, feature_name]
    if filter_.shape == ():
        move_data.at[id_, feature_name] = values
    else:
        values_feature[filter_] = values
        move_data.at[id_, feature_name] = values_feature

    if not inplace:
        return move_data
    else:
        return None


def feature_values_using_filter_and_indexes(
    move_data, id_, feature_name, filter_, idxs, values, inplace=True
):
    """
    Create or update move and stop by radius.

    Parameters
    ----------
    move_data : dataframe
       The input trajectories data.
    id_ : String
        Indicates the index to be changed.
    feature_name : String
        The name of the column that the user wants to change values for.
    filter_ : Array
        Indicates the rows with the index "id_" of the "feature_name"
        that must be changed.
    idxs : array like of indexes
        Indexes to atribute value
    values : array like
        The new values to be set to the selected feature.
    inplace: boolean, optional(True by default)
        if set to true the original dataframe will be altered,
        otherwise the alteration will be made in a copy, that will be returned.
    move_data : pandas.core.frame.DataFrame
        Represents the dataset with contains lat, long and datetime.

    Returns
    -------
    dataframe or None
        A copy of the original dataframe, with the alterations
        done by the function. (When inplace is False)

    """

    if not inplace:
        move_data = PandasMoveDataFrame(data=move_data.to_data_frame())

    values_feature = move_data.at[id_, feature_name]
    values_feature_filter = values_feature[filter_]
    values_feature_filter[idxs] = values
    values_feature[filter_] = values_feature_filter
    move_data.at[id_, feature_name] = values_feature

    if not inplace:
        return move_data
    else:
        return None
