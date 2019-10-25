from pymove.core.dataframe import PandasMoveDataFrame


def feature_values_using_filter(move_data, id_, feature_name, filter_, values, inplace=True):
    """
    Parameters
    ----------
    move_data : pandas.core.frame.DataFrame
        Represents the dataset with contains lat, long and datetime.

    id_ : String
        ?

    feature_name : String
        ?.

    filter_ : ?
        ?.

    values : ?
        ?.

    Returns
    -------

    Notes
    -----
    equivalent to: move_data.at[id_, feature_name][filter_] = values
    e.g. move_data.at[tid, "time"][filter_nodes] = intp_result.astype(np.int64)
    dataframe must be indexed by id_: move_data.set_index(index_name, inplace=True)
    """
    if not inplace:
        move_data = PandasMoveDataFrame(data=move_data.to_DataFrame())

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


def feature_values_using_filter_and_indexes(move_data, id_, feature_name, filter_, idxs, values, inplace=True):
    """
    Create or update move and stop by radius.

    Parameters
    ----------
    move_data : pandas.core.frame.DataFrame
        Represents the dataset with contains lat, long and datetime.

    id_ : String
        ?

    feature_name : String
        ?.

    filter_ : ?
        ?.

    idxs: ?
        ?.

    values : ?
        ?.


    Returns
    -------
    """
    if not inplace:
        move_data = PandasMoveDataFrame(data=move_data.to_DataFrame())
        
    values_feature = move_data.at[id_, feature_name]
    values_feature_filter = values_feature[filter_]
    values_feature_filter[idxs] = values
    values_feature[filter_] = values_feature_filter
    move_data.at[id_, feature_name] = values_feature
    
    if not inplace:
        return move_data
    else:
        return None