#TODO complementar oq ela faz
#TODO trocar nome da func
from pymove.core.dataframe import PandasMoveDataFrame


def feature_values_using_filter(df, id_, feature_name, filter_, values, inplace = True):
    """
    ?
    equivalent of: df.at[id_, feature_name][filter_] = values
    e.g. df.at[tid, 'time'][filter_nodes] = intp_result.astype(np.int64)
    dataframe must be indexed by id_: df.set_index(index_name, inplace=True)

    Parameters
    ----------
    df_ : pandas.core.frame.DataFrame
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


    Examples
    --------
    -

    >>> from pymove.utils.transformations import change_df_feature_values_using_filter
    >>> change_df_feature_values_using_filter(df, -, -, -, -)

    """
    if inplace == False:
        df = PandasMoveDataFrame(data = df.to_DataFrame())

    values_feature = df.at[id_, feature_name]
    if filter_.shape == ():
        df.at[id_, feature_name] = values
    else:
        values_feature[filter_] = values
        df.at[id_, feature_name] = values_feature
    if inplace == False:
        return df

#TODO complementar oq ela faz
#TODO trocar nome da func
def feature_values_using_filter_and_indexes(df, id_, feature_name, filter_, idxs, values, inplace = True):
    """
    ?
    Create or update move and stop by radius.

    Parameters
    ----------
    df_ : pandas.core.frame.DataFrame
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


    Examples
    --------
    -

    >>> from pymove.utils.transformations import change_df_feature_values_using_filter_and_indexes
    >>> change_df_feature_values_using_filter_and_indexes(df)

    """
    if inplace == False:
        df = PandasMoveDataFrame(data = df.to_DataFrame())
    values_feature = df.at[id_, feature_name]
    values_feature_filter = values_feature[filter_]
    values_feature_filter[idxs] = values
    values_feature[filter_] = values_feature_filter
    df.at[id_, feature_name] = values_feature
    if inplace == False:
        return df