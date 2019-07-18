
import pandas as pd

def change_df_feature_values_using_filter(df, id_, feature_name, filter_, values):
    """
    equivalent of: df.at[id_, feature_name][filter_] = values
    e.g. df.at[tid, 'time'][filter_nodes] = intp_result.astype(np.int64)
    dataframe must be indexed by id_: df.set_index(index_name, inplace=True)
    """
    values_feature = df.at[id_, feature_name]
    if filter_.shape == ():
        df.at[id_, feature_name] = values
    else:
        values_feature[filter_] = values
        df.at[id_, feature_name] = values_feature


def change_df_feature_values_using_filter_and_indexes(df, id_, feature_name, filter_, idxs, values):
    """
    equivalent of: df.at[id_, feature_name][filter_][idxs] = values
    e.g. df.at[tid, 'deleted'][filter_][idx_not_in_ascending_order] = True
    dataframe must be indexed by id_: df.set_index(index_name, inplace=True)
    """
    values_feature = df.at[id_, feature_name]
    values_feature_filter = values_feature[filter_]
    values_feature_filter[idxs] = values
    values_feature[filter_] = values_feature_filter
    df.at[id_, feature_name] = values_feature


