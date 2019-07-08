import pandas as pd


def drop_consecutive_duplicates(df, subset=None, keep='first', inplace=False):
    if keep == 'first':
        n = 1
    else:
        n = -1
        
    
    if subset is None:
        filter_ = (df.shift(n) != df).any(axis=1)
    else:
        filter_ = (df[subset].shift(n) != df[subset]).any(axis=1)

    if inplace:
        df.drop( index=df[~filter_].index, inplace=True )
        return df
    else:
        return df.loc[ filter_ ]


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

def tranform_speed_from_mS_to_kmH(df_, label_speed, replace=True):
    """
    transformed speed 
    """
    df_[label_speed+'_kmH'] = df_[label_speed].transform(lambda row: row*3.6)
    if replace:
        del df_[label_speed] 


def transform_speed_from_kmH_to_mS(df_, label_speed, replace=True):
    df_[label_speed+'_ms'] = df_[label_speed].transform(lambda row: row/3.6)
    if replace:
        del df_[label_speed] 

""" transform distances """
def transform_dist_from_meters_to_kilometers(df_, label_distance, replace=True):
    df_[label_distance+'_km'] = df_[label_distance].transform(lambda row: row/1000)
    if replace:
        del df_[label_distance] 

def transform_dist_from_to_kilometers_to_meters(df_, label_distance, replace=True):
    df_[label_distance+'_m'] = df_[label_distance].transform(lambda row: row*1000)
    if replace:
        del df_[label_distance] 


""" transform time """
def transform_time_from_seconds_to_minutes(df_, label_time, replace=True):
    df_[label_time+'_m'] = df_[label_time].transform(lambda row: row/60.0)
    if replace:
        del df_[label_time]

def transform_time_from_minute_to_seconds(df_, label_time, replace=True):
    df_[label_time+'_s'] = df_[label_time].apply(lambda row: row*60.0)
    if replace:
        del df_[label_time]

def transform_time_from_minute_to_hours(df_, label_time, replace=True):
    df_[label_time+'_h'] = df_[label_time].apply(lambda row: row/60.0)
    if replace:
        del df_[label_time]

def transform_time_from_hours_to_minute(df_, label_time, replace=True):
    df_[label_time+'_m'] = df_[label_time].apply(lambda row: row*60.0)
    if replace:
        del df_[label_time]


def transform_time_from_seconds_to_hours(df_, label_time, replace=True):
    df_[label_time+'_h'] = df_[label_time].apply(lambda row: row/3600.0)
    if replace:
        del df_[label_time]

def transform_time_from_hours_to_seconds(df_, label_time, replace=True):
    df_[label_time+'_s'] = df_[label_time].apply(lambda row: row*3600.0)
    if replace:
        del df_[label_time]