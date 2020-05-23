import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

import pymove
from pymove import MoveDataFrame, read_csv
from pymove.utils.constants import (
    DATE,
    DATETIME,
    HOUR,
    LATITUDE,
    LONGITUDE,
    PERIOD,
    TID,
    TRAJ_ID,
    UID,
)

list_data = [[39.984094, 116.319236, '2008-10-23 05:53:05', 1],
             [39.984198, 116.319322, '2008-10-23 05:53:06', 1],
             [39.984224, 116.319402, '2008-10-23 05:53:11', 2],
             [39.984224, 116.319402, '2008-10-23 05:53:11', 2]]

move_df = MoveDataFrame(data=list_data, latitude=LATITUDE, longitude=LONGITUDE, datetime=DATETIME, traj_id=TRAJ_ID)

move_df_nan = MoveDataFrame(data=list_data, latitude=LATITUDE, longitude=LONGITUDE, datetime=DATETIME, traj_id=TRAJ_ID)

def _has_columns(data):
    """
    Checks whether the received dataset has LATITUDE, LONGITUDE, 'datetime' columns.

    Parameters
    ----------
    data : dict, list, numpy array or pandas.core.DataFrame.
        Input trajectory data.

    Returns
    -------
    bool
        Represents whether or not you have the required columns.

    """
    if LATITUDE in data and LONGITUDE in data and DATETIME in data:
        return True
    return False


def _validate_move_data_frame_data(data):
    """
        Converts the columns type to the default type used by PyMove lib.

        Parameters
        ----------
        data : dict, list, numpy array or pandas.core.DataFrame.
            Input trajectory data.

        Returns
        -------

        """
    try:
        if data.dtypes.lat != 'float64':
            return False
        if data.dtypes.lon != 'float64':
            return False
        if data.dtypes.datetime != 'datetime64[ns]':
            return False
        return True
    except AttributeError:
        print(AttributeError)


def test_move_data_frame_from_list():
    """
    Tests the creation of a MoveDataFrame from a list.
    Checks if the collumns latitude, longitude and datetime are created
    and converted to the default types used by PyMove lib.
    """
    list_data = [[39.984094, 116.319236, '2008-10-23 05:53:05', 1],
             [39.984198, 116.319322, '2008-10-23 05:53:06', 1],
             [39.984224, 116.319402, '2008-10-23 05:53:11', 1],
             [39.984224, 116.319402, '2008-10-23 05:53:11', 1]]
    move_df = MoveDataFrame(data=list_data, latitude=LATITUDE, longitude=LONGITUDE, datetime=DATETIME, traj_id=TRAJ_ID)
    assert _has_columns(move_df)
    assert _validate_move_data_frame_data(move_df)


def test_move_data_frame_from_file():
    """
        Tests the creation of a MoveDataFrame from a file.
        Checks if the collumns latitude, longitude and datetime are created
        and converted to the default types used by PyMove lib.
    """

    move_df = read_csv('examples/geolife_sample.csv')
    assert _has_columns(move_df)
    assert _validate_move_data_frame_data(move_df)


def test_move_data_frame_from_dict():
    """
        Tests the creation of a MoveDataFrame from a dictionary.
        Checks if the collumns latitude, longitude and datetime are created
        and converted to the default types used by PyMove lib.
    """
    dict_data = {
        LATITUDE: [39.984198, 39.984224, 39.984094],
        LONGITUDE: [116.319402, 116.319322, 116.319402],
        'datetime': ['2008-10-23 05:53:11', '2008-10-23 05:53:06', '2008-10-23 05:53:06']
    }
    move_df = MoveDataFrame(data=dict_data, latitude=LATITUDE, longitude=LONGITUDE, datetime=DATETIME, traj_id=TRAJ_ID)
    assert _has_columns(move_df)
    assert _validate_move_data_frame_data(move_df)


def test_move_data_frame_from_data_frame():
    """
        Tests the creation of a MoveDataFrame from a pandas dataframe.
        Checks if the collumns latitude, longitude and datetime are created
        and converted to the default types used by PyMove lib.
    """
    df = pd.read_csv('examples/geolife_sample.csv', parse_dates=['datetime'])
    move_df = MoveDataFrame(data=df, latitude='lat', longitude=LONGITUDE, datetime=DATETIME)
    assert _has_columns(move_df)
    assert _validate_move_data_frame_data(move_df)


def test_attribute_error_from_data_frame():
    """
        Checks if MoveDataFrame raises a error message when the columns latitude, longitude and datetime
        are missing from the data.
    """
    df = pd.read_csv('pymove/tests/geolife_sample_erro_lat.csv', parse_dates=['datetime'])
    try:
        MoveDataFrame(data=df, latitude=LATITUDE, longitude=LONGITUDE, datetime=DATETIME)
        raise AssertionError('AttributeError error not raised by MoveDataFrame')
    except AttributeError as e:
        pass

    df = pd.read_csv('pymove/tests/geolife_sample_erro_lon.csv', parse_dates=['datetime'])
    try:
        MoveDataFrame(data=df, latitude=LATITUDE, longitude=LONGITUDE, datetime=DATETIME)
        raise AssertionError('AttributeError error not raised by MoveDataFrame')
    except AttributeError as e:
        pass

    df = pd.read_csv('pymove/tests/geolife_sample_erro_datetime.csv', parse_dates=['datetime_erro'])

    try:
        MoveDataFrame(data=df, latitude=LATITUDE, longitude=LONGITUDE, datetime=DATETIME)
        raise AssertionError('AttributeError error not raised by MoveDataFrame')
    except AttributeError as e:
        pass


def test_number_users():
    #Test if the correct number of users is return when the MoveDataFrame has one or multiple users.

    assert move_df.get_users_number() == 1

    move_df[UID] = [1, 1, 2, 3]
    assert move_df.get_users_number() == 3

    move_df.drop(UID, axis=1, inplace = True)


def test_to_numpy():
    # Test if the MoveDataFrame data is converted to numpy array format.

    import numpy
    move_numpy = move_df.to_numpy()
    assert type(move_numpy) is numpy.ndarray


def test_to_dict():
    # Test if the MoveDataFrame data is converted to dict format.

    move_dict = move_df.to_dict()
    assert type(move_df.to_dict()) is dict


def test_to_grid():
    # Test if the MoveDataFrame data is converted to grid format.

    import pymove
    assert type(move_df.to_grid(8)) is pymove.core.grid.Grid


def test_to_data_frame():
    # Test if the MoveDataFrame data is converted to pandas dataFrame format.

    import pandas
    assert type(move_df.to_DataFrame()) is pandas.DataFrame


def test_generate_tid_based_on_id_datetime():
    #Test the function generate_tid_based_on_id_datetime is creating the tid feature.

    #Check if the inplace option is working and the tid feature is created in a copy of PandasMoveDataFrame.
    #Check if the return object of the generate function is a PandasMoveDataFrame,
    #Test if the original PandasMoveDataFrame remains unchanged.
    new_move_df = move_df.generate_tid_based_on_id_datetime(inplace=False)
    assert_array_equal(new_move_df[TID], ['12008102305', '12008102305', '22008102305', '22008102305'])
    assert type(new_move_df) is pymove.core.dataframe.PandasMoveDataFrame
    assert TID not in move_df

    # Check if the tid feature is created in the original PandasMoveDataFrame when inplace = True.
    # Check if the original MoveDataFrame is still a PandasMoveDataFrame.
    move_df.generate_tid_based_on_id_datetime()
    assert_array_equal(move_df[TID], ['12008102305', '12008102305', '22008102305', '22008102305'])
    assert type(move_df) is pymove.core.dataframe.PandasMoveDataFrame

    #Drops the created column
    move_df.drop(TID, axis=1, inplace=True)


def test_generate_date_features():
    # Test the function generate_date_features.

    # Check if the inplace option is working and the date feature is created in a copy of PandasMoveDataFrame.
    # Check if the return object of the generate function is a PandasMoveDataFrame,
    # Test if the original PandasMoveDataFrame remains unchanged.
    new_move_df = move_df.generate_date_features(inplace=False)
    assert_array_equal(new_move_df[DATE].astype(str), ['2008-10-23', '2008-10-23', '2008-10-23', '2008-10-23'])
    assert type(new_move_df) is pymove.core.dataframe.PandasMoveDataFrame
    assert DATE not in move_df

    # Check if the date feature is created in the original PandasMoveDataFrame when inplace = True.
    # Check if the original MoveDataFrame is still a PandasMoveDataFrame.
    move_df.generate_date_features()
    assert_array_equal(move_df[DATE].astype(str), ['2008-10-23', '2008-10-23', '2008-10-23', '2008-10-23'])
    assert type(move_df) is pymove.core.dataframe.PandasMoveDataFrame

    #Drops the created column
    move_df.drop(DATE, axis=1, inplace=True)


def test_generate_hour_features():
    # Test the function generate_hour_features.

    # Check if the inplace option is working and the hour feature is created in a copy of the original
    # PandasMoveDataFrame.
    # Check if the return object of the generate function is a PandasMoveDataFrame,
    # Test if the original PandasMoveDataFrame remains unchanged.
    new_move_df = move_df.generate_hour_features(inplace=False)
    assert new_move_df[HOUR].tolist() == [5, 5, 5, 5]
    assert type(new_move_df) is pymove.core.dataframe.PandasMoveDataFrame
    assert HOUR not in move_df

    # Check if the hour feature is created in the original PandasMoveDataFrame when inplace = True.
    # Check if the original MoveDataFrame is still a PandasMoveDataFrame.
    move_df.generate_hour_features()
    assert move_df[HOUR].tolist() == [5, 5, 5, 5]
    assert type(move_df) is pymove.core.dataframe.PandasMoveDataFrame

    # Drops the created column
    move_df.drop(HOUR, axis=1, inplace=True)


def test_generate_day_of_the_week_features():
    # Test the function day_of_the_week_features.

    # Check if the inplace option is working and the date of the week feature is created in a copy of the original
    # PandasMoveDataFrame.
    # Check if the return object of the generate function is a PandasMoveDataFrame,
    # Test if the original PandasMoveDataFrame remains unchanged.
    new_move_df = move_df.generate_day_of_the_week_features(inplace=False)
    assert_array_equal(new_move_df['day'], ['Thursday', 'Thursday', 'Thursday', 'Thursday'])
    assert type(new_move_df) is pymove.core.dataframe.PandasMoveDataFrame
    print(move_df)
    assert 'day' not in move_df

    # Check if the date of the week feature is created in the original PandasMoveDataFrame
    # PandasMoveDataFrame when inplace = True.
    # Check if the original MoveDataFrame is still a PandasMoveDataFrame.
    move_df.generate_day_of_the_week_features()
    assert_array_equal(move_df['day'].tolist(), ['Thursday', 'Thursday', 'Thursday', 'Thursday'])
    assert type(move_df) is pymove.core.dataframe.PandasMoveDataFrame

    # Drops the created column
    move_df.drop('day', axis=1, inplace=True)


def test_generate_weekend_features():
    # Test the function test_generate_weekend_features.

    # Check if the inplace option is working and the weekend feature is created in a copy of the original
    # PandasMoveDataFrame.
    # Check if the return object of the generate function is a PandasMoveDataFrame,
    # Test if the original PandasMoveDataFrame remains unchanged.
    new_move_df = move_df.generate_weekend_features(inplace=False)
    assert_array_equal(new_move_df['weekend'], [0, 0, 0, 0])
    assert type(new_move_df) is pymove.core.dataframe.PandasMoveDataFrame
    assert 'weekend' not in move_df

    # Check if the weekend feature is created in the original PandasMoveDataFrame
    # when inplace = True.
    # Check if the original MoveDataFrame is still a PandasMoveDataFrame.
    move_df.generate_weekend_features()
    assert_array_equal(move_df['weekend'], [0, 0, 0, 0])
    assert type(move_df) is pymove.core.dataframe.PandasMoveDataFrame

    # Drops the created column
    move_df.drop('weekend', axis=1, inplace=True)


def test_generate_time_of_day_features():
    # Test the function generate_time_of_day_features.

    # Check if the inplace option is working and the time of day feature is created in a copy of the original
    # PandasMoveDataFrame.
    # Check if the return object of the generate function is a PandasMoveDataFrame,
    # Test if the original PandasMoveDataFrame remains unchanged.
    new_move_df = move_df.generate_time_of_day_features(inplace = False)
    assert_array_equal(new_move_df[PERIOD],['Early morning','Early morning','Early morning','Early morning'])
    assert type(new_move_df) is pymove.core.dataframe.PandasMoveDataFrame
    assert PERIOD not in move_df

    # Check if the time of day feature is created in the original PandasMoveDataFrame
    # when inplace = True.
    # Check if the MoveDataFrame is a PandasMoveDataFrame.
    move_df.generate_time_of_day_features()
    assert_array_equal(move_df[PERIOD], ['Early morning','Early morning','Early morning','Early morning'])
    assert type(move_df) is pymove.core.dataframe.PandasMoveDataFrame

    # Drops the created column
    move_df.drop(PERIOD, axis=1, inplace = True)


def test_generate_datetime_in_format_cyclical():
    # Test the function generate_datetime_in_format_cyclical.

    # Check if the inplace option is working and the hour_sin and hour_cos feature is begin created in a copy
    # of the original PandasMoveDataFrame.
    # Check if the return object of the generate function is a PandasMoveDataFrame,
    # Test if the original PandasMoveDataFrame remains unchanged.
    new_move_df = move_df.generate_datetime_in_format_cyclical(inplace = False)
    assert_array_equal(new_move_df['hour_sin'],[0.9790840876823229, 0.9790840876823229,0.9790840876823229, 0.9790840876823229])
    assert_array_equal(new_move_df['hour_cos'], [0.20345601305263375,0.20345601305263375,0.20345601305263375,0.20345601305263375])
    assert type(new_move_df) is pymove.core.dataframe.PandasMoveDataFrame
    assert 'hour_sin' not in move_df
    assert 'hour_cos' not in move_df

    # Check if the hour_sin and hour_cos feature are created in the original PandasMoveDataFrame
    #when inplace = True.
    # Check if the original MoveDataFrame is still a PandasMoveDataFrame.
    move_df.generate_datetime_in_format_cyclical()
    assert_array_equal(move_df['hour_sin'], [0.9790840876823229, 0.9790840876823229,0.9790840876823229, 0.9790840876823229])
    assert_array_equal(move_df['hour_cos'], [0.20345601305263375,0.20345601305263375,0.20345601305263375,0.20345601305263375])
    assert type(move_df) is pymove.core.dataframe.PandasMoveDataFrame

    # Drops the created column
    move_df.drop('hour_sin', axis=1, inplace = True)
    move_df.drop('hour_cos', axis=1, inplace = True)


def test_generate_dist_features():
    # Test the function generate_dist_features.

    # Check if the inplace option is working and the dist features are created in a copy of the original
    # PandasMoveDataFrame.
    # Check if the return object of the generate function is a PandasMoveDataFrame,
    # Test if the original PandasMoveDataFrame remains unchanged.
    import pandas as pd
    df = pd.read_csv('examples/geolife_sample.csv', parse_dates=['datetime'], nrows=5)
    df_move = MoveDataFrame(data=df, latitude=LATITUDE, longitude=LONGITUDE, datetime=DATETIME)
    new_df_move = df_move.generate_dist_features(inplace=False)
    assert_array_equal(new_df_move['dist_to_prev'].astype(str),
                       ['nan', '14.015318782639952', '7.345483960534693', '1.6286216204832726', '2.4484945931533275'])
    assert_array_equal(new_df_move['dist_to_next'].astype(str),
                       ['14.015318782639952', '7.345483960534693', '1.6286216204832726', '2.4484945931533275', 'nan'])
    assert_array_equal(new_df_move['dist_prev_to_next'].astype(str),
                       ['nan', '20.082061827224607', '5.929779944096936', '1.2242472060393084', 'nan'])
    assert type(new_df_move) is pymove.core.dataframe.PandasMoveDataFrame
    assert 'dist_to_prev' not in df_move
    assert 'dist_to_next' not in df_move
    assert 'dist_prev_to_next' not in df_move

    # Check if the dist features are created in the original PandasMoveDataFrame when inplace = True.
    # Check if the original MoveDataFrame is still a PandasMoveDataFrame.
    df_move.generate_dist_features()
    assert_array_equal(df_move['dist_to_prev'].astype(str),
                       ['nan', '14.015318782639952', '7.345483960534693', '1.6286216204832726', '2.4484945931533275'])
    assert_array_equal(df_move['dist_to_next'].astype(str),
                       ['14.015318782639952', '7.345483960534693', '1.6286216204832726', '2.4484945931533275', 'nan'])
    assert_array_equal(df_move['dist_prev_to_next'].astype(str),
                       ['nan', '20.082061827224607', '5.929779944096936', '1.2242472060393084', 'nan'])
    assert type(df_move) is pymove.core.dataframe.PandasMoveDataFrame



def test_generate_dist_time_speed_features():
    # Test the function generate_dist_time_speed_features(.

    # Check if the inplace option is working and the dist, time and speed features are created in a copy of the original
    # PandasMoveDataFrame.
    # Check if the return object of the generate function is a PandasMoveDataFrame,
    # Test if the original PandasMoveDataFrame remains unchanged.
    import pandas as pd
    df = pd.read_csv('examples/geolife_sample.csv', parse_dates=['datetime'], nrows=5)
    df_move = MoveDataFrame(data=df, latitude=LATITUDE, longitude=LONGITUDE, datetime=DATETIME)
    new_df_move = df_move.generate_dist_time_speed_features(inplace=False)
    assert_array_equal(new_df_move['dist_to_prev'].astype(str),
                       ['nan', '14.015318782639952', '7.345483960534693', '1.6286216204832726', '2.4484945931533275'])
    assert_array_equal(new_df_move['time_to_prev'].astype(str), ['nan', '1.0', '5.0', '5.0', '5.0'])
    assert_array_equal(new_df_move['speed_to_prev'].astype(str),
                       ['nan', '14.015318782639952', '1.4690967921069387', '0.3257243240966545', '0.4896989186306655'])
    assert type(new_df_move) is pymove.core.dataframe.PandasMoveDataFrame
    assert 'dist_to_prev' not in df_move
    assert 'time_to_prev' not in df_move
    assert 'speed_to_prev' not in df_move

    # Check if the dist, time and speed features are created in the original PandasMoveDataFrame when inplace = True.
    # Check if the original MoveDataFrame is still a PandasMoveDataFrame.
    df_move.generate_dist_time_speed_features()
    assert_array_equal(df_move['dist_to_prev'].astype(str),
                       ['nan', '14.015318782639952', '7.345483960534693', '1.6286216204832726', '2.4484945931533275'])
    assert_array_equal(df_move['time_to_prev'].astype(str), ['nan', '1.0', '5.0', '5.0', '5.0'])
    assert_array_equal(df_move['speed_to_prev'].astype(str),
                       ['nan', '14.015318782639952', '1.4690967921069387', '0.3257243240966545', '0.4896989186306655'])
    assert type(df_move) is pymove.core.dataframe.PandasMoveDataFrame



def test_generate_move_and_stop_by_radius():
    # Test the function generate_move_and_stop_by_radius.

    # Check if the inplace option is working and the stop feature is created in a copy of the original
    # PandasMoveDataFrame.
    # Check if the return object of the generate function is a PandasMoveDataFrame,
    # Test if the original PandasMoveDataFrame remains unchanged.
    df = pd.read_csv('examples/geolife_sample.csv', parse_dates=['datetime'], nrows=5)
    df_move = MoveDataFrame(data=df, latitude=LATITUDE, longitude=LONGITUDE, datetime=DATETIME)
    new_df_move = df_move.generate_move_and_stop_by_radius(inplace =False)
    assert_array_equal(new_df_move['situation'].astype(str),['nan', 'move', 'move', 'move', 'move'])
    assert type(new_df_move) is pymove.core.dataframe.PandasMoveDataFrame
    assert 'situation' not in df_move

    # Check if the stop feature is created in the original PandasMoveDataFrame when inplace = True.
    # Check if the original MoveDataFrame is still a PandasMoveDataFrame.
    df_move.generate_move_and_stop_by_radius()
    assert_array_equal(df_move['situation'].astype(str),['nan', 'move', 'move', 'move', 'move'])
    assert type(move_df) is pymove.core.dataframe.PandasMoveDataFrame
    assert 'situation' in df_move


def test_loc():
    assert move_df.loc[0,TRAJ_ID] == 1
    assert_array_equal(move_df.loc[move_df[LONGITUDE] > 116.319321].astype(str), [['39.984222', '116.319405', '2008-10-23 05:53:11', '2'],
                                                                              ['39.984222', '116.319405', '2008-10-23 05:53:11', '2']])


def test_iloc():
    assert move_df.iloc[0].all() == move_df.loc[0].all()
    assert_array_equal(move_df.iloc[:3], move_df.loc[:2])


def test_at():
    assert move_df.at[0, TRAJ_ID] == 1


def test_values():
    assert_array_equal(move_df, [[39.984092712402344, 116.3192367553711,
                                        pd.Timestamp('2008-10-23 05:53:05'), 1],
                                       [39.98419952392578, 116.31932067871094,
                                        pd.Timestamp('2008-10-23 05:53:06'), 1],
                                       [39.984222412109375, 116.31940460205078,
                                        pd.Timestamp('2008-10-23 05:53:11'), 2],
                                       [39.984222412109375, 116.31940460205078,
                                        pd.Timestamp('2008-10-23 05:53:11'), 2]])


def test_columns():
    assert_array_equal(move_df.columns, [LATITUDE, LONGITUDE, 'datetime', TRAJ_ID])


def test_index():
    assert_array_equal(move_df.index, [0, 1, 2, 3])


def test_dtypes():
    assert move_df.dtypes.astype(str).tolist() == ['float32', 'float32', 'datetime64[ns]', 'int64']


def test_shape():
    assert move_df.shape == (4,4)


def test_len():
    assert move_df.len() == 4


def test_head():
    assert_array_equal(move_df.head(2), [[39.984092712402344, 116.3192367553711,
                                          pd.Timestamp('2008-10-23 05:53:05'), 1],
                                          [39.98419952392578, 116.31932067871094,
                                          pd.Timestamp('2008-10-23 05:53:06'), 1]])
    assert_array_equal(move_df.head(-1), [[39.984092712402344, 116.3192367553711,
                                                 pd.Timestamp('2008-10-23 05:53:05'), 1],
                                                 [39.98419952392578, 116.31932067871094,
                                                 pd.Timestamp('2008-10-23 05:53:06'), 1],
                                                 [39.984222412109375, 116.31940460205078,
                                                 pd.Timestamp('2008-10-23 05:53:11'), 2]])


def test_time_interval():
    assert move_df.time_interval() == pd.Timedelta('0 days 00:00:06')


def test_get_bbox():
    assert_array_equal(str(move_df.get_bbox()), str((39.984093, 116.31924, 39.984222, 116.319405)))


def test_min():
    assert_array_equal(move_df.min(),[39.984092712402344, 116.3192367553711, pd.Timestamp('2008-10-23 05:53:05'), 1])


def test_max():
    assert_array_equal(move_df.max(), [39.984222412109375, 116.31940460205078, pd.Timestamp('2008-10-23 05:53:11'), 2])


def test_count():
    assert_array_equal(move_df.count(), [4,4,4,4])


def test_group_by():
    assert_array_equal(move_df.groupby(TRAJ_ID).mean().astype(str), [[ '39.984146', '116.319275'],
                                                              [ '39.984222', '116.319405']])


def test_select_dtypes():
    assert_array_equal(move_df.select_dtypes(include='int64'), [[1],[1],[2],[2]])


def test_sort_values():
    move_df.loc[0,TRAJ_ID] = 4
    sorted_move_df = move_df.sort_values(by=[TRAJ_ID])
    assert_array_equal(sorted_move_df, [[39.98419952392578, 116.31932067871094,
                                        pd.Timestamp('2008-10-23 05:53:06'), 1],
                                       [39.984222412109375, 116.31940460205078,
                                        pd.Timestamp('2008-10-23 05:53:11'), 2],
                                       [39.984222412109375, 116.31940460205078,
                                        pd.Timestamp('2008-10-23 05:53:11'), 2],
                                       [39.984092712402344, 116.3192367553711,
                                        pd.Timestamp('2008-10-23 05:53:05'), 4]])
    move_df.loc[0, TRAJ_ID]=1


def test_drop():
    #Creating a new column to use in the drop tests
    move_df[UID] = [1, 1, 2, 3]

    # Check if the column 'uid' is dropped in a copy of the original PandasMoveDataFrame
    # Test if the original PandasMoveDataFrame remains unchanged.
    move_test = move_df.drop(UID, axis=1)
    assert UID not in move_test
    assert UID in move_df

    move_test = move_df.drop([0, 1])
    assert move_test.len() == 2

    # Check if the column 'uid' is dropped from the original PandasMoveDataFrame when inplace = True
    move_df.drop(columns=[UID], axis=1, inplace=True)
    assert UID not in move_df

    #Checks if MoveDataFrame raises an error message when the user tries to drop the columns latitude,
    # longitude or datetime from the data.
    try:
        move_df.drop(columns=[LATITUDE], axis=1, inplace=True)
        raise AssertionError('AttributeError error not raised by MoveDataFrame')
    except AttributeError as e:
        pass

    try:
        move_df.drop(columns=[LONGITUDE], axis=1, inplace=True)
        raise AssertionError('AttributeError error not raised by MoveDataFrame')
    except AttributeError as e:
        pass

    try:
        move_df.drop(columns=['datetime'], axis=1, inplace=True)
        raise AssertionError('AttributeError error not raised by MoveDataFrame')
    except AttributeError as e:
        pass

    assert type(move_df) is pymove.core.dataframe.PandasMoveDataFrame


def test_duplicated():
    assert_array_equal(move_df.duplicated(TRAJ_ID), [False, True, False, True])
    assert_array_equal(move_df.duplicated(subset=['datetime'], keep='last'), [False, False, True, False])


def test_drop_duplicated():
    list_data = [[39.984094, 116.319236, '2008-10-23 05:53:05', 1],
                 [39.984198, 116.319322, '2008-10-23 05:53:06', 1],
                 [39.984224, 116.319402, '2008-10-23 05:53:11', 2],
                 [39.984224, 116.319402, '2008-10-23 05:53:11', 2]]
    move_df_test = MoveDataFrame(data=list_data, latitude=LATITUDE, longitude=LONGITUDE, datetime=DATETIME, traj_id=TRAJ_ID)

    # Check if the duplicated values are dropped in a copy of the original PandasMoveDataFrame when inplace = False
    # Check if the return object of the generate function is a PandasMoveDataFrame,
    move_test = move_df.drop_duplicates()
    assert_array_equal(move_test, [[39.984092712402344, 116.3192367553711,
                                    pd.Timestamp('2008-10-23 05:53:05'), 1],
                                   [39.98419952392578, 116.31932067871094,
                                    pd.Timestamp('2008-10-23 05:53:06'), 1],
                                   [39.984222412109375, 116.31940460205078,
                                    pd.Timestamp('2008-10-23 05:53:11'), 2]])
    assert type(move_df_test) is pymove.core.dataframe.PandasMoveDataFrame

    # Check if the duplicated values are dropped from the original PandasMoveDataFrame when inplace = True
    move_df_test.drop_duplicates(inplace=True)
    assert_array_equal(move_df_test, [[39.984092712402344, 116.3192367553711,
                                       pd.Timestamp('2008-10-23 05:53:05'), 1],
                                      [39.98419952392578, 116.31932067871094,
                                       pd.Timestamp('2008-10-23 05:53:06'), 1],
                                      [39.984222412109375, 116.31940460205078,
                                       pd.Timestamp('2008-10-23 05:53:11'), 2]])


def test_convert_to():

    #Test converting a PandasMoveDataFrame to a DaskDataFame and vice versa
    assert move_df.get_type() == 'pandas'
    move_df_dask = move_df.convert_to('dask')
    assert move_df_dask.get_type() == 'dask'
    move_df_pandas = move_df_dask.convert_to('pandas')
    assert move_df_pandas.get_type() == 'pandas'

    #Test converting a MoveDataFrame to is current type.
    move_df_pandas = move_df_dask.convert_to('pandas')
    assert move_df_pandas.get_type() == 'pandas'


def test_get_type():
    assert move_df.get_type() == 'pandas'
    move_df_dask = move_df.convert_to('dask')
    assert move_df_dask.get_type() == 'dask'


def test_all():
    assert_array_equal(move_df.all(), [True, True, True, True])


def test_any():
    assert_array_equal(move_df.all(), [True, True, True, True])
    move_df['teste'] = [False, True, True, True]
    assert_array_equal(move_df.all(), [True, True, True, True, False])
    move_df.drop('teste',axis=1, inplace=True)


def test_isna():
    test = move_df_nan.isna()
    assert test.any(axis=None) == False
    move_df_nan.loc[0,LATITUDE] = np.nan
    test = move_df_nan.isna()
    assert test.any(axis=None) == True


def test_fillna():
    #Test the function fillna
    test = move_df_nan.fillna(0)
    assert test.isna().any(axis=None) == False
    #Check if the return object from the function fillna is a PandasMoveDataFrame
    assert test.get_type() == 'pandas'
    #Check is the original MoveDataFrame remains unchanged with the nan values.
    assert move_df_nan.isna().any(axis=None) == True


def test_dropna():
    # Check if the nan values are dropped in a copy of the original PandasMoveDataFrame when inplace = False
    # Check if the return object of the generate function is a PandasMoveDataFrame,
    test =  move_df_nan.dropna()
    assert test.len() == 3
    assert test.get_type() == 'pandas'

    # Check if the nan values are dropped in the original PandasMoveDataFrame when inplace = True
    move_df_nan.dropna(inplace=True)
    assert move_df_nan.len() == 3


def test_sample():
    sample_test = move_df[LATITUDE].sample(n=3, random_state=1)
    assert_array_equal(sample_test.astype(str), ['39.984222', '39.984222', '39.984093'])


def test_isin():
    move_df_copy = move_df.copy()
    assert move_df.isin(move_df_copy).all(axis=None) == True
    move_df_copy.loc[0,LATITUDE] = 0
    assert move_df.isin(move_df_copy).all(axis=None) == False


def test_append():
    assert_array_equal(move_df.append(move_df), [[39.984092712402344, 116.3192367553711,
                                                         pd.Timestamp('2008-10-23 05:53:05'), 1],
                                                        [39.98419952392578, 116.31932067871094,
                                                         pd.Timestamp('2008-10-23 05:53:06'), 1],
                                                        [39.984222412109375, 116.31940460205078,
                                                         pd.Timestamp('2008-10-23 05:53:11'), 2],
                                                        [39.984222412109375, 116.31940460205078,
                                                         pd.Timestamp('2008-10-23 05:53:11'), 2],
                                                        [39.984092712402344, 116.3192367553711,
                                                         pd.Timestamp('2008-10-23 05:53:05'), 1],
                                                        [39.98419952392578, 116.31932067871094,
                                                         pd.Timestamp('2008-10-23 05:53:06'), 1],
                                                        [39.984222412109375, 116.31940460205078,
                                                         pd.Timestamp('2008-10-23 05:53:11'), 2],
                                                        [39.984222412109375, 116.31940460205078,
                                                         pd.Timestamp('2008-10-23 05:53:11'), 2]])


def test_nunique():
    assert_array_equal(move_df.nunique(), [3, 3, 3, 2])


def test_join():
    move_df_test = MoveDataFrame(data=list_data, latitude=LATITUDE, longitude=LONGITUDE, datetime=DATETIME, traj_id=TRAJ_ID)
    other = pd.DataFrame({'key': ['K0', 'K1', 'K2']})
    result = move_df_test.join(other)
    assert_array_equal(result['key'].astype(str), ['K0', 'K1', 'K2', 'nan'])


def test_astype():
    move_df.lat = move_df.lat.astype('float64')
    assert move_df.dtypes.lat == 'float64'
    move_df.lat = move_df.lat.astype('float32')
    assert move_df.dtypes.lat == 'float32'


def test_set_index():
    # Check if the index is set in a copy of the original PandasMoveDataFrame when the inplace option = false.
    # Check if the return object of the function is a PandasMoveDataFrame,
    # Test if the original PandasMoveDataFrame remains unchanged.
    move_test = move_df_nan.set_index(TRAJ_ID)
    assert move_test.index.name == TRAJ_ID
    assert move_test.get_type() == 'pandas'
    assert move_df_nan.index.name != TRAJ_ID

    # Check if the index is set in the original PandasMoveDataFrame when the inplace option = true.
    move_df_nan.set_index(keys=TRAJ_ID, inplace=True)
    assert move_df_nan.index.name == TRAJ_ID

    # Checks if MoveDataFrame raises an error message when the user tries to drop the columns latitude,
    # longitude or datetime from the data when setting a new index.
    try:
        move_df_nan.set_index(keys=LATITUDE, drop=True, inplace=True)
        raise AssertionError('AttributeError error not raised by MoveDataFrame')
    except AttributeError as e:
        pass


def test_reset_index():

    # Check if the index is reset in a copy of the original PandasMoveDataFrame when the inplace option = false.
    # Check if the return object of the function is a PandasMoveDataFrame,
    # Test if the original PandasMoveDataFrame remains unchanged.
    move_test = move_df_nan.reset_index()
    assert move_test.index.name != TRAJ_ID
    assert move_test.get_type() == 'pandas'
    assert move_df_nan.index.name == TRAJ_ID

    # Check if the index is reset in the original PandasMoveDataFrame when the inplace option = true.
    move_df_nan.reset_index(inplace=True)
    assert move_df_nan.index.name != TRAJ_ID


def test_unique():
    assert_array_equal(move_df.id.unique(), [1,2])


def test_write_file():
    import pymove
    move_df.write_file('pymove/tests/test_file.csv')
    move_df_test = pymove.read_csv('pymove/tests/test_file.csv')
    assert_array_equal(move_df, move_df_test)


def test_to_csv():
    import pymove
    move_df.to_csv(file_name='pymove/tests/test_csv_file.csv', index = False)
    move_df_test = pymove.read_csv('pymove/tests/test_csv_file.csv')
    assert_array_equal(move_df, move_df_test)


def test_plot_traj_id():
    move_df.generate_tid_based_on_id_datetime()
    df, img = move_df.plot_traj_id(move_df[TID][0])
    assert_array_equal(df, [[39.984092712402344, 116.3192367553711,
                                    pd.Timestamp('2008-10-23 05:53:05'), 1, '12008102305'],
                                   [39.98419952392578, 116.31932067871094,
                                    pd.Timestamp('2008-10-23 05:53:06'), 1, '12008102305']])
    move_df.drop(TID, axis=1, inplace=True)
