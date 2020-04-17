# coding=utf-8
import time
import dask
import os
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dask.dataframe import DataFrame
from pymove.utils.conversions import lat_meters
from pymove.utils.distances import haversine
from pymove.core.grid import Grid
from pymove.core import MoveDataFrameAbstractModel
from pymove.utils.trajectories import format_labels, shift, progress_update
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
    TYPE_DASK,
    TB,
    GB,
    MB,
    KB,
    B,
    HOUR_SIN,
    HOUR_COS
)
from pymove.utils.mem import begin_operation, end_operation


class MoveDataFrame():
    @staticmethod
    def __new__(
        self,
        data,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        datetime=DATETIME,
        traj_id=TRAJ_ID,
        type="pandas",
        n_partitions=1
    ):
        self.type = type

        if type == 'pandas':
            return PandasMoveDataFrame(data, latitude, longitude, datetime, traj_id)
        if type == 'dask':
            return DaskMoveDataFrame(data, latitude, longitude, datetime, traj_id, n_partitions)


class PandasMoveDataFrame(pd.DataFrame, MoveDataFrameAbstractModel):
    def __init__(self, data, latitude=LATITUDE, longitude=LONGITUDE, datetime=DATETIME, traj_id=TRAJ_ID):
        """
        Checks whether past data has 'lat', 'lon', 'datetime' columns, and renames it with the PyMove lib standard.
        After starts the attributes of the class.
        - self._data : Represents trajectory data.
        - self._type : Represents the type of layer below the data structure.
        - self._last_operation : Represents the last operation performed.

        Parameters
        ----------
        data : dict, list, numpy array or pandas.core.DataFrame
            Input trajectory data.

        latitude : String, optional, default 'lat'.
            Represents column name latitude.

        longitude : String, optional, default 'lon'.
            Represents column name longitude.

        datetime : String, optional, default 'datetime'.
            Represents column name datetime.

        traj_id : String, optional, default 'id'.
            Represents column name trajectory id.

        Returns
        -------
        
        Raises
        ------
            AttributeError if the data doesn't contains one of the columns LATITUDE, LONGITUDE, DATETIME

        """
        if isinstance(data, dict):
            data = pd.DataFrame.from_dict(data)
        elif (isinstance(data, list) or isinstance(data, np.ndarray)) and len(data) >= 4:
            zip_list = [LATITUDE, LONGITUDE, DATETIME, TRAJ_ID]
            for i in range(len(data[0])):
                try:
                    zip_list[i] = zip_list[i]
                except KeyError:
                    zip_list.append(i)
            data = pd.DataFrame(data, columns=zip_list)

        mapping_columns = format_labels(data, traj_id, latitude, longitude, datetime)
        tdf = data.rename(columns=mapping_columns)

        if self._has_columns(tdf):
            self._validate_move_data_frame(tdf)
            self._data = tdf
            self._type = TYPE_PANDAS
            self.last_operation = None
        else:
            raise AttributeError("Could not instantiate new MoveDataFrame because data has missing columns")

    @staticmethod
    def _has_columns(data):
        """
        Checks whether the received dataset has 'lat', 'lon', 'datetime' columns.

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

    @staticmethod
    def _validate_move_data_frame(data):
        """
        Converts the column type to the default type used by PyMove lib.

        Parameters
        ----------
        data : dict, list, numpy array or pandas.core.DataFrame.
            Input trajectory data.

        Returns
        -------

        """
        try:
            if data.dtypes.lat != 'float32':
                data.lat.astype('float32')
            if data.dtypes.lon != 'float32':
                data.lon.astype('float32')
            if data.dtypes.datetime != 'datetime64[ns]':
                data.datetime.astype('datetime64[ns]')
        except AttributeError:
            print(AttributeError)

    @property
    def lat(self):
        """
        Checks for the 'lat' column and returns its value.

        """
        if LATITUDE not in self:
            raise AttributeError("The MoveDataFrame does not contain the column '%s.'" % LATITUDE)
        return self._data[LATITUDE]

    @property
    def lng(self):
        """
        Checks for the 'lon' column and returns its value.

        """
        if LONGITUDE not in self:
            raise AttributeError("The MoveDataFrame does not contain the column '%s.'" % LONGITUDE)
        return self._data[LONGITUDE]

    @property
    def datetime(self):
        """
        Checks for the 'datetime' column and returns its value.

        """
        if DATETIME not in self:
            raise AttributeError("The MoveDataFrame does not contain the column '%s.'" % DATETIME)
        return self._data[DATETIME]

    @property
    def loc(self):
        """
        Access a group of rows and columns by label(s) or a boolean array. .loc[] is primarily label based, but may
        also be used with a boolean array.

        Allowed inputs are:
        - A single label, e.g. 5 or 'a', (note that 5 is interpreted as a label of the index, and never as an integer
        position along the index).
        - A list or array of labels, e.g. ['a', 'b', 'c'].
        - A slice object with labels, e.g. 'a':'f'.
            Warning Note that contrary to usual python slices, both the start and the stop are included
            A boolean array of the same length as the axis being sliced, e.g. [True, False, True].
        - A callable function with one argument (the calling Series or DataFrame) and that returns valid output for
        indexing (one of the above)

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.loc.html

        """
        operation = begin_operation('loc')
        loc_ = self._data.loc
        self.last_operation = end_operation(operation)

        return loc_

    @property
    def iloc(self):
        """
        Purely integer-location based indexing for selection by position. .iloc[] is primarily integer position based
        (from 0 to length-1 of the axis), but may also be used with a boolean array.

        Allowed inputs are:
        - An integer, e.g. 5.
        - A list or array of integers, e.g. [4, 3, 0].
        - A slice object with ints, e.g. 1:7.
        - A boolean array.
        - A callable function with one argument (the calling Series or DataFrame) and that returns valid output for
        indexing (one of the above). This is useful in method chains, when you don’t have a reference to the calling
        object, but would like to base your selection on some value.

        .iloc will raise IndexError if a requested indexer is out-of-bounds, except slice indexers which allow
        out-of-bounds indexing (this conforms with python/numpy slice semantics).

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iloc.html

        """
        operation = begin_operation('iloc')
        iloc_ = self._data.iloc
        self.last_operation = end_operation(operation)

        return iloc_

    @property
    def at(self):
        """
        Access a single value for a row/column label pair. Similar to loc, in that both provide label-based lookups.
        Use at if you only need to get or set a single value in a DataFrame or Series.

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.at.html#pandas.DataFrame.at

        """
        operation = begin_operation('at')
        at_ = self._data.at
        self.last_operation = end_operation(operation)

        return at_

    @property
    def values(self):
        """
        Return a Numpy representation of the DataFrame.
        Only the values in the DataFrame will be returned, the axes labels will be removed.
        Warning We recommend using DataFrame.to_numpy() instead.

        Returns
        -------
        numpy.ndarray
            The values of the DataFrame.

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.values.html

        """
        operation = begin_operation('values')
        values_ = self._data.values
        self.last_operation = end_operation(operation)

        return values_

    @property
    def columns(self):
        """
        The column labels of the DataFrame.

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.columns.html#pandas.DataFrame.columns

        """
        return self._data.columns

    @property
    def index(self):
        """
        The index (row labels) of the DataFrame.

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.index.html#pandas.DataFrame.index

        """
        operation = begin_operation('index')
        index_ = self._data.index
        self.last_operation = end_operation(operation)

        return index_

    @property
    def dtypes(self):
        """
        Return the dtypes in the DataFrame. This returns a Series with the data type of each column.
        The result’s index is the original DataFrame’s columns. Columns with mixed types are stored with the object
        dtype. See the User Guide for more.

        Returns
        -------
        pandas.Series
            The data type of each column.

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dtypes.html

        """
        operation = begin_operation('dtypes')
        dtypes_ = self._data.dtypes
        self.last_operation = end_operation(operation)
        return dtypes_

    @property
    def shape(self):
        """
        Return a tuple representing the dimensionality of the DataFrame.

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.shape.html

        """
        operation = begin_operation('shape')
        shape_ = self._data.shape
        self.last_operation = end_operation(operation)
        return shape_

    def len(self):
        """
        Returns the length/row numbers in trajectory data.

        Parameters
        ---------

        Returns
        -------
        len_ : int
            Represents the trajectory data length.

        """
        operation = begin_operation('len')
        len_ = self._data.shape[0]
        self.last_operation = end_operation(operation)

        return len_

    def unique(self, values):
        """
        Return unique values of Series object. Uniques are returned in order of appearance. Hash table-based unique,
        therefore does NOT sort.

        Returns
        -------
        ndarray or ExtensionArray
            The unique values returned as a NumPy array.

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.unique.html

        """
        operation = begin_operation('unique')
        unique_ = self._data.unique(values)
        self.last_operation = end_operation(operation)

        return unique_

    def __setitem__(self, attr, value):
        self.__dict__['_data'][attr] = value

    def __getitem__(self, name):
        try:
            item = self.__dict__['_data'][name]
            if (isinstance(item, pd.DataFrame) and self._has_columns(item)):
                return PandasMoveDataFrame(item)
            return item
        except Exception as e:
            raise e

    def head(self, n=5):
        """
        Return the first n rows.

        This function returns the first n rows for the object based on position. It is useful for quickly testing if
        your object has the right type of data in it.

        Parameters
        ----------
        n : int, default 5
            Number of rows to select.

        Returns
        -------
        head_ : same type as caller
            The first n rows of the caller object.

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.head.html

        """
        operation = begin_operation('head')
        head_ = self._data.head(n)
        self.last_operation = end_operation(operation)

        return head_

    def get_users_number(self):
        """
        Check and return number of users in trajectory data.

        Parameters
        ----------

        Returns
        -------
        number_ : int
            Represents the number of users in trajectory data.

        """
        operation = begin_operation('get_users_numbers')

        if UID in self._data:
            number_ = self._data[UID].nunique()
        else:
            number_ = 1
        self.last_operation = end_operation(operation)

        return number_

    def to_numpy(self):
        """
        Converts trajectory data to numpy array format.

        Parameters
        ----------

        Returns
        -------
        numpy_ : np.array
            Represents the trajectory in numpy array format.

        """
        operation = begin_operation('to_numpy')
        numpy_ = self._data.values
        self.last_operation = end_operation(operation)

        return numpy_

    def to_dict(self):
        """
        Converts trajectory data to dict format.

        Parameters
        ----------

        Returns
        -------
        dict_ : dict
            Represents the trajectory in dict format.

        """
        operation = begin_operation('to_dict')
        dict_ = self._data.to_dict()
        self.last_operation = end_operation(operation)

        return dict_

    def to_grid(self, cell_size, meters_by_degree=lat_meters(-3.8162973555)):
        """
        Converts trajectory data to grid format.

        Parameters
        ----------
        cell_size : float.
            Represents grid cell size.

        meters_by_degree : float, optional, default lat_meters(-3.8162973555).
            Represents the corresponding meters of lat by degree.

        Returns
        -------
        grid_ : pymove.core.grid
            Represents the trajectory in grid format.

        """
        operation = begin_operation('to_grid')
        grid_ = Grid(self, cell_size, meters_by_degree)
        self.last_operation = end_operation(operation)

        return grid_

    def to_DataFrame(self):
        """
        Converts trajectory data to DataFrame format.

        Parameters
        ----------

        Returns
        -------
        pandas.core.DataFrame
            Represents the trajectory in DataFrame format.

        """
        operation = begin_operation('to_DataFrame')
        data_ = self._data
        self.last_operation = end_operation(operation)

        return data_

    def generate_tid_based_on_id_datatime(self, str_format="%Y%m%d%H", sort=True, inplace=True):
        """
        Create or update trajectory id based on id and datetime.

        Parameters
        ----------
        str_format : String, optional, default "%Y%m%d%H".

        sort : bool, optional, default True.
	    If sort == True the dataframe will be sorted.

        inplace : bool, optional, default True.
            Represents whether the operation will be performed on the data provided or in a copy.

        Returns
        -------
        PandasMoveDataFrame or None
            Object with new features or None if ``inplace=True``.

        """
        operation = begin_operation('generate_tid_based_on_id_datatime')

        if inplace:
            data_ = self._data
        else:
            data_ = self._data.copy()

        try:
            print('\nCreating or updating tid feature...\n')
            if sort is True:
                print('...Sorting by {} and {} to increase performance\n'.format(TRAJ_ID, DATETIME))
                data_.sort_values([TRAJ_ID, DATETIME], inplace=True)

            data_[TID] = data_[TRAJ_ID].astype(str) + data_[DATETIME].dt.strftime(str_format)
            print('\n...tid feature was created...\n')

            if inplace:
                self.last_operation = end_operation(operation)
                return None

            self.last_operation = end_operation(operation)
            data_ = PandasMoveDataFrame(data=data_)
            return data_
        except Exception as e:
            self.last_operation = end_operation(operation)
            raise e

    def generate_date_features(self, inplace=True):
        """
        Create or update date feature based on datetime.

        Parameters
        ----------
        inplace : bool, optional, default True.
            Represents whether the operation will be performed on the data provided or in a copy.

        Returns
        -------
        PandasMoveDataFrame or None
            Object with new features or None if ``inplace=True``.
        """
        operation = begin_operation('generate_date_features')

        if inplace:
            data_ = self._data
        else:
            data_ = self._data.copy()

        try:
            print('Creating date features...')
            if DATETIME in data_:
                data_['date'] = data_[DATETIME].dt.date
                print('..Date features was created...\n')

            if inplace:
                self.last_operation = end_operation(operation)
                return None

            data_ = PandasMoveDataFrame(data=data_)
            self.last_operation = end_operation(operation)
            return data_
        except Exception as e:
            self.last_operation = end_operation(operation)
            raise e

    def generate_hour_features(self, inplace=True):
        """
        Create or update hour feature based on datetime.

        Parameters
        ----------
        inplace : bool, optional, default True.
            Represents whether the operation will be performed on the data provided or in a copy.

        Returns
        -------
        PandasMoveDataFrame or None
            Object with new features or None if ``inplace=True``.
        """
        operation = begin_operation('generate_hour_features')

        if inplace:
            data_ = self._data
        else:
            data_ = self._data

        try:
            print('\nCreating or updating a feature for hour...\n')
            if DATETIME in data_:
                data_['hour'] = data_[DATETIME].dt.hour
                print('...Hour feature was created...\n')

            if inplace:
                self.last_operation = end_operation(operation)
                return None

            data_ = PandasMoveDataFrame(data=data_)
            self.last_operation = end_operation(operation)
            return data_
        except Exception as e:
            self.last_operation = end_operation(operation)
            raise e

    def generate_day_of_the_week_features(self, inplace=True):
        """
        Create or update a feature day of the week from datatime.

        Parameters
        ----------
        inplace : bool, optional, default True.
            Represents whether the operation will be performed on the data provided or in a copy.

        Returns
        -------
        PandasMoveDataFrame or None
            Object with new features or None if ``inplace=True``.
        """
        operation = begin_operation('generate_day_of_the_week_features')

        if inplace:
            data_ = self._data
        else:
            data_ = self._data.copy()

        try:
            print('\nCreating or updating day of the week feature...\n')
            data_[DAY] = data_[DATETIME].dt.day_name()
            print('...the day of the week feature was created...\n')

            if inplace:
                self.last_operation = end_operation(operation)
                return None

            data_ = PandasMoveDataFrame(data=data_)
            self.last_operation = end_operation(operation)
            return data_
        except Exception as e:
            self.last_operation = end_operation(operation)
            raise e

    def generate_weekend_features(self, create_day_of_week=False, inplace=True):
        """
        Create or update the feature weekend to the dataframe, if this resource is set to 1 it indicates that the
        given day is is the weekend, otherwise, it is a day of the week.

        Parameters
        ----------
        create_day_of_week : bool, optional (False by default).
            Indicates if the column day should be keeped in the dataframe. If set to False the column will be dropped.

        inplace : bool, optional, default True.
            Indicates whether the operation will be performed on the data provided or in a copy.

        Returns
        ----------
        PandasMoveDataFrame or None
            Object with new features or None if ``inplace=True``.
        """
        operation = begin_operation('generate_weekend_features')
        
        try:
            if inplace:
                self.generate_day_of_the_week_features(inplace=inplace)
                data_ = self._data
            else:
                df = self.generate_day_of_the_week_features(inplace=inplace)
                data_ = df._data

            print('Creating or updating a feature for weekend\n')
            if 'day' in data_:
                index_fds = data_[(data_[DAY] == 'Saturday') | (data_[DAY] == 'Sunday')].index
                data_['weekend'] = 0
                data_.at[index_fds, 'weekend'] = 1
                print('...Weekend was set as 1 or 0...\n')
                if not create_day_of_week:
                    print('...dropping colum day\n')
                    del data_['day']

            if inplace:
                self.last_operation = end_operation(operation)
                return None

            data_ = PandasMoveDataFrame(data=data_)
            self.last_operation = end_operation(operation)
            return data_
        except Exception as e:
            self.last_operation = end_operation(operation)
            raise e

    def generate_time_of_day_features(self, inplace=True):
        """
        Create a feature time of day or period from datatime.

        Parameters
        ----------
         inplace : bool, optional, default True.
            Represents whether the operation will be performed on the data provided or in a copy.

        Returns
        -------
        PandasMoveDataFrame or None
            Object with new features or None if ``inplace=True``.

        Examples
        --------
        - datetime1 = 2019-04-28 02:00:56 -> period = early morning
        - datetime2 = 2019-04-28 08:00:56 -> period = morning
        - datetime3 = 2019-04-28 14:00:56 -> period = afternoon
        - datetime4 = 2019-04-28 20:00:56 -> period = evening

        """
        operation = begin_operation('generate_time_of_day_features')

        if inplace:
            data_ = self._data
        else:
            data_ = self._data.copy()

        try:
            periods = [
                '\n'
                'Creating or updating period feature',
                '...Early morning from 0H to 6H',
                '...Morning from 6H to 12H',
                '...Afternoon from 12H to 18H',
                '...Evening from 18H to 24H'
                '\n'
            ]
            print('\n'.join(periods))

            conditions = [(data_[DATETIME].dt.hour >= 0) & (data_[DATETIME].dt.hour < 6),
                          (data_[DATETIME].dt.hour >= 6) & (data_[DATETIME].dt.hour < 12),
                          (data_[DATETIME].dt.hour >= 12) & (data_[DATETIME].dt.hour < 18),
                          (data_[DATETIME].dt.hour >= 18) & (data_[DATETIME].dt.hour < 24)]
            choices = ['Early morning', 'Morning', 'Afternoon', 'Evening']
            data_[PERIOD] = np.select(conditions, choices, 'undefined')
            print('...the period of day feature was created')

            if inplace:
                self.last_operation = end_operation(operation)
                return None

            data_ = PandasMoveDataFrame(data=data_)
            self.last_operation = end_operation(operation)
            return data_
        except Exception as e:
            self.last_operation = end_operation(operation)
            raise e

    def generate_datetime_in_format_cyclical(self, label_datetime=DATETIME, inplace=True):
        """
        Create or update column with cyclical datetime feature.

        Parameters
        ----------
        label_datetime : String, optional, default 'datetime.
            Represents column id type.

        inplace : bool, optional, default True.
            Represents whether the operation will be performed on the data provided or in a copy.

        Returns
        -------
        PandasMoveDataFrame or None
            Object with new features or None if ``inplace=True``.
        
        References
        ----------
        # https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/
        # https://www.avanwyk.com/encoding-cyclical-features-for-deep-learning/
        """
        operation = begin_operation('generate_datetime_in_format_cyclical')
        
        if inplace:
            data_ = self._data
        else:
            data_ = self._data.copy()

        try:
            print('Encoding cyclical continuous features - 24-hour time')
            if label_datetime in self._data:
                hours = data_[label_datetime].dt.hour
                data_[HOUR_SIN] = np.sin(2 * np.pi * hours / 23.0)
                data_[HOUR_COS] = np.cos(2 * np.pi * hours / 23.0)
                print('...hour_sin and  hour_cos features were created...\n')

            if inplace:
                self.last_operation = end_operation(operation)
                return None

            data_ = PandasMoveDataFrame(data=data_)
            self.last_operation = end_operation(operation)
            return data_
        except Exception as e:
            self.last_operation = end_operation(operation)
            raise e

    def generate_dist_features(self, label_id=TRAJ_ID, label_dtype=np.float64, sort=True, inplace=True):
        """
        Create the three distance in meters to an GPS point P (lat, lon).

        Parameters
        ----------
        label_id : String, optional, default 'id'.
            Represents name of column of trajectore's id.

        label_dtype : type, optional, default np.float64.
            Represents column id type.

        sort : bool, optional, default True.
            If sort == True the dataframe will be sorted.

        inplace : bool, optional, default True.
            Represents whether the operation will be performed on the data provided or in a copy.

        Returns
        -------
        PandasMoveDataFrame or None
            Object with new features or None if ``inplace=True``.

        Examples
        --------
        Example:    P to P.next = 2 meters
                    P to P.previous = 1 meter
                    P.previous to P.next = 1 meters

        """
        operation = begin_operation('generate_dist_features')

        if inplace:
            data_ = self._data
        else:
            data_ = self._data.copy()

        try:
            print('\nCreating or updating distance features in meters...\n')
            start_time = time.time()

            if sort is True:
                print('...Sorting by {} and {} to increase performance\n'.format(label_id, DATETIME))
                data_.sort_values([label_id, DATETIME], inplace=True)

            if data_.index.name is None:
                print('...Set {} as index to increase attribution performance\n'.format(label_id))
                data_.set_index(label_id, inplace=True)

            """ create ou update columns"""
            data_[DIST_TO_PREV] = label_dtype(-1.0)
            data_[DIST_TO_NEXT] = label_dtype(-1.0)
            data_[DIST_PREV_TO_NEXT] = label_dtype(-1.0)

            ids = data_.index.unique()
            selfsize = data_.shape[0]
            curr_perc_int = -1
            start_time = time.time()
            deltatime_str = ''
            sum_size_id = 0
            size_id = 0
            for idx in ids:
                curr_lat = data_.at[idx, LATITUDE]
                curr_lon = data_.at[idx, LONGITUDE]

                size_id = curr_lat.size

                if size_id <= 1:
                    #print('...id:{}, must have at least 2 GPS points\n'.format(idx))
                    data_.at[idx, DIST_TO_PREV] = np.nan

                else:
                    prev_lat = shift(curr_lat, 1)
                    prev_lon = shift(curr_lon, 1)
                    # compute distance from previous to current point
                    data_.at[idx, DIST_TO_PREV] = haversine(prev_lat, prev_lon, curr_lat, curr_lon)

                    next_lat = shift(curr_lat, -1)
                    next_lon = shift(curr_lon, -1)
                    # compute distance to next point
                    data_.at[idx, DIST_TO_NEXT] = haversine(curr_lat, curr_lon, next_lat, next_lon)

                    # using pandas shift in a large dataset: 7min 21s
                    # using numpy shift above: 33.6 s

                    # use distance from previous to next
                    data_.at[idx, DIST_PREV_TO_NEXT] = haversine(prev_lat, prev_lon, next_lat, next_lon)

                    sum_size_id += size_id
                    curr_perc_int, est_time_str = progress_update(sum_size_id, selfsize, start_time, curr_perc_int,
                                                                  step_perc=20)
            data_.reset_index(inplace=True)
            print('...Reset index\n')
            print('..Total Time: {}'.format((time.time() - start_time)))

            if inplace:
                self.last_operation = end_operation(operation)
                return None

            data_ = PandasMoveDataFrame(data=data_)
            self.last_operation = end_operation(operation)
            return data_
        except Exception as e:
            print('label_id:{}\nidx:{}\nsize_id:{}\nsum_size_id:{}'.format(label_id, idx, size_id, sum_size_id))
            self.last_operation = end_operation(operation)
            raise e

    def generate_dist_time_speed_features(self, label_id=TRAJ_ID, label_dtype=np.float64, sort=True, inplace=True):
        """
        Firstly, create the three distance to an GPS point P (lat, lon).
        After, create two time features to point P: time to previous and time to next.
        Lastly, create two features to speed using time and distance features.

        Parameters
        ----------
        label_id : String, optional, default 'id'.
            Represents name of column of trajectore's id.

        label_dtype : type, optional, default np.float64.
            Represents column id type.

        sort : bool, optional, default True.
            If sort == True the dataframe will be sorted.

        inplace : bool, optional, default True.
            Represents whether the operation will be performed on the data provided or in a copy.

        Returns
        -------
        PandasMoveDataFrame or None
            Object with new features or None if ``inplace=True``.

        Examples
        --------
        Example:    dist_to_prev =  248.33 meters, dist_to_prev 536.57 meters
                    time_to_prev = 60 seconds, time_prev = 60.0 seconds
                    speed_to_prev = 4.13 m/s, speed_prev = 8.94 m/s.

        """
        operation = begin_operation('generate_dist_time_speed_features')

        if inplace:
            data_ = self._data
        else:
            data_ = self._data.copy()

        try:
            print('\nCreating or updating distance, time and speed features in meters by seconds\n')
            start_time = time.time()

            if sort is True:
                print('...Sorting by {} and {} to increase performance\n'.format(label_id, DATETIME))
                data_.sort_values([label_id, DATETIME], inplace=True)

            if data_.index.name is None:
                print('...Set {} as index to a higher peformance\n'.format(label_id))
                data_.set_index(label_id, inplace=True)

            """create new feature to distance"""
            data_[DIST_TO_PREV] = label_dtype(-1.0)

            """create new feature to time"""
            data_[TIME_TO_PREV] = label_dtype(-1.0)

            """create new feature to speed"""
            data_[SPEED_TO_PREV] = label_dtype(-1.0)

            ids = data_.index.unique()
            selfsize = data_.shape[0]
            curr_perc_int = -1
            sum_size_id = 0
            size_id = 0

            for idx in ids:
                curr_lat = data_.at[idx, LATITUDE]
                curr_lon = data_.at[idx, LONGITUDE]

                size_id = curr_lat.size

                if size_id <= 1:
                    #print('...id:{}, must have at least 2 GPS points\n'.format(idx))
                    data_.at[idx, DIST_TO_PREV] = np.nan
                    data_.at[idx, TIME_TO_PREV] = np.nan
                    data_.at[idx, SPEED_TO_PREV] = np.nan
                else:
                    prev_lat = shift(curr_lat, 1)
                    prev_lon = shift(curr_lon, 1)
                    prev_lon = shift(curr_lon, 1)
                    # compute distance from previous to current point
                    data_.at[idx, DIST_TO_PREV] = haversine(prev_lat, prev_lon, curr_lat, curr_lon)

                    time_ = data_.at[idx, DATETIME].astype(label_dtype)
                    time_prev = (time_ - shift(time_, 1)) * (10 ** -9)
                    data_.at[idx, TIME_TO_PREV] = time_prev

                    """ set time_to_next"""
                    # time_next = (ut.shift(time_, -1) - time_)*(10**-9)
                    # self.at[idx, dic_features_label['time_to_next']] = time_next

                    "set speed features"
                    data_.at[idx, SPEED_TO_PREV] = data_.at[idx, DIST_TO_PREV]/time_prev  # unit: m/s

                    sum_size_id += size_id
                    curr_perc_int, est_time_str = progress_update(sum_size_id, selfsize, start_time, curr_perc_int,
                                                                  step_perc=20)
            print('...Reset index...\n')
            data_.reset_index(inplace=True)
            print('..Total Time: {:.3f}'.format((time.time() - start_time)))

            if inplace:
                self.last_operation = end_operation(operation)
                return None

            data_ = PandasMoveDataFrame(data=data_)
            self.last_operation = end_operation(operation)
            return data_
        except Exception as e:
            print('label_id:{}\nidx:{}\nsize_id:{}\nsum_size_id:{}'.format(label_id, idx, size_id, sum_size_id))
            self.last_operation = end_operation(operation)
            raise e

    def generate_move_and_stop_by_radius(self, radius=0, target_label=DIST_TO_PREV, inplace=True):
        """
        Create or update column with move and stop points by radius.

        Parameters
        ----------
        radius : int, optional, default 0.
            Represents radius.

        target_label : String, optional, default 'dist_to_prev.
            Represents column id type.

        inplace : bool, optional, default True.
            Represents whether the operation will be performed on the data provided or in a copy.

        Returns
        -------
        PandasMoveDataFrame or None
            Object with new features or None if ``inplace=True``.
        """
        operation = begin_operation('generate_move_and_stop_by_radius')

        if inplace:
            if DIST_TO_PREV not in self._data:
                self.generate_dist_features(inplace=inplace)
            data_ = self._data
        else:
            if DIST_TO_PREV not in self._data:
                df = self.generate_dist_features(inplace=inplace)
            data_ = df._data

        try:
            print('\nCreating or updating features MOVE and STOPS...\n')
            conditions = (data_[target_label] > radius), (data_[target_label] <= radius)
            choices = ['move', 'stop']

            data_['situation'] = np.select(conditions, choices, np.nan)
            print('\n....There are {} stops to this parameters\n'.format(data_[data_['situation'] == 'stop'].shape[0]))

            if inplace:
                self.last_operation = end_operation(operation)
                return None

            data_ = PandasMoveDataFrame(data=data_)
            self.last_operation = end_operation(operation)
            return data_
        except Exception as e:
            self.last_operation = end_operation(operation)
            raise e

    def time_interval(self):
        """
        Get time difference between max and min datetime in trajectory data.

        Parameters
        ----------

        Returns
        -------
        time_diff : datetime64
            Represents the time difference.

        """
        operation = begin_operation('time_interval')
        time_diff = self._data[DATETIME].max() - self._data[DATETIME].min()
        self.last_operation = end_operation(operation)

        return time_diff

    def get_bbox(self):
        """
        A bounding box (usually shortened to bbox) is an area defined by two longitudes and two latitudes, where:
            - Latitude is a decimal number between -90.0 and 90.0.
            - Longitude is a decimal number between -180.0 and 180.0.
        They usually follow the standard format of:
        - bbox = left, bottom, right, top
        - bbox = min Longitude , min Latitude , max Longitude , max Latitude
        Parameters
        ----------

        Returns
        -------
        bbox_ : tuple
            Represents a bound box, that is a tuple of 4 values with the min and max limits of latitude e longitude.

        Examples
        --------
        (22.147577, 113.54884299999999, 41.132062, 121.156224)

        """
        operation = begin_operation('get_bbox')

        try:
            bbox_ = (self._data[LATITUDE].min(), self._data[LONGITUDE].min(), self._data[LATITUDE].max(),
                     self._data[LONGITUDE].max())

            self.last_operation = end_operation(operation)

            return bbox_
        except Exception as e:
            self.last_operation = end_operation(operation)
            raise e

    def plot_all_features(self, dtype=np.float64, figsize=(21, 15), return_fig=True, save_fig=False, name='features.png'):
        """
        Generate a visualization for each columns that type is equal dtype.

        Parameters
        ----------
        figsize : tuple, optional, default (21, 15).
            Represents dimensions of figure.

        dtype : type, optional, default np.float64.
            Represents column type.

        return_fig : bool, optional, default True.
            Represents whether or not to save the generated picture.

        save_fig : bool, optional, default False.
            Represents whether or not to save the generated picture.

        name : String, optional, default 'features.png'.
            Represents name of a file.

        Returns
        -------
        fig : matplotlib.pyplot.figure or None
            The generated picture.
        """
        operation = begin_operation('plot_all_features')

        try:
            col_float = self._data.select_dtypes(include=[dtype]).columns
            tam = col_float.size
            if tam > 0:
                fig, ax = plt.subplots(tam, 1, figsize=figsize)
                ax_count = 0
                for col in col_float:
                    ax[ax_count].set_title(col)
                    self._data[col].plot(subplots=True, ax=ax[ax_count])
                    ax_count += 1

                if save_fig:
                    plt.savefig(fname=name, fig=fig)

            self.last_operation = end_operation(operation)
            
            if return_fig:
                return fig
        except Exception as e:
            self.last_operation = end_operation(operation)
            raise e

    def plot_trajs(self, markers='o', markersize=20, figsize=(10, 10), return_fig=True, save_fig=False, name='trajectories.png'):
        """Generate a visualization that show trajectories.

        Parameters
        ----------
        figsize : tuple, optional, default (10, 10).
            Represents dimensions of figure.

        markers : String, optional, default 'o'.
            Represents visualization type marker.

        markersize : int, optional, default 20.
            Represents visualization size marker.

        return_fig : bool, optional, default True.
            Represents whether or not to save the generated picture.

        save_fig : bool, optional, default False.
            Represents whether or not to save the generated picture.

        name : String, optional, default 'trajectories.png'.
            Represents name of a file.

        Returns
        -------
        fig : matplotlib.pyplot.figure or None
            The generated picture.
        """

        operation = begin_operation('plot_trajs')

        fig = plt.figure(figsize=figsize)

        ids = self._data['id'].unique()
        for id_ in ids:
            selfid = self._data[self._data['id'] == id_]
            plt.plot(selfid[LONGITUDE], selfid[LATITUDE], markers, markersize=markersize)

        if save_fig:
            plt.savefig(fname=name, fig=fig)

        self.last_operation = end_operation(operation)
        
        if return_fig:
            return fig

    def plot_traj_id(self, tid, highlight=None, figsize=(10, 10), return_fig=True, save_fig=False, name=None):
        """Generate a visualization that shows a trajectory with the specified tid.

        Parameters
        ----------
        tid : String.
            Represents the trajectory tid.

        highlight: String, optional, default None.
            Name of the feature to highlight on plot. 
            If value of feature is 1, it will be highlighted as green marker 

        figsize : tuple, optional, default (10,10).
            Represents dimensions of figure.

        return_fig : bool, optional, default True.
            Represents whether or not to save the generated picture.

        save_fig : bool, optional, default False.
            Represents whether or not to save the generated picture.

        name : String, optional, default None.
            Represents name of a file.


        Returns
        -------
        move_data : pymove.core.MoveDataFrameAbstract subclass.
            Trajectory with the specified tid.

        fig : matplotlib.pyplot.figure or None
            The generated picture.

        Raises
        ------
        KeyError if the dataframe does not contains the TID feature
        IndexError if there is no trajectory with the tid passed
        """

        operation = begin_operation('plot_traj_id')

        if TID not in self._data:
            self.last_operation = end_operation(operation)
            raise KeyError("TID feature not in dataframe")

        df_ = self._data[self._data[TID] == tid]
        
        if not len(df_):
            self.last_operation = end_operation(operation)
            raise IndexError(f"No trajectory with tid {tid} in dataframe")
        
        fig = plt.figure(figsize=figsize)

        plt.plot(df_.iloc[0][LONGITUDE], df_.iloc[0][LATITUDE], 'yo', markersize=20)  # start point
        plt.plot(df_.iloc[-1][LONGITUDE], df_.iloc[-1][LATITUDE], 'yX', markersize=20)  # end point

        if (not highlight) or (highlight not in df_):
            plt.plot(df_[LONGITUDE], df_[LATITUDE])
            plt.plot(df_.loc[:, LONGITUDE], df_.loc[:, LATITUDE], 'r.', markersize=8)  # points
        else:
            filter_ = df_[highlight] == 1
            selfnodes = df_.loc[filter_]
            selfpoints = df_.loc[~filter_]
            plt.plot(selfnodes[LONGITUDE], selfnodes[LATITUDE], linewidth=3)
            plt.plot(selfpoints[LONGITUDE], selfpoints[LATITUDE])
            plt.plot(selfnodes[LONGITUDE], selfnodes[LATITUDE], 'go', markersize=10)  # nodes
            plt.plot(selfpoints[LONGITUDE], selfpoints[LATITUDE], 'r.', markersize=8)  # points

        if save_fig:
            if not name:
                name = f'trajectory_{tid}.png'
            plt.savefig(fname=name, fig=fig)

        df_ = PandasMoveDataFrame(df_)
        
        self.last_operation = end_operation(operation)

        if return_fig:
            return df_, fig
        return df_

    def show_trajectories_info(self):
        """
        Show dataset information from dataframe, this is number of rows, datetime interval, and bounding box.

        Parameters
        ----------
        self : pandas.core.frame.DataFrame
            Represents the dataset with contains lat, long and datetime.
            
        Examples
        --------
        ======================= INFORMATION ABOUT DATASET =======================
        Number of Points: 217654
        Number of IDs objects: 2
        Start Date:2008-10-23 05:53:05     End Date:2009-03-19 05:46:37
        Bounding Box:(22.147577, 113.54884299999999, 41.132062, 121.156224)
        =========================================================================
        """

        operation = begin_operation('show_trajectories_info')

        try:
            print('\n======================= INFORMATION ABOUT DATASET =======================\n')
            print('Number of Points: {}\n'.format(self._data.shape[0]))

            if TRAJ_ID in self._data:
                print('Number of IDs objects: {}\n'.format(self._data[TRAJ_ID].nunique()))

            if TID in self._data:
                print('Number of TIDs trajectory: {}\n'.format(self._data[TID].nunique()))

            if DATETIME in self._data:
                print('Start Date:{}     End Date:{}\n'.format(self._data[DATETIME].min(),
                                                               self._data[DATETIME].max()))

            if LATITUDE and LONGITUDE in self._data:
                print('Bounding Box:{}\n'.format(
                    self.get_bbox()))  # bbox return =  Lat_min , Long_min, Lat_max, Long_max)

            if TIME_TO_PREV in self._data:
                print(
                    'Gap time MAX:{}     Gap time MIN:{}\n'.format(
                        round(self._data[TIME_TO_PREV].max(), 3),
                        round(self._data[TIME_TO_PREV].min(), 3)))

            if SPEED_TO_PREV in self._data:
                print('Speed MAX:{}    Speed MIN:{}\n'.format(round(self._data[SPEED_TO_PREV].max(), 3),
                                                              round(self._data[SPEED_TO_PREV].min(), 3)))

            if DIST_TO_PREV in self._data:
                print('Distance MAX:{}    Distance MIN:{}\n'.format(
                    round(self._data[DIST_TO_PREV].max(), 3),
                    round(self._data[DIST_TO_PREV].min(),
                          3)))

            print('\n=========================================================================\n')

            self.last_operation = end_operation(operation)
        except Exception as e:
            self.last_operation = end_operation(operation)
            raise e

    def min(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        """Returns the minimum values for the requested axis of the dataframe.

        Parameters
        ----------
        axis: int, None by default, {index (0), columns (1)}.
            Axis for the function to be applied on.
        skipna: bool, optional, default None.
            Exclude NA/null values when computing the result.
        level: int or str, optional, default None.
            If the axis is a MultiIndex (hierarchical), count along a particular level, collapsing into a Series.
        numeric_only: bool, optional (None by default)
            Include only float, int, boolean columns.
            If None, will attempt to use everything, then use only numeric data.
        kwargs:
            Additional keyword arguments to be passed to the function

        Returns
        -------
        mim: Series or DataFrame (if level specified)
            The minimum values for the request axis.

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.min.html
        """
        operation = begin_operation('min')
        _min = self._data.min(axis, skipna, level, numeric_only, **kwargs)
        self.last_operation = end_operation(operation)

        return _min

    def max(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        """Returns the maximum  values for the requested axis of the dataframe.

        Parameters
        ----------
        axis: int, None by default, {index (0), columns (1)}
            Axis for the function to be applied on.
        skipna: bool, optional(None by default).
            Exclude NA/null values when computing the result.
        level: int or str, optional (None by default).
            If the axis is a MultiIndex (hierarchical), count along a particular level, collapsing into a Series.
        numeric_only: bool, optional (None by default).
            Include only float, int, boolean columns.
            If None, will attempt to use everything, then use only numeric data.
        kwargs: keywords.
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        max: Series or DataFrame (if level specified)
            The maximum values for the request axis.

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.max.html
        """
        operation = begin_operation('max')
        _max = self._data.max(axis, skipna, level, numeric_only, **kwargs)
        self.last_operation = end_operation(operation)

        return _max

    def count(self, axis=0, level=None, numeric_only=False):
        """Uses the pandas's function count, to count the number of non-NA cells for each column or row.

        Parameters
        ----------
        axis: int, None by default, {index (0), columns (1)}
            if set to 0 or'index', will count for each column.
            if set to 1 or'columns', will count for each row.
        level: int or str, optional (None by default)
            If the axis is a MultiIndex (hierarchical), count along a particular level, collapsing into a DataFrame.
            A str specifies the level name
        numeric_only: bool, optional (False by default)
            If set to true, only float, int or boolean data, will be included.

        Returns
        --------
        A series or a DataFrame. 
            The number of non-NA/null entries for each column/row. If level is specified returns a DataFrame.

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.count.html
        """
        operation = begin_operation('count')
        _count = self._data.count(axis, level, numeric_only)
        self.last_operation = end_operation(operation)

        return _count

    def groupby(
        self,
        by=None,
        axis=0,
        level=None,
        as_index=True,
        sort=True,
        group_keys=True,
        squeeze=False,
        observed=False,
        **kwargs
    ):
        """Groups DataFrame using a mapper or by a Series of columns.
        A groupby operation involves some combination of splitting the object, applying a function, and
        combining the results. This can be used to group large amounts of data and compute operations on these groups.

        Parameters
        ----------
        by : mapping, function, label, or list of labels, optional(None by default)
            Used to determine the groups for the groupby. If by is a function, it’s called on each value of the
            object’s index. If a dict or Series is passed, the Series or dict VALUES will be used to determine the
            groups (the Series’ values are first aligned; see .align() method). If an ndarray is passed, the values
            are used as-is determine the groups. A label or list of labels may be passed to group by the columns in
            self. Notice that a tuple is interpreted as a (single) key.
        axis : int, None by default, {index (0), columns (1)}
            Split along rows (0) or columns (1).
        level : Integer, level name, or sequence of such, optional (default None)
            If the axis is a MultiIndex (hierarchical), group by a particular level or levels.
        as_index : boolean, optional (default True)
            For aggregated output, return object with group labels as the index. Only relevant for DataFrame input.
             as_index=False is effectively “SQL-style” grouped output.
        sort : boolean, optional (default True)
            Sort group keys. Get better performance by turning this off. Note this does not influence the order
            of observations within each group. Groupby preserves the order of rows within each group.
        group_keys : boolean, default True
            When calling apply, add group keys to index to identify pieces.
        squeeze : boolean, optional (default False)
            Reduce the dimensionality of the return type if possible, otherwise return a consistent type.
        observed : boolean, optional (default False)
            This only applies if any of the groupers are Categoricals. If True: only show observed values for
            categorical groupers. If False: show all values for categorical groupers.
        **kwargs
            Optional, only accepts keyword argument ‘mutated’ and is passed to groupby.

        Returns
        -------
        DataFrameGroupBy:
            Returns groupby object that contains information about the groups.

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html
        """
        operation = begin_operation('groupby')
        _groupby = self._data.groupby(by, axis, level, as_index, sort, group_keys, squeeze, observed, **kwargs)
        self.last_operation = end_operation(operation)

        return _groupby

    def plot(self, *args, **kwargs):
        """Makes a plot of _data.

        Parameters
        ----------
        args:
        kwargs: keywords
            Additional keyword arguments to be passed to the function

        Returns:
        --------
        class:`matplotlib.axes.Axes` or numpy.ndarray of them
            If the backend is not the default matplotlib one, the return value
            will be the object returned by the backend.

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html
        """
        operation = begin_operation('plot')
        _plot = self._data.plot(*args, **kwargs)
        self.last_operation = end_operation(operation)

        return _plot

    def select_dtypes(self, include=None, exclude=None):
        """Returns a subset of the _data's columns based on the column dtypes.

        Parameters
        ----------
        include: scalar or list-like
            A selection of dtypes or strings to be included/excluded.
        exclude: scalar or list-like
            A selection of dtypes or strings to be included/excluded.

        Returns
        --------
            DataFrame
                The subset of the _data including the dtypes in
                include and excluding the dtypes in exclude.

        Notes
        -----
        One of the parameters: include or exclude must be supplied.

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.select_dtypes.html
        """
        operation = begin_operation('select_dtypes')
        _select_dtypes = self._data.select_dtypes(include, exclude)
        self.last_operation = end_operation(operation)

        return _select_dtypes
    
    def astype(self, dtype, copy=True, errors='raise', **kwargs):
        """Cast a pandas object to a specified dtype.

        Parameters
        ----------
        dtype: data type, or dict of column name -> data type
            Use a numpy.dtype or Python type to cast entire pandas object to the same type. Alternatively,
            use {col: dtype, …}, where col is a column label and dtype is a numpy.dtype or Python type to
            cast one or more of the DataFrame’s columns to column-specific types.
        copy: bool, optional(True by default)
            Return a copy when copy=True (be very careful setting copy=False as changes to values then
            may propagate to other pandas objects).
        errors: str, optional('raise' by default), options: 'raise', 'ignore'
            Control raising of exceptions on invalid data for provided dtype.
            - raise : allow exceptions to be raised
            - ignore : suppress exceptions. On error return original object
        kwargs:
             keyword arguments to pass on to the constructor

        Returns
        -------
        DataFrame:
            Casted _data to specified type.

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.astype.html
        """
        
        if not copy and type(dtype) == str:
            raise AttributeError("Could not change lat, lon, and datetime type.")
        elif not copy and type(dtype) == dict:
            keys = set(list(dtype.keys()))
            columns = set(['lat','lon','datetime'])
            if keys & columns:
                raise AttributeError("Could not change lat, lon, and datetime type.")
                
        operation = begin_operation('astype')
        _astype = self._data.astype(dtype, copy, errors, **kwargs)
        self.last_operation = end_operation(operation)

        return _astype

    def sort_values(self, by, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last'):
        """Sorts the values of the _data, along an axis.

        Parameters
        ----------
        by: str or list of str
            Name or list of names to sort the _data by.
        axis: Integer, None by default, {index (0), columns (1)}
            if set to 0 or'index', will count for each column.
            if set to 1 or'columns', will count for each row
        ascending: boolean or list of beoolean, default True.
            Sort ascending vs. descending. Specify list for multiple sort orders.
            If this is a list of bools, must match the length of the by.
        inplace: Boolean, optional(False by default)
            if set to true the original dataframe will be altered, the duplicates will be dropped in place,
            otherwise the operation will be made in a copy, that will be returned.
        kind: String, optional ('quicksort' by default), options: ‘quicksort’, ‘mergesort’, ‘heapsort’.
            Choice of sorting algorithm.
            For DataFrames, this option is only applied when sorting on a single column or label.
        na_position: String, optional('last' by default), options: ‘first’, ‘last’.
            If 'first' puts NaNs at the beginning;
            If last puts NaNs at the end.

        Returns
        -------
        PandasDataframe or None
            Object with sorted values or None if ``inplace=True``.

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_values.html
        """
        operation = begin_operation('sort_values')
        _sort_values = self._data.sort_values(by, axis, ascending, inplace, kind, na_position)

        if inplace:
            self.last_operation = end_operation(operation)
            return None
        self.last_operation = end_operation(operation)
        return PandasMoveDataFrame(data=_sort_values)

    def reset_index(self, level=None, drop=False, inplace=False, col_level=0, col_fill=''):
        """Resets the DataFrame's index, and use the default one.
        One or more levels can be removed, if the DataFrame has a MultiIndex.

        Parameters
        ----------
        level: int, str, tuple, or list. Optional (None by default)
            Only the levels specify will be removed from the index. If set to None, all levels are removed.
        drop: boolean, optional (False by default)
            Do not try to insert index into dataframe columns. This resets the index to the default integer index.
        inplace: bool, optional(False by default)
            Modify the DataFrame in place (do not create a new object).
        col_level: int or str, default 0
            If the columns have multiple levels, determines which level the labels are inserted into.
            By default it is inserted into the first level..
        col_fill: object, default ‘’
            If the columns have multiple levels, determines how the other levels are named.
            If None then the index name is repeated.

        Returns
        -------
        PandasMoveDataFrame or None
            Object with a new index or None if ``inplace=True``.

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.reset_index.html
        """
        operation = begin_operation('reset_index')
        _reset_index = self._data.reset_index(level, drop, inplace, col_level, col_fill)

        if inplace:
            self.last_operation = end_operation(operation)
            return None
        self.last_operation = end_operation(operation)
        return PandasMoveDataFrame(data=_reset_index)

    def set_index(self, keys, drop=True, append=False, inplace=False, verify_integrity=False):
        """Set the DataFrame index (row labels) using one or more existing columns or arrays (of the correct length).

        Parameters
        ----------
        keys: str or array of String.
            label or array-like or list of labels/arrays. This parameter can be either a single column key, a single
            array of the same length as the calling DataFrame, or a list containing an arbitrary combination of
            column keys and arrays.
        drop: bool, optional (True by defautl)
            Delete columns to be used as the new index.
        append: bool, optional (False by defautl)
            Whether to append columns to existing index.
        inplace: bool, optional (False by defautl)
            Modify the DataFrame in place (do not create a new object).
        verify_integrity: bool, optional (False by defautl)
            Check the new index for duplicates. Otherwise defer the check until necessary.
            Setting to False will improve the performance of this method.

        Returns
        -------
            DataFrame
                _data with changed row labels.

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.set_index.html
        """
        if inplace and drop:
            if type(keys) == str:
                aux = set([keys])
            else:
                aux = set(keys)
            columns = set(['lat','lon','datetime'])
            if aux & columns:
                raise AttributeError("Could not change lat, lon, and datetime type.")

        operation = begin_operation('set_index')
        _set_index = self._data.set_index(keys, drop, append, inplace, verify_integrity)
        self.last_operation = end_operation(operation)

        if (isinstance(_set_index, pd.DataFrame) and self._has_columns(_set_index)):
            return PandasMoveDataFrame(_set_index)
        return _set_index

    def drop(self, labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise'):
        """Remove rows or columns by specifying label names and corresponding axis, or by specifying directly index or
           column names. When using a multi-index, labels on different levels can be removed by specifying the level.

        Parameters
        ----------
        labels: str or array of String
            Index or column labels to drop.
        axis: str or int, optional (0 by default)
            Whether to drop labels from the index (0 or ‘index’) or columns (1 or ‘columns’).
        index: str or array of string, optional (None by defautl)
            Alternative to specifying axis (labels, axis=0 is equivalent to index=labels).
        columns: str or array of string, optional (None by defautl)
            Alternative to specifying axis (labels, axis=1 is equivalent to columns=labels).
        level: int or str, optional (None by defautl)
            For MultiIndex, level from which the labels will be removed.
        inplace: bool, optional (False by defautl)
            If True, do operation inplace and return None. Otherwise, make a copy, do operations and return.
        errors: str, optional ('raise' by default), options: ‘ignore’, ‘raise’.
            If ‘ignore’, suppress error and only existing labels are dropped.

        Returns
        -------
            DataFrame
                _data without the removed index or column labels.

        Raises
        ------
            KeyError
            If any of the labels is not found in the selected axis.

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html
        """
        
        if inplace:
            _labels1 = set()
            _labels2 = set()
            if labels != None:
                if type(labels) == str:
                    _labels1 = set([labels])
                else:
                    _labels1 = set(labels)
            elif columns != None:
                if type(columns) == str:
                    _labels2 = set([columns])
                else:
                    _labels2 = set(columns)
            _columns = set(['lat', 'lon', 'datetime'])
            if (axis==1 or axis=='columns' or columns) and (_labels1.union(_labels2) & _columns):
                raise AttributeError("Could not drop columns lat, lon, and datetime.")
     
        operation = begin_operation('drop')
        _drop = self._data.drop(labels, axis, index, columns, level, inplace, errors)
        self.last_operation = end_operation(operation)

        if (isinstance(_drop, pd.DataFrame)) and self._has_columns(_drop):
            return PandasMoveDataFrame(_drop)
        return _drop

    def duplicated(self, subset=None, keep='first'):
        """Returns boolean Series denoting duplicate rows, optionally only considering certain columns.

        Parameters
        ----------
        subset: str, array of str, optional (None by default)
            Only consider certain columns for identifying duplicates, by default use all of the columns
        keep: str, optional ('first' by default), options: first’, ‘last’, False.
            first : Mark duplicates as True except for the first occurrence.
            last : Mark duplicates as True except for the last occurrence.
            False : Mark all duplicates as True.

        Returns
        -------
            Series denoting duplicated rows.

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.duplicated.html
        """
        operation = begin_operation('duplicated')
        _duplicated = self._data.duplicated(subset, keep)
        self.last_operation = end_operation(operation)

        return _duplicated

    def drop_duplicates(self, subset=None, keep='first', inplace=False):
        """Uses the pandas's function drop_duplicates, to remove duplicated rows from data.

        Parameters
        ----------
        subset: int or str, optional (None by default)
            Only consider certain columns for identifying duplicates, by default use all of the columns
        keep: str, optional('first' by default)
            - first : Drop duplicates except for the first occurrence.
            - last : Drop duplicates except for the last occurrence.
            - False : Drop all duplicates.
        Inplace: bool, optional(False by default)
            Whether to drop duplicates in place or to return a copy

        Returns
        --------
        PandasMoveDataFrame or None
            Object with duplicated rows or None if ``inplace=True``.

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html
        """
        operation = begin_operation('drop_duplicates')
        _drop_duplicates = self._data.drop_duplicates(subset, keep, inplace)

        if inplace:
            self.last_operation = end_operation(operation)
            return None

        self.last_operation = end_operation(operation)
        return PandasMoveDataFrame(data=_drop_duplicates)

    def shift(self, periods=1, freq=None, axis=0, fill_value=None):
        """Shift index by desired number of periods with an optional time freq.

        Parameters
        ----------
        periods: int, optional (1 by default)
            Number of periods to shift. Can be positive or negative.
        freq: pandas.DateOffset, pandas.Timedelta or string, optional(None by default).
            Offset to use from the tseries module or time rule (e.g. ‘EOM’). If freq is specified then the index values
            are shifted but the data is not realigned. That is, use freq if you would like to extend the index when
            shifting and preserve the original data. When freq is not passed, shift the index without realigning the
            data. If freq is passed (in this case, the index must be date or datetime, or it will raise a
            NotImplementedError), the index will be increased using the periods and the freq.
        axis: int, str, optional (0 by default), options: 0 or ‘index’, 1 or ‘columns’, None
            Shift direction.
        fill_value: object, optional(None by default)
            The scalar value to use for newly introduced missing values. the default depends on the dtype of self.
            For numeric data, np.nan is used. For datetime, timedelta, or period data, etc.
            NaT is used. For extension dtypes, self.dtype.na_value is used.

        Returns
        -------
            PandasMoveDataFrame
                A copy of the original object, shifed.

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.shift.html
        """
        operation = begin_operation('shift')
        _shift = self._data.shift(periods, freq, axis, fill_value)
        _shift = PandasMoveDataFrame(data=_shift)
        self.last_operation = end_operation(operation)

        return PandasMoveDataFrame(data=_shift)

    def all(self, axis=0, bool_only=None, skipna=True, level=None, **kwargs):
        """Inidicates if all elements are True, potentially over an axis.
        Returns True unless there at least one element within the Dataframe axis that is False or equivalent.

        Prameters
        ---------
        axis: int or str, optional (0 by default), options: 0 or ‘index’, 1 or ‘columns’, None
            Indicate which axis or axes should be reduced.
            - 0 / ‘index’ : reduce the index, return a Series whose index is the original column labels.
            - 1 / ‘columns’ : reduce the columns, return a Series whose index is the original index.
            - None : reduce all axes, return a scalar.
        bool_only: bool, optional (None by defautl)
            Include only boolean columns.
            If None, will attempt to use everything, then use only boolean data
        skipna: bool, optional (True by defautl)
            Exclude NA/null values. If the entire row/column is NA and skipna is True, then the result will be True,
            as for an empty row/column. If skipna is False, then NA are treated as True,
            because these are not equal to zero.
        level: int or String(level name), optional (default None)
            If the axis is a MultiIndex (hierarchical), count along a particular level, collapsing into a Series.
        kwargs: any, default None
            Additional keywords have no effect but might be accepted for compatibility with NumPy.

        Returns
        -------
        Series or DataFrame
            If level is specified, then, DataFrame is returned; otherwise, Series is returned.

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.all.html
        """
        operation = begin_operation('all')
        _all = self._data.all(axis, bool_only, skipna, level, **kwargs)
        self.last_operation = end_operation(operation)

        return _all

    def any(self, axis=0, bool_only=None, skipna=True, level=None, **kwargs):
        """Inidicates if any element is True, potentially over an axis.
        Returns False unless there at least one element within the Dataframe axis that is True or equivalent.

        Prameters
        ---------
        axis: int or str, optional (0 by default), options: 0 or ‘index’, 1 or ‘columns’, None
            Indicate which axis or axes should be reduced.
            - 0 / ‘index’ : reduce the index, return a Series whose index is the original column labels.
            - 1 / ‘columns’ : reduce the columns, return a Series whose index is the original index.
            - None : reduce all axes, return a scalar.
        bool_only: bool, optional (None by defautl)
            Include only boolean columns.
            If None, will attempt to use everything, then use only boolean data
        skipna: bool, optional (True by defautl)
            Exclude NA/null values. If the entire row/column is NA and skipna is True, then the result will be False,
            as for an empty row/column. If skipna is False, then NA are treated as True,
            because these are not equal to zero.
        level: int or String(level name), optional (default None)
            If the axis is a MultiIndex (hierarchical), count along a particular level, collapsing into a Series.
        kwargs: any, default None
            Additional keywords have no effect but might be accepted for compatibility with NumPy.

        Returns
        -------
        Series or DataFrame
            If level is specified, then, DataFrame is returned; otherwise, Series is returned.

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.any.html
        """
        operation = begin_operation('any')
        _any = self._data.any(axis, bool_only, skipna, level, **kwargs)
        self.last_operation = end_operation(operation)

        return _any

    def isna(self):
        """Detect missing values.

        Return a boolean same-sized object indicating if the values are NA.
        NA values, such as None or numpy.NaN, gets mapped to True values.
        Everything else gets mapped to False values.
        Characters such as empty strings '' or numpy.inf are not considered NA values 
        (unless you set pandas.options.mode.use_inf_as_na = True).

        Returns
        -------
        DataFrame:
            DataFrame of booleans showing for each element in DataFrame that indicates 
            whether an element is not an NA value.
        """
        operation = begin_operation('isna')
        _isna = self._data.isna()
        self.last_operation = end_operation(operation)

        return _isna

    def fillna(self, value=None, method=None, axis=None, inplace=False, limit=None, downcast=None):
        """Fill NA/NaN values using the specified method.

        Parameters
        ----------
        value : scalar, dict, Series, or DataFrame
            Value to use to fill holes (e.g. 0), alternately a
            dict/Series/DataFrame of values specifying which value to use for
            each index (for a Series) or column (for a DataFrame).  Values not
            in the dict/Series/DataFrame will not be filled. This value cannot
            be a list.
        method : {'backfill', 'bfill', 'pad', 'ffill', None}, default None
            Method to use for filling holes in reindexed Series
            pad / ffill: propagate last valid observation forward to next valid
            backfill / bfill: use next valid observation to fill gap.
        axis : {0 or 'index', 1 or 'columns'}
            Axis along which to fill missing values.
        inplace : bool, default False
            If True, fill in-place. Note: this will modify any
            other views on this object (e.g., a no-copy slice for a column in a
            DataFrame).
        limit : int, default None
            If method is specified, this is the maximum number of consecutive
            NaN values to forward/backward fill. In other words, if there is
            a gap with more than this number of consecutive NaNs, it will only
            be partially filled. If method is not specified, this is the
            maximum number of entries along the entire axis where NaNs will be
            filled. Must be greater than 0 if not None.
        downcast : dict, default is None
            A dict of item->dtype of what to downcast if possible,
            or the string 'infer' which will try to downcast to an appropriate
            equal type (e.g. float64 to int64 if possible).

        Returns
        -------
        PandasMoveDataFrame or None
            Object with missing values filled or None if ``inplace=True``.

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html
        """
        operation = begin_operation('fillna')
        _fillna = self._data.fillna(value, method, axis, inplace, limit, downcast)

        if inplace:
            self.last_operation = end_operation(operation)
            return None

        self.last_operation = end_operation(operation)
        return PandasMoveDataFrame(data=_fillna)

    def dropna(self, axis=0, how='any', thresh=None, subset=None, inplace=False):
        """Removes missing data.

        Prameters
        ---------
        axis: int or str, optional (0 by default), options: 0 or ‘index’, 1 or ‘columns’, None
            Determine if rows or columns which contain missing values are removed.
            - 0, or ‘index’ : Drop rows which contain missing values.
            - 1, or ‘columns’ : Drop columns which contain missing value.
        how: str, optional ('any' by default), options: ‘any’, ‘all’
            Determine if row or column is removed from DataFrame, when we have at least one NA or all NA.
               - ‘any’ : If any NA values are present, drop that row or column.
               - ‘all’ : If all values are NA, drop that row or column.
        thresh: int, optional (None by default)
            Require that many non-NA values.
        subset: array-like, optional (None by default)
            Labels along other axis to consider,
            e.g. if you are dropping rows these would be a list of columns to include.
        inplace: bool, optional (default False)
            If True, do operation inplace and return None

        Returns
        -------
        PandasMoveDataFrame or None
            Object with NA entries dropped or None if ``inplace=True``.

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dropna.html
        """
        
        if inplace:
            if (axis==1 or axis=='columns'):
                columns = ['lat', 'lon', 'datetime']
                data = self._data[columns]
                if data.isnull().values.any():
                    raise AttributeError("Could not drop columns lat, lon, and datetime.")

        operation = begin_operation('dropna')
        _dropna = self._data.dropna(axis, how, thresh, subset, inplace)
        self.last_operation = end_operation(operation)

        if (isinstance(_dropna, pd.DataFrame) and self._has_columns(_dropna)):
            return PandasMoveDataFrame(_dropna)
        return _dropna

    def sample(self, n=None, frac=None, replace=False, weights=None, random_state=None, axis=None):
        """Return a random sample of items from an axis of object.

        You can use `random_state` for reproducibility.

        Parameters
        ----------
        n : int, optional
            Number of items from axis to return. Cannot be used with `frac`.
            Default = 1 if `frac` = None.
        frac : float, optional
            Fraction of axis items to return. Cannot be used with `n`.
        replace : bool, default False
            Allow or disallow sampling of the same row more than once.
        weights : str or ndarray-like, optional
            Default 'None' results in equal probability weighting.
            If passed a Series, will align with target object on index. Index
            values in weights not found in sampled object will be ignored and
            index values in sampled object not in weights will be assigned
            weights of zero.
            If called on a DataFrame, will accept the name of a column
            when axis = 0.
            Unless weights are a Series, weights must be same length as axis
            being sampled.
            If weights do not sum to 1, they will be normalized to sum to 1.
            Missing values in the weights column will be treated as zero.
            Infinite values not allowed.
        random_state : int or numpy.random.RandomState, optional
            Seed for the random number generator (if int), or numpy RandomState
            object.
        axis : {0 or ‘index’, 1 or ‘columns’, None}, default None
            Axis to sample. Accepts axis number or name. Default is stat axis
            for given data type (0 for Series and DataFrames).

        Returns
        -------
        PandasMoveDataFrame
            A new object of same type as caller containing `n` items randomly
            sampled from the caller object.

        See Also
        --------
        numpy.random.choice: Generates a random sample from a given 1-D numpy
            array.

        Notes
        -----
        If `frac` > 1, `replacement` should be set to `True`.

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sample.html
        """
        operation = begin_operation('sample')
        _sample = self._data.sample(n, frac, replace, weights, random_state, axis)
        _sample = PandasMoveDataFrame(data=_sample)
        self.last_operation = end_operation(operation)

        return PandasMoveDataFrame(data=_sample)

    def isin(self, values):
        """Determines whether each element in the DataFrame is contained in values.

        values : iterable, Series, DataFrame or dict
            The result will only be true at a location if all the labels match. If values is a Series,
            that’s the index. If values is a dict, the keys must be the column names, which must match.
            If values is a DataFrame, then both the index and column labels must match.

        Returns
        -------
        DataFrame:
            DataFrame of booleans showing whether each element in the DataFrame is contained in values

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.isin.html
        """
        operation = begin_operation('isin')
        _isin = self._data.isin(values)
        self.last_operation = end_operation(operation)

        return _isin

    def append(self, other, ignore_index=False, verify_integrity=False, sort=None):
        """Append rows of other to the end of caller, returning a new object.
        Columns in other that are not in the caller are added as new columns

        Parameters
        ----------
        other : DataFrame or Series/dict-like object, or list of these
            The data to append.
        ignore_index : bool, optional(False by default)
            If True, do not use the index labels.
        verify_integrity : bool, optional (False by default)
            If True, raise ValueError on creating index with duplicates.
        sort : bool, optional (None by default)
            Sort columns if the columns of self and other are not aligned. The default sorting is deprecated and will
            change to not-sorting in a future version of pandas. Explicitly pass sort=True to silence the warning and
            sort. Explicitly pass sort=False to silence the warning and not sort.

        Returns
        -------
        PandasMoveDataFrame
            A dataframe containing rows from both the caller and `other`.

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.append.html
        """
        operation = begin_operation('append')

        if isinstance(other, PandasMoveDataFrame):
            other = other._data
            
        _append = self._data.append(other, ignore_index, verify_integrity, sort)
        _append = PandasMoveDataFrame(data=_append)
        self.last_operation = end_operation(operation)

        return _append

    def join(self, other, on=None, how='left', lsuffix='', rsuffix='', sort=False):
        """Join columns of other, returning a new object.

        Join columns with `other` PandasMoveDataFrame either on index or on a key
        column. Efficiently join multiple DataFrame objects by index at once by
        passing a list.

        Parameters
        ----------
        other : DataFrame, Series, or list of DataFrame
            Index should be similar to one of the columns in this one. If a
            Series is passed, its name attribute must be set, and that will be
            used as the column name in the resulting joined DataFrame.
        on : str, list of str, or array-like, optional
            Column or index level name(s) in the caller to join on the index
            in `other`, otherwise joins index-on-index. If multiple
            values given, the `other` DataFrame must have a MultiIndex. Can
            pass an array as the join key if it is not already contained in
            the calling DataFrame. Like an Excel VLOOKUP operation.
        how : {'left', 'right', 'outer', 'inner'}, default 'left'
            How to handle the operation of the two objects.

            * left: use calling frame's index (or column if on is specified)
            * right: use `other`'s index.
            * outer: form union of calling frame's index (or column if on is
            specified) with `other`'s index, and sort it.
            lexicographically.
            * inner: form intersection of calling frame's index (or column if
            on is specified) with `other`'s index, preserving the order
            of the calling's one.
        lsuffix : str, default ''
            Suffix to use from left frame's overlapping columns.
        rsuffix : str, default ''
            Suffix to use from right frame's overlapping columns.
        sort : bool, default False
            Order result DataFrame lexicographically by the join key. If False,
            the order of the join key depends on the join type (how keyword).

        Returns
        -------
        PandasMoveDataFrame
            A dataframe containing columns from both the caller and `other`.

        Notes
        -----
        Parameters `on`, `lsuffix`, and `rsuffix` are not supported when
        passing a list of `DataFrame` objects.
        
        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.join.html
        """
        operation = begin_operation('join')

        if isinstance(other, PandasMoveDataFrame):
            other = other._data
            
        _join = self._data.join(other, on, how, lsuffix, rsuffix, sort)
        _join = PandasMoveDataFrame(data=_join) 
        self.last_operation = end_operation(operation)
        
        return _join

    def nunique(self, axis=0, dropna=True):
        """Count distinct observations over requested axis.

        Parameters
        ----------
         axis : int or str, optional (0 by default), options: 0 or ‘index’, 1 or ‘columns’, None
            The axis to use. 0 or ‘index’ for row-wise, 1 or ‘columns’ for column-wise.
        dropna : bool, optional (default True)
            Don’t include NaN in the counts.

        Returns
        -------
        Series
            Return Series with number of distinct observations

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.unique.html
        """
        operation = begin_operation('nunique')
        _nunique = self._data.nunique(axis, dropna)
        self.last_operation = end_operation(operation)

        return _nunique

    def write_file(self, file_name, separator=','):
        """
        Write trajectory data to a new file.

        Parameters
        ----------
        file_name : String.
            Represents the filename.

        separator : String, optional, default ','.
            Represents the informations separator in a new file.

        Returns
        -------

        """
        operation = begin_operation('write_file')
        self._data.to_csv(file_name, sep=separator, encoding='utf-8', index=False)
        self.last_operation = end_operation(operation)

    def to_csv(self, file_name, sep=',', index=True, encoding=None):
        """Write object to a comma-separated values (csv) file.

        Parameters
        ----------
        file_name: str
            File path or object
        sep: str
            String of length 1. Field delimiter for the output file.
        index: bool
            Boolean indicating whether to save row indexes
        encoding: str, optional (None default)
            A string representing the encoding to use in the output file, defaults to ‘utf-8’

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html
        """
        operation = begin_operation('to_csv')
        self._data.to_csv(file_name, sep=sep, index=index, encoding=encoding)
        self.last_operation = end_operation(operation)

    def convert_to(self, new_type):
        """Convert an object from one type to another specified by the user.

        Parameters
        ----------
        new_type: str
            The type for which the object will be converted.

        Returns
        -------
        A subclass of MoveDataFrameAbstractModel
            The converted object.
        """
        operation = begin_operation('convet_to')

        if (new_type == "dask"):
            _dask = DaskMoveDataFrame(
                                      self._data,
                                      latitude=LATITUDE,
                                      longitude=LONGITUDE,
                                      datetime=DATETIME,
                                      traj_id=TRAJ_ID,
                                      n_partitions=1)
            self.last_operation = end_operation(operation)
            return _dask
        elif (new_type == "pandas"):
            self.last_operation = end_operation(operation)
            return self

    def get_type(self):
        """Returns the type of the object.

        Returns
        -------
        A string representing the type of the object.
        """
        operation = begin_operation('get_type')
        type_ = self._type
        self.last_operation = end_operation(operation)
        return type_


class DaskMoveDataFrame(DataFrame, MoveDataFrameAbstractModel):
    def __init__(self, data, latitude=LATITUDE, longitude=LONGITUDE, datetime=DATETIME, traj_id=TRAJ_ID,
                 n_partitions=1):
        # formatar os labels que foram return 0ados pro que usado no pymove -> format_labels
        # renomeia as colunas do dado return 0ado pelo novo dict
        # cria o dataframe
        # AttributeError if the data doesn't contains one of the columns LATITUDE, LONGITUDE, DATETIME
        mapping_columns = format_labels(data, traj_id, latitude, longitude, datetime)
        dsk = data.rename(columns=mapping_columns)

        if self._has_columns(dsk):
            self._validate_move_data_frame(dsk)
            self._data = dask.dataframe.from_pandas(dsk, npartitions=n_partitions)
            self._type = TYPE_DASK
            self.last_operation = None
        else:
            raise AttributeError("Could not instantiate new MoveDataFrame because data has missing columns")

    @staticmethod
    def _has_columns(data):
        """
        Checks whether past data has 'lat', 'lon', 'datetime' columns.

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

    @staticmethod
    def _validate_move_data_frame(data):
        """
        Converts the column type to the default PyMove lib used.

        Parameters
        ----------
        data : dict, list, numpy array or pandas.core.DataFrame.
            Input trajectory data.

        Returns
        -------
        """
        try:
            if data.dtypes.lat != 'float32':
                data.lat.astype('float32')
            if data.dtypes.lon != 'float32':
                data.lon.astype('float32')
            if data.dtypes.datetime != 'datetime64[ns]':
                data.lon.astype('datetime64[ns]')
        except AttributeError as erro:
            print(erro)

    @property
    def lat(self):
        """
        Checks for the 'lat' column and returns its value.

        """
        if LATITUDE not in self:
            raise AttributeError("The MoveDataFrame does not contain the column '%s.'" % LATITUDE)
        return self[LATITUDE]

    @property
    def lng(self):
        """
        Checks for the 'lon' column and returns its value.

        """
        if LONGITUDE not in self:
            raise AttributeError("The MoveDataFrame does not contain the column '%s.'" % LONGITUDE)
        return self[LONGITUDE]

    @property
    def datetime(self):
        """
        Checks for the 'datetime' column and returns its value

        """
        if DATETIME not in self:
            raise AttributeError("The MoveDataFrame does not contain the column '%s.'" % DATETIME)
        return self[DATETIME]

    @property
    def loc(self):
        # Access a group of rows and columns by label(s) or a boolean array.
        raise NotImplementedError("To be implemented")
    
    @property
    def iloc(self):
        # Purely integer-location based indexing for selection by position.
        raise NotImplementedError("To be implemented")
    
    @property
    def at(self):
        # Access a single value for a row/column label pair.
        raise NotImplementedError("To be implemented")

    @property
    def values(self):
        # Return a Numpy representation of the DataFrame.
        raise NotImplementedError("To be implemented")
    
    @property
    def columns(self):
        # The column labels of the DataFrame.
        raise NotImplementedError("To be implemented")
    
    @property
    def index(self):
        # The index (row labels) of the DataFrame.
        raise NotImplementedError("To be implemented")
    
    @property
    def dtypes(self):
        # Return the dtypes in the DataFrame.
        raise NotImplementedError("To be implemented")
    
    @property
    def shape(self):
        # Return a tuple representing the dimensionality of the DataFrame.
        raise NotImplementedError("To be implemented")

    def len(self):
        """
        Returns the length/row numbers in trajectory data.

        Parameters
        ---------

        Returns
        -------
        int
            Represents the trajectory data length.

        """
        raise NotImplementedError("To be implemented")

    def unique(self):
        """
        Return unique values of Series object. Uniques are returned in order of appearance. Hash table-based unique,
        therefore does NOT sort.

        Returns
        -------
        """
        raise NotImplementedError("To be implemented")

    def head(self, n=5, npartitions=1, compute=True):
        """
        Return the first n rows.

        This function returns the first n rows for the object based on position. It is useful for quickly testing if
        your object has the right type of data in it.

        Parameters
        ----------
        n : int, optional, default 5
            Number of rows to select.

        npartitions : int, optional, default 1.
            Represents the number partitions.

        compute : bool, optional, default True.
            Represents ?

        Returns
        -------
        same type as caller
            The first n rows of the caller object.

        """
        return self._data.head(n, npartitions, compute)

    def get_users_number(self):
        """
        Check and return number of users in trajectory data.

        Parameters
        ----------

        Returns
        -------
        int
            Represents the number of users in trajectory data.

        """
        raise NotImplementedError("To be implemented")

    def to_numpy(self):
        """
        Converts trajectory data to numpy array format.

        Parameters
        ----------

        Returns
        -------
        np.array
            Represents the trajectory in numpy array format.

        """
        raise NotImplementedError("To be implemented")

    def to_dict(self):
        """
        Converts trajectory data to dict format.

        Parameters
        ----------

        Returns
        -------
        dict
            Represents the trajectory in dict format.

        """
        raise NotImplementedError("To be implemented")

    def to_grid(self):
        """
        Converts trajectory data to grid format.

        Parameters
        ----------

        Returns
        -------
        pymove.core.grid
            Represents the trajectory in grid format.

        """
        raise NotImplementedError("To be implemented")

    def to_DataFrame(self):
        """
        Converts trajectory data to DataFrame format.

        Parameters
        ----------

        Returns
        -------
        dask.dataframe.DataFrame
            Represents the trajectory in DataFrame format.

        """
        self._last_operation_time_duration = 0
        self._last_operation_name = 'to_DataFrame'
        self._last_operation_mem_usage = 0
        return self._data

    def generate_tid_based_on_id_datatime(self):
        """
        Create or update trajectory id based on id e datetime.

        Parameters
        ----------

        Returns
        -------

        """
        raise NotImplementedError("To be implemented")

    def generate_date_features(self):
        """
        Create or update date feature.

        Parameters
        ----------

        Returns
        -------

        """
        raise NotImplementedError("To be implemented")

    def generate_hour_features(self):
        """
        Create or update hour feature.

        Parameters
        ----------

        Returns
        -------

        """
        raise NotImplementedError("To be implemented")

    def generate_day_of_the_week_features(self):
        """
        Create or update a feature day of the week from datatime.

        Parameters
        ----------

        Returns
        -------

        """
        raise NotImplementedError("To be implemented")

    def generate_weekend_features(self):
        """
        Create or update the feature weekend to the dataframe, if this resource is set to 1 it indicates that the
        given day is is the weekend, otherwise, it is a day of the week.

        Parameters
        ----------
        
        Returns
        ----------
        """
        raise NotImplementedError("To be implemented")

    def generate_time_of_day_features(self):
        """
        Create a feature time of day or period from datatime.

        Parameters
        ----------

        Returns
        -------

        Examples
        --------
        - datetime1 = 2019-04-28 02:00:56 -> period = early morning
        - datetime2 = 2019-04-28 08:00:56 -> period = morning
        - datetime3 = 2019-04-28 14:00:56 -> period = afternoon
        - datetime4 = 2019-04-28 20:00:56 -> period = evening

        """
        raise NotImplementedError("To be implemented")

    def generate_datetime_in_format_cyclical(self):
        """
        Create or update column with cyclical datetime feature.

        Parameters
        ----------
        
        Returns
        -------
        
        References
        ----------
        # https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/
        # https://www.avanwyk.com/encoding-cyclical-features-for-deep-learning/
        """
        raise NotImplementedError("To be implemented")

    def generate_dist_features(self):
        """
        Create the three distance in meters to an GPS point P (lat, lon).

        Parameters
        ----------

        Returns
        -------

        Examples
        --------
        Example:    P to P.next = 2 meters
                    P to P.previous = 1 meter
                    P.previous to P.next = 1 meters

        """
        raise NotImplementedError("To be implemented")

    def generate_dist_time_speed_features(self):
        """
        Firstly, create three distance to an GPS point P (lat, lon).
        After, create two feature to time between two P: time to previous and time to next.
        Lastly, create two feature to speed using time and distance features.

        Parameters
        ----------

        Returns
        -------

        Examples
        --------
        Example:    dist_to_prev =  248.33 meters, dist_to_prev 536.57 meters
                    time_to_prev = 60 seconds, time_prev = 60.0 seconds
                    speed_to_prev = 4.13 m/s, speed_prev = 8.94 m/s.

        """
        raise NotImplementedError("To be implemented")

    def generate_move_and_stop_by_radius(self):
        """
        Create or update column with move and stop points by radius.

        Parameters
        ----------

        Returns
        -------

        """
        raise NotImplementedError("To be implemented")

    def time_interval(self):
        """
        Get time difference between max and min datetime in trajectory data.

        Parameters
        ----------

        Returns
        -------

        """
        raise NotImplementedError("To be implemented")

    def get_bbox(self):
        """
        A bounding box (usually shortened to bbox) is an area defined by two longitudes and two latitudes, where:
            - Latitude is a decimal number between -90.0 and 90.0.
            - Longitude is a decimal number between -180.0 and 180.0.
        They usually follow the standard format of:
        - bbox = left, bottom, right, top
        - bbox = min Longitude , min Latitude , max Longitude , max Latitude
        Parameters
        ----------

        Returns
        -------
        tuple
            Represents a bound box, that is a tuple of 4 values with the min and max limits of latitude e longitude.

        Examples
        --------
        (22.147577, 113.54884299999999, 41.132062, 121.156224)

        """
        raise NotImplementedError("To be implemented")

    def plot_all_features(self):
        """
        Generate a visualization for each columns that type is equal dtype.

        Parameters
        ----------

        Returns
        -------
        """
        raise NotImplementedError("To be implemented")

    def plot_trajs(self):
        #Generate a visualization that show trajectories.
        raise NotImplementedError("To be implemented")

    def plot_traj_id(self):
        # Generate a visualization that shows a trajectory with the specified tid.
        raise NotImplementedError("To be implemented")

    def show_trajectories_info(self):
        # Show dataset information from dataframe, this is number of rows, datetime interval, and bounding box.
        raise NotImplementedError("To be implemented")

    def min(self, axis=None, skipna=True, split_every=False, out=None):
        """Return the minimum of the values for the requested axis..

        Parameters
        ----------
        axis: int, None by default, {index (0), columns (1)}.
            Axis for the function to be applied on.
        skipna: bool, optional, default None.
            Exclude NA/null values when computing the result.
        split_every:
        out:

        Returns
        -------
        max:Series or DataFrame (if level specified)
            The minimum values for the request axis.

        References
        ----------
        https://docs.dask.org/en/latest/dataframe-api.html#dask.dataframe.DataFrame.min
        """
        return self._data.min(axis, skipna, split_every, out)

    def max(self, axis=None, skipna=True, split_every=False, out=None):
        """Return the maximum of the values for the requested axis..

        Parameters
        ----------
        axis: int, None by default, {index (0), columns (1)}.
            Axis for the function to be applied on.
        skipna: bool, optional, default None.
            Exclude NA/null values when computing the result.
        split_every:
        out:

        Returns
        -------
        max:Series or DataFrame (if level specified)
            The maximum values for the request axis.

        References
        ----------
        https://docs.dask.org/en/latest/dataframe-api.html#dask.dataframe.DataFrame.max
        """
        return self._data.max(axis, skipna, split_every, out)

    def count(self):
        # CountS the non-NA cells for each column or row.
        raise NotImplementedError("To be implemented")

    def groupby(self, by=None, **kwargs):
        """Groups dask DataFrame using a mapper or by a Series of columns.

        Parameters
        ----------
        by : mapping, function, label, or list of labels, optional(None by default)
            Used to determine the groups for the groupby.
        **kwargs
            Optional, only accepts keyword argument ‘mutated’ and is passed to groupby.

        Returns
        -------
        DataFrameGroupBy:
            Returns groupby object that contains information about the groups.

        References
        ----------
        https://docs.dask.org/en/latest/dataframe-api.html#dask.dataframe.DataFrame.groupby
        """

        return self._data.groupby(by)

    def plot(self):
        # Plot the data of the dask DataFrame.
        raise NotImplementedError("To be implemented")

    def select_dtypes(self):
        # Returns a subset of the dask DataFrame columns based on the column dtypes.
        raise NotImplementedError("To be implemented")

    def astype(self):
        # Casts a dask object to a specified dtype.
        raise NotImplementedError("To be implemented")

    def sort_values(self):
        # Sorts the values of the dask DataFrame.
        raise NotImplementedError("To be implemented")

    def reset_index(self):
        # Resets the dask DataFrame's index, and use the default one.
        raise NotImplementedError("To be implemented")

    def set_index(self):
        # Set the dask DataFrame index (row labels) using one or more existing columns or arrays (of the correct length).
        raise NotImplementedError("To be implemented")

    def drop(self):
        # Drops specified rows or columns of the dask Dataframe.
        raise NotImplementedError("To be implemented")

    def duplicated(self):
        # Returns boolean Series denoting duplicate rows, optionally only considering certain columns.
        raise NotImplementedError("To be implemented")

    def drop_duplicates(self):
        # Removes duplicated rows from the data.
        raise NotImplementedError("To be implemented")

    def shift(self):
        # Shifts dask dataFrame index by desired number of periods with an optional time freq.
        raise NotImplementedError("To be implemented")

    def all(self):
        # Inidicates if all elements are True, potentially over an axis.
        # Returns True unless there at least one element within the Dataframe axis that is False or equivalent.
        raise NotImplementedError("To be implemented")

    def any(self):
        # Inidicates if any element is True, potentially over an axis.
        # Returns False unless there at least one element within the dask Dataframe axis that is True or equivalent.
        raise NotImplementedError("To be implemented")

    def isna(self):
        # Detect missing values
        raise NotImplementedError("To be implemented")

    def fillna(self):
        # Fills missing data in the dask DataFrame
        raise NotImplementedError("To be implemented")

    def dropna(self):
        # Removes missing data from dask DataFrame.
        raise NotImplementedError("To be implemented")

    def sample(self):
        # Samples data from the dask DataFrame
        raise NotImplementedError("To be implemented")

    def isin(self):
        # Determines whether each element in the dask DataFrame is contained in values.
        raise NotImplementedError("To be implemented")

    def append(self):
        # Append rows of other to the end of caller, returning a new object.
        # Columns in other that are not in the caller are added as new columns
        raise NotImplementedError("To be implemented")

    def join(self):
        # Join columns of another DataFrame.
        raise NotImplementedError("To be implemented")

    def nunique(self):
        # Count distinct observations over requested axis.
        raise NotImplementedError("To be implemented")

    def write_file(self):
        """
        Write trajectory data to a new file.

        Parameters
        ----------

        Returns
        -------

        """
        raise NotImplementedError("To be implemented")

    def to_csv(self):
        # Write object to a comma-separated values (csv) file.
        raise NotImplementedError("To be implemented")

    def convert_to(self, new_type):
        """Convert an object from one type to another specified by the user.

        Parameters
        ----------
        new_type: str
            The type for which the object will be converted.

        Returns
        -------
        A subclass of MoveDataFrameAbstractModel
            The converted object.
        """

        if (new_type == "dask"):
            return self
        elif (new_type == "pandas"):
            df_pandas = self._data.compute()
            return PandasMoveDataFrame(df_pandas, latitude=LATITUDE, longitude=LONGITUDE, datetime=DATETIME, traj_id=TRAJ_ID)

    def get_type(self):
        """Returns the type of the object.

        Returns
        -------
        A string representing the type of the object.
        """
        return self._type
