# coding=utf-8
import time
import dask
import resource
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dask.dataframe import DataFrame
from pymove.core.grid import lat_meters
from pymove.utils.distances import haversine
from pymove.core.grid import create_virtual_grid
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
    B
)


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
    #TODO: Documentar
    def __init__(self, data, latitude=LATITUDE, longitude=LONGITUDE, datetime=DATETIME, traj_id=TRAJ_ID):
        # formatar os labels que foram passados pro que usado no pymove -> format_labels
        # renomeia as colunas do dado passado pelo novo dict
        # cria o dataframe
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
            self._last_operation_name = ''
            self._last_operation_mem_usage = 0
            self._last_operation_time_duration = 0
        else:
            print("Could not instantiate new MoveDataFrame because data has missing columns")

    @staticmethod
    def _has_columns(self, data):
        if LATITUDE in data and LONGITUDE in data and DATETIME in data:
            return True
        return False

    @staticmethod
    def _validate_move_data_frame(self, data):
        # chama a função de validação
        # deverá verificar se tem as colunas e os tipos
        try:
            if (data.dtypes.lat != 'float32'):
                data.lat.astype('float32')
            if (data.dtypes.lon != 'float32'):
                data.lon.astype('float32')
            if (data.dtypes.datetime != 'datetime64[ns]'):
                data.datetime.astype('datetime64[ns]')
        except AttributeError as erro:
            print(erro)

    @property
    def lat(self):
        if LATITUDE not in self:
            raise AttributeError("The MoveDataFrame does not contain the column '%s.'" % LATITUDE)
        return self._data[LATITUDE]

    @property
    def lng(self):
        if LONGITUDE not in self:
            raise AttributeError("The MoveDataFrame does not contain the column '%s.'" % LONGITUDE)
        return self._data[LONGITUDE]

    @property
    def datetime(self):
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
        self._last_operation_time_duration = 0
        self._last_operation_name = 'loc'
        self._last_operation_mem_usage = 0
        return self._data.loc

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
        self._last_operation_time_duration = 0
        self._last_operation_name = 'iloc'
        self._last_operation_mem_usage = 0
        return self._data.iloc

    @property
    def at(self):
        """
        Access a single value for a row/column label pair. Similar to loc, in that both provide label-based lookups.
        Use at if you only need to get or set a single value in a DataFrame or Series.

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.at.html#pandas.DataFrame.at

        """
        self._last_operation_time_duration = 0
        self._last_operation_name = 'at'
        self._last_operation_mem_usage = 0
        return self._data.at

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
        self._last_operation_time_duration = 0
        self._last_operation_name = 'values'
        self._last_operation_mem_usage = 0
        return self._data.values

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
        self._last_operation_time_duration = 0
        self._last_operation_name = 'index'
        self._last_operation_mem_usage = 0
        return self._data.index

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
        self._last_operation_time_duration = 0
        self._last_operation_name = 'dtypes'
        self._last_operation_mem_usage = 0
        return self._data.dtypes

    @property
    def shape(self):
        """
        Return a tuple representing the dimensionality of the DataFrame.

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.shape.html

        """
        self._last_operation_time_duration = 0
        self._last_operation_name = 'shape'
        self._last_operation_mem_usage = 0
        return self._data.shape

    @property
    def isin(self):
        """
        Whether each element in the DataFrame is contained in values.

        Parameters
        ----------
        values : iterable, Series, DataFrame or dict
            The result will only be true at a location if all the labels match. If values is a Series, that’s the index.
            If values is a dict, the keys must be the column names, which must match. If values is a DataFrame, then
            both the index and column labels must match.

        Returns
        -------
        DataFrame
            DataFrame of booleans showing whether each element in the DataFrame is contained in values.

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.isin.html

        """
        self._last_operation_time_duration = 0
        self._last_operation_name = 'isin'
        self._last_operation_mem_usage = 0
        return self._data.isin

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
        self._last_operation_time_duration = 0
        self._last_operation_name = 'unique'
        self._last_operation_mem_usage = 0
        return self._data.unique(values)

    def __setitem__(self, attr, value):
        self.__dict__['_data'][attr] = value

    def __getitem__(self, name):
        try:
            return self.__dict__['_data'][name]
        except Exception as e:
            raise e

    def head(self, n=5):
        _head = self._data.head(n)
        self._last_operation_time_duration = 0
        self._last_operation_name = 'head'
        self._last_operation_mem_usage = 0

        return _head

    def get_users_number(self):

        if UID in self._data:
            returno = self._data[UID].nunique()
        else:
            retorno = 1

        self._last_operation_time_duration = 0
        self._last_operation_name = 'get_users_number'
        self._last_operation_mem_usage = 0
        return retorno

    def to_numpy(self):
        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        _numpy = self._data.values

        finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self._last_operation_time_duration = time.time() - start
        self._last_operation_name = 'to_numpy'
        self._last_operation_mem_usage = finish - init
        return _numpy

    def write_file(self, file_name, separator=','):
        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        self._data.to_csv(file_name, sep=separator, encoding='utf-8', index=False)

        finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self._last_operation_time_duration = time.time() - start
        self._last_operation_name = 'write_file'
        self._last_operation_mem_usage = finish - init

    def len(self):

        _len = self._data.shape[0]

        self._last_operation_time_duration = 0
        self._last_operation_name = 'len'
        self._last_operation_mem_usage = 0
        return _len

    def to_dict(self):

        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        _dict = self._data.to_dict()

        finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        self._last_operation_time_duration = time.time() - start
        self._last_operation_name = 'to_dict'
        self._last_operation_mem_usage = finish - init
        return _dict

    def to_grid(self, cell_size, meters_by_degree=lat_meters(-3.8162973555)):

        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        _grid = create_virtual_grid(cell_size, self.get_bbox(), meters_by_degree)

        finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self._last_operation_time_duration = time.time() - start
        self._last_operation_name = 'to_grid'
        self._last_operation_mem_usage = finish - init
        return _grid

    def to_DataFrame(self):
        self._last_operation_time_duration = 0
        self._last_operation_name = 'to_DataFrame'
        self._last_operation_mem_usage = 0
        return self._data

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
        self : pandas.core.frame.DataFrame
            Represents the dataset with contains lat, long and datetime.

        Returns
        -------
        bbox : tuple
            Represents a bound box, that is a tuple of 4 values with the min and max limits of latitude e longitude.
        Examples
        --------
        (22.147577, 113.54884299999999, 41.132062, 121.156224)
        """

        try:
            _bbox = (self._data[LATITUDE].min(), self._data[LONGITUDE].min(), self._data[LATITUDE].max(),
                     self._data[LONGITUDE].max())

            self._last_operation_time_duration = 0
            self._last_operation_name = 'get_bbox'
            self._last_operation_mem_usage = 0
            return _bbox
        except Exception as e:
            self._last_operation_time_duration = 0
            self._last_operation_name = 'get_bbox'
            self._last_operation_mem_usage = 0
            raise e

    def generate_tid_based_on_id_datatime(self, str_format="%Y%m%d%H", sort=True, inplace=True):
        """
        Create or update trajectory id based on id e datetime.
        Parameters
        ----------
        self : pandas.core.frame.DataFrame
            Represents the dataset with contains lat, long and datetime.
        str_format : String
            Contains informations about virtual grid, how
                - lon_min_x: longitude mínima.
                - lat_min_y: latitude miníma.
                - grid_size_lat_y: tamanho da grid latitude.
                - grid_size_lon_x: tamanho da longitude da grid.
                - cell_size_by_degree: tamanho da célula da Grid.
            If value is none, the function ask user by dic_grid.
        sort : boolean
            Represents the state of dataframe, if is sorted. By default it's true.
        Returns
        -------
        Examples
        --------
        ID = M00001 and datetime = 2019-04-28 00:00:56  -> tid = M000012019042800
        >>> from pymove.utils.transformations import generate_tid_based_on_id_datatime
        >>> generate_tid_based_on_id_datatime(df)
        """
        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        if inplace:
            _data = self._data
        else:
            _data = PandasMoveDataFrame(data=self._data)

        try:
            print('\nCreating or updating tid feature...\n')
            if sort is True:
                print('...Sorting by {} and {} to increase performance\n'.format(TRAJ_ID, DATETIME))
                _data.sort_values([TRAJ_ID, DATETIME], inplace=True)

            _data[TID] = _data[TRAJ_ID].astype(str) + _data[DATETIME].dt.strftime(str_format)
            print('\n...tid feature was created...\n')

            finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self._last_operation_time_duration = time.time() - start
            self._last_operation_name = 'generate_tid_based_on_id_datatime'
            self._last_operation_mem_usage = finish - init

            if inplace == False:
                return _data
        except Exception as e:
            finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self._last_operation_time_duration = time.time() - start
            self._last_operation_name = 'generate_tid_based_on_id_datatime'
            self._last_operation_mem_usage = finish - init
            raise e

    # TODO complementar oq ela faz
    def generate_date_features(self, inplace=True):
        """
        Create or update date feature.

        Parameters
        ----------
        self : pandas.core.frame.DataFrame
            Represents the dataset with contains lat, long and datetime.

        Returns
        -------

        Examples
        --------
        >>> from pymove.utils.transformations import generate_date_features
        >>> generate_date_features(df)
        """
        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        if inplace:
            _data = self._data
        else:
            _data = PandasMoveDataFrame(data=self._data)

        try:
            print('Creating date features...')
            if DATETIME in _data:
                _data['date'] = _data[DATETIME].dt.date
                print('..Date features was created...\n')

            finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self._last_operation_time_duration = time.time() - start
            self._last_operation_name = 'generate_date_features'
            self._last_operation_mem_usage = finish - init

            if inplace == False:
                return _data
        except Exception as e:
            finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self._last_operation_time_duration = time.time() - start
            self._last_operation_name = 'generate_date_features'
            self._last_operation_mem_usage = finish - init
            raise e

    # TODO complementar oq ela faz
    def generate_hour_features(self, inplace=True):
        """
        Create or update hour feature.
        Parameters
        ----------
        self : pandas.core.frame.DataFrame
            Represents the dataset with contains lat, long and datetime.
        Returns
        -------
        Examples
        --------
        >>> from pymove.utils.transformations import generate_hour_features
        >>> generate_date_features(df)
        """
        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        if inplace:
            _data = self._data
        else:
            _data = PandasMoveDataFrame(data=self._data)

        try:
            print('\nCreating or updating a feature for hour...\n')
            if DATETIME in _data:
                _data['hour'] = _data[DATETIME].dt.hour
                print('...Hour feature was created...\n')

            finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self._last_operation_time_duration = time.time() - start
            self._last_operation_name = 'generate_hour_features'
            self._last_operation_mem_usage = finish - init

            if inplace == False:
                return _data
        except Exception as e:
            finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self._last_operation_time_duration = time.time() - start
            self._last_operation_name = 'generate_hour_features'
            self._last_operation_mem_usage = finish - init
            raise e

    # TODO: botar inplace
    def generate_day_of_the_week_features(self, inplace=True):
        """
        Create or update a feature day of the week from datatime.
        Parameters
        ----------
        self : pandas.core.frame.DataFrame
            Represents the dataset with contains lat, long and datetime.
        Returns
        -------
        Examples
        --------
        Exampĺe: datetime = 2019-04-28 00:00:56  -> day = Sunday
        >>> from pymove.utils.transformations import generate_day_of_the_week_features
        >>> generate_day_of_the_week_features(df)
        """
        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        if inplace:
            _data = self._data
        else:
            _data = PandasMoveDataFrame(data=self._data)

        try:
            print('\nCreating or updating day of the week feature...\n')
            _data[DAY] = _data[DATETIME].dt.day_name()
            print('...the day of the week feature was created...\n')
            finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self._last_operation_time_duration = time.time() - start
            self._last_operation_name = 'generate_day_of_the_week_features'
            self._last_operation_mem_usage = finish - init

            if inplace == False:
                return _data
        except Exception as e:
            finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self._last_operation_time_duration = time.time() - start
            self._last_operation_name = 'generate_day_of_the_week_features'
            self._last_operation_mem_usage = finish - init
            raise e

    # TODO: botar inplace
    def generate_time_of_day_features(self, inplace=True):
        """
        Create a feature time of day or period from datatime.
        Parameters
        ----------
        self : pandas.core.frame.DataFrame
            Represents the dataset with contains lat, long and datetime.
        Returns
        -------
        Examples
        --------
        - datetime1 = 2019-04-28 02:00:56 -> period = early morning
        - datetime2 = 2019-04-28 08:00:56 -> period = morning
        - datetime3 = 2019-04-28 14:00:56 -> period = afternoon
        - datetime4 = 2019-04-28 20:00:56 -> period = evening
        >>> from pymove.utils.transformations import generate_time_of_day_features
        >>> generate_time_of_day_features(df)
        """
        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        if inplace:
            _data = self._data
        else:
            _data = PandasMoveDataFrame(data=self._data)

        try:
            print(
                '\nCreating or updating period feature\n...early morning from 0H to 6H\n...morning from 6H to 12H\n...afternoon from 12H to 18H\n...evening from 18H to 24H')
            conditions = [(_data[DATETIME].dt.hour >= 0) & (_data[DATETIME].dt.hour < 6),
                          (_data[DATETIME].dt.hour >= 6) & (_data[DATETIME].dt.hour < 12),
                          (_data[DATETIME].dt.hour >= 12) & (_data[DATETIME].dt.hour < 18),
                          (_data[DATETIME].dt.hour >= 18) & (_data[DATETIME].dt.hour < 24)]
            choices = ['early morning', 'morning', 'afternoon', 'evening']
            _data[PERIOD] = np.select(conditions, choices, 'undefined')
            print('...the period of day feature was created')

            finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self._last_operation_time_duration = time.time() - start
            self._last_operation_name = 'generate_time_of_day_features'
            self._last_operation_mem_usage = finish - init

            if inplace == False:
                return _data
        except Exception as e:
            finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self._last_operation_time_duration = time.time() - start
            self._last_operation_name = 'generate_time_of_day_features'
            self._last_operation_mem_usage = finish - init
            raise e

    # TODO complementar oq ela faz
    def generate_dist_features(self, label_id=TRAJ_ID, label_dtype=np.float64, sort=True, inplace=True):
        """
         Create three distance in meters to an GPS point P (lat, lon).
        Parameters
        ----------
        self : pandas.core.frame.DataFrame
            Represents the dataset with contains lat, long and datetime.
        label_id : String
            Represents name of column of trajectore's id. By default it's 'id'.
        label_dtype : String
            Represents column id type. By default it's np.float64.
        sort : boolean
            Represents the state of dataframe, if is sorted. By default it's true.
        Returns
        -------
        Examples
        --------
        Example:    P to P.next = 2 meters
                    P to P.previous = 1 meter
                    P.previous to P.next = 1 meters
        >>> from pymove.utils.transformations import generate_dist_features
        >>> generate_dist_features(df)
        """
        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        if inplace:
            _data = self._data
        else:
            _data = PandasMoveDataFrame(data=self._data)

        try:
            print('\nCreating or updating distance features in meters...\n')
            start_time = time.time()

            if sort is True:
                print('...Sorting by {} and {} to increase performance\n'.format(label_id, DATETIME))
                _data.sort_values([label_id, DATETIME], inplace=True)

            if _data.index.name is None:
                print('...Set {} as index to increase attribution performance\n'.format(label_id))
                _data.set_index(label_id, inplace=True)

            """ create ou update columns"""
            _data[DIST_TO_PREV] = label_dtype(-1.0)
            _data[DIST_TO_NEXT] = label_dtype(-1.0)
            _data[DIST_PREV_TO_NEXT] = label_dtype(-1.0)

            ids = _data.index.unique()
            selfsize = _data.shape[0]
            curr_perc_int = -1
            start_time = time.time()
            deltatime_str = ''
            sum_size_id = 0
            size_id = 0
            for idx in ids:
                curr_lat = _data.at[idx, LATITUDE]
                curr_lon = _data.at[idx, LONGITUDE]

                size_id = curr_lat.size

                if size_id <= 1:
                    print('...id:{}, must have at least 2 GPS points\n'.format(idx))
                    _data.at[idx, DIST_TO_PREV] = np.nan

                else:
                    prev_lat = shift(curr_lat, 1)
                    prev_lon = shift(curr_lon, 1)
                    # compute distance from previous to current point
                    _data.at[idx, DIST_TO_PREV] = haversine(prev_lat, prev_lon, curr_lat, curr_lon)

                    next_lat = shift(curr_lat, -1)
                    next_lon = shift(curr_lon, -1)
                    # compute distance to next point
                    _data.at[idx, DIST_TO_NEXT] = haversine(curr_lat, curr_lon, next_lat, next_lon)

                    # using pandas shift in a large dataset: 7min 21s
                    # using numpy shift above: 33.6 s

                    # use distance from previous to next
                    _data.at[idx, DIST_PREV_TO_NEXT] = haversine(prev_lat, prev_lon, next_lat, next_lon)

                    sum_size_id += size_id
                    curr_perc_int, est_time_str = progress_update(sum_size_id, selfsize, start_time, curr_perc_int,
                                                                  step_perc=20)
            _data.reset_index(inplace=True)
            print('...Reset index\n')
            print('..Total Time: {}'.format((time.time() - start_time)))

            finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self._last_operation_time_duration = time.time() - start
            self._last_operation_name = 'generate_dist_features'
            self._last_operation_mem_usage = finish - init

            if inplace == False:
                return _data
        except Exception as e:
            print('label_id:{}\nidx:{}\nsize_id:{}\nsum_size_id:{}'.format(label_id, idx, size_id, sum_size_id))
            finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self._last_operation_time_duration = time.time() - start
            self._last_operation_name = 'generate_dist_features'
            self._last_operation_mem_usage = finish - init
            raise e

    def generate_dist_time_speed_features(self, label_id=TRAJ_ID, label_dtype=np.float64, sort=True, inplace=True):
        """
        Firstly, create three distance to an GPS point P (lat, lon)
        After, create two feature to time between two P: time to previous and time to next
        Lastly, create two feature to speed using time and distance features
        Parameters
        ----------
        self : pandas.core.frame.DataFrame
            Represents the dataset with contains lat, long and datetime.
        label_id : String
            Represents name of column of trajectore's id. By default it's 'id'.
        label_dtype : String
            Represents column id type. By default it's np.float64.
        sort : boolean
            Represents the state of dataframe, if is sorted. By default it's true.
        inplace : boolean
            Represents te of dataframe, if is sorted. By default it's true.
        Returns
        -------
        Examples
        --------
        Example:    dist_to_prev =  248.33 meters, dist_to_prev 536.57 meters
                    time_to_prev = 60 seconds, time_prev = 60.0 seconds
                    speed_to_prev = 4.13 m/s, speed_prev = 8.94 m/s.
        >>> from pymove.utils.transformations import generate_dist_time_speed_features
        >>> generate_dist_time_speed_features(df)
        """
        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        if inplace:
            _data = self._data
        else:
            _data = PandasMoveDataFrame(data=self._data)

        try:
            print('\nCreating or updating distance, time and speed features in meters by seconds\n')
            start_time = time.time()

            if sort is True:
                print('...Sorting by {} and {} to increase performance\n'.format(label_id, DATETIME))
                _data.sort_values([label_id, DATETIME], inplace=True)

            if _data.index.name is None:
                print('...Set {} as index to a higher peformance\n'.format(label_id))
                _data.set_index(label_id, inplace=True)

            """create new feature to time"""
            _data[DIST_TO_PREV] = label_dtype(-1.0)

            """create new feature to time"""
            _data[TIME_TO_PREV] = label_dtype(-1.0)

            """create new feature to speed"""
            _data[SPEED_TO_PREV] = label_dtype(-1.0)

            ids = _data.index.unique()
            selfsize = _data.shape[0]
            curr_perc_int = -1
            sum_size_id = 0
            size_id = 0

            for idx in ids:
                curr_lat = _data.at[idx, LATITUDE]
                curr_lon = _data.at[idx, LONGITUDE]

                size_id = curr_lat.size

                if size_id <= 1:
                    print('...id:{}, must have at least 2 GPS points\n'.format(idx))
                    _data.at[idx, DIST_TO_PREV] = np.nan
                    _data.at[idx, TIME_TO_PREV] = np.nan
                    _data.at[idx, SPEED_TO_PREV] = np.nan
                else:
                    prev_lat = shift(curr_lat, 1)
                    prev_lon = shift(curr_lon, 1)
                    prev_lon = shift(curr_lon, 1)
                    # compute distance from previous to current point
                    _data.at[idx, DIST_TO_PREV] = haversine(prev_lat, prev_lon, curr_lat, curr_lon)

                    time_ = _data.at[idx, DATETIME].astype(label_dtype)
                    time_prev = (time_ - shift(time_, 1)) * (10 ** -9)
                    _data.at[idx, TIME_TO_PREV] = time_prev

                    """ set time_to_next"""
                    # time_next = (ut.shift(time_, -1) - time_)*(10**-9)
                    # self.at[idx, dic_features_label['time_to_next']] = time_next

                    "set Speed features"
                    _data.at[idx, SPEED_TO_PREV] = _data.at[idx, DIST_TO_PREV] / (time_prev)  # unit: m/s

                    sum_size_id += size_id
                    curr_perc_int, est_time_str = progress_update(sum_size_id, selfsize, start_time, curr_perc_int,
                                                                  step_perc=20)
            print('...Reset index...\n')
            _data.reset_index(inplace=True)
            print('..Total Time: {:.3f}'.format((time.time() - start_time)))

            finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self._last_operation_time_duration = time.time() - start
            self._last_operation_name = 'generate_dist_time_speed_features'
            self._last_operation_mem_usage = finish - init

            if inplace == False:
                return _data
        except Exception as e:
            print('label_id:{}\nidx:{}\nsize_id:{}\nsum_size_id:{}'.format(label_id, idx, size_id, sum_size_id))
            finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self._last_operation_time_duration = time.time() - start
            self._last_operation_name = 'generate_dist_time_speed_features'
            self._last_operation_mem_usage = finish - init
            raise e

    def generate_move_and_stop_by_radius(self, radius=0, target_label=DIST_TO_PREV, inplace=True):
        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        if inplace:
            _data = self._data
        else:
            _data = PandasMoveDataFrame(data=self._data)

        if DIST_TO_PREV not in self._data:
            _data.generate_dist_features()

        try:
            print('\nCreating or updating features MOVE and STOPS...\n')
            conditions = (_data[target_label] > radius), (_data[target_label] <= radius)
            choices = ['move', 'stop']

            _data["situation"] = np.select(conditions, choices, np.nan)
            print('\n....There are {} stops to this parameters\n'.format(_data[_data["situation"] == 'stop'].shape[0]))

            finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self._last_operation_time_duration = time.time() - start
            self._last_operation_name = 'generate_move_and_stop_by_radius'
            self._last_operation_mem_usage = finish - init

            if inplace == False:
                return _data
        except Exception as e:
            finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self._last_operation_time_duration = time.time() - start
            self._last_operation_name = 'generate_move_and_stop_by_radius'
            self._last_operation_mem_usage = finish - init
            raise e

    def time_interval(self):

        time_diff = self._data[DATETIME].max() - self._data[DATETIME].min()

        self._last_operation_time_duration = 0
        self._last_operation_name = 'time_interval'
        self._last_operation_mem_usage = 0
        return time_diff

    def plot_all_features(self, figsize=(21, 15), dtype=np.float64, save_fig=True, name='features.png'):
        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        try:
            col_float = self._data.select_dtypes(include=[dtype]).columns
            tam = col_float.size
            if (tam > 0):
                fig, ax = plt.subplots(tam, 1, figsize=figsize)
                ax_count = 0
                for col in col_float:
                    ax[ax_count].set_title(col)
                    self._data[col].plot(subplots=True, ax=ax[ax_count])
                    ax_count += 1

                if save_fig:
                    plt.savefig(fname=name, fig=fig)

            finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self._last_operation_time_duration = time.time() - start
            self._last_operation_name = 'plot_all_features'
            self._last_operation_mem_usage = finish - init
        except Exception as e:
            finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self._last_operation_time_duration = time.time() - start
            self._last_operation_name = 'plot_all_features'
            self._last_operation_mem_usage = finish - init
            raise e

    def plot_trajs(self, figsize=(10, 10), return_fig=True, markers='o', markersize=20):
        """Generate a visualization that show trajectories.

        Parameters
        ----------
        figsize : tuple, optional, default (10, 10).
            Represents dimensions of figure.
        return_fig : bool, optional, default True.
            Represents whether or not to save the generated picture.
        markers : String, optional, default 'o'.
            Represents visualization type marker.
        markersize : int, optional, default 20.
            Represents visualization size marker.

        Returns
        -------
        fig : matplotlib.pyplot.figure
            The generated picture.
        """

        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        fig = plt.figure(figsize=figsize)

        ids = self._data["id"].unique()
        for id_ in ids:
            selfid = self._data[self._data["id"] == id_]

            plt.plot(selfid[LONGITUDE], selfid[LATITUDE], markers, markersize=markersize)

            finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self._last_operation_time_duration = time.time() - start
            self._last_operation_name = 'plot_trajs'
            self._last_operation_mem_usage = finish - init
        if return_fig:
            finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self._last_operation_time_duration = time.time() - start
            self._last_operation_name = 'plot_trajs'
            self._last_operation_mem_usage = finish - init
            return fig

    def plot_traj_id(self, tid, figsize=(10, 10)):
        """Generate a visualization that shows a trajectory with the specified tid.

        Parameters
        ----------
        tid : String.
            Represents the trajectory tid.
        figsize : tuple, optional, default (10,10).
            Represents dimensions of figure.

        Returns
        -------
        move_data : pymove.core.MoveDataFrameAbstract subclass.
            Trajectory with the specified tid.

        fig : matplotlib.pyplot.figure
            The generated picture.
        """

        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        fig = plt.figure(figsize=figsize)

        if TID not in self._data:
            self.generate_tid_based_on_id_datatime()

        df_ = self._data[self._data[TID] == tid]

        plt.plot(df_.iloc[0][LONGITUDE], df_.iloc[0][LATITUDE], 'yo', markersize=20)  # start point
        plt.plot(df_.iloc[-1][LONGITUDE], df_.iloc[-1][LATITUDE], 'yX', markersize=20)  # end point

        if 'isNode' not in self:
            plt.plot(df_[LONGITUDE], df_[LATITUDE])
            plt.plot(df_.loc[:, LONGITUDE], df_.loc[:, LATITUDE], 'r.', markersize=8)  # points
        else:
            filter_ = df_['isNode'] == 1
            selfnodes = df_.loc[filter_]
            selfpoints = df_.loc[~filter_]
            plt.plot(selfnodes[LONGITUDE], selfnodes[LATITUDE], linewidth=3)
            plt.plot(selfpoints[LONGITUDE], selfpoints[LATITUDE])
            plt.plot(selfnodes[LONGITUDE], selfnodes[LATITUDE], 'go', markersize=10)  # nodes
            plt.plot(selfpoints[LONGITUDE], selfpoints[LATITUDE], 'r.', markersize=8)  # points

        finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self._last_operation_time_duration = time.time() - start
        self._last_operation_name = 'plot_traj_id'
        self._last_operation_mem_usage = finish - init
        return df_, fig

    def show_trajectories_info(self):
        """
        Show dataset information from dataframe, this is number of rows, datetime interval, and bounding box.

        Parameters
        ----------
        self : pandas.core.frame.DataFrame
            Represents the dataset with contains lat, long and datetime.

        dic_labels : dict
            Represents mapping of column's header between values passed on params.

        Examples
        --------
        ======================= INFORMATION ABOUT DATASET =======================
        Number of Points: 217654
        Number of IDs objects: 2
        Start Date:2008-10-23 05:53:05     End Date:2009-03-19 05:46:37
        Bounding Box:(22.147577, 113.54884299999999, 41.132062, 121.156224)
        =========================================================================
        """

        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

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

            finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self._last_operation_time_duration = time.time() - start
            self._last_operation_name = 'show_trajectories_info'
            self._last_operation_mem_usage = finish - init
        except Exception as e:
            finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self._last_operation_time_duration = time.time() - start
            self._last_operation_name = 'show_trajectories_info'
            self._last_operation_mem_usage = finish - init
            raise e

    def min(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        """Returns and returns the minimum values for the requested axis of the dataframe.

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
        _min = self._data.min(axis, skipna, level, numeric_only, **kwargs)

        self._last_operation_time_duration = 0
        self._last_operation_name = 'min'
        self._last_operation_mem_usage = 0

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

        _max = self._data.max(axis, skipna, level, numeric_only, **kwargs)

        self._last_operation_time_duration = 0
        self._last_operation_name = 'max'
        self._last_operation_mem_usage = 0

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
        
        _count = self._data.count(axis, level, numeric_only)

        self._last_operation_time_duration = 0
        self._last_operation_name = 'count'
        self._last_operation_mem_usage = 0
        
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

        _groupby = self._data.groupby(by, axis, level, as_index, sort, group_keys, squeeze, observed, **kwargs)

        self._last_operation_time_duration = 0
        self._last_operation_name = 'groupby'
        self._last_operation_mem_usage = 0
        return _groupby

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
        PandasMoveDataFrame with the duplicates dropped.

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html
        """

        _drop_duplicates = self._data.drop_duplicates(subset, keep, inplace)

        self._last_operation_time_duration = 0
        self._last_operation_name = 'drop_duplicates'
        self._last_operation_mem_usage = 0
        return _drop_duplicates

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
        PandasMoveDataFrame
            with _data with a new index.

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.reset_index.html
        """

        _reset_index = self._data.reset_index(level, drop, inplace, col_level, col_fill)

        self._last_operation_time_duration = 0
        self._last_operation_name = 'reset_index'
        self._last_operation_mem_usage = 0

        return _reset_index

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

        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        _plot = self._data.plot(*args, **kwargs)

        finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self._last_operation_time_duration = time.time() - start
        self._last_operation_name = 'plot'
        self._last_operation_mem_usage = finish - init

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

        _select_dtypes = self._data.select_dtypes(include, exclude)

        self._last_operation_time_duration = 0
        self._last_operation_name = 'select_dtypes'
        self._last_operation_mem_usage = 0

        return _select_dtypes

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
        PandasDataframe:
            _data with sorted values if inplace=False, None otherwise.

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_values.html
        """

        _sort_values = self._data.sort_values(by, axis, ascending, inplace, kind, na_position)

        self._last_operation_time_duration = 0
        self._last_operation_name = '_sort_values'
        self._last_operation_mem_usage = 0

        return _sort_values

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

        _astype = self._data.astype(dtype, copy, errors, **kwargs)

        self._last_operation_time_duration = 0
        self._last_operation_name = 'astype'
        self._last_operation_mem_usage = 0

        return _astype

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

        _set_index = self._data.set_index(keys, drop, append, inplace, verify_integrity)

        self._last_operation_time_duration = 0
        self._last_operation_name = '_set_index'
        self._last_operation_mem_usage = 0

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

        _drop = self._data.drop(labels, axis, index, columns, level, inplace, errors)

        self._last_operation_time_duration = 0
        self._last_operation_name = 'drop'
        self._last_operation_mem_usage = 0

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

        _duplicated = self._data.duplicated(subset, keep)

        self._last_operation_time_duration = 0
        self._last_operation_name = 'duplicated'
        self._last_operation_mem_usage = 0

        return _duplicated

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
            DataFrame
                A copy of the original object, shifed.

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.shift.html
        """

        _shift = self._data.shift(periods, freq, axis, fill_value)

        self._last_operation_time_duration = 0
        self._last_operation_name = 'shift'
        self._last_operation_mem_usage = 0

        return _shift

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

        _any = self._data.any(axis, bool_only, skipna, level, **kwargs)

        self._last_operation_time_duration = 0
        self._last_operation_name = 'any'
        self._last_operation_mem_usage = 0

        return _any

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
        subset: array-like, optional (None bu default)
            Labels along other axis to consider,
            e.g. if you are dropping rows these would be a list of columns to include.
        inplace: bool, optional (default False)
            If True, do operation inplace and return None

        Returns
        -------
        DataFrame:
            _data with NA entries dropped from it.

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dropna.html
        """
        _dropna = self._data.dropna(axis, how, thresh, subset, inplace)

        self._last_operation_time_duration = 0
        self._last_operation_name = 'dropna'
        self._last_operation_mem_usage = 0

        return _dropna

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

        _isin = self._data.isin(values)

        self._last_operation_time_duration = 0
        self._last_operation_name = 'isin'
        self._last_operation_mem_usage = 0
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
        DataFrame
            _data with the appended data.

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.append.html
        """
        _append = self._data.append(other, ignore_index, verify_integrity, sort)

        self._last_operation_time_duration = 0
        self._last_operation_name = 'append'
        self._last_operation_mem_usage = 0

        return _append

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
        _nunique = self._data.nunique(axis, dropna)

        self._last_operation_time_duration = 0
        self._last_operation_name = 'nunique'
        self._last_operation_mem_usage = 0

        return _nunique

    def to_csv(self, file_name, sep=',', encoding=None):
        """Write object to a comma-separated values (csv) file.

        Parameters
        ----------
        file_name: str
            File path or object
        sep: str
            String of length 1. Field delimiter for the output file.
        encoding: str, optional (None default)
            A string representing the encoding to use in the output file, defaults to ‘utf-8’

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html
        """

        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        self._data.to_csv(file_name, sep, encoding)

        finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self._last_operation_time_duration = time.time() - start
        self._last_operation_name = 'to_csv'
        self._last_operation_mem_usage = finish - init

    # TODO: Ajeitar esse bug e deixar esse como central. erro nao entendi
    # def to_csv(self, path_or_buf=None, sep=',', na_rep='', float_format=None, columns=None, header=True, index=True,
    #          index_label=None, mode='w', encoding=None, compression='infer', quoting=None, quotechar='"',
    #          line_terminator=None, chunksize=None, date_format=None, doublequote=True, escapechar=None, decimal='.'):
    #   self._data.to_csv(path_or_buf, sep, na_rep, float_format, columns, header, index,
    #    index_label, mode, encoding, compression, quoting, quotechar,
    #    line_terminator, chunksize, date_format, doublequote, escapechar, decimal)
    # self._data.to_csv("teste3.csv")]

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

        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self._last_operation_name = 'convert_to'

        if (new_type == "dask"):
            _dask = DaskMoveDataFrame(
                                      self._data,
                                      latitude=LATITUDE,
                                      longitude=LONGITUDE,
                                      datetime=DATETIME,
                                      traj_id=TRAJ_ID,
                                      n_partitions=1)
            finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self._last_operation_time_duration = time.time() - start
            self._last_operation_mem_usage = finish - init

            return _dask

        elif (new_type == "pandas"):

            finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self._last_operation_time_duration = time.time() - start
            self._last_operation_mem_usage = finish - init

            return self._data

    def get_type(self):
        """Returns the type of the object.

        Returns
        -------
        A string representing the type of the object.
        """
        self._last_operation_time_duration = 0
        self._last_operation_name = 'get_type'
        self._last_operation_mem_usage = 0
        return self._type

    def last_operation_time(self):
        """Returns the execution time of the last function, called to the PandasMoveDataFrame object.

        Returns
        -------
        A string of the execution time of the last function, called to the PandasMoveDataFrame object.
        """
        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        return self._last_operation_time_duration

    def last_operation_name(self):
        """Returns the name of the last function, called to the PandasMoveDataFrame object.

        Returns
        -------
        A string of the name of the last function, called to the PandasMoveDataFrame object.
        """

        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        return self._last_operation_name

    def last_operation(self):
        """Returns the memory usage calculation, in bytes, of the last function,
           called to the PandasMoveDataFrame object.

        Returns
        -------
        A string of the memory usage calculation, in bytes, of the last function,
        called to the PandasMoveDataFrame object.
        """

        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        return self.self._last_operation_mem_usage

    def mem(self, format):
        """Returns the memory usage calculation of the last function, called to the PandasMoveDataFrame object,
        in the data format required by the user.

        Parameters
        ----------
        format : str
            The data format to which the memory calculation must be converted to.

        Returns
        -------
        A string of the memory usage calculation of the last function, called to the PandasMoveDataFrame object.
        """

        switcher = {
            B: self._last_operation_dict["mem_usage"],
            KB: self._last_operation_dict["mem_usage"] / 1024,
            MB: self._last_operation_dict["mem_usage"] / (1024 * 2),
            GB: self._last_operation_dict["mem_usage"] / (1024 * 3),
            TB: self._last_operation_dict["mem_usage"] / (1024 * 4),
        }

        return switcher[format]


class DaskMoveDataFrame(DataFrame, MoveDataFrameAbstractModel):
    def __init__(self, data, latitude=LATITUDE, longitude=LONGITUDE, datetime=DATETIME, traj_id=TRAJ_ID,
                 n_partitions=1):
        # formatar os labels que foram return 0ados pro que usado no pymove -> format_labels
        # renomeia as colunas do dado return 0ado pelo novo dict
        # cria o dataframe
        mapping_columns = format_labels(data, traj_id, latitude, longitude, datetime)
        dsk = data.rename(columns=mapping_columns)

        if self._has_columns(dsk):
            self._validate_move_data_frame(dsk)
            self._data = dask.dataframe.from_pandas(dsk, npartitions=n_partitions)
            self._type = TYPE_DASK
            self._last_operation_name = ''
            self._last_operation_mem_usage = 0
            self._last_operation_time_duration = 0
        else:
            print("erroo")

    def _has_columns(self, data):
        if (LATITUDE in data and LONGITUDE in data and DATETIME in data):
            return True
        return False

    def _validate_move_data_frame(self, data):
        # chama a função de validação
        # deverá verificar se tem as colunas e os tipos
        try:
            if (data.dtypes.lat != 'float32'):
                data.lat.astype('float32')
            if (data.dtypes.lon != 'float32'):
                data.lon.astype('float32')
            if (data.dtypes.datetime != 'datetime64[ns]'):
                data.lon.astype('datetime64[ns]')
        except AttributeError as erro:
            print(erro)

    @property
    def lat(self):
        if LATITUDE not in self:
            raise AttributeError("The MoveDataFrame does not contain the column '%s.'" % LATITUDE)
        return self[LATITUDE]

    @property
    def lng(self):
        if LONGITUDE not in self:
            raise AttributeError("The MoveDataFrame does not contain the column '%s.'" % LONGITUDE)
        return self[LONGITUDE]

    @property
    def datetime(self):
        if DATETIME not in self:
            raise AttributeError("The MoveDataFrame does not contain the column '%s.'" % DATETIME)
        return self[DATETIME]

    def head(self, n=5, npartitions=1, compute=True):
        return self._data.head(n, npartitions, compute)

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
            return self._data
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

    def get_users_number(self):
        return 0

    def time_interval(self):
        return 0

    def to_numpy(self):
        return 0

    def write_file(self):
        return 0

    def len(self):
        return 0

    def to_dict(self):
        return 0

    def to_grid(self):
        return 0

    def generate_tid_based_on_id_datatime(self):
        return 0

    def generate_date_features(self):
        return 0

    def generate_hour_features(self):
        return 0

    def generate_day_of_the_week_features(self):
        return 0

    def generate_time_of_day_features(self):
        return 0

    def generate_dist_features(self):
        return 0

    def generate_dist_time_speed_features(self):
        return 0

    def generate_move_and_stop_by_radius(self):
        return 0

    def time_interval(self):
        return 0

    def get_bbox(self):
        return 0

    def plot_all_features(self):
        return 0

    def plot_trajs(self):
        #Generate a visualization that show trajectories.
        
        return 0

    def plot_traj_id(self):
        # Generate a visualization that shows a trajectory with the specified tid.
        return 0

    def show_trajectories_info(self):
        # Show dataset information from dataframe, this is number of rows, datetime interval, and bounding box.
        return 0

    def count(self):
        # CountS the non-NA cells for each column or row.
        return 0

    def reset_index(self):
        # Resets the dask DataFrame's index, and use the default one.
        return 0

    def plot(self):
        # Plot the data of the dask DataFrame.
        return 0

    def drop_duplicates(self):
        # Removes duplicated rows from the data.
        return 0

    def select_dtypes(self):
        # Returns a subset of the dask DataFrame columns based on the column dtypes.
        return 0

    def sort_values(self):
        # Sorts the values of the dask DataFrame.
        return 0

    def astype(self):
        # Casts a dask object to a specified dtype.
        return 0

    def set_index(self):
        # Set the dask DataFrame index (row labels) using one or more existing columns or arrays (of the correct length).
        return 0

    def drop(self):
        # Drops specified rows or columns of the dask Dataframe.
        return 0

    def duplicated(self):
        # Returns boolean Series denoting duplicate rows, optionally only considering certain columns.
        return 0

    def shift(self):
        # Shifts dask dataFrame index by desired number of periods with an optional time freq.
        return 0

    def any(self):
        # Inidicates if any element is True, potentially over an axis.
        # Returns False unless there at least one element within the dask Dataframe axis that is True or equivalent.
        return 0

    def dropna(self):
        # Removes missing data from dask DataFrame.
        return 0

    def isin(self):
        # Determines whether each element in the dask DataFrame is contained in values.
        return 0

    def append(self):
        # Append rows of other to the end of caller, returning a new object.
        # Columns in other that are not in the caller are added as new columns
        return 0

    def nunique(self):
        # Count distinct observations over requested axis.
        return 0

    def to_csv(self):
        # Write object to a comma-separated values (csv) file.
        return 0

    def at(self):
        return 0

    def to_DataFrame(self):
        return 0

    def unique(self, values):
        return 0

    def last_operation_time(self):
        """Returns the execution time of the last function, called to the PandasMoveDataFrame object.

        Returns
        -------
        A string of the execution time of the last function, called to the PandasMoveDataFrame object.
        """

        return self._last_operation_time_duration

    def last_operation_name(self):
        """Returns the name of the last function, called to the PandasMoveDataFrame object.

        Returns
        -------
        A string of the name of the last function, called to the PandasMoveDataFrame object.
        """

        return self._last_operation_name

    def last_operation(self):
        """Returns the memory usage calculation, in bytes, of the last function,
           called to the PandasMoveDataFrame object.

        Returns
        -------
        A string of the memory usage calculation, in bytes, of the last function,
        called to the PandasMoveDataFrame object.
        """

        return self._last_operation_dict

    def mem(self, format):
        """Returns the memory usage calculation of the last function, called to the PandasMoveDataFrame object,
        in the data format required by the user.

        Parameters
        ----------
        format : str
            The data format to which the memory calculation must be converted to.

        Returns
        -------
        A string of the memory usage calculation of the last function, called to the PandasMoveDataFrame object.
        """
        
        return self._last_operation_dict['mem_usage']  # TODO ver a formula