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
    B)



class MoveDataFrame():
    @staticmethod
    def __new__(self, data, latitude=LATITUDE, longitude=LONGITUDE, datetime=DATETIME, traj_id=TRAJ_ID, type="pandas",
                n_partitions=1):
        self.type = type

        if type == 'pandas':
            return PandasMoveDataFrame(data, latitude, longitude, datetime, traj_id)
        if type == 'dask':
            return DaskMoveDataFrame(data, latitude, longitude, datetime, traj_id, n_partitions)

# TODO: tirar o data do format_labels
class PandasMoveDataFrame(pd.DataFrame, MoveDataFrameAbstractModel):  # dask sua estrutura de dados
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
            self._last_operation_dict = {'name': '', 'time': '', 'mem_usage': ''}
        else:
            print("Could not instantiate new MoveDataFrame because data has missing columns")

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
        return self._data.loc

    @property
    def iloc(self):
        return self._data.iloc

    @property
    def at(self):
        return self._data.at

    @property
    def values(self):
        return self._data.values

    @property
    def columns(self):
        return self._data.columns

    @property
    def index(self):
        return self._data.index

    @property
    def dtypes(self):
        return self._data.dtypes

    @property
    def shape(self):
        return self._data.shape

    @property
    def isin(self):
        return self._data.isin

    def unique(self, values):
        return self._data.unique(values)

    def __setitem__(self, attr, value):
        self.__dict__['_data'][attr] = value

    def __getitem__(self, name):
        try:
            return self.__dict__['_data'][name]
        except Exception as e:
            raise e

    def head(self, n=5):
        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        _head = self._data.head(n)

        finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self._last_operation_dict['time'] = time.time() - start
        self._last_operation_dict['name'] = 'head'
        self._last_operation_dict['mem_usage'] = finish - init

        return _head

    def get_users_number(self):
        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        if UID in self._data:
            returno = self._data[UID].nunique()
        else:
            retorno = 1

        finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self._last_operation_dict['time'] = time.time() - start
        self._last_operation_dict['name'] = 'get_users_number'
        self._last_operation_dict['mem_usage'] = finish - init
        return retorno

    def to_numpy(self):
        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        _numpy = self._data.values

        finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self._last_operation_dict['time'] = time.time() - start
        self._last_operation_dict['name'] = 'to_numpy'
        self._last_operation_dict['mem_usage'] = finish - init
        return _numpy

    def write_file(self, file_name, separator=','):
        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        self._data.to_csv(file_name, sep=separator, encoding='utf-8', index=False)

        finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self._last_operation_dict['time'] = time.time() - start
        self._last_operation_dict['name'] = 'write_file'
        self._last_operation_dict['mem_usage'] = finish - init

    def len(self):
        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        _len = self._data.shape[0]

        finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self._last_operation_dict['time'] = time.time() - start
        self._last_operation_dict['name'] = 'len'
        self._last_operation_dict['mem_usage'] = finish - init
        return _len

    def to_dict(self):
        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        _dict = self._data.to_dict()

        finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self._last_operation_dict['time'] = time.time() - start
        self._last_operation_dict['name'] = 'to_dict'
        self._last_operation_dict['mem_usage'] = finish - init
        return _dict

    def to_grid(self, cell_size, meters_by_degree=lat_meters(-3.8162973555)):
        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        _grid = create_virtual_grid(cell_size, self.get_bbox(), meters_by_degree)
        finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self._last_operation_dict['time'] = time.time() - start
        self._last_operation_dict['name'] = 'to_grid'
        self._last_operation_dict['mem_usage'] = finish - init
        return _grid

    def to_DataFrame(self):
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
        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        try:
            _bbox = (self._data[LATITUDE].min(), self._data[LONGITUDE].min(), self._data[LATITUDE].max(),
                     self._data[LONGITUDE].max())

            finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self._last_operation_dict['time'] = time.time() - start
            self._last_operation_dict['name'] = 'get_bbox'
            self._last_operation_dict['mem_usage'] = finish - init
            return _bbox
        except Exception as e:
            finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self._last_operation_dict['time'] = time.time() - start
            self._last_operation_dict['name'] = 'get_bbox'
            self._last_operation_dict['mem_usage'] = finish - init
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
            self._last_operation_dict['time'] = time.time() - start
            self._last_operation_dict['name'] = 'generate_tid_based_on_id_datatime'
            self._last_operation_dict['mem_usage'] = finish - init

            if inplace == False:
                return _data
        except Exception as e:
            finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self._last_operation_dict['time'] = time.time() - start
            self._last_operation_dict['name'] = 'generate_tid_based_on_id_datatime'
            self._last_operation_dict['mem_usage'] = finish - init
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
            self._last_operation_dict['time'] = time.time() - start
            self._last_operation_dict['name'] = 'generate_date_features'
            self._last_operation_dict['mem_usage'] = finish - init

            if inplace == False:
                return _data
        except Exception as e:
            finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self._last_operation_dict['time'] = time.time() - start
            self._last_operation_dict['name'] = 'generate_date_features'
            self._last_operation_dict['mem_usage'] = finish - init
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
            self._last_operation_dict['time'] = time.time() - start
            self._last_operation_dict['name'] = 'generate_hour_features'
            self._last_operation_dict['mem_usage'] = finish - init

            if inplace == False:
                return _data
        except Exception as e:
            finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self._last_operation_dict['time'] = time.time() - start
            self._last_operation_dict['name'] = 'generate_hour_features'
            self._last_operation_dict['mem_usage'] = finish - init
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
            self._last_operation_dict['time'] = time.time() - start
            self._last_operation_dict['name'] = 'generate_day_of_the_week_features'
            self._last_operation_dict['mem_usage'] = finish - init

            if inplace == False:
                return _data
        except Exception as e:
            finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self._last_operation_dict['time'] = time.time() - start
            self._last_operation_dict['name'] = 'generate_day_of_the_week_features'
            self._last_operation_dict['mem_usage'] = finish - init
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
            self._last_operation_dict['time'] = time.time() - start
            self._last_operation_dict['name'] = 'generate_time_of_day_features'
            self._last_operation_dict['mem_usage'] = finish - init

            if inplace == False:
                return _data
        except Exception as e:
            finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self._last_operation_dict['time'] = time.time() - start
            self._last_operation_dict['name'] = 'generate_time_of_day_features'
            self._last_operation_dict['mem_usage'] = finish - init
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
            self._last_operation_dict['time'] = time.time() - start
            self._last_operation_dict['name'] = 'generate_dist_features'
            self._last_operation_dict['mem_usage'] = finish - init

            if inplace == False:
                return _data
        except Exception as e:
            print('label_id:{}\nidx:{}\nsize_id:{}\nsum_size_id:{}'.format(label_id, idx, size_id, sum_size_id))
            finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self._last_operation_dict['time'] = time.time() - start
            self._last_operation_dict['name'] = 'generate_dist_features'
            self._last_operation_dict['mem_usage'] = finish - init
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
            self._last_operation_dict['time'] = time.time() - start
            self._last_operation_dict['name'] = 'generate_dist_time_speed_features'
            self._last_operation_dict['mem_usage'] = finish - init

            if inplace == False:
                return _data
        except Exception as e:
            print('label_id:{}\nidx:{}\nsize_id:{}\nsum_size_id:{}'.format(label_id, idx, size_id, sum_size_id))
            finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self._last_operation_dict['time'] = time.time() - start
            self._last_operation_dict['name'] = 'generate_dist_time_speed_features'
            self._last_operation_dict['mem_usage'] = finish - init
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
            self._last_operation_dict['time'] = time.time() - start
            self._last_operation_dict['name'] = 'generate_move_and_stop_by_radius'
            self._last_operation_dict['mem_usage'] = finish - init

            if inplace == False:
                return _data
        except Exception as e:
            finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self._last_operation_dict['time'] = time.time() - start
            self._last_operation_dict['name'] = 'generate_move_and_stop_by_radius'
            self._last_operation_dict['mem_usage'] = finish - init
            raise e

    def time_interval(self):
        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        time_diff = self._data[DATETIME].max() - self._data[DATETIME].min()

        finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self._last_operation_dict['time'] = time.time() - start
        self._last_operation_dict['name'] = 'time_interval'
        self._last_operation_dict['mem_usage'] = finish - init
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
            self._last_operation_dict['time'] = time.time() - start
            self._last_operation_dict['name'] = 'plot_all_features'
            self._last_operation_dict['mem_usage'] = finish - init
        except Exception as e:
            finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self._last_operation_dict['time'] = time.time() - start
            self._last_operation_dict['name'] = 'plot_all_features'
            self._last_operation_dict['mem_usage'] = finish - init
            raise e

    def plot_trajs(self, figsize=(10, 10), return_fig=True, markers='o', markersize=20):
        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        fig = plt.figure(figsize=figsize)
        ids = self._data["id"].unique()

        for id_ in ids:
            selfid = self._data[self._data["id"] == id_]
            plt.plot(selfid[LONGITUDE], selfid[LATITUDE], markers, markersize=markersize)

            finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self._last_operation_dict['time'] = time.time() - start
            self._last_operation_dict['name'] = 'plot_trajs'
            self._last_operation_dict['mem_usage'] = finish - init
        if return_fig:
            finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self._last_operation_dict['time'] = time.time() - start
            self._last_operation_dict['name'] = 'plot_trajs'
            self._last_operation_dict['mem_usage'] = finish - init
            return fig

    def plot_traj_id(self, tid, figsize=(10, 10)):
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
        self._last_operation_dict['time'] = time.time() - start
        self._last_operation_dict['name'] = 'plot_traj_id'
        self._last_operation_dict['mem_usage'] = finish - init
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
        Returns
        -------
        Examples
        --------
        >>> from pymove.utils.utils import show_trajectories_info
        >>> show_trajectories_info(df)
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
            self._last_operation_dict['time'] = time.time() - start
            self._last_operation_dict['name'] = 'show_trajectories_info'
            self._last_operation_dict['mem_usage'] = finish - init
        except Exception as e:
            finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self._last_operation_dict['time'] = time.time() - start
            self._last_operation_dict['name'] = 'show_trajectories_info'
            self._last_operation_dict['mem_usage'] = finish - init
            raise e

    def min(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        _min = self._data.min(axis, skipna, level, numeric_only, **kwargs)

        finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self._last_operation_dict['time'] = time.time() - start
        self._last_operation_dict['name'] = 'min'
        self._last_operation_dict['mem_usage'] = finish - init
        return _min

    def max(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        _max = self._data.max(axis, skipna, level, numeric_only, **kwargs)

        finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self._last_operation_dict['time'] = time.time() - start
        self._last_operation_dict['name'] = 'max'
        self._last_operation_dict['mem_usage'] = finish - init
        return _max

    def count(self, axis=0, level=None, numeric_only=False):
        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        _count = self._data.count(axis, level, numeric_only)

        finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self._last_operation_dict['time'] = time.time() - start
        self._last_operation_dict['name'] = 'count'
        self._last_operation_dict['mem_usage'] = finish - init
        return _count

    def groupby(self, by=None, axis=0, level=None, as_index=True, sort=True, group_keys=True, squeeze=False,
                observed=False, **kwargs):
        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        _groupby = self._data.groupby(by, axis, level, as_index, sort, group_keys, squeeze, observed, **kwargs)

        finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self._last_operation_dict['time'] = time.time() - start
        self._last_operation_dict['name'] = 'groupby'
        self._last_operation_dict['mem_usage'] = finish - init
        return _groupby

    def drop_duplicates(self, subset=None, keep='first', inplace=False):
        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        _drop_duplicates = self._data.drop_duplicates(subset, keep, inplace)

        finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self._last_operation_dict['time'] = time.time() - start
        self._last_operation_dict['name'] = 'drop_duplicates'
        self._last_operation_dict['mem_usage'] = finish - init
        return _drop_duplicates

    def reset_index(self, level=None, drop=False, inplace=False, col_level=0, col_fill=''):
        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        _reset_index = self._data.reset_index(level, drop, inplace, col_level, col_fill)

        finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self._last_operation_dict['time'] = time.time() - start
        self._last_operation_dict['name'] = 'reset_index'
        self._last_operation_dict['mem_usage'] = finish - init
        return _reset_index

    # TODO: duvida sobre erro quando sem paraetros, perguntar dd
    def plot(self, *args, **kwargs):
        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        _plot = self._data.plot(*args, **kwargs)

        finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self._last_operation_dict['time'] = time.time() - start
        self._last_operation_dict['name'] = 'plot'
        self._last_operation_dict['mem_usage'] = finish - init
        return _plot

    def select_dtypes(self, include=None, exclude=None):
        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        _select_dtypes = self._data.select_dtypes(include, exclude)

        finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self._last_operation_dict['time'] = time.time() - start
        self._last_operation_dict['name'] = 'select_dtypes'
        self._last_operation_dict['mem_usage'] = finish - init
        return _select_dtypes

    def sort_values(self, by, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last'):
        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        _sort_values = self._data.sort_values(by, axis, ascending, inplace, kind, na_position)

        finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self._last_operation_dict['time'] = time.time() - start
        self._last_operation_dict['name'] = '_sort_values'
        self._last_operation_dict['mem_usage'] = finish - init
        return _sort_values

    def astype(self, dtype, copy=True, errors='raise', **kwargs):
        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        _astype = self._data.astype(dtype, copy, errors, **kwargs)

        finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self._last_operation_dict['time'] = time.time() - start
        self._last_operation_dict['name'] = 'astype'
        self._last_operation_dict['mem_usage'] = finish - init
        return _astype

    def set_index(self, keys, drop=True, append=False, inplace=False, verify_integrity=False):
        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        _set_index = self._data.set_index(keys, drop, append, inplace, verify_integrity)

        finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self._last_operation_dict['time'] = time.time() - start
        self._last_operation_dict['name'] = '_set_index'
        self._last_operation_dict['mem_usage'] = finish - init
        return _set_index

    def drop(self, labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise'):
        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        _drop = self._data.drop(labels, axis, index, columns, level, inplace, errors)

        finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self._last_operation_dict['time'] = time.time() - start
        self._last_operation_dict['name'] = 'drop'
        self._last_operation_dict['mem_usage'] = finish - init
        return _drop

    def duplicated(self, subset=None, keep='first'):
        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        _duplicated = self._data.duplicated(subset, keep)

        finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self._last_operation_dict['time'] = time.time() - start
        self._last_operation_dict['name'] = 'duplicated'
        self._last_operation_dict['mem_usage'] = finish - init
        return _duplicated

    def shift(self, periods=1, freq=None, axis=0, fill_value=None):
        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        _shift = self._data.shift(periods, freq, axis, fill_value)

        finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self._last_operation_dict['time'] = time.time() - start
        self._last_operation_dict['name'] = 'shift'
        self._last_operation_dict['mem_usage'] = finish - init
        return _shift

    def any(self, axis=0, bool_only=None, skipna=True, level=None, **kwargs):
        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        _any = self._data.any(axis, bool_only, skipna, level, **kwargs)

        finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self._last_operation_dict['time'] = time.time() - start
        self._last_operation_dict['name'] = 'any'
        self._last_operation_dict['mem_usage'] = finish - init
        return _any

    def dropna(self, axis=0, how='any', thresh=None, subset=None, inplace=False):
        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        _dropna = self._data.dropna(axis, how, thresh, subset, inplace)

        finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self._last_operation_dict['time'] = time.time() - start
        self._last_operation_dict['name'] = 'dropna'
        self._last_operation_dict['mem_usage'] = finish - init
        return _dropna

    def isin(self, values):
        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        _isin = self._data.isin(values)

        finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self._last_operation_dict['time'] = time.time() - start
        self._last_operation_dict['name'] = 'isin'
        self._last_operation_dict['mem_usage'] = finish - init
        return _isin

    def append(self, other, ignore_index=False, verify_integrity=False, sort=None):
        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        _append = self._data.append(other, ignore_index, verify_integrity, sort)

        finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self._last_operation_dict['time'] = time.time() - start
        self._last_operation_dict['name'] = 'append'
        self._last_operation_dict['mem_usage'] = finish - init
        return _append

    def nunique(self, axis=0, dropna=True):
        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        _nunique = self._data.nunique(axis, dropna)

        finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self._last_operation_dict['time'] = time.time() - start
        self._last_operation_dict['name'] = 'nunique'
        self._last_operation_dict['mem_usage'] = finish - init

        return _nunique

    # TODO: botar os parâmetros
    def to_csv(self, file_name, sep=',', encoding=None):
        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        self._data.to_csv(file_name, sep, encoding)

        finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self._last_operation_dict['time'] = time.time() - start
        self._last_operation_dict['name'] = 'to_csv'
        self._last_operation_dict['mem_usage'] = finish - init

    # TODO: Ajeitar esse bug e deixar esse como central. erro nao entendi
    # def to_csv(self, path_or_buf=None, sep=',', na_rep='', float_format=None, columns=None, header=True, index=True,
    #          index_label=None, mode='w', encoding=None, compression='infer', quoting=None, quotechar='"',
    #          line_terminator=None, chunksize=None, date_format=None, doublequote=True, escapechar=None, decimal='.'):
    #   self._data.to_csv(path_or_buf, sep, na_rep, float_format, columns, header, index,
    #    index_label, mode, encoding, compression, quoting, quotechar,
    #    line_terminator, chunksize, date_format, doublequote, escapechar, decimal)
    # self._data.to_csv("teste3.csv")]

    def convert_to(self, new_type):
        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self._last_operation_dict['name'] = 'convert_to'
        if (new_type == "dask"):
            from pymove.core.DaskMoveDataFrame import DaskMoveDataFrame as dm
            _dask = dm(self._data, latitude=LATITUDE, longitude=LONGITUDE, datetime=DATETIME, traj_id=TRAJ_ID,
                       n_partitions=1)
            finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self._last_operation_dict['time'] = time.time() - start
            self._last_operation_dict['mem_usage'] = finish - init
            return _dask
        elif (new_type == "pandas"):
            finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self._last_operation_dict['time'] = time.time() - start
            self._last_operation_dict['mem_usage'] = finish - init
            return self._data

    def get_type(self):
        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        finish = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self._last_operation_dict['time'] = time.time() - start
        self._last_operation_dict['name'] = 'get_type'
        self._last_operation_dict['mem_usage'] = finish - init
        return self._type

    def last_operation_time(self):
        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        return self._last_operation_dict['time']

    def last_operation_name(self):
        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        return self._last_operation_dict['name']

    def last_operation(self):
        start = time.time()
        init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        return self._last_operation_dict

    def mem(self, format):
        switcher = {
            B: self._last_operation_dict["mem_usage"],
            KB: self._last_operation_dict["mem_usage"] / 1024,
            MB: self._last_operation_dict["mem_usage"] / (1024 * 2),
            GB: self._last_operation_dict["mem_usage"] / (1024 * 3),
            TB: self._last_operation_dict["mem_usage"] / (1024 * 4),
        }

        return switcher[format]


class DaskMoveDataFrame(DataFrame, MoveDataFrameAbstractModel):  # dask sua estrutura de dados
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
            self._last_operation_dict = {'name': '', 'time': '', 'mem_usage': ''}
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
        return self._data.min(axis, skipna, split_every, out)

    def max(self, axis=None, skipna=True, split_every=False, out=None):
        return self._data.max(axis, skipna, split_every, out)

    def groupby(self, by=None, **kwargs):
        return self._data.groupby(by)

    def convert_to(self, new_type):
        if (new_type == "dask"):
            return self._data
        elif (new_type == "pandas"):
            df_pandas = self._data.compute()
            from pymove.core.PandasMoveDataFrame import PandasMoveDataFrame as pm
            return pm(df_pandas, latitude=LATITUDE, longitude=LONGITUDE, datetime=DATETIME, traj_id=TRAJ_ID)

    def get_type(self):
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
        return 0

    def plot_traj_id():
        return 0

    def show_trajectories_info(self):
        return 0

    def count(self):
        return 0

    def reset_index(self):
        return 0

    def plot(self):
        return 0

    def drop_duplicates(self):
        return 0

    def select_dtypes(self):
        return 0

    def sort_values(self):
        return 0

    def astype(self):
        return 0

    def set_index(self):
        return 0

    def drop(self):
        return 0

    def duplicated(self):
        return 0

    def shift(self):
        return 0

    def any(self):
        return 0

    def dropna(self):
        return 0

    def isin(self):
        return 0

    def append(self):
        return 0

    def nunique(self):
        return 0

    def to_csv(self):
        return 0

    def last_operation_time(self):
        return self._last_operation_dict['time']

    def last_operation_name(self):
        return self._last_operation_dict['name']

    def last_operation(self):
        return self._last_operation_dict

    def mem(self, format):
        return self._last_operation_dict['mem_usage']  # TODO ver a formula