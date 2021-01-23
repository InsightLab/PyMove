import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from pymove.core import MoveDataFrameAbstractModel
from pymove.core.dataframe import MoveDataFrame
from pymove.core.grid import Grid
from pymove.utils.constants import (
    DATE,
    DATETIME,
    DAY,
    DAY_PERIODS,
    DIST_PREV_TO_NEXT,
    DIST_TO_NEXT,
    DIST_TO_PREV,
    HOUR,
    HOUR_COS,
    HOUR_SIN,
    LATITUDE,
    LOCAL_LABEL,
    LONGITUDE,
    MOVE,
    PERIOD,
    SITUATION,
    SPEED_PREV_TO_NEXT,
    SPEED_TO_NEXT,
    SPEED_TO_PREV,
    STOP,
    TID,
    TIME_PREV_TO_NEXT,
    TIME_TO_NEXT,
    TIME_TO_PREV,
    TRAJ_ID,
    TYPE_DASK,
    TYPE_PANDAS,
    UID,
    WEEK_DAYS,
    WEEK_END,
)
from pymove.utils.conversions import lat_meters
from pymove.utils.distances import haversine
from pymove.utils.log import progress_bar
from pymove.utils.mem import begin_operation, end_operation
from pymove.utils.trajectories import shift


class PandasMoveDataFrame(pd.DataFrame, MoveDataFrameAbstractModel):
    def __init__(
        self,
        data,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        datetime=DATETIME,
        traj_id=TRAJ_ID,
    ):
        """
        Checks whether past data has 'lat', 'lon', 'datetime' columns,
        and renames it with the PyMove lib standard. After starts the
        attributes of the class.

        - self._mgr : Represents trajectory data.
        - self._type : Represents the type of layer below the data structure.
        - self.last_operation : Represents the last operation performed.

        Parameters
        ----------
        data : dict, list, numpy array or pandas.core.DataFrame.
            Input trajectory data.
        latitude : str, optional, default 'lat'.
            Represents column name latitude.
        longitude : str, optional, default 'lon'.
            Represents column name longitude.
        datetime : str, optional, default 'datetime'.
            Represents column name datetime.
        traj_id : str, optional, default 'id'.
            Represents column name trajectory id.

        Raises
        ------
        KeyError
            If missing one of lat, lon, datetime columns
        ValueError, ParserError
            If the data types can't be converted.

        """

        if isinstance(data, dict):
            data = pd.DataFrame.from_dict(data)
        elif isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        elif (
            isinstance(data, list) or isinstance(data, np.ndarray)
        ):
            zip_list = [LATITUDE, LONGITUDE, DATETIME, TRAJ_ID]
            for i in range(len(data[0])):
                try:
                    zip_list[i] = zip_list[i]
                except KeyError:
                    zip_list.append(i)
            data = pd.DataFrame(data, columns=zip_list)

        columns = MoveDataFrame.format_labels(
            traj_id, latitude, longitude, datetime
        )
        tdf = data.rename(columns=columns)

        if MoveDataFrame.has_columns(tdf):
            MoveDataFrame.validate_move_data_frame(tdf)
            super(PandasMoveDataFrame, self).__init__(tdf)
            self._type = TYPE_PANDAS
            self.last_operation = None
        else:

            raise AttributeError(
                'Couldn\'t instantiate MoveDataFrame because data has missing columns.'
            )

    @property
    def lat(self):
        """Checks for the 'lat' column and returns its value."""
        if LATITUDE not in self:
            raise AttributeError(
                "The MoveDataFrame does not contain the column '%s.'"
                % LATITUDE
            )
        return self[LATITUDE]

    @property
    def lng(self):
        """Checks for the 'lon' column and returns its value."""
        if LONGITUDE not in self:
            raise AttributeError(
                "The MoveDataFrame does not contain the column '%s.'"
                % LONGITUDE
            )
        return self[LONGITUDE]

    @property
    def datetime(self):
        """Checks for the 'datetime' column and returns its value."""
        if DATETIME not in self:
            raise AttributeError(
                "The MoveDataFrame does not contain the column '%s.'"
                % DATETIME
            )
        return self[DATETIME]

    def rename(
            self,
            mapper=None,
            index=None,
            columns=None,
            axis=None,
            copy=True,
            inplace=False
    ):
        """
        Alter axes labels.

        Function / dict values must be unique (1-to-1).
        Labels not contained in a dict / Series will be left as-is.
        Extra labels listed don’t throw an error.

        Parameters
        ----------
        mapper: dict-like or function
            Dict-like or functions transformations to apply to that axis’ values.
            Use either mapper and axis to specify the axis to target
            with mapper, or index and columns.

        index: dict-like or function
            Alternative to specifying axis
            (mapper, axis=0 is equivalent to index=mapper).

        columns: dict-like or function
            Alternative to specifying axis
            (mapper, axis=1 is equivalent to columns=mapper).

        axis: int or str
            Axis to target with mapper.
            Can be either the axis name (‘index’, ‘columns’) or number (0, 1).
            The default is ‘index’.

        copy: bool, default True
            Also copy underlying data.

        inplace: bool, default False
            Whether to return a new DataFrame.
            If True then value of copy is ignored.

        Returns
        -------
        PandasMoveDataFrame or None
            DataFrame with the renamed axis labels.

        Raises
        ------
        AttributeError
            If trying to rename a required column inplace

        """

        rename_ = super().rename(
            mapper=mapper, index=index, columns=columns, axis=axis, copy=copy
        )

        if inplace:
            if MoveDataFrame.has_columns(rename_):
                self._mgr = rename_._mgr
                self._item_cache = dict()
                rename_ = None
            else:
                raise AttributeError(
                    'Could not rename columns lat, lon, and datetime.'
                )
        if rename_ is not None and MoveDataFrame.has_columns(rename_):
            rename_ = PandasMoveDataFrame(data=rename_)
        return rename_

    def len(self):
        """
        Returns the length/row numbers in trajectory data.

        Returns
        -------
        int
            Represents the trajectory data length.

        """
        return self.shape[0]

    def __getitem__(self, key):
        """Retrieves and item from this object."""
        item = super().__getitem__(key)
        if (
            isinstance(item, pd.DataFrame)
            and MoveDataFrame.has_columns(item)
        ):
            return PandasMoveDataFrame(item)
        return item

    def head(self, n=5):
        """
        Return the first n rows.

        This function returns the first n rows for the object
        based on position. It is useful for quickly testing if
        your object has the right type of data in it.

        Parameters
        ----------
        n : int, default 5.
            Number of rows to select.

        Returns
        -------
        same type as caller
            The first n rows of the caller object.

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.head.html

        """
        head_ = super().head(n=n)
        return PandasMoveDataFrame(data=head_)

    def tail(self, n=5):
        """
        Return the last n rows.

        This function returns the last n rows for the object
        based on position. It is useful for quickly testing if
        your object has the right type of data in it.

        Parameters
        ----------
        n : int, default 5.
            Number of rows to select.

        Returns
        -------
        same type as caller
            The last n rows of the caller object.

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.tail.html

        """
        tail_ = super().tail(n=n)
        return PandasMoveDataFrame(data=tail_)

    def get_users_number(self):
        """
        Check and return number of users in trajectory data.

        Returns
        -------
        int
            Represents the number of users in trajectory data.

        """

        operation = begin_operation('get_users_numbers')

        if UID in self:
            number_ = self[UID].nunique()
        else:
            number_ = 1
        self.last_operation = end_operation(operation)

        return number_

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
        pymove.core.grid
            Represents the trajectory in grid format.

        """
        operation = begin_operation('to_grid')
        grid_ = Grid(
            data=self, cell_size=cell_size, meters_by_degree=meters_by_degree
        )
        self.last_operation = end_operation(operation)
        return grid_

    def to_data_frame(self):
        """
        Converts trajectory data to DataFrame format.

        Returns
        -------
        pandas.core.DataFrame
            Represents the trajectory in DataFrame format.

        """
        return pd.DataFrame(self)

    def to_dicrete_move_df(self, local_label=LOCAL_LABEL):
        """
        Generate a discrete dataframe move.

        Parameters
        ----------
        local_label : str, optional, default 'local_label'.
            Represents the column name of feature local label.

        Returns
        -------
        pymove.core.pandas.PandasDiscreteMoveDataFrame
            Represents an PandasMoveDataFrame discretized.
        """

        operation = begin_operation('to_discrete_move_df')

        if local_label not in self:
            raise ValueError(
                'columns {} not in df'.format(local_label)
            )

        self.last_operation = end_operation(operation)

        from pymove.core.pandas_discrete import PandasDiscreteMoveDataFrame
        return PandasDiscreteMoveDataFrame(
            self, LATITUDE, LONGITUDE, DATETIME, TRAJ_ID, local_label
        )

    def copy(self, deep=True):
        """
        Make a copy of this object’s indices and data.

        When deep=True (default), a new object will be created with a copy
        of the calling object data and indices. Modifications to the
        data or indices of the copy will not be reflected in the original
        object (see notes below).
        When deep=False, a new object will be created without copying the calling
        object data or index (only references to the data and index are copied).
        Any changes to the data of the original will be reflected in the
        shallow copy (and vice versa).

        Parameters
        ----------
        deep : bool, default True
            Make a deep copy, including a copy of the data and the indices.
            With deep=False neither the indices nor the data are copied.

        Returns
        -------
        Series or DataFrame
            Object type matches caller.

        Notes
        -----
        When deep=True, data is copied but actual Python objects will not be
        copied recursively, only the reference to the object.
        This is in contrast to copy.deepcopy in the Standard Library, which
        recursively copies object data (see examples below).
        While Index objects are copied when deep=True, the underlying
        numpy array is not copied for performance reasons. Since Index is
        immutable, the underlying data can be safely shared and a
        copy is not needed.

        """
        copy_ = super().copy(deep=deep)
        return PandasMoveDataFrame(data=copy_)

    def generate_tid_based_on_id_datetime(
        self, str_format='%Y%m%d%H', sort=True, inplace=True
    ):
        """
        Create or update trajectory id based on id and datetime.

        Parameters
        ----------
        str_format : str, optional, default "%Y%m%d%H".
            Format to consider the datetime
        sort : bool, optional, default True.
            If sort == True the dataframe will be sorted.
        inplace : bool, optional, default True.
            Represents whether the operation will be performed on
            the data provided or in a copy.

        Returns
        -------
        PandasMoveDataFrame or None
            Object with new features or None if ``inplace=True``.

        """

        operation = begin_operation('generate_tid_based_on_id_datetime')
        columns = set(self.columns)
        try:
            print('\nCreating or updating tid feature...\n')
            if sort is True:
                print(
                    '...Sorting by %s and %s to increase performance\n'
                    % (TRAJ_ID, DATETIME)
                )

                self.sort_values([TRAJ_ID, DATETIME], inplace=True)

            self[TID] = self[TRAJ_ID].astype(str) + self[
                DATETIME
            ].dt.strftime(str_format)
            print('\n...tid feature was created...\n')

            if inplace:
                self.last_operation = end_operation(operation)
                return None

            mdf = self.copy()
            drop = set(self.columns) - columns
            self.drop(columns=[*drop], inplace=True)
            self.last_operation = end_operation(operation)
            return mdf
        except Exception as e:
            drop = set(self.columns) - columns
            self.drop(columns=[*drop], inplace=True)
            self.last_operation = end_operation(operation)
            raise e

    def generate_date_features(self, inplace=True):
        """
        Create or update date feature based on datetime.

        Parameters
        ----------
        inplace : bool, optional, default True.
            Represents whether the operation will be performed
            on the data provided or in a copy.

        Returns
        -------
        PandasMoveDataFrame or None
            Object with new features or None if ``inplace=True``.

        """

        operation = begin_operation('generate_date_features')
        columns = set(self.columns)
        try:
            print('Creating date features...')
            if DATETIME in self:
                self[DATE] = self[DATETIME].dt.date
                print('..Date features was created...\n')

            if inplace:
                self.last_operation = end_operation(operation)
                return None

            mdf = self.copy()
            drop = set(self.columns) - columns
            self.drop(columns=[*drop], inplace=True)
            self.last_operation = end_operation(operation)
            return mdf
        except Exception as e:
            drop = set(self.columns) - columns
            self.drop(columns=[*drop], inplace=True)
            self.last_operation = end_operation(operation)
            raise e

    def generate_hour_features(self, inplace=True):
        """
        Create or update hour feature based on datetime.

        Parameters
        ----------
        inplace : bool, optional, default True.
            Represents whether the operation will be performed
            on the data provided or in a copy.

        Returns
        -------
        PandasMoveDataFrame or None
            Object with new features or None if ``inplace=True``.

        """

        operation = begin_operation('generate_hour_features')
        columns = set(self.columns)

        try:
            print('\nCreating or updating a feature for hour...\n')
            if DATETIME in self:
                self[HOUR] = self[DATETIME].dt.hour
                print('...Hour feature was created...\n')

            if inplace:
                self.last_operation = end_operation(operation)
                return None

            mdf = self.copy()
            drop = set(self.columns) - columns
            self.drop(columns=[*drop], inplace=True)
            self.last_operation = end_operation(operation)
            return mdf
        except Exception as e:
            drop = set(self.columns) - columns
            self.drop(columns=[*drop], inplace=True)
            self.last_operation = end_operation(operation)
            raise e

    def generate_day_of_the_week_features(self, inplace=True):
        """
        Create or update a feature day of the week from datatime.

        Parameters
        ----------
        inplace : bool, optional, default True.
            Represents whether the operation will be performed
            on the data provided or in a copy.

        Returns
        -------
        PandasMoveDataFrame or None
            Object with new features or None if ``inplace=True``.

        """

        operation = begin_operation('generate_day_of_the_week_features')
        columns = set(self.columns)

        try:
            print('\nCreating or updating day of the week feature...\n')
            self[DAY] = self[DATETIME].dt.day_name()
            print('...the day of the week feature was created...\n')

            if inplace:
                self.last_operation = end_operation(operation)
                return None

            mdf = self.copy()
            drop = set(self.columns) - columns
            self.drop(columns=[*drop], inplace=True)
            self.last_operation = end_operation(operation)
            return mdf
        except Exception as e:
            drop = set(self.columns) - columns
            self.drop(columns=[*drop], inplace=True)
            self.last_operation = end_operation(operation)
            raise e

    def generate_weekend_features(
        self, create_day_of_week=False, inplace=True
    ):
        """
        Create or update the feature weekend to the dataframe,
        if this resource indicates that the given day is the
        weekend, otherwise, it is a day of the week.

        Parameters
        ----------
        create_day_of_week : bool, optional (default False).
            Indicates if the column day should be keeped in the dataframe.
            If set to False the column will be dropped.
        inplace : bool, optional, default True.
            Indicates whether the operation will be performed on
            the data provided or in a copy.

        Returns
        ----------
        PandasMoveDataFrame or None
            Object with new features or None if ``inplace=True``.

        """

        operation = begin_operation('generate_weekend_features')
        columns = set(self.columns)
        try:
            self.generate_day_of_the_week_features(inplace=True)

            print('Creating or updating a feature for weekend\n')
            if DAY in self:
                fds = (self[DAY] == WEEK_DAYS[5]) | (self[DAY] == WEEK_DAYS[6])
                index_fds = self[fds].index
                self[WEEK_END] = 0
                self.at[index_fds, WEEK_END] = 1
                print('...Weekend was set as 1 or 0...\n')
                if not create_day_of_week:
                    print('...dropping colum day\n')
                    del self[DAY]

            if inplace:
                self.last_operation = end_operation(operation)
                return None

            mdf = self.copy()
            drop = set(self.columns) - columns
            self.drop(columns=[*drop], inplace=True)
            self.last_operation = end_operation(operation)
            return mdf
        except Exception as e:
            drop = set(self.columns) - columns
            self.drop(columns=[*drop], inplace=True)
            self.last_operation = end_operation(operation)
            raise e

    def generate_time_of_day_features(self, inplace=True):
        """
        Create a feature time of day or period from datatime.

        Parameters
        ----------
         inplace : bool, optional, default True.
            Represents whether the operation will be performed on
            the data provided or in a copy.

        Returns
        -------
        PandasMoveDataFrame or None
            Object with new features or None if ``inplace=True``.

        Examples
        --------
        - datetime1 = 2019-04-28 02:00:56 -> period = Early Morning
        - datetime2 = 2019-04-28 08:00:56 -> period = Morning
        - datetime3 = 2019-04-28 14:00:56 -> period = Afternoon
        - datetime4 = 2019-04-28 20:00:56 -> period = Evening

        """

        operation = begin_operation('generate_time_of_day_features')
        columns = set(self.columns)

        try:
            periods = [
                '\n' 'Creating or updating period feature',
                '...Early morning from 0H to 6H',
                '...Morning from 6H to 12H',
                '...Afternoon from 12H to 18H',
                '...Evening from 18H to 24H' '\n',
            ]
            print('\n'.join(periods))

            hours = self[DATETIME].dt.hour
            conditions = [
                (hours >= 0) & (hours < 6),
                (hours >= 6) & (hours < 12),
                (hours >= 12) & (hours < 18),
                (hours >= 18) & (hours < 24),
            ]
            self[PERIOD] = np.select(conditions, DAY_PERIODS, 'undefined')
            print('...the period of day feature was created')

            if inplace:
                self.last_operation = end_operation(operation)
                return None

            mdf = self.copy()
            drop = set(self.columns) - columns
            self.drop(columns=[*drop], inplace=True)
            self.last_operation = end_operation(operation)
            return mdf
        except Exception as e:
            drop = set(self.columns) - columns
            self.drop(columns=[*drop], inplace=True)
            self.last_operation = end_operation(operation)
            raise e

    def generate_datetime_in_format_cyclical(
        self, label_datetime=DATETIME, inplace=True
    ):
        """
        Create or update column with cyclical datetime feature.

        Parameters
        ----------
        label_datetime : str, optional, default 'datetime'.
            Represents column id type.
        inplace : bool, optional, default True.
            Represents whether the operation will be performed on
            the data provided or in a copy.

        Returns
        -------
        PandasMoveDataFrame or None
            Object with new features or None if ``inplace=True``.

        References
        ----------
        https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/
        https://www.avanwyk.com/encoding-cyclical-features-for-deep-learning/

        """

        operation = begin_operation('generate_datetime_in_format_cyclical')
        columns = set(self.columns)

        try:
            print('Encoding cyclical continuous features - 24-hour time')
            if label_datetime in self:
                hours = self[label_datetime].dt.hour
                self[HOUR_SIN] = np.sin(2 * np.pi * hours / 23.0)
                self[HOUR_COS] = np.cos(2 * np.pi * hours / 23.0)
                print('...hour_sin and  hour_cos features were created...\n')

            if inplace:
                self.last_operation = end_operation(operation)
                return None

            mdf = self.copy()
            drop = set(self.columns) - columns
            self.drop(columns=[*drop], inplace=True)
            self.last_operation = end_operation(operation)
            return mdf
        except Exception as e:
            drop = set(self.columns) - columns
            self.drop(columns=[*drop], inplace=True)
            self.last_operation = end_operation(operation)
            raise e

    @staticmethod
    def _prepare_generate_data(data_, sort, label_id):
        """
        Processes the data and create variables for generate methods.

        Parameters
        ----------
        data_ : dataframe
            Dataframe to be processed.
        sort : bool
            Whether to sort the data.
        label_id : str
            Name of the label feature.

        Returns
        -------
        array
            data_ unique ids.
        int
            sum size of id.
        int
            size of id.
        int
            starting index

        """

        if sort is True:
            print(
                '...Sorting by %s and %s to increase performance\n'
                % (label_id, DATETIME),
                flush=True,
            )
            data_.sort_values([label_id, DATETIME], inplace=True)

        if data_.index.name is None:
            print(
                '...Set %s as index to a higher performance\n'
                % label_id,
                flush=True,
            )
            data_.set_index(label_id, inplace=True)

        ids = data_.index.unique()
        sum_size_id = 0
        size_id = 0
        idx = None

        return ids, sum_size_id, size_id, idx

    def _return_generated_data(self, data_, columns, operation, inplace):
        """
        Finishes the generate methods.

        Parameters
        ----------
        data_ : dataframe
            Dataframe with the generated features.
        columns: set
            Set with columns before operation
        operation : dict
            initial stats of the operation.
        inplace : bool, optional, default True.
            Represents whether the operation will be performed on
            the data provided or in a copy.

        Returns
        -------
        PandasMoveDataFrame or None
            Object with new features or None if ``inplace=True``.

        """
        print('...Reset index...\n')

        data_.reset_index(inplace=True)
        if inplace:
            self.last_operation = end_operation(operation)
            return None
        data_ = self.copy()
        drop = set(self.columns) - columns
        self.drop(columns=[*drop], inplace=True)
        self.last_operation = end_operation(operation)
        return data_

    def generate_dist_time_speed_features(
        self, label_id=TRAJ_ID, label_dtype=np.float64, sort=True, inplace=True
    ):
        """
        Firstly, create the three distance to an GPS point P (lat, lon). After,
        create two time features to point P: time to previous and time to next.
        Lastly, create two features to speed using time and distance features.

        Parameters
        ----------
        label_id : str, optional, default 'id'.
            Represents name of column of trajectories id.
        label_dtype : type, optional, default np.float64.
            Represents column id type.
        sort : bool, optional, default True.
            If sort == True the dataframe will be sorted.
        inplace : bool, optional, default True.
            Represents whether the operation will be performed on
            the data provided or in a copy.

        Returns
        -------
        PandasMoveDataFrame or None
            Object with new features or None if ``inplace=True``.

        Examples
        --------
        - dist_to_prev =  248.33 meters, dist_to_prev 536.57 meters
        - time_to_prev = 60 seconds, time_prev = 60.0 seconds
        - speed_to_prev = 4.13 m/srs, speed_prev = 8.94 m/srs.

        """

        operation = begin_operation('generate_dist_time_speed_features')
        columns = set(self.columns)
        ids, sum_size_id, size_id, idx = self._prepare_generate_data(
            self, sort, label_id
        )

        try:
            message = '\nCreating or updating distance, time and speed features'
            message += ' in meters by seconds\n'
            print(
                message
            )

            # create new feature to distance
            self[DIST_TO_PREV] = label_dtype(-1.0)

            # create new feature to time
            self[TIME_TO_PREV] = label_dtype(-1.0)

            # create new feature to speed
            self[SPEED_TO_PREV] = label_dtype(-1.0)

            for idx in progress_bar(
                ids, desc='Generating distance, time and speed features'
            ):
                curr_lat = self.at[idx, LATITUDE]
                curr_lon = self.at[idx, LONGITUDE]

                size_id = curr_lat.size

                if size_id <= 1:
                    self.at[idx, DIST_TO_PREV] = np.nan
                    self.at[idx, TIME_TO_PREV] = np.nan
                    self.at[idx, SPEED_TO_PREV] = np.nan
                else:
                    prev_lat = shift(curr_lat, 1)
                    prev_lon = shift(curr_lon, 1)
                    # compute distance from previous to current point
                    self.at[idx, DIST_TO_PREV] = haversine(
                        prev_lat, prev_lon, curr_lat, curr_lon
                    )

                    time_ = self.at[idx, DATETIME].values.astype(label_dtype)
                    time_prev = (time_ - shift(time_, 1)) * (10 ** -9)
                    self.at[idx, TIME_TO_PREV] = time_prev

                    # set speed features
                    self.at[idx, SPEED_TO_PREV] = (
                        self.at[idx, DIST_TO_PREV] / time_prev
                    )  # unit: m/srs

            return self._return_generated_data(
                self, columns, operation, inplace
            )

        except Exception as e:
            print(
                'label_tid:%s\nidx:%s\nsize_id:%s\nsum_size_id:%s'
                % (label_id, idx, size_id, sum_size_id)
            )
            drop = set(self.columns) - columns
            self.drop(columns=[*drop], inplace=True)
            self.last_operation = end_operation(operation)
            raise e

    def generate_dist_features(
        self, label_id=TRAJ_ID, label_dtype=np.float64, sort=True, inplace=True
    ):
        """
        Create the three distance in meters to an GPS point P.

        Parameters
        ----------
        label_id : str, optional, default 'id'.
            Represents name of column of trajectories id.
        label_dtype : type, optional, default np.float64.
            Represents column id type.
        sort : bool, optional, default True.
            If sort == True the dataframe will be sorted.
        inplace : bool, optional, default True.
            Represents whether the operation will be performed on
            the data provided or in a copy.

        Returns
        -------
        PandasMoveDataFrame or None
            Object with new features or None if ``inplace=True``.

        Examples
        --------
        - P to P.next = 2 meters
        - P to P.previous = 1 meter
        - P.previous to P.next = 1 meters

        """

        operation = begin_operation('generate_dist_features')
        columns = set(self.columns)
        ids, sum_size_id, size_id, idx = self._prepare_generate_data(
            self, sort, label_id
        )

        try:
            print('\nCreating or updating distance features in meters...\n')

            # create ou update columns
            self[DIST_TO_PREV] = label_dtype(-1.0)
            self[DIST_TO_NEXT] = label_dtype(-1.0)
            self[DIST_PREV_TO_NEXT] = label_dtype(-1.0)

            for idx in progress_bar(ids, desc='Generating distance features'):
                curr_lat = self.at[idx, LATITUDE]
                curr_lon = self.at[idx, LONGITUDE]

                size_id = curr_lat.size

                if size_id <= 1:
                    self.at[idx, DIST_TO_PREV] = np.nan

                else:
                    prev_lat = shift(curr_lat, 1)
                    prev_lon = shift(curr_lon, 1)
                    # compute distance from previous to current point
                    self.at[idx, DIST_TO_PREV] = haversine(
                        prev_lat, prev_lon, curr_lat, curr_lon
                    )

                    next_lat = shift(curr_lat, -1)
                    next_lon = shift(curr_lon, -1)
                    # compute distance to next point
                    self.at[idx, DIST_TO_NEXT] = haversine(
                        curr_lat, curr_lon, next_lat, next_lon
                    )

                    # using pandas shift in a large dataset: 7min 21s
                    # using numpy shift above: 33.6 srs

                    # use distance from previous to next
                    self.at[idx, DIST_PREV_TO_NEXT] = haversine(
                        prev_lat, prev_lon, next_lat, next_lon
                    )

            return self._return_generated_data(
                self, columns, operation, inplace
            )

        except Exception as e:
            print(
                'label_tid:%s\nidx:%s\nsize_id:%s\nsum_size_id:%s'
                % (label_id, idx, size_id, sum_size_id)
            )
            drop = set(self.columns) - columns
            self.drop(columns=[*drop], inplace=True)
            self.last_operation = end_operation(operation)
            raise e

    def generate_time_features(
        self, label_id=TRAJ_ID, label_dtype=np.float64, sort=True, inplace=True
    ):
        """
        Create the three time in seconds to an GPS point P.

        Parameters
        ----------
        label_id : str, optional, default 'id'.
            Represents name of column of trajectories id.
        label_dtype : type, optional, default np.float64.
            Represents column id type_.
        sort : bool, optional, default True.
            If sort == True the dataframe will be sorted.
        inplace : bool, optional, default True.
            Represents whether the operation will be performed on
            the data provided or in a copy.

        Returns
        -------
        PandasMoveDataFrame or None
            Object with new features or None if ``inplace=True``.

        Examples
        --------
        - P to P.next = 5 seconds
        - P to P.previous = 15 seconds
        - P.previous to P.next = 20 seconds

        """

        operation = begin_operation('generate_time_features')
        columns = set(self.columns)
        ids, sum_size_id, size_id, idx = self._prepare_generate_data(
            self, sort, label_id
        )

        try:
            print(
                '\nCreating or updating time features seconds\n'
            )

            # create new feature to time
            self[TIME_TO_PREV] = label_dtype(-1.0)
            self[TIME_TO_NEXT] = label_dtype(-1.0)
            self[TIME_PREV_TO_NEXT] = label_dtype(-1.0)

            for idx in progress_bar(
                ids, desc='Generating time features'
            ):
                curr_time = self.at[idx, DATETIME].values.astype(label_dtype)

                size_id = curr_time.size

                if size_id <= 1:
                    self.at[idx, TIME_TO_PREV] = np.nan
                else:
                    prev_time = shift(curr_time, 1)
                    time_prev = (curr_time - prev_time) * (10 ** -9)
                    self.at[idx, TIME_TO_PREV] = time_prev

                    next_time = shift(curr_time, -1)
                    time_prev = (next_time - curr_time) * (10 ** -9)
                    self.at[idx, TIME_TO_NEXT] = time_prev

                    time_prev_to_next = (next_time - prev_time) * (10 ** -9)
                    self.at[idx, TIME_PREV_TO_NEXT] = time_prev_to_next

            return self._return_generated_data(
                self, columns, operation, inplace
            )

        except Exception as e:
            print(
                'label_tid:%s\nidx:%s\nsize_id:%s\nsum_size_id:%s'
                % (label_id, idx, size_id, sum_size_id)
            )
            drop = set(self.columns) - columns
            self.drop(columns=[*drop], inplace=True)
            self.last_operation = end_operation(operation)
            raise e

    def generate_speed_features(
            self,
            label_id=TRAJ_ID,
            label_dtype=np.float64,
            sort=True,
            inplace=True
    ):
        """
        Create the three speed in meter by seconds to an GPS point P.

        Parameters
        ----------
        label_id : str, optional, default 'id'.
            Represents name of column of trajectories id.
        label_dtype : type, optional, default np.float64.
            Represents column id type_.
        sort : bool, optional, default True.
            If sort == True the dataframe will be sorted.
        inplace : bool, optional, default True.
            Represents whether the operation will be performed on
            the data provided or in a copy.

        Returns
        -------
        PandasMoveDataFrame or None
            Object with new features or None if ``inplace=True``.

        Examples
        --------
        - P to P.next = 1 meter/seconds
        - P to P.previous = 3 meter/seconds
        - P.previous to P.next = 2 meter/seconds

        """

        operation = begin_operation('generate_speed_features')
        columns = set(self.columns)

        try:
            print(
                '\nCreating or updating speed features meters by seconds\n'
            )

            dist_cols = [DIST_TO_PREV, DIST_TO_NEXT, DIST_PREV_TO_NEXT]
            time_cols = [TIME_TO_PREV, TIME_TO_NEXT, TIME_PREV_TO_NEXT]

            dists = self.generate_dist_features(
                label_id, label_dtype, sort, inplace=False
            )[dist_cols]
            times = self.generate_time_features(
                label_id, label_dtype, sort, inplace=False
            )[time_cols]

            self[SPEED_TO_PREV] = dists[DIST_TO_PREV] / times[TIME_TO_PREV]
            self[SPEED_TO_NEXT] = dists[DIST_TO_NEXT] / times[TIME_TO_NEXT]

            d_prev_next = dists[DIST_TO_PREV] + dists[DIST_TO_NEXT]
            self[SPEED_PREV_TO_NEXT] = d_prev_next / times[TIME_PREV_TO_NEXT]
            self._prepare_generate_data(
                self, sort, label_id
            )
            return self._return_generated_data(
                self, columns, operation, inplace
            )

        except Exception as e:
            drop = set(self.columns) - columns
            self.drop(columns=[*drop], inplace=True)
            self.last_operation = end_operation(operation)
            raise e

    def generate_move_and_stop_by_radius(
        self, radius=0, target_label=DIST_TO_PREV, inplace=True
    ):
        """
        Create or update column with move and stop points by radius.

        Parameters
        ----------
        radius : int, optional, default 0.
            Represents radius.
        target_label : str, optional, default 'dist_to_prev.
            Represents column id type.
        inplace : bool, optional, default True.
            Represents whether the operation will be performed on
            the data provided or in a copy.

        Returns
        -------
        PandasMoveDataFrame or None
            Object with new features or None if ``inplace=True``.

        """

        operation = begin_operation('generate_move_and_stop_by_radius')
        columns = set(self.columns)

        try:
            self.generate_dist_features(inplace=True)

            print('\nCreating or updating features MOVE and STOPS...\n')
            conditions = (
                (self[target_label] > radius),
                (self[target_label] <= radius),
            )
            choices = [MOVE, STOP]

            self[SITUATION] = np.select(conditions, choices, np.nan)
            print(
                '\n....There are %s stops to this parameters\n'
                % (self[self[SITUATION] == STOP].shape[0])
            )

            if inplace:
                self.last_operation = end_operation(operation)
                return None

            mdf = self.copy()
            drop = set(self.columns) - columns
            self.drop(columns=[*drop], inplace=True)
            self.last_operation = end_operation(operation)
            return mdf
        except Exception as e:
            drop = set(self.columns) - columns
            self.drop(columns=[*drop], inplace=True)
            self.last_operation = end_operation(operation)
            raise e

    def time_interval(self):
        """
        Get time difference between max and min datetime in trajectory data.

        Returns
        -------
        datetime64
            Represents the time difference.

        """

        operation = begin_operation('time_interval')
        time_diff = self[DATETIME].max() - self[DATETIME].min()
        self.last_operation = end_operation(operation)

        return time_diff

    def get_bbox(self):
        """
        A bounding box (usually shortened to bbox) is an area defined by two
        longitudes and two latitudes, where:

            - Latitude is a decimal number between -90.0 and 90.0.
            - Longitude is a decimal number between -180.0 and 180.0.
        They usually follow the standard format of:
        - bbox = left, bottom, right, top
        - bbox = min Longitude , min Latitude , max Longitude , max Latitude

        Returns
        -------
        tuple
            Represents a bound box, that is a tuple of 4 values with
            the min and max limits of latitude e longitude.

        Examples
        --------
        (22.147577, 113.54884299999999, 41.132062, 121.156224)

        """

        operation = begin_operation('get_bbox')

        bbox_ = (
            self[LATITUDE].min(),
            self[LONGITUDE].min(),
            self[LATITUDE].max(),
            self[LONGITUDE].max(),
        )

        self.last_operation = end_operation(operation)

        return bbox_

    def plot_all_features(
        self,
        dtype=np.float64,
        figsize=(21, 15),
        return_fig=True,
        save_fig=False,
        name='features.png',
    ):
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
        name : str, optional, default 'features.png'.
            Represents name of a file.

        Returns
        -------
        matplotlib.pyplot.figure or None
            The generated picture.

        Raises
        ------
        AttributeError
            If there are no columns with the specified type

        """

        operation = begin_operation('plot_all_features')

        try:
            col_dtype = self.select_dtypes(include=[dtype]).columns
            tam = col_dtype.size
            if not tam:
                raise AttributeError('No columns with dtype %s.' % dtype)

            fig, ax = plt.subplots(tam, 1, figsize=figsize)
            ax_count = 0
            for col in col_dtype:
                ax[ax_count].set_title(col)
                self[col].plot(subplots=True, ax=ax[ax_count])
                ax_count += 1

            if save_fig:
                plt.savefig(fname=name, fig=fig)

            self.last_operation = end_operation(operation)

            if return_fig:
                return fig
        except Exception as e:
            self.last_operation = end_operation(operation)
            raise e

    def plot_trajs(
        self,
        markers='o',
        markersize=20,
        figsize=(10, 10),
        return_fig=True,
        save_fig=False,
        name='trajectories.png',
    ):
        """
        Generate a visualization that show trajectories.

        Parameters
        ----------
        figsize : tuple, optional, default (10, 10).
            Represents dimensions of figure.
        markers : str, optional, default 'o'.
            Represents visualization type marker.
        markersize : int, optional, default 20.
            Represents visualization size marker.
        return_fig : bool, optional, default True.
            Represents whether or not to save the generated picture.
        save_fig : bool, optional, default False.
            Represents whether or not to save the generated picture.
        name : str, optional, default 'trajectories.png'.
            Represents name of a file.

        Returns
        -------
        matplotlib.pyplot.figure or None
            The generated picture.

        """

        operation = begin_operation('plot_trajs')

        fig = plt.figure(figsize=figsize)

        ids = self['id'].unique()
        for id_ in ids:
            self_id = self[self['id'] == id_]
            plt.plot(
                self_id[LONGITUDE],
                self_id[LATITUDE],
                markers,
                markersize=markersize,
            )

        if save_fig:
            plt.savefig(fname=name, fig=fig)

        self.last_operation = end_operation(operation)

        if return_fig:
            return fig

    def plot_traj_id(
        self,
        tid,
        label=TID,
        feature=None,
        value=None,
        figsize=(10, 10),
        return_fig=True,
        save_fig=False,
        name=None,
    ):
        """
        Generate a visualization that shows a trajectory with the specified tid.

        Parameters
        ----------
        tid : any.
            Represents the trajectory tid.
        label : str, optional, default 'traj_id'.
            Feature with trajectories tids.
        feature : str, optional, default None.
            Name of the feature to highlight on plot.
        value : any, optional, default None.
            Value of the feature to be highlighted as green marker
        figsize : tuple, optional, default (10,10).
            Represents dimensions of figure.
        return_fig : bool, optional, default True.
            Represents whether or not to save the generated picture.
        save_fig : bool, optional, default False.
            Represents whether or not to save the generated picture.
        name : str, optional, default None.
            Represents name of a file.

        Returns
        -------
        pymove.core.MoveDataFrameAbstract subclass
            Trajectory with the specified tid.
        matplotlib.pyplot.figure or None
            The generated picture.

        Raises
        ------
        KeyError
            If the dataframe does not contains the TID feature
        IndexError
            If there is no trajectory with the tid passed

        """

        operation = begin_operation('plot_traj_id')

        if label not in self:
            self.last_operation = end_operation(operation)
            raise KeyError('%s feature not in dataframe' % label)

        df_ = self[self[label] == tid]

        if not len(df_):
            self.last_operation = end_operation(operation)
            raise IndexError(f'No trajectory with tid {tid} in dataframe')

        fig = plt.figure(figsize=figsize)

        plt.plot(
            df_.iloc[0][LONGITUDE], df_.iloc[0][LATITUDE], 'yo', markersize=23
        )  # start point
        plt.plot(
            df_.iloc[-1][LONGITUDE], df_.iloc[-1][LATITUDE], 'yX', markersize=23
        )  # end point

        if (not feature) or (not value) or (feature not in df_):
            plt.plot(df_[LONGITUDE], df_[LATITUDE])
            plt.plot(
                df_.loc[:, LONGITUDE], df_.loc[:, LATITUDE], 'r.', markersize=8
            )
        else:
            filter_ = df_[feature] == value
            df_nodes = df_.loc[filter_]
            df_points = df_.loc[~filter_]
            plt.plot(df_[LONGITUDE], df_[LATITUDE], linewidth=3)
            plt.plot(
                df_nodes[LONGITUDE], df_nodes[LATITUDE], 'gs', markersize=13
            )
            plt.plot(
                df_points[LONGITUDE], df_points[LATITUDE], 'r.', markersize=8
            )

        if save_fig:
            if not name:
                name = 'trajectory_%s.png' % tid
            plt.savefig(fname=name, fig=fig)

        df_ = PandasMoveDataFrame(df_)
        self.last_operation = end_operation(operation)

        if return_fig:
            return df_, fig
        return df_

    def show_trajectories_info(self):
        """
        Show dataset information from dataframe, this is number of rows,
        datetime interval, and bounding box.

        Examples
        --------
        ====================== INFORMATION ABOUT DATASET ======================
        Number of Points: 217654
        Number of IDs objects: 2
        Start Date:2008-10-23 05:53:05     End Date:2009-03-19 05:46:37
        Bounding Box:(22.147577, 113.54884299999999, 41.132062, 121.156224)
        =======================================================================
        """

        operation = begin_operation('show_trajectories_info')

        try:
            message = ('=' * 22) + ' INFORMATION ABOUT DATASET ' + ('=' * 22)
            print(
                '\n%s\n' % message
            )
            print('Number of Points: %s\n' % self.shape[0])

            if TRAJ_ID in self:
                print(
                    'Number of IDs objects: %s\n'
                    % self[TRAJ_ID].nunique()
                )

            if TID in self:
                print(
                    'Number of TIDs trajectory: %s\n'
                    % self[TID].nunique()
                )

            if DATETIME in self:
                dt_max = self[DATETIME].max()
                dt_min = self[DATETIME].min()
                print(
                    'Start Date:%s     End Date:%s\n'
                    % (dt_min, dt_max)
                )

            if LATITUDE and LONGITUDE in self:
                print(
                    'Bounding Box:%s\n' % (self.get_bbox(),)
                )  # bbox return =  Lat_min , Long_min, Lat_max, Long_max

            if TIME_TO_PREV in self:
                t_max = round(self[TIME_TO_PREV].max(), 3)
                t_min = round(self[TIME_TO_PREV].min(), 3)
                print(
                    'Gap time MAX:%s     Gap time MIN:%s\n'
                    % (t_max, t_min)
                )

            if SPEED_TO_PREV in self:
                s_max = round(self[SPEED_TO_PREV].max(), 3)
                s_min = round(self[SPEED_TO_PREV].min(), 3)
                print(
                    'Speed MAX:%s    Speed MIN:%s\n'
                    % (s_max, s_min)
                )

            if DIST_TO_PREV in self:
                d_max = round(self[DIST_TO_PREV].max(), 3)
                d_min = round(self[DIST_TO_PREV].min(), 3)
                print(
                    'Distance MAX:%s    Distance MIN:%s\n'
                    % (d_max, d_min)
                )

            print(
                '\n%s\n' % ('=' * len(message))
            )

            self.last_operation = end_operation(operation)
        except Exception as e:
            self.last_operation = end_operation(operation)
            raise e

    def astype(self, dtype, copy=True, errors='raise'):
        """
        Cast a pandas object to a specified dtype.

        Parameters
        ----------
        dtype: data type, or dict of column name -> data type
            Use a numpy.dtype or Python type to cast entire pandas object
            to the same type. Alternatively, use {col: dtype, …},
            where col is a column label and dtype is a numpy.dtype
            or Python type to cast one or more of the DataFrame
            columns to column-specific types.
        copy: bool, optional, default None
            Return a copy when copy=True (be very careful setting
            copy=False as changes to values then
            may propagate to other pandas objects).
        errors: 'raise', 'ignore', optional, default raise
            Control raising of exceptions on invalid data for provided dtype.
            - raise : allow exceptions to be raised
            - ignore : suppress exceptions. On error return original object

        Returns
        -------
        DataFrame
            Casted object to specified type.

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.astype.html

        Raises
        ------
        AttributeError
            If trying to change required types inplace

        """

        if not copy and isinstance(dtype, str):
            raise AttributeError(
                'Could not change lat, lon, and datetime type.'
            )
        elif not copy and isinstance(dtype, dict):
            keys = set(list(dtype.keys()))
            columns = {LATITUDE, LONGITUDE, DATETIME}
            if keys & columns:
                raise AttributeError(
                    'Could not change lat, lon, and datetime type.'
                )

        return super().astype(dtype=dtype, copy=copy, errors=errors)

    def sort_values(
        self,
        by,
        axis=0,
        ascending=True,
        inplace=False,
        kind='quicksort',
        na_position='last',
    ):
        """
        Sorts the values of the _data, along an axis.

        Parameters
        ----------
        by: str or list of str
            Name or list of names to sort the _data by.
        axis: Integer, optional, default None, {index (0), columns (1)}
            if set to 0 or 'index', will count for each column.
            if set to 1 or 'columns', will count for each row
        ascending: boolean or list of boolean, default True.
            Sort ascending vs. descending. Specify list for
            multiple sort orders.
            If this is a list of bool, must match the length of the by.
        inplace: Boolean, optional, default False
            if set to true the original dataframe will be altered,
            the duplicates will be dropped in place,
            otherwise the operation will be made in a copy,
            that will be returned.
        kind: 'quicksort', 'mergesort', 'heapsort', default 'quicksort'.
            Choice of sorting algorithm.
            For DataFrames, this option is only applied when sorting
            on a single column or label.
        na_position: 'first', 'last'.
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

        _sort_values = super().sort_values(
            by=by, axis=axis, ascending=ascending,
            inplace=False, kind=kind, na_position=na_position
        )

        if inplace:
            self._mgr = _sort_values._mgr
            self._item_cache = dict()
            return None
        return PandasMoveDataFrame(data=_sort_values)

    def reset_index(
        self, level=None, drop=False, inplace=False, col_level=0, col_fill=''
    ):
        """
        Resets the DataFrame's index, and use the default one. One or more
        levels can be removed, if the DataFrame has a MultiIndex.

        Parameters
        ----------
        level: int, str, tuple, or list. Optional, default None
            Only the levels specify will be removed from the index.
            If set to None, all levels are removed.
        drop: boolean, optional, default False
            Do not try to insert index into dataframe columns.
            This resets the index to the default integer index.
        inplace: bool, optional, default False
            Modify the DataFrame in place (do not create a new object).
        col_level: int or str, default 0
            If the columns have multiple levels, determines which level
            the labels are inserted into.
            By default it is inserted into the first level..
        col_fill: object, default ''
            If the columns have multiple levels, determines how
            the other levels are named.
            If None then the index name is repeated.

        Returns
        -------
        PandasMoveDataFrame or None
            Object with a resetted indexes or None if ``inplace=True``.

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.reset_index.html

        """

        _reset_index = super().reset_index(
            level=level, drop=drop, inplace=False, col_level=col_level, col_fill=col_fill
        )

        if inplace:
            self._mgr = _reset_index._mgr
            self._item_cache = dict()
            return None
        return PandasMoveDataFrame(data=_reset_index)

    def set_index(
        self,
        keys,
        drop=True,
        append=False,
        inplace=False,
        verify_integrity=False,
    ):
        """
        Set the DataFrame index (row labels) using one or more existing columns
        or arrays (of the correct length).

        Parameters
        ----------
        keys: str or array of str.
            label or array-like or list of labels/arrays.
            This parameter can be either a single column key, a single
            array of the same length as the calling DataFrame,
            or a list containing an arbitrary combination of
            column keys and arrays.
        drop: bool, optional (True by default)
            Delete columns to be used as the new index.
        append: bool, optional (False by default)
            Whether to append columns to existing index.
        inplace: bool, optional (False by default)
            Modify the DataFrame in place (do not create a new object).
        verify_integrity: bool, optional (False by default)
            Check the new index for duplicates.
            Otherwise defer the check until necessary.
            Setting to False will improve the performance of this method.

        Returns
        -------
        PandasMoveDataFrame or None
            Object with a new index or None if ``inplace=True``.

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.set_index.html

        Raises
        ------
        AttributeError
            If trying to change required columns types

        """

        if inplace and drop:
            if isinstance(keys, str):
                aux = {keys}
            else:
                aux = set(keys)
            columns = {LATITUDE, LONGITUDE, DATETIME}
            if aux & columns:
                raise AttributeError(
                    'Could not change lat, lon, and datetime type.'
                )

        _set_index = super().set_index(
            keys=keys, drop=drop, append=append,
            inplace=False, verify_integrity=verify_integrity
        )
        if inplace:
            self._mgr = _set_index._mgr
            self._item_cache = dict()
            _set_index = None
        if _set_index is not None and MoveDataFrame.has_columns(_set_index):
            _set_index = PandasMoveDataFrame(data=_set_index)

        return _set_index

    def drop(
        self,
        labels=None,
        axis=0,
        index=None,
        columns=None,
        level=None,
        inplace=False,
        errors='raise',
    ):
        """
        Remove rows or columns by specifying label names and corresponding axis,
        or by specifying directly index or column names. When using a multi-
        index, labels on different levels can be removed by specifying the
        level.

        Parameters
        ----------
        labels: str or array of str
            Index or column labels to drop.
        axis: str or int, optional, default 0
            Whether to drop labels from the index (0 or 'index')
            or columns (1 or 'columns').
        index: str or array of str, optional (None by default)
            Alternative to specifying axis
            (labels, axis=0 is equivalent to index=labels).
        columns: str or array of str, optional (None by default)
            Alternative to specifying axis
            (labels, axis=1 is equivalent to columns=labels).
        level: int or str, optional (None by default)
            For MultiIndex, level from which the labels will be removed.
        inplace: bool, optional (False by default)
            If True, do operation inplace and return None.
            Otherwise, make a copy, do operations and return.
        errors:'ignore', 'raise', optional, default 'raise'
            If 'ignore', suppress error and only existing labels are dropped.

        Returns
        -------
        PandasMoveDataFrame or None
            Object without the removed index or column labels.

        Raises
        ------
        AttributeError
            If trying to drop a required column inplace
        KeyError
            If any of the labels is not found in the selected axis.

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html

        """

        if inplace:
            _labels1 = set()
            _labels2 = set()
            if labels is not None:
                if isinstance(labels, str):
                    _labels1 = {labels}
                else:
                    _labels1 = set(labels)
            elif columns is not None:
                if isinstance(columns, str):
                    _labels2 = {columns}
                else:
                    _labels2 = set(columns)
            _columns = {LATITUDE, LONGITUDE, DATETIME}
            if (
                    (axis == 1 or axis == 'columns' or columns)
                    and (_labels1.union(_labels2) & _columns)
            ):
                raise AttributeError(
                    'Could not drop columns lat, lon, and datetime.'
                )

        _drop = super().drop(
            labels=labels, axis=axis, index=index, columns=columns,
            level=level, inplace=False, errors=errors
        )

        if inplace:
            self._mgr = _drop._mgr
            self._item_cache = dict()
            _drop = None
        if _drop is not None and MoveDataFrame.has_columns(_drop):
            _drop = PandasMoveDataFrame(data=_drop)

        return _drop

    def drop_duplicates(self, subset=None, keep='first', inplace=False):
        """
        Uses the pandas's function drop_duplicates, to remove duplicated rows
        from data.

        Parameters
        ----------
        subset: int or str, optional, default None
            Only consider certain columns for identifying duplicates,
            by default use all of the columns
        keep: 'first', 'lasts', False, optional, default 'first'
            - first : Drop duplicates except for the first occurrence.
            - last : Drop duplicates except for the last occurrence.
            - False : Drop all duplicates.
        inplace: bool, optional, default False
            Whether to drop duplicates in place or to return a copy

        Returns
        -------
        PandasMoveDataFrame or None
            Object with duplicated rows or None if ``inplace=True``.

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html

        """
        _drop_duplicates = super().drop_duplicates(
            subset=subset, keep=keep, inplace=False
        )

        if inplace:
            self._mgr = _drop_duplicates._mgr
            self._item_cache = dict()
            return None
        return PandasMoveDataFrame(data=_drop_duplicates)

    def shift(self, periods=1, freq=None, axis=0, fill_value=None):
        """
        Shift index by desired number of periods with an optional time freq.

        Parameters
        ----------
        periods: int, optional, default 1
            Number of periods to shift. Can be positive or negative.
        freq: pandas.DateOffset, pandas.Timedelta or str, optional, default None
            Offset to use from the series module or time rule (e.g. 'EOM').
            If freq is specified then the index values are shifted but
            the data is not realigned. That is, use freq if you would like
            to extend the index when shifting and preserve the original data.
            When freq is not passed, shift the index without realigning the
            data. If freq is passed (in this case, the index must be
            date or datetime, or it will raise a NotImplementedError),
            the index will be increased using the periods and the freq.
        axis: 0 or 'index', 1 or 'columns', None, optional, default 0
            Shift direction.
        fill_value: object, optional, default None
            The scalar value to use for newly introduced missing values.
            The default depends on the dtype of self.
            For numeric data, np.nan is used.
            For datetime, timedelta, or period data, etc.
            NaT is used. For extension dtypes, self.dtype.na_value is used.

        Returns
        -------
        PandasMoveDataFrame
            A copy of the original object, shifted.

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.shift.html

        """
        _shift = super().shift(
            periods=periods, freq=freq, axis=axis, fill_value=fill_value
        )
        return PandasMoveDataFrame(data=_shift)

    def fillna(
        self,
        value=None,
        method=None,
        axis=None,
        inplace=False,
        limit=None,
        downcast=None,
    ):
        """
        Fill NA/NaN values using the specified method.

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
            or the str 'infer' which will try to downcast to an appropriate
            equal type (e.g. float64 to int64 if possible).

        Returns
        -------
        PandasMoveDataFrame or None
            Object with missing values filled or None if ``inplace=True``.

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html

        """
        _fillna = super().fillna(
            value=value, method=method, axis=axis,
            inplace=False, limit=limit, downcast=downcast
        )

        if inplace:
            self._mgr = _fillna._mgr
            self._item_cache = dict()
            return None
        return PandasMoveDataFrame(data=_fillna)

    def dropna(
        self, axis=0, how='any', thresh=None, subset=None, inplace=False
    ):
        """
        Removes missing data.

        Parameters
        ----------
        axis: 0 or 'index', 1 or 'columns', None, optional, default 0
            Determine if rows or columns are removed.
            - 0, or 'index' : Drop rows which contain missing values.
            - 1, or 'columns' : Drop columns which contain missing value.
        how: str, optional, default 'any', options: 'any', 'all'
            Determine if row or column is removed from DataFrame,
            when we have at least one NA or all NA.
               - 'any' : If any NA values are present, drop that row or column.
               - 'all' : If all values are NA, drop that row or column.
        thresh: int, optional, default None
            Require that many non-NA values.
        subset: array-like, optional, default None
            Labels along other axis to consider,
            e.g. if you are dropping rows these would be a
            list of columns to include.
        inplace: bool, optional (default False)
            If True, do operation inplace and return None

        Returns
        -------
        PandasMoveDataFrame or None
            Object with NA entries dropped or None if ``inplace=True``.

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dropna.html

        Raises
        ------
        AttributeError
            If trying to drop required columns inplace

        """

        if inplace:
            if axis == 1 or axis == 'columns':
                columns = [LATITUDE, LONGITUDE, DATETIME]
                data = self[columns]
                if data.isnull().values.any():
                    raise AttributeError(
                        'Could not drop columns lat, lon, and datetime.'
                    )

        _dropna = super().dropna(
            axis=axis, how=how, thresh=thresh, subset=subset, inplace=False)

        if inplace:
            self._mgr = _dropna._mgr
            self._item_cache = dict()
            _dropna = None
        if _dropna is not None and MoveDataFrame.has_columns(_dropna):
            _dropna = PandasMoveDataFrame(data=_dropna)

        return _dropna

    def sample(
        self,
        n=None,
        frac=None,
        replace=False,
        weights=None,
        random_state=None,
        axis=None,
    ):
        """
        Return a random sample of items from an axis of object.

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
        axis : {0 or 'index', 1 or 'columns', None}, default None
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
        _sample = super().sample(
            n=n, frac=frac, replace=replace, weights=weights,
            random_state=random_state, axis=axis
        )
        return PandasMoveDataFrame(data=_sample)

    def isin(self, values):
        """
        Determines whether each element in the DataFrame is contained in values.

        values : iterable, Series, DataFrame or dict
            The result will only be true at a location if all the labels match.
            If values is a Series, that'srs the index.
            If values is a dict, the keys must be the
            column names, which must match.
            If values is a DataFrame, then both the
            index and column labels must match.

        Returns
        -------
        DataFrame:
            DataFrame of booleans showing whether
            each element in the DataFrame is contained in values

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.isin.html

        """
        return pd.DataFrame(self).isin(values)

    def append(
        self, other, ignore_index=False, verify_integrity=False, sort=None
    ):
        """
        Append rows of other to the end of caller, returning a new object.
        Columns in other that are not in the caller are added as new columns.

        Parameters
        ----------
        other : DataFrame or Series/dict-like object, or list of these
            The data to append.
        ignore_index : bool, optional, default False
            If True, do not use the index labels.
        verify_integrity : bool, optional, default False
            If True, raise ValueError on creating index with duplicates.
        sort : bool, optional, default None
            Sort columns if the columns of self and other are not aligned.
            The default sorting is deprecated and will
            change to not-sorting in a future version of pandas.
            Explicitly pass sort=True to silence the warning and sort.
            Explicitly pass sort=False to silence the warning and not sort.

        Returns
        -------
        PandasMoveDataFrame
            A dataframe containing rows from both the caller and `other`.

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.append.html

        """
        if isinstance(other, PandasMoveDataFrame):
            other = pd.DataFrame(other)

        _append = super().append(
            other=other, ignore_index=ignore_index,
            verify_integrity=verify_integrity, sort=sort
        )
        return PandasMoveDataFrame(data=_append)

    def join(
        self, other, on=None, how='left', lsuffix='', rsuffix='', sort=False
    ):
        """
        Join columns of other, returning a new object.

        Join columns with `other` PandasMoveDataFrame either on index or
        on a key column. Efficiently join multiple DataFrame objects
        by index at once by passing a list.

        Parameters
        ----------
        other : DataFrame, Series, or list of DataFrame
            Index should be similar to one of the columns in this one. If a
            Series is passed, its name attribute must be set, and that will be
            used as the column name in the resulting joined DataFrame.
        on : str, list of str, or array-like, optional
            Column or index level name(srs) in the caller to join on the index
            in `other`, otherwise joins index-on-index. If multiple
            values given, the `other` DataFrame must have a MultiIndex. Can
            pass an array as the join key if it is not already contained in
            the calling DataFrame. Like an Excel VLOOKUP operation.
        how : {'left', 'right', 'outer', 'inner'}, default 'left'
            How to handle the operation of the two objects.

            * left: use calling frame index (or column if on is specified)
            * right: use `other` index.
            * outer: form union of calling frame index (or column if on is
            specified) with `other` index, and sort it.
            lexicographically.
            * inner: form intersection of calling frame index (or column if
            on is specified) with `other` index, preserving the order
            of the calling one.
        lsuffix : str, default ''
            Suffix to use from left frame overlapping columns.
        rsuffix : str, default ''
            Suffix to use from right frame overlapping columns.
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
        if isinstance(other, PandasMoveDataFrame):
            other = other._data

        _join = super().join(
            other=other, on=on, how=how,
            lsuffix=lsuffix, rsuffix=rsuffix, sort=sort)
        return PandasMoveDataFrame(data=_join)

    def merge(
            self,
            right,
            how='inner',
            on=None,
            left_on=None,
            right_on=None,
            left_index=False,
            right_index=False,
            sort=False,
            suffixes=('_x', '_y'),
            copy=True,
            indicator=False,
            validate=None
    ):
        """
        Merge DataFrame or named Series objects with a database-style join.

        The join is done on columns or indexes. If joining columns on columns,
        the DataFrame indexes will be ignored. Otherwise if joining indexes
        on indexes or indexes on a column or columns, the index will be passed on.

        Parameters
        ----------
        right: DataFrame or named Series
            Object to merge with.
        how: {‘left’, ‘right’, ‘outer’, ‘inner’}, default ‘inner’
            Type of merge to be performed.
            left: use only keys from left frame, similar to a SQL left outer join;
                preserve key order.
            right: use only keys from right frame, similar to a SQL right outer join;
                preserve key order.
            outer: use union of keys from both frames, similar to a SQL full outer join;
                sort keys lexicographically.
            inner: use intersection of keys from both frames, similar to a SQL inner join;
                preserve the order of the left keys.
        on: label or list
            Column or index level names to join on. These must be found in both
            DataFrames. If on is None and not merging on indexes then this defaults
            to the intersection of the columns in both DataFrames.
        left_on: label or list, or array-like
            Column or index level names to join on in the left DataFrame. Can
            also be an array or list of arrays of the length of the left DataFrame.
            These arrays are treated as if they are columns.
        right_on: label or list, or array-like
            Column or index level names to join on in the right DataFrame.
            Can also be an array or list of arrays of the length of the right DataFrame.
            These arrays are treated as if they are columns.
        left_index: bool, default False
            Use the index from the left DataFrame as the join key(s).
            If it is a MultiIndex, the number of keys in the other DataFrame
            (either the index or a number of columns) must match the number of levels.
        right_index: bool, default False
            Use the index from the right DataFrame as the join key.
            Same caveats as left_index.
        sort: bool, default False
            Sort the join keys lexicographically in the result DataFrame.
            If False, the order of the join keys depends on the join type (how keyword).
        suffixes: tuple of (str, str), default (‘_x’, ‘_y’)
            Suffix to apply to overlapping column names in the left and right side,
            respectively. To raise an exception on overlapping columns use (False, False)
        copy: bool, default True
            If False, avoid copy if possible.
        indicator: bool or str, default False
            If True, adds a column to output DataFrame called '_merge' with
            information on the source of each row. If string, column with
            information on source of each row will be added to output DataFrame,
            and column will be named value of string. Information column is
            Categorical-type and takes on a value of 'left_only' for observations
            whose merge key only appears in ‘left’ DataFrame, 'right_only' for
            observations whose merge key only appears in ‘right’ DataFrame,
            and 'both' if the observation’s merge key is found in both.
        validate: str, optional
            If specified, checks if merge is of specified type.
            'one_to_one' or '1:1': check if merge keys are unique in both
                left and right datasets.
            'one_to_many' or '1:m': check if merge keys are unique in left dataset.
            'many_to_one' or 'm:1': check if merge keys are unique in right dataset.
            'many_to_many' or 'm:m': allowed, but does not result in checks.

        Returns
        -------
        PandasMoveDataFrame
            A DataFrame of the two merged objects.

        References
        ----------
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html?highlight=merge#pandas.DataFrame.merge

        """
        if isinstance(right, PandasMoveDataFrame):
            right = right._data

        _merge = super().merge(
            right=right, how=how, on=on, left_on=left_on, right_on=right_on,
            left_index=left_index, right_index=right_index, sort=sort,
            suffixes=suffixes, copy=copy, indicator=indicator, validate=validate
        )
        return PandasMoveDataFrame(data=_merge)

    def write_file(self, file_name, separator=','):
        """
        Write trajectory data to a new file.

        Parameters
        ----------
        file_name : str.
            Represents the filename.
        separator : str, optional, default ','.
            Represents the information separator in a new file.

        """
        self.to_csv(
            file_name, sep=separator, encoding='utf-8', index=False
        )

    def convert_to(self, new_type):
        """
        Convert an object from one type to another specified by the user.

        Parameters
        ----------
        new_type: 'pandas' or 'dask'
            The type for which the object will be converted.

        Returns
        -------
        A subclass of MoveDataFrameAbstractModel
            The converted object.

        """

        operation = begin_operation('convet_to')

        if new_type == TYPE_DASK:
            _dask = MoveDataFrame(
                self,
                latitude=LATITUDE,
                longitude=LONGITUDE,
                datetime=DATETIME,
                traj_id=TRAJ_ID,
                type_=TYPE_DASK,
                n_partitions=1,
            )
            self.last_operation = end_operation(operation)
            return _dask
        elif new_type == TYPE_PANDAS:
            self.last_operation = end_operation(operation)
            return self

    def get_type(self):
        """
        Returns the type of the object.

        Returns
        -------
        str
            A string representing the type of the object.
        """
        type_ = self._type
        return type_
