from typing import TYPE_CHECKING, Dict, List, Optional, Text, Union

import dask
import numpy as np
import pandas as pd
from dask.dataframe import DataFrame

from pymove.core import MoveDataFrameAbstractModel
from pymove.core.dataframe import MoveDataFrame
from pymove.utils.constants import (
    DATETIME,
    LATITUDE,
    LONGITUDE,
    TRAJ_ID,
    TYPE_DASK,
    TYPE_PANDAS,
)

if TYPE_CHECKING:
    from pymove.core.pandas import PandasMoveDataFrame


class DaskMoveDataFrame(DataFrame, MoveDataFrameAbstractModel):
    def __init__(
        self,
        data: Union[DataFrame, List, Dict],
        latitude: Optional[Text] = LATITUDE,
        longitude: Optional[Text] = LONGITUDE,
        datetime: Optional[Text] = DATETIME,
        traj_id: Optional[Text] = TRAJ_ID,
        n_partitions: Optional[int] = 1,
    ):
        """
        Checks whether past data has 'lat', 'lon', 'datetime' columns, and
        renames it with the PyMove lib standard. After starts the attributes of
        the class.

        - self._data : Represents trajectory data.
        - self._type : Represents the type of layer below the data structure.
        - self.last_operation : Represents the last operation performed.

        Parameters
        ----------
        data : dict, list, numpy array or pandas.core.DataFrame
            Input trajectory data.
        latitude : str, optional, default 'lat'.
            Represents column name latitude.
        longitude : str, optional, default 'lon'.
            Represents column name longitude.
        datetime : str, optional, default 'datetime'.
            Represents column name datetime.
        traj_id : str, optional, default 'id'.
            Represents column name trajectory id.
        n_partitions : int, optional, default 1.
            Number of partitions of the dask dataframe.

        Raises
        ------
        KeyError
            If missing one of lat, lon, datetime columns
        ValueError, ParserError
            If the data types can't be converted.

        """

        if isinstance(data, dict):
            data = pd.DataFrame.from_dict(data)
        elif (
            (isinstance(data, list) or isinstance(data, np.ndarray))
            and len(data) >= 4
        ):
            zip_list = [LATITUDE, LONGITUDE, DATETIME, TRAJ_ID]
            for i in range(len(data[0])):
                try:
                    zip_list[i] = zip_list[i]
                except KeyError:
                    zip_list.append(i)
            data = pd.DataFrame(data, columns=zip_list)

        mapping_columns = MoveDataFrame.format_labels(
            traj_id, latitude, longitude, datetime
        )
        dsk = data.rename(columns=mapping_columns)

        if MoveDataFrame.has_columns(dsk):
            MoveDataFrame.validate_move_data_frame(dsk)
            self._data = dask.dataframe.from_pandas(
                dsk, npartitions=n_partitions
            )
            self._type = TYPE_DASK
            self.last_operation = None
        else:
            raise AttributeError(
                'Couldn\'t instantiate MoveDataFrame because data has missing columns.'
            )

    @property
    def lat(self):
        """
        Checks for the LATITUDE column and returns its value.

        Returns
        -------
        Series
            LATITUDE column

        Raises
        ------
        AttributeError
            If the LATITUDE column is not present in the DataFrame
        """
        if LATITUDE not in self.columns:
            raise AttributeError(
                "The MoveDataFrame does not contain the column '%s.'"
                % LATITUDE
            )
        return self._data[LATITUDE]

    @property
    def lng(self):
        """
        Checks for the LONGITUDE column and returns its value.

        Returns
        -------
        Series
            LONGITUDE column

        Raises
        ------
        AttributeError
            If the LONGITUDE column is not present in the DataFrame
        """
        if LONGITUDE not in self.columns:
            raise AttributeError(
                "The MoveDataFrame does not contain the column '%s.'"
                % LONGITUDE
            )
        return self._data[LONGITUDE]

    @property
    def datetime(self):
        """
        Checks for the DATETIME column and returns its value.

        Returns
        -------
        Series
            DATETIME column

        Raises
        ------
        AttributeError
            If the DATETIME column is not present in the DataFrame
        """
        if DATETIME not in self.columns:
            raise AttributeError(
                "The MoveDataFrame does not contain the column '%s.'"
                % DATETIME
            )
        return self._data[DATETIME]

    @property
    def loc(self):
        """Access a group of rows and columns by label(srs) or a boolean array."""
        raise NotImplementedError('To be implemented')

    @property
    def iloc(self):
        """Purely integer-location based indexing for selection by position."""
        raise NotImplementedError('To be implemented')

    @property
    def at(self):
        """Access a single value for a row/column label pair."""
        raise NotImplementedError('To be implemented')

    @property
    def values(self):
        """Return a Numpy representation of the DataFrame."""
        raise NotImplementedError('To be implemented')

    @property
    def columns(self):
        """The column labels of the DataFrame."""
        return self._data.columns

    @property
    def index(self):
        """The row labels of the DataFrame."""
        raise NotImplementedError('To be implemented')

    @property
    def dtypes(self):
        """Return the dtypes in the DataFrame."""
        return self._data.dtypes

    @property
    def shape(self):
        """Return a tuple representing the dimensionality of the DataFrame."""
        raise NotImplementedError('To be implemented')

    def rename(self):
        """Alter axes labels.."""
        raise NotImplementedError('To be implemented')

    def len(self):
        """Returns the length/row numbers in trajectory data."""
        raise NotImplementedError('To be implemented')

    def unique(self):
        """Return unique values of Series object."""
        raise NotImplementedError('To be implemented')

    def head(
        self,
        n: Optional[int] = 5,
        npartitions: Optional[int] = 1,
        compute: Optional[bool] = True
    ) -> DataFrame:
        """
        Return the first n rows.

        This function returns the first n rows for the object based on position.
        It is useful for quickly testing if
        your object has the right type of data in it.

        Parameters
        ----------
        n : int, optional, default 5
            Number of rows to select.
        npartitions : int, optional, default 1.
            Represents the number partitions.
        compute : bool, optional, default True.
            Wether to perform the operation

        Returns
        -------
        same type as caller
            The first n rows of the caller object.

        """
        return self._data.head(n, npartitions, compute)

    def tail(
        self,
        n: Optional[int] = 5,
        npartitions: Optional[int] = 1,
        compute: Optional[bool] = True
    ) -> DataFrame:
        """
        Return the last n rows.

        This function returns the last n rows for the object based on position.
        It is useful for quickly testing if
        your object has the right type of data in it.

        Parameters
        ----------
        n : int, optional, default 5
            Number of rows to select.
        npartitions : int, optional, default 1.
            Represents the number partitions.
        compute : bool, optional, default True.
            ?

        Returns
        -------
        same type as caller
            The last n rows of the caller object.

        """
        return self._data.tail(n, npartitions, compute)

    def get_users_number(self):
        """Check and return number of users in trajectory data."""
        raise NotImplementedError('To be implemented')

    def to_numpy(self):
        """Converts trajectory data to numpy array format."""
        raise NotImplementedError('To be implemented')

    def to_dict(self):
        """Converts trajectory data to dict format."""
        raise NotImplementedError('To be implemented')

    def to_grid(self):
        """Converts trajectory data to grid format."""
        raise NotImplementedError('To be implemented')

    def to_data_frame(self) -> DataFrame:
        """
        Converts trajectory data to DataFrame format.

        Returns
        -------
        dask.dataframe.DataFrame
            Represents the trajectory in DataFrame format.

        """

        return self._data

    def info(self):
        """Print a concise summary of a DataFrame."""
        raise NotImplementedError('To be implemented')

    def describe(self):
        """Generate descriptive statistics."""
        raise NotImplementedError('To be implemented')

    def memory_usage(self):
        """Return the memory usage of each column in bytes."""
        raise NotImplementedError('To be implemented')

    def copy(self):
        """Make a copy of this object’srs indices and data."""
        raise NotImplementedError('To be implemented')

    def generate_tid_based_on_id_datetime(self):
        """Create or update trajectory id based on id e datetime."""
        raise NotImplementedError('To be implemented')

    def generate_date_features(self):
        """Create or update date feature."""
        raise NotImplementedError('To be implemented')

    def generate_hour_features(self):
        """Create or update hour feature."""
        raise NotImplementedError('To be implemented')

    def generate_day_of_the_week_features(self):
        """Create or update a feature day of the week from datatime."""
        raise NotImplementedError('To be implemented')

    def generate_weekend_features(self):
        """Create or update the feature weekend to the dataframe."""
        raise NotImplementedError('To be implemented')

    def generate_time_of_day_features(self):
        """Create a feature time of day or period from datatime."""
        raise NotImplementedError('To be implemented')

    def generate_datetime_in_format_cyclical(self):
        """Create or update column with cyclical datetime feature."""
        raise NotImplementedError('To be implemented')

    def generate_dist_time_speed_features(self):
        """Creates features of distance, time and speed between points."""
        raise NotImplementedError('To be implemented')

    def generate_dist_features(self):
        """Create the three distance in meters to an GPS point P."""
        raise NotImplementedError('To be implemented')

    def generate_time_features(self):
        """Create the three time in seconds to an GPS point P."""
        raise NotImplementedError('To be implemented')

    def generate_speed_features(self):
        """Create the three speed in meters by seconds to an GPS point P."""
        raise NotImplementedError('To be implemented')

    def generate_move_and_stop_by_radius(self):
        """Create or update column with move and stop points by radius."""
        raise NotImplementedError('To be implemented')

    def time_interval(self):
        """Get time difference between max and min datetime in trajectory."""
        raise NotImplementedError('To be implemented')

    def get_bbox(self):
        """Creates the bounding box of the trajectories."""
        raise NotImplementedError('To be implemented')

    def plot_all_features(self):
        """Generate a visualization for each column that type is equal dtype."""
        raise NotImplementedError('To be implemented')

    def plot_trajs(self):
        """Generate a visualization that show trajectories."""
        raise NotImplementedError('To be implemented')

    def plot_traj_id(self):
        """Generate a visualization for a trajectory with the specified tid."""
        raise NotImplementedError('To be implemented')

    def show_trajectories_info(self):
        """Show dataset information from dataframe."""
        raise NotImplementedError('To be implemented')

    def min(self):
        """Return the minimum of the values for the requested axis."""
        raise NotImplementedError('To be implemented')

    def max(self):
        """Return the maximum of the values for the requested axis."""
        raise NotImplementedError('To be implemented')

    def count(self):
        """Counts the non-NA cells for each column or row."""
        raise NotImplementedError('To be implemented')

    def groupby(self):
        """Groups dask DataFrame using a mapper or by a Series of columns."""
        raise NotImplementedError('To be implemented')

    def plot(self):
        """Plot the data of the dask DataFrame."""
        raise NotImplementedError('To be implemented')

    def select_dtypes(self):
        """Returns a subset of the columns based on the column dtypes."""
        raise NotImplementedError('To be implemented')

    def astype(self):
        """Casts a dask object to a specified dtype."""
        raise NotImplementedError('To be implemented')

    def sort_values(self):
        """Sorts the values of the dask DataFrame."""
        raise NotImplementedError('To be implemented')

    def reset_index(self):
        """Resets the dask DataFrame'srs index, and use the default one."""
        raise NotImplementedError('To be implemented')

    def set_index(self):
        """Set of row labels using one or more existing columns or arrays."""
        raise NotImplementedError('To be implemented')

    def drop(self):
        """Drops specified rows or columns of the dask Dataframe."""
        raise NotImplementedError('To be implemented')

    def duplicated(self):
        """Returns boolean Series denoting duplicate rows."""
        raise NotImplementedError('To be implemented')

    def drop_duplicates(self):
        """Removes duplicated rows from the data."""
        raise NotImplementedError('To be implemented')

    def shift(self):
        """Shifts by desired number of periods with an optional time freq."""
        raise NotImplementedError('To be implemented')

    def all(self):
        """Indicates if all elements are True, potentially over an axis."""
        raise NotImplementedError('To be implemented')

    def any(self):
        """Indicates if any element is True, potentially over an axis."""
        raise NotImplementedError('To be implemented')

    def isna(self):
        """Detect missing values."""
        raise NotImplementedError('To be implemented')

    def fillna(self):
        """Fills missing data in the dask DataFrame."""
        raise NotImplementedError('To be implemented')

    def dropna(self):
        """Removes missing data from dask DataFrame."""
        raise NotImplementedError('To be implemented')

    def sample(self):
        """Samples data from the dask DataFrame."""
        raise NotImplementedError('To be implemented')

    def isin(self):
        """Determines whether each element is contained in values."""
        raise NotImplementedError('To be implemented')

    def append(self):
        """Append rows of other to the end of caller, returning a new object."""
        raise NotImplementedError('To be implemented')

    def join(self):
        """Join columns of another DataFrame."""
        raise NotImplementedError('To be implemented')

    def merge(self):
        """Merge columns of another DataFrame."""
        raise NotImplementedError('To be implemented')

    def nunique(self):
        """Count distinct observations over requested axis."""
        raise NotImplementedError('To be implemented')

    def write_file(self):
        """Write trajectory data to a new file."""
        raise NotImplementedError('To be implemented')

    def to_csv(self):
        """Write object to a comma-separated values (csv) file."""
        raise NotImplementedError('To be implemented')

    def convert_to(
        self, new_type: Text
    ) -> Union['PandasMoveDataFrame', 'DaskMoveDataFrame']:
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

        if new_type == TYPE_DASK:
            return self
        elif new_type == TYPE_PANDAS:
            df_pandas = self._data.compute()
            return MoveDataFrame(
                df_pandas,
                latitude=LATITUDE,
                longitude=LONGITUDE,
                datetime=DATETIME,
                traj_id=TRAJ_ID,
                type_=TYPE_PANDAS
            )

    def get_type(self) -> Text:
        """
        Returns the type of the object.

        Returns
        -------
        str
            A string representing the type of the object.

        """

        return self._type
