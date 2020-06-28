
from dateutil.parser._parser import ParserError

from pymove.utils.constants import (
    DATETIME,
    LATITUDE,
    LONGITUDE,
    TRAJ_ID,
    TYPE_DASK,
    TYPE_PANDAS,
)


class MoveDataFrame:
    @staticmethod
    def __new__(
        self,
        data,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        datetime=DATETIME,
        traj_id=TRAJ_ID,
        type_=TYPE_PANDAS,
        n_partitions=1,
    ):
        """
        Creates the PyMove dataframe, which must contain latitude, longitude and datetime.
        The dataframe can be a pandas or dask dataframe.

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
        self.type = type_

        if type_ == TYPE_PANDAS:
            from pymove.core.pandas import PandasMoveDataFrame
            return PandasMoveDataFrame(
                data, latitude, longitude, datetime, traj_id
            )
        if type_ == TYPE_DASK:
            from pymove.core.dask import DaskMoveDataFrame
            return DaskMoveDataFrame(
                data, latitude, longitude, datetime, traj_id, n_partitions
            )

    @staticmethod
    def has_columns(data):
        """
        Checks whether the received dataset has 'lat', 'lon', 'datetime'
        columns.

        Parameters
        ----------
        data : DataFrame.
            Input trajectory data.

        Returns
        -------
        bool
            Represents whether or not you have the required columns.

        """

        cols = data.columns
        if LATITUDE in cols and LONGITUDE in cols and DATETIME in cols:
            return True
        return False

    @staticmethod
    def validate_move_data_frame(data):
        """
        Converts the column type to the default type used by PyMove lib.

        Parameters
        ----------
        data : DataFrame.
            Input trajectory data.

        Raises
        ------
        KeyError
            If missing one of lat, lon, datetime columns
        ValueError, ParserError
            If the data types can't be converted.

        """

        try:
            if data.dtypes[LATITUDE] != 'float64':
                data[LATITUDE] = data[LATITUDE].astype('float64')
            if data.dtypes[LONGITUDE] != 'float64':
                data[LONGITUDE] = data[LONGITUDE].astype('float64')
            if data.dtypes[DATETIME] != 'datetime64[ns]':
                data[DATETIME] = data[DATETIME].astype('datetime64[ns]')
        except KeyError as e:
            print('dataframe missing one of lat, lon, datetime columns.')
            raise e
        except (ValueError, ParserError) as e:
            print('dtypes cannot be converted.')
            raise e

    @staticmethod
    def format_labels(current_id, current_lat, current_lon, current_datetime):
        """
        Format the labels for the PyMove lib pattern labels output
        lat, lon and datatime.

        Parameters
        ----------
        current_id : String.
            Represents the column name of feature id.
        current_lat : String.
            Represents the column name of feature latitude.
        current_lon : String.
            Represents the column name of feature longitude.
        current_datetime : String.
            Represents the column name of feature datetime.

        Returns
        -------
        dict
            Represents a dict with mapping current columns of data
            to format of PyMove column.

        """

        return {
            current_id: TRAJ_ID,
            current_lon: LONGITUDE,
            current_lat: LATITUDE,
            current_datetime: DATETIME
        }
