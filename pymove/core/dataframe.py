from typing import Dict, List, Optional, Text, Union

from dateutil.parser._parser import ParserError
from pandas.core.frame import DataFrame

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
        data: Union[DataFrame, Dict, List],
        latitude: Optional[Text] = LATITUDE,
        longitude: Optional[Text] = LONGITUDE,
        datetime: Optional[Text] = DATETIME,
        traj_id: Optional[Text] = TRAJ_ID,
        type_: Optional[Text] = TYPE_PANDAS,
        n_partitions: Optional[int] = 1,
    ):
        """
        Creates the PyMove dataframe, which must contain latitude, longitude and datetime.
        The dataframe can be a pandas or dask dataframe.

        Parameters
        ----------
        data : DataFrame or PandasMoveDataFrame or dict or list
            Input trajectory data.
        latitude : str, optional
            Represents column name latitude, by default LATITUDE
        longitude : str, optional
            Represents column name longitude, by default LONGITUDE
        datetime : str, optional
            Represents column name datetime, by default DATETIME
        traj_id : str, optional
            Represents column name trajectory id, by default TRAJ_ID
        type_ : str, optional
            Number of partitions of the dask dataframe, by default TYPE_PANDAS
        n_partitions : Optional[int], optional
            Amount of partitions for dask dataframe, by default 1

        Raises
        ------
        KeyError
            If missing one of lat, lon, datetime columns
        ValueError, ParserError
            If the data types can't be converted.

        """

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
    def has_columns(data: DataFrame) -> bool:
        """
        Checks whether the received dataset has 'lat', 'lon', 'datetime'
        columns.

        Parameters
        ----------
        data : DataFrame
            Input trajectory data

        Returns
        -------
        bool
            Represents whether or not you have the required columns

        """

        cols = data.columns
        if LATITUDE in cols and LONGITUDE in cols and DATETIME in cols:
            return True
        return False

    @staticmethod
    def validate_move_data_frame(data: DataFrame):
        """
        Converts the column type to the default type used by PyMove lib.

        Parameters
        ----------
        data : DataFrame
            Input trajectory data

        Raises
        ------
        KeyError
            If missing one of lat, lon, datetime columns
        ValueError, ParserError
            If the data types can't be converted

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
    def format_labels(
        current_id: Text, current_lat: Text, current_lon: Text, current_datetime: Text
    ) -> Dict:
        """
        Format the labels for the PyMove lib pattern labels output
        lat, lon and datatime.

        Parameters
        ----------
        current_id : str
            Represents the column name of feature id
        current_lat : str
            Represents the column name of feature latitude
        current_lon : str
            Represents the column name of feature longitude
        current_datetime : str
            Represents the column name of feature datetime

        Returns
        -------
        Dict
            Represents a dict with mapping current columns of data
            to format of PyMove column.

        """

        return {
            current_id: TRAJ_ID,
            current_lon: LONGITUDE,
            current_lat: LATITUDE,
            current_datetime: DATETIME
        }
