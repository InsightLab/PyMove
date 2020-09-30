import numpy as np
import pandas as pd

from pymove.core.dataframe import MoveDataFrame
from pymove.core.grid import Grid
from pymove.core.pandas import PandasMoveDataFrame
from pymove.utils.constants import (
    DATETIME,
    GRID_ID,
    INDEX_GRID,
    LATITUDE,
    LOCAL_LABEL,
    LONGITUDE,
    TRAJ_ID,
    TYPE_PANDAS,
)
from pymove.utils.mem import begin_operation, end_operation


class PandasDiscreteMoveDataFrame(PandasMoveDataFrame):
    def __init__(
        self,
        data,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        datetime=DATETIME,
        traj_id=TRAJ_ID,
        local_label=LOCAL_LABEL
    ):
        """
        Checks whether past data has 'lat', 'lon', 'datetime' and 'local_label'
        columns, and renames it with the PyMove lib standard. After starts the
        attributes of the class.

        - self._data : Represents trajectory data.
        - self._type : Represents the type of layer below the data structure.
        - self.last_operation : Represents the last operation perfomed.

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
        local_label : str, optional, default 'local_label'.
            Represents column name local label

        Raises
        ------
        KeyError
            If missing one of lat, lon, datetime, local_label columns
        ValueError, ParserError
            If the data types can't be converted.

        """

        super()

        if isinstance(data, dict):
            data = pd.DataFrame.from_dict(data)
        elif (
            (isinstance(data, list) or isinstance(data, np.ndarray))
            and len(data) >= 5
        ):
            zip_list = [LATITUDE, LONGITUDE, DATETIME, TRAJ_ID, local_label]
            for i in range(len(data[0])):
                try:
                    zip_list[i] = zip_list[i]
                except KeyError:
                    zip_list.append(i)
            data = pd.DataFrame(data, columns=zip_list)

        mapping_columns = MoveDataFrame.format_labels(
            traj_id, latitude, longitude, datetime
        )
        tdf = data.rename(columns=mapping_columns)

        if local_label not in tdf:
            raise ValueError(
                '{} column not in dataframe'.format(local_label)
            )

        if MoveDataFrame.has_columns(tdf):
            MoveDataFrame.validate_move_data_frame(tdf)
            self._data = tdf
            self._type = TYPE_PANDAS
            self.last_operation = None
        else:
            raise AttributeError(
                'Couldn\'t instantiate MoveDataFrame because data has missing columns'
            )

    def discretize_based_grid(self, region_size=1000):
        """
        Discrete space in cells of the same size,
        assigning a unique id to each cell.

        Parameters
        ----------
        region_size: number, optional, default 1000
            Size of grid'srs cell.
        """

        operation = begin_operation('discretize based on grid')
        print('\nDiscretizing dataframe...')
        try:
            grid = Grid(self, cell_size=region_size)
            grid.create_update_index_grid_feature(self)
            self.reset_index(drop=True, inplace=True)
            self.last_operation = end_operation(operation)

        except Exception as e:
            self.last_operation = end_operation(operation)
            raise e
