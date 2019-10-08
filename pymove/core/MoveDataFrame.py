import abc
from pymove.utils.traj_utils import format_labels
from pymove.core.PandasMoveDataFrame import PandasMoveDataFrame as pm
from pymove.core.DaskMoveDataFrame import DaskMoveDataFrame as dm
from pymove.utils.constants import LATITUDE, LONGITUDE, DATETIME, TRAJ_ID


class MoveDataFrame():
    @staticmethod
    def __new__(self, data, latitude=LATITUDE, longitude=LONGITUDE, datetime=DATETIME, traj_id=TRAJ_ID, type="pandas",
                n_partitions=1):
        self.type = type
        if type == 'pandas':
            return pm(data, latitude=LATITUDE, longitude=LONGITUDE, datetime=DATETIME, traj_id=TRAJ_ID)
        if type == 'dask':
            return dm(data, latitude=LATITUDE, longitude=LONGITUDE, datetime=DATETIME, traj_id=TRAJ_ID, n_partitions=1)


