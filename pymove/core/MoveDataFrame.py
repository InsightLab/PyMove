import abc
from pymove.core.PandasMoveDataFrame import PandasMoveDataFrame as pm
from pymove.utils.traj_utils import format_labels

LATITUDE = 'lat'
LONGITUDE = 'lon'
DATETIME = 'datetime'
TRAJ_ID = 'id'
ID = "id"
TID = "tid"
DIST_TO_PREV = 'dist_to_prev'
 
class MoveDataFrame():
    @staticmethod
    def __new__(self, data, latitude=LATITUDE, longitude=LONGITUDE, datetime=DATETIME, traj_id = TRAJ_ID, type="pandas"):
        self.type = type
        if type == 'pandas':            
            return pm(data, latitude=LATITUDE, longitude=LONGITUDE, datetime=DATETIME, traj_id = TRAJ_ID)



    