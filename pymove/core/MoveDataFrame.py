import abc
from pymove.core.PandasMoveDataFrame import PandasMoveDataFrame as pm
from pymove.utils.utils import format_labels

LATITUDE = 'lat'
LONGITUDE = 'lon'
DATETIME = 'datetime'
TRAJ_ID = 'id'
ID = "id"
TID = "tid"
DIST_TO_PREV = 'dist_to_prev'
class MoveDataFrame():
    def __init__(self, data, latitude=LATITUDE, longitude=LONGITUDE, datetime=DATETIME, traj_id = TRAJ_ID, type="pandas"):
        self.type = type
        if type == 'pandas':
            
            self.data = pm(data, latitude=LATITUDE, longitude=LONGITUDE, datetime=DATETIME, traj_id = TRAJ_ID)
            # super(MoveDataFrame, self).__init__(self.data)
        # if type == 'dask':
        #     self.data = DaskMoveDataFrame()
 
    def head(self):
        return self.data.head()
    
    def bla(self):
        self.data.bla()

#     #ddza
    # def read_file():
    #     self.data.read_file()

#     #ddza
#     @abc.abstractmethod
#     def get_user_number(self):
#         pass

#     #ddza
#     @abc.abstractmethod
#     def time_interval():
#         pass  

#     #ddza
#     @abc.abstractmethod
#     def to_csv():
#         pass

#     #ddza
#     @abc.abstractmethod
#     def to_numpy():
#         pass          

#     #arina
#     @abc.abstractmethod   
#     def write_file(self):
#         pass

#     #arina
#     @abc.abstractmethod
#     def len():
#         pass

#     #arina
#     @abc.abstractmethod
#     def __getattr__(self):
#         pass

#     #arina
#     @abc.abstractmethod
#     def __setattr__(self):
#         pass

#     #arina
#     @abc.abstractmethod
#     def to_dict():
#         pass

#     #arina
#     @abc.abstractmethod
#     def to_grid():
#         pass


# #####################

#     # Primeiros 7 andreza, os outros 7 arina
#     @abc.abstractmethod
#     def with_tid_based_on_id_datatime():
#         pass

#     @abc.abstractmethod
#     def with_date_features():
#         pass

#     @abc.abstractmethod
#     def with_hour_features():
#         pass

#     @abc.abstractmethod
#     def with_day_of_the_week_features():
#         pass

#     @abc.abstractmethod
#     def with_time_of_day_features():
#         pass

#     @abc.abstractmethod
#     def with_dist_features():
#         pass

#     @abc.abstractmethod
#     def with_dist_time_speed_features():
#         pass

#     @abc.abstractmethod
#     def with_move_and_stop_by_radius():
#         pass

#     @abc.abstractmethod
#     def time_interval():
#         pass
    
#         @abc.abstractmethod
#     def get_bbox():
#         pass

#     @abc.abstractmethod   
#     def plot_all_features():
#         pass

#     @abc.abstractmethod   
#     def plot_trajs(self):
#         pass

#     @abc.abstractmethod
#     def plot_traj_id():
#         pass

#     @abc.abstractmethod
#     def show_trajectories_info(self):
#         pass

# #####################

#     #arina
#     #@property
#     def lat(self):
#         pass

#     #arina
#     #@property
#     def lng(self):
#         pass

#     #ddza
#     #@property
#     def datetime(self):
#         pass

    