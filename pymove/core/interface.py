import abc
import numpy as np
from pymove.utils.constants import LATITUDE, LONGITUDE, DATETIME, TRAJ_ID
from pymove.core.grid import lat_meters

class MoveDataFrameAbstractModel(abc.ABC):
    @abc.abstractmethod
    def head(self, n=5):
        pass

#     #ddza
    @abc.abstractmethod
    def get_users_number(self):
        pass

#     #ddza
    @abc.abstractmethod
    def time_interval(self):
        pass  

#     #ddza
    @abc.abstractmethod
    def to_numpy(self):
        pass          

    #arina
    @abc.abstractmethod   
    def write_file(self,  file_name, separator = ','):
        pass

#     #arina
    @abc.abstractmethod
    def len(self):
        pass

#     #arina
#     @abc.abstractmethod
#     def __getattr__(self):
#         pass

#     #arina
#     @abc.abstractmethod
#     def __setattr__(self):
#         pass

    #arina
    @abc.abstractmethod
    def to_dict():
        pass

    #arina
    @abc.abstractmethod
    def to_grid(self, cell_size, meters_by_degree = lat_meters(-3.8162973555)):
        pass

#     # Primeiros 7 andreza, os outros 7 arina
    @abc.abstractmethod
    def generate_tid_based_on_id_datatime(self, str_format="%Y%m%d%H", sort=True):
        pass

    @abc.abstractmethod
    def generate_date_features(self):
        pass

    @abc.abstractmethod
    def generate_hour_features(self):
        pass

    @abc.abstractmethod
    def generate_day_of_the_week_features(self):
        pass

    @abc.abstractmethod
    def generate_time_of_day_features(self):
        pass

    @abc.abstractmethod
    def generate_dist_features(self, label_id=TRAJ_ID, label_dtype=np.float64, sort=True):
        pass

    @abc.abstractmethod
    def generate_dist_time_speed_features(self, label_id=TRAJ_ID, label_dtype=np.float64, sort=True):
        pass

    @abc.abstractmethod
    def generate_move_and_stop_by_radius():
        pass

    @abc.abstractmethod
    def time_interval():
        pass
    
    @abc.abstractmethod
    def get_bbox():
        pass

    @abc.abstractmethod   
    def plot_all_features():
        pass

    @abc.abstractmethod   
    def plot_trajs(self):
        pass

    @abc.abstractmethod
    def plot_traj_id():
        pass

    @abc.abstractmethod
    def show_trajectories_info(self):
        pass
