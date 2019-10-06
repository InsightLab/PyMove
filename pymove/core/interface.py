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
    def to_dict(self):
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

    @abc.abstractmethod
    def min(self):
        pass

    @abc.abstractmethod
    def mac(self):
        pass

    #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Index.unique.html#pandas.Index.unique
    #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.unique.html
    #example in transformations
    @abc.abstractmethod
    def unique(self, level=None):
        pass


    @abc.abstractmethod
    def count(self, axis=0, level=None, numeric_only=False ):
        pass

    @abc.abstractmethod
    def reset_index(self,  level=None, drop=False, inplace=False, col_level=0, col_fill=''):
        pass

    @abc.abstractmethod
    def groupby(self, by=None, axis=0, level=None, as_index=True, sort=True, group_keys=True, squeeze=False,
                observed=False, **kwargs):


    @abc.abstractmethod
    def plot(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def drop_duplicates(self, subset=None, keep='first', inplace=False):
        pass

    @abc.abstractmethod
    def select_dtypes(self, include=None, exclude=None):
        pass

    @abc.abstractmethod
    def sort_values(self, by, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last'):
        pass

    @abc.abstractmethod
    def astype(self, dtype, copy=True, errors='raise', **kwargs):
        pass

    @abc.abstractmethod
    def set_index(self, keys, drop=True, append=False, inplace=False, verify_integrity=False):
        pass

    #duvida nesse aqui
    @abc.abstractmethod
    def index(self, labes):
        pass


    @abc.abstractmethod
    def drop(self, labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise'):
        pass

    @abc.abstractmethod
    def duplicated(self, subset=None, keep='first'):
        pass

    @abc.abstractmethod
    def shift(self, periods=1, freq=None, axis=0, fill_value=None):
        pass

    @abc.abstractmethod
    def any(self, axis=0, bool_only=None, skipna=True, level=None, **kwargs):
        pass

    @abc.abstractmethod
    def dropna(self, axis=0, how='any', thresh=None, subset=None, inplace=False):
        pass

    @abc.abstractmethod
    def isin(self, values):
        pass

    @abc.abstractmethod
    def append(self, other, ignore_index=False, verify_integrity=False, sort=None):
        pass

    @abc.abstractmethod
    def nunique(self, axis=0, dropna=True):
        pass

    @abc.abstractmethod
    def to_csv(self, path_or_buf=None, sep=', ', na_rep='', float_format=None, columns=None, header=True, index=True,
         index_label=None, mode='w', encoding=None, compression='infer', quoting=None, quotechar='"',
         line_terminator=None, chunksize=None, date_format=None, doublequote=True, escapechar=None, decimal='.'):
        pass

    # duvida columns

    # duvida loc iloc values

    # duvida shape, at, in, dtype

