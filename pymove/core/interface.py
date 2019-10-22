import abc
import numpy as np
from pymove.utils.constants import LATITUDE, LONGITUDE, DATETIME, TRAJ_ID
from pymove.core.grid import lat_meters

class MoveDataFrameAbstractModel(abc.ABC):
	@property
	def loc(self):
		pass

	@property
	def iloc(self):
		pass

	@property
	def at(self):
		pass

	@property
	def values(self):
		pass

	@property
	def columns(self):
		pass

	@property
	def index(self):
		pass

	@property
	def dtypes(self):
		pass

	@property
	def shape(self):
		pass

	def unique(self, values):
		pass

	@abc.abstractmethod
	def head(self):
		pass
	
	@abc.abstractmethod
	def get_users_number(self):
	    pass
	
	@abc.abstractmethod
	def time_interval(self):
	    pass
	
	@abc.abstractmethod
	def to_numpy(self):
	    pass
	
	@abc.abstractmethod
	def write_file(self):
	    pass
	
	@abc.abstractmethod
	def len(self):
	    pass
	
	@abc.abstractmethod
	def to_dict(self):
	    pass
	
	@abc.abstractmethod
	def to_grid(self):

	@abc.abstractmethod
	def to_DataFrame(self):
		pass

	@abc.abstractmethod
	def generate_tid_based_on_id_datatime(self):
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
	def generate_dist_features(self):
	    pass
	
	@abc.abstractmethod
	def generate_dist_time_speed_features(self):
	    pass
	
	@abc.abstractmethod
	def generate_move_and_stop_by_radius(self):
	    pass
	
	@abc.abstractmethod
	def time_interval(self):
	    pass
	
	@abc.abstractmethod
	def get_bbox(self):
	    pass
	
	@abc.abstractmethod
	def plot_all_features(self):
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
	def max(self):
		pass

	@abc.abstractmethod
	def count(self):
	    pass
	
	@abc.abstractmethod
	def reset_index(self):
	    pass
	
	@abc.abstractmethod
	def groupby(self):
	    pass
	
	@abc.abstractmethod
	def plot(self):
	    pass
	
	@abc.abstractmethod
	def drop_duplicates(self):
	    pass
	
	@abc.abstractmethod
	def select_dtypes(self):
	    pass

	@abc.abstractmethod
	def sort_values(self):
	    pass
	
	@abc.abstractmethod
	def astype(self):
	    pass
	
	@abc.abstractmethod
	def set_index(self):
	    pass
	
	@abc.abstractmethod
	def drop(self):
	    pass
	
	@abc.abstractmethod
	def duplicated(self):
	    pass
	
	@abc.abstractmethod
	def shift(self):
	    pass
	
	@abc.abstractmethod
	def any(self):
	    pass
	
	@abc.abstractmethod
	def dropna(self):
	    pass
	
	@abc.abstractmethod
	def isin(self):
	    pass
	
	@abc.abstractmethod
	def append(self):
	    pass
	
	@abc.abstractmethod
	def nunique(self):
	    pass
	
	@abc.abstractmethod
	def to_csv(self):
	    pass

	@abc.abstractmethod
	def convert_to(self, new_type):
		pass

	@abc.abstractmethod
	def get_type(self):
		pass
	
	@abc.abstractmethod
	def last_operation_time(self):
		pass

	@abc.abstractmethod
	def last_operation_name(self):
		pass

	@abc.abstractmethod
	def last_operation(self):
		pass

	@abc.abstractmethod
	def mem(self, format):
		pass

