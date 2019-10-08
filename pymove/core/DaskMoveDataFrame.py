import dask
from dask.dataframe import DataFrame
from pymove.utils.traj_utils import format_labels, shift, progress_update
from pymove.utils.constants import LATITUDE, LONGITUDE, DATETIME, TRAJ_ID
from pymove.utils.constants import (
	LATITUDE,
	LONGITUDE,
	DATETIME,
	TRAJ_ID,
	TID,
	UID,
	TIME_TO_PREV,
	SPEED_TO_PREV,
	DIST_TO_PREV,
	DIST_PREV_TO_NEXT,
	DIST_TO_NEXT,
	DAY,
	PERIOD,
	TYPE_DASK)
from pymove.core import MoveDataFrameAbstractModel


class DaskMoveDataFrame(DataFrame, MoveDataFrameAbstractModel):  # dask sua estrutura de dados
	def __init__(self, data, latitude=LATITUDE, longitude=LONGITUDE, datetime=DATETIME, traj_id=TRAJ_ID,
				 n_partitions=1):
		# formatar os labels que foram return 0ados pro que usado no pymove -> format_labels
		# renomeia as colunas do dado return 0ado pelo novo dict
		# cria o dataframe
		mapping_columns = format_labels(data, traj_id, latitude, longitude, datetime)
		dsk = data.rename(columns=mapping_columns)

		if self._has_columns(dsk):
			self._validate_move_data_frame(dsk)
			self._data = dask.dataframe.from_pandas(dsk, npartitions = n_partitions)
			self._type = TYPE_DASK
			self._last_operation_dict = {'name': '', 'time': '', 'mem_usage': ''}
		else:
			print("erroo")

	def _has_columns(self, data):
		if (LATITUDE in data and LONGITUDE in data and DATETIME in data):
			return True
		return False

	def _validate_move_data_frame(self, data):
		# chama a função de validação
		# deverá verificar se tem as colunas e os tipos
		try:
			if (data.dtypes.lat != 'float32'):
				data.lat.astype('float32')
			if (data.dtypes.lon != 'float32'):
				data.lon.astype('float32')
			if (data.dtypes.datetime != 'datetime64[ns]'):
				data.lon.astype('datetime64[ns]')
		except AttributeError as erro:
			print(erro)

	@property
	def lat(self):
		if LATITUDE not in self:
			raise AttributeError("The MoveDataFrame does not contain the column '%s.'" % LATITUDE)
		return self[LATITUDE]

	@property
	def lng(self):
		if LONGITUDE not in self:
			raise AttributeError("The MoveDataFrame does not contain the column '%s.'" % LONGITUDE)
		return self[LONGITUDE]

	@property
	def datetime(self):
		if DATETIME not in self:
			raise AttributeError("The MoveDataFrame does not contain the column '%s.'" % DATETIME)
		return self[DATETIME]

	def head(self, n=5, npartitions=1, compute=True):
		return self._data.head(n, npartitions, compute)

	def min(self, axis=None, skipna=True, split_every=False, out=None):
		return self._data.min(axis, skipna, split_every, out)

	def max(self, axis=None, skipna=True, split_every=False, out=None):
		return self._data.max(axis, skipna, split_every, out)

	def groupby(self, by=None, **kwargs):
		return self._data.groupby(by)

	def to_dask(self):
		return self._data

	def to_pandas(self):
		df_pandas = self._data.compute()
		from pymove.core.PandasMoveDataFrame import PandasMoveDataFrame as pm
		return pm(df_pandas, latitude=LATITUDE, longitude=LONGITUDE, datetime=DATETIME, traj_id=TRAJ_ID)

	def get_type(self):
		return self._type

	 
	def get_users_number(self):
	    return 0
	
	 
	def time_interval(self):
	    return 0
	
	 
	def to_numpy(self):
	    return 0
	
	 
	def write_file(self):
	    return 0
	
	 
	def len(self):
	    return 0
	

	 
	def to_dict(self):
	    return 0
	
	 
	def to_grid(self):
	    return 0
	
	 
	def generate_tid_based_on_id_datatime(self):
	    return 0
	
	 
	def generate_date_features(self):
	    return 0
	
	 
	def generate_hour_features(self):
	    return 0
	
	 
	def generate_day_of_the_week_features(self):
	    return 0
	
	 
	def generate_time_of_day_features(self):
	    return 0
	
	 
	def generate_dist_features(self):
	    return 0
	
	 
	def generate_dist_time_speed_features(self):
	    return 0
	
	 
	def generate_move_and_stop_by_radius(self):
	    return 0
	
	 
	def time_interval(self):
	    return 0
	
	 
	def get_bbox(self):
	    return 0
	
	 
	def plot_all_features(self):
	    return 0
	
	 
	def plot_trajs(self):
	    return 0
	
	 
	def plot_traj_id():
	    return 0
	
	 
	def show_trajectories_info(self):
	    return 0
	
	def count(self):
	    return 0
	

	def reset_index(self):
	    return 0
	

	def plot(self):
	    return 0
	

	def drop_duplicates(self):
	    return 0
	

	def select_dtypes(self):
	    return 0
	
	def sort_values(self):
	    return 0
	
	 
	def astype(self):
	    return 0
	
	 
	def set_index(self):
	    return 0
	
	 
	def drop(self):
	    return 0
	
	 
	def duplicated(self):
	    return 0
	
	 
	def shift(self):
	    return 0
	
	 
	def any(self):
	    return 0
	
	 
	def dropna(self):
	    return 0
	
	 
	def isin(self):
	    return 0
	
	 
	def append(self):
	    return 0
	
	 
	def nunique(self):
	    return 0
	
	 
	def to_csv(self):
	    return 0

	def last_operation_time(self):
		return self._last_operation_dict['time']

	def last_operation_name(self):
		return self._last_operation_dict['name']

	def last_operation(self):
		return self._last_operation_dict

	def mem(self, format):
		return self._last_operation_dict['mem_usage']  #TODO ver a formula