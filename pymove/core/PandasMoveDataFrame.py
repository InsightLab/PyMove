import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymove.utils.traj_utils import format_labels, shift, progress_update
from pymove.core.grid import lat_meters
from pymove.utils import transformations
from pymove.core import MoveDataFrameAbstractModel
from pymove.core.grid import create_virtual_grid
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
	PERIOD)
from pymove.utils.transformations import haversine
from pymove.core import indexes


#TODO: tirar o data do format_labels
#TODO: mover constantes para um arquivo
class PandasMoveDataFrame(pd.DataFrame,MoveDataFrameAbstractModel): # dask sua estrutura de dados
	def __init__(self, data, latitude=LATITUDE, longitude=LONGITUDE, datetime=DATETIME, traj_id = TRAJ_ID):
		# formatar os labels que foram passados pro que usado no pymove -> format_labels
		# renomeia as colunas do dado passado pelo novo dict
		# cria o dataframe
	
		mapping_columns = format_labels(data, traj_id, latitude, longitude, datetime)
		tdf = data.rename(columns=mapping_columns)
		
		if self._has_columns(tdf):
			self._validate_move_data_frame(tdf)
			#pd.DataFrame.__init__(self, tdf)
			self._data = tdf

	def _has_columns(self, data):
		if(LATITUDE in data and LONGITUDE in data and DATETIME in data):
			return True
		return False

	def _validate_move_data_frame(self, data):
		# chama a função de validação   
			# deverá verificar se tem as colunas e os tipos
		try:
			if(data.dtypes.lat != 'float32'):
				data.lat.astype('float32') 
			if(data.dtypes.lon != 'float32'):
				data.lon.astype('float32') 
			if(data.dtypes.datetime != 'datetime64[ns]'):
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
			raise AttributeError("The MoveDataFrame does not contain the column '%s.'"%LONGITUDE)
		return self[LONGITUDE]

	@property
	def datetime(self):
		if DATETIME not in self:
			raise AttributeError("The MoveDataFrame does not contain the column '%s.'"%DATETIME)
		return self[DATETIME]

	def head(self, n=5):
		return self._data.head(n)

	def get_users_number(self):
		if UID in self._data:
			return self._data[UID].nunique()
		return 1 

	def to_numpy(self):
		return self._data.values

	def write_file(self, file_name, separator = ','):
		self._data.to_csv(file_name, sep=separator, encoding='utf-8', index=False)

	def len(self):
		return self._data.shape[0]

	#pocurar jeito mais otimizado de fazer
	def to_dict(self):
		df = self._data.copy()
		data_dict = df.to_dict() 
		return data_dict

	def to_grid(self, cell_size, meters_by_degree = lat_meters(-3.8162973555)):
		return create_virtual_grid(cell_size, self.get_bbox(), meters_by_degree) 

	def get_bbox(self):
		"""
		A bounding box (usually shortened to bbox) is an area defined by two longitudes and two latitudes, where:
			- Latitude is a decimal number between -90.0 and 90.0. 
			- Longitude is a decimal number between -180.0 and 180.0.
		They usually follow the standard format of: 
		- bbox = left, bottom, right, top 
		- bbox = min Longitude , min Latitude , max Longitude , max Latitude 
		Parameters
		----------
		self : pandas.core.frame.DataFrame
			Represents the dataset with contains lat, long and datetime.
		
		Returns
		-------
		bbox : tuple
			Represents a bound box, that is a tuple of 4 values with the min and max limits of latitude e longitude.
		Examples
		--------
		(22.147577, 113.54884299999999, 41.132062, 121.156224)
		"""
		try:
			return (self._data[LATITUDE].min(), self._data[LONGITUDE].min(), self._data[LATITUDE].max(),self._data[LONGITUDE].max())
		except Exception as e:
			raise e

	def generate_tid_based_on_id_datatime(self, str_format="%Y%m%d%H", sort=True):
		"""
		Create or update trajectory id based on id e datetime.  
		Parameters
		----------
		self : pandas.core.frame.DataFrame
			Represents the dataset with contains lat, long and datetime.
		str_format : String
			Contains informations about virtual grid, how
				- lon_min_x: longitude mínima.
				- lat_min_y: latitude miníma. 
				- grid_size_lat_y: tamanho da grid latitude. 
				- grid_size_lon_x: tamanho da longitude da grid.
				- cell_size_by_degree: tamanho da célula da Grid.
			If value is none, the function ask user by dic_grid.
		sort : boolean
			Represents the state of dataframe, if is sorted. By default it's true.
		Returns
		-------
		Examples
		--------
		ID = M00001 and datetime = 2019-04-28 00:00:56  -> tid = M000012019042800
		>>> from pymove.utils.transformations import generate_tid_based_on_id_datatime
		>>> generate_tid_based_on_id_datatime(df)
		"""
		try:
			print('\nCreating or updating tid feature...\n')
			if sort is True:
				print('...Sorting by {} and {} to increase performance\n'.format(TRAJ_ID, DATETIME))
				self._data.sort_values([TRAJ_ID, DATETIME], inplace=True)

			self._data[TID] = self._data[TRAJ_ID].astype(str) + self._data[DATETIME].dt.strftime(str_format)
			print('\n...tid feature was created...\n')

		except Exception as e:
			raise e

	# TODO complementar oq ela faz
	# TODO botar o check pra replace
	def generate_date_features(self):
		"""
		Create or update date feature.  
		Parameters
		----------
		self : pandas.core.frame.DataFrame
			Represents the dataset with contains lat, long and datetime.
		Returns
		-------
		Examples
		--------
		>>> from pymove.utils.transformations import generate_date_features
		>>> generate_date_features(df)
		"""
		try:
			print('Creating date features...')
			if DATETIME in self._data:
				self._data['date'] = self._data[DATETIME].dt.date
				print('..Date features was created...\n')
		except Exception as e:
			raise e

	# TODO complementar oq ela faz
	# TODO botar o check pra replace
	def generate_hour_features(self):
		"""
		Create or update hour feature.  
		Parameters
		----------
		self : pandas.core.frame.DataFrame
			Represents the dataset with contains lat, long and datetime.
		Returns
		-------
		Examples
		--------
		>>> from pymove.utils.transformations import generate_hour_features
		>>> generate_date_features(df)
		"""
		try:
			print('\nCreating or updating a feature for hour...\n')
			if DATETIME in self._data:
				self._data['hour'] = self._data[DATETIME].dt.hour
				print('...Hour feature was created...\n')
		except Exception as e:
			raise e

	# TODO botar o check pra replace
	def generate_day_of_the_week_features(self):
		"""
		Create or update a feature day of the week from datatime.  
		Parameters
		----------
		self : pandas.core.frame.DataFrame
			Represents the dataset with contains lat, long and datetime.
		Returns
		-------
		Examples
		--------
		Exampĺe: datetime = 2019-04-28 00:00:56  -> day = Sunday
		>>> from pymove.utils.transformations import generate_day_of_the_week_features
		>>> generate_day_of_the_week_features(df)
		"""
		try:
			print('\nCreating or updating day of the week feature...\n')
			self._data[DAY] = self._data[DATETIME].dt.day_name()
			print('...the day of the week feature was created...\n')
		except Exception as e:
			raise e

	# TODO botar o check pra replace
	def generate_time_of_day_features(self):
		"""
		Create a feature time of day or period from datatime.
		Parameters
		----------
		self : pandas.core.frame.DataFrame
			Represents the dataset with contains lat, long and datetime.
		Returns
		-------
		Examples
		--------
		- datetime1 = 2019-04-28 02:00:56 -> period = early morning
		- datetime2 = 2019-04-28 08:00:56 -> period = morning
		- datetime3 = 2019-04-28 14:00:56 -> period = afternoon
		- datetime4 = 2019-04-28 20:00:56 -> period = evening
		>>> from pymove.utils.transformations import generate_time_of_day_features
		>>> generate_time_of_day_features(df)
		"""
		try:
			print(
				'\nCreating or updating period feature\n...early morning from 0H to 6H\n...morning from 6H to 12H\n...afternoon from 12H to 18H\n...evening from 18H to 24H')
			conditions = [(self._data[DATETIME].dt.hour >= 0) & (self._data[DATETIME].dt.hour < 6),
						  (self._data[DATETIME].dt.hour >= 6) & (self._data[DATETIME].dt.hour < 12),
						  (self._data[DATETIME].dt.hour >= 12) & (self._data[DATETIME].dt.hour < 18),
						  (self._data[DATETIME].dt.hour >= 18) & (self._data[DATETIME].dt.hour < 24)]
			choices = ['early morning', 'morning', 'afternoon', 'evening']
			self._data[PERIOD] = np.select(conditions, choices, 'undefined')
			print('...the period of day feature was created')
		except Exception as e:
			raise e

	# TODO complementar oq ela faz
	# TODO botar o check pra replace
	def generate_dist_features(self, label_id=TRAJ_ID, label_dtype=np.float64, sort=True):
		"""
		 Create three distance in meters to an GPS point P (lat, lon).
		Parameters
		----------
		self : pandas.core.frame.DataFrame
			Represents the dataset with contains lat, long and datetime.
		label_id : String
			Represents name of column of trajectore's id. By default it's 'id'.
		label_dtype : String
			Represents column id type. By default it's np.float64.
		sort : boolean
			Represents the state of dataframe, if is sorted. By default it's true.
		Returns
		-------
		Examples
		--------
		Example:    P to P.next = 2 meters
					P to P.previous = 1 meter
					P.previous to P.next = 1 meters
		>>> from pymove.utils.transformations import generate_dist_features
		>>> generate_dist_features(df)
		"""
		try:
			print('\nCreating or updating distance features in meters...\n')
			start_time = time.time()

			if sort is True:
				print('...Sorting by {} and {} to increase performance\n'.format(label_id, DATETIME))
				self._data.sort_values([label_id, DATETIME], inplace=True)

			if self._data.index.name is None:
				print('...Set {} as index to increase attribution performance\n'.format(label_id))
				self._data.set_index(label_id, inplace=True)

			""" create ou update columns"""
			self._data[DIST_TO_PREV] = label_dtype(-1.0)
			self._data[DIST_TO_NEXT] = label_dtype(-1.0)
			self._data[DIST_PREV_TO_NEXT] = label_dtype(-1.0)

			ids = self._data.index.unique()
			selfsize = self._data.shape[0]
			curr_perc_int = -1
			start_time = time.time()
			deltatime_str = ''
			sum_size_id = 0
			size_id = 0
			for idx in ids:
				curr_lat = self._data.at[idx, LATITUDE]
				curr_lon = self._data.at[idx, LONGITUDE]

				size_id = curr_lat.size

				if size_id <= 1:
					print('...id:{}, must have at least 2 GPS points\n'.format(idx))
					self._data.at[idx, DIST_TO_PREV] = np.nan

				else:
					prev_lat = shift(curr_lat, 1)
					prev_lon = shift(curr_lon, 1)
					# compute distance from previous to current point
					self._data.at[idx, DIST_TO_PREV] = haversine(prev_lat, prev_lon, curr_lat, curr_lon)

					next_lat = shift(curr_lat, -1)
					next_lon = shift(curr_lon, -1)
					# compute distance to next point
					self._data.at[idx, DIST_TO_NEXT] = haversine(curr_lat, curr_lon, next_lat, next_lon)

					# using pandas shift in a large dataset: 7min 21s
					# using numpy shift above: 33.6 s

					# use distance from previous to next
					self._data.at[idx, DIST_PREV_TO_NEXT] = haversine(prev_lat, prev_lon, next_lat, next_lon)

					sum_size_id += size_id
					curr_perc_int, est_time_str = progress_update(sum_size_id, selfsize, start_time, curr_perc_int,
																  step_perc=20)
			self._data.reset_index(inplace=True)
			print('...Reset index\n')
			print('..Total Time: {}'.format((time.time() - start_time)))
		except Exception as e:
			print('label_id:{}\nidx:{}\nsize_id:{}\nsum_size_id:{}'.format(label_id, idx, size_id, sum_size_id))
			raise e

	# TODO botar o check pra replace
	def generate_dist_time_speed_features(self, label_id=TRAJ_ID, label_dtype=np.float64, sort=True):
		"""
		Firstly, create three distance to an GPS point P (lat, lon)
		After, create two feature to time between two P: time to previous and time to next 
		Lastly, create two feature to speed using time and distance features
		Parameters
		----------
		self : pandas.core.frame.DataFrame
			Represents the dataset with contains lat, long and datetime.
		label_id : String
			Represents name of column of trajectore's id. By default it's 'id'.
		label_dtype : String
			Represents column id type. By default it's np.float64.
		sort : boolean
			Represents the state of dataframe, if is sorted. By default it's true.
		Returns
		-------
		Examples
		--------
		Example:    dist_to_prev =  248.33 meters, dist_to_prev 536.57 meters
					time_to_prev = 60 seconds, time_prev = 60.0 seconds
					speed_to_prev = 4.13 m/s, speed_prev = 8.94 m/s.
		>>> from pymove.utils.transformations import generate_dist_time_speed_features
		>>> generate_dist_time_speed_features(df)
		"""
		try:

			print('\nCreating or updating distance, time and speed features in meters by seconds\n')
			start_time = time.time()

			if sort is True:
				print('...Sorting by {} and {} to increase performance\n'.format(label_id, DATETIME))
				self._data.sort_values([label_id, DATETIME], inplace=True)

			if self._data.index.name is None:
				print('...Set {} as index to a higher peformance\n'.format(label_id))
				self._data.set_index(label_id, inplace=True)

			"""create new feature to time"""
			self._data[DIST_TO_PREV] = label_dtype(-1.0)

			"""create new feature to time"""
			self._data[TIME_TO_PREV] = label_dtype(-1.0)

			"""create new feature to speed"""
			self._data[SPEED_TO_PREV] = label_dtype(-1.0)

			ids = self._data.index.unique()
			selfsize = self._data.shape[0]
			curr_perc_int = -1
			sum_size_id = 0
			size_id = 0

			for idx in ids:
				curr_lat = self._data.at[idx, LATITUDE]
				curr_lon = self._data.at[idx, LONGITUDE]

				size_id = curr_lat.size

				if size_id <= 1:
					print('...id:{}, must have at least 2 GPS points\n'.format(idx))
					self._data.at[idx, DIST_TO_PREV] = np.nan
					self._data.at[idx, TIME_TO_PREV] = np.nan
					self._data.at[idx, SPEED_TO_PREV] = np.nan
				else:
					prev_lat = shift(curr_lat, 1)
					prev_lon = shift(curr_lon, 1)
					prev_lon = shift(curr_lon, 1)
					# compute distance from previous to current point
					self._data.at[idx, DIST_TO_PREV] = haversine(prev_lat, prev_lon, curr_lat, curr_lon)

					time_ = self._data.at[idx, DATETIME].astype(label_dtype)
					time_prev = (time_ - shift(time_, 1)) * (10 ** -9)
					self._data.at[idx, TIME_TO_PREV] = time_prev

					""" set time_to_next"""
					# time_next = (ut.shift(time_, -1) - time_)*(10**-9)
					# self.at[idx, dic_features_label['time_to_next']] = time_next

					"set Speed features"
					self._data.at[idx, SPEED_TO_PREV] = self._data.at[idx, DIST_TO_PREV] / (time_prev)  # unit: m/s

					sum_size_id += size_id
					curr_perc_int, est_time_str = progress_update(sum_size_id, selfsize, start_time, curr_perc_int,
																  step_perc=20)
			print('...Reset index...\n')
			self._data.reset_index(inplace=True)
			print('..Total Time: {:.3f}'.format((time.time() - start_time)))
		except Exception as e:
			print('label_id:{}\nidx:{}\nsize_id:{}\nsum_size_id:{}'.format(label_id, idx, size_id, sum_size_id))
			raise e

	def generate_move_and_stop_by_radius(self, radius=0, target_label=DIST_TO_PREV):
		if DIST_TO_PREV not in self._data:
			self._data.generate_dist_features()
		try:
			print('\nCreating or updating features MOVE and STOPS...\n')
			conditions = (self._data[target_label] > radius), (self._data[target_label] <= radius)
			choices = ['move', 'stop']

			self._data["situation"] = np.select(conditions, choices, np.nan)
			print('\n....There are {} stops to this parameters\n'.format(self._data[self._data["situation"] == 'stop'].shape[0]))
		except Exception as e:
			raise e

	# def generate_date_features(self):
	# 	try:
	# 		print('Creating date features...')
	# 		if DATETIME in df_:
	# 			DATE = df_[DATETIME].dt.date
	# 			print('..Date features was created...\n')
	# 	except Exception as e:
	#     	raise e

	def time_interval(self):
		time_diff = self._data[DATETIME].max() - self._data[DATETIME].min()
		return time_diff

	def plot_all_features(self, figsize=(21, 15), dtype=np.float64, save_fig=True, name='features.png'):
		try:
			col_float = self._data.select_dtypes(include=[dtype]).columns
			tam = col_float.size
			if (tam > 0):
				fig, ax = plt.subplots(tam, 1, figsize=figsize)
				ax_count = 0
				for col in col_float:
					ax[ax_count].set_title(col)
					self._data[col].plot(subplots=True, ax=ax[ax_count])
					ax_count += 1

				if save_fig:
					plt.savefig(fname=name, fig=fig)
		except Exception as e:
			raise e

	def plot_trajs(self, figsize=(10,10), return_fig=True, markers= 'o',markersize=20):
		fig = plt.figure(figsize=figsize)
		ids = self._data["id"].unique()
		
		for id_ in ids:
			selfid = self._data[ self._data["id"] == id_ ]
			plt.plot(selfid[LONGITUDE], selfid[LATITUDE], markers, markersize=markersize)
		if return_fig:
			return fig

	def plot_traj_id(self, tid, figsize=(10,10)):
		fig = plt.figure(figsize=figsize)
		if TID not in self._data:
			self.generate_tid_based_on_id_datatime()
		self._data = self._data[self._data[TID] == tid ]
		plt.plot(self._data.iloc[0][LONGITUDE], self._data.iloc[0][LATITUDE], 'yo', markersize=20)             # start point
		plt.plot(self._data.iloc[-1][LONGITUDE], self._data.iloc[-1][LATITUDE], 'yX', markersize=20)           # end point
		
		if 'isNode'not in self:
			plt.plot(self._data[LONGITUDE], self._data[LATITUDE])
			plt.plot(self._data.loc[:, LONGITUDE], self._data.loc[:, LATITUDE], 'r.', markersize=8)  # points
		else:
			filter_ = self._data['isNode'] == 1
			selfnodes = self._data.loc[filter_]
			selfpoints = self._data.loc[~filter_]
			plt.plot(selfnodes[LONGITUDE], selfnodes[LATITUDE], linewidth=3)
			plt.plot(selfpoints[LONGITUDE], selfpoints[LATITUDE])
			plt.plot(selfnodes[LONGITUDE], selfnodes[LATITUDE], 'go', markersize=10)   # nodes
			plt.plot(selfpoints[LONGITUDE], selfpoints[LATITUDE], 'r.', markersize=8)  # points
		return self._data, fig

	def show_trajectories_info(self):
		"""
		Show dataset information from dataframe, this is number of rows, datetime interval, and bounding box.
		Parameters
		----------
		self : pandas.core.frame.DataFrame
			Represents the dataset with contains lat, long and datetime.
		dic_labels : dict
			Represents mapping of column's header between values passed on params.
		Returns
		-------
		Examples
		--------
		>>> from pymove.utils.utils import show_trajectories_info
		>>> show_trajectories_info(df)
		======================= INFORMATION ABOUT DATASET =======================
		Number of Points: 217654
		Number of IDs objects: 2
		Start Date:2008-10-23 05:53:05     End Date:2009-03-19 05:46:37
		Bounding Box:(22.147577, 113.54884299999999, 41.132062, 121.156224)
		=========================================================================
		"""
		try:
			print('\n======================= INFORMATION ABOUT DATASET =======================\n')
			print('Number of Points: {}\n'.format(self._data.shape[0]))
			if TRAJ_ID in self._data:
				print('Number of IDs objects: {}\n'.format(self._data[TRAJ_ID].nunique()))
			if TID in self._data:
				print('Number of TIDs trajectory: {}\n'.format(self._data[TID].nunique()))
			if DATETIME in self._data:
				print('Start Date:{}     End Date:{}\n'.format(self._data[DATETIME].min(),
															   self._data[DATETIME].max()))
			if LATITUDE and LONGITUDE in self._data:
				print('Bounding Box:{}\n'.format(
					self.get_bbox()))  # bbox return =  Lat_min , Long_min, Lat_max, Long_max)
			if TIME_TO_PREV in self._data:
				print(
					'Gap time MAX:{}     Gap time MIN:{}\n'.format(
						round(self._data[TIME_TO_PREV].max(), 3),
						round(self._data[TIME_TO_PREV].min(), 3)))
			if SPEED_TO_PREV in self._data:
				print('Speed MAX:{}    Speed MIN:{}\n'.format(round(self._data[SPEED_TO_PREV].max(), 3),
															  round(self._data[SPEED_TO_PREV].min(), 3)))
			if DIST_TO_PREV in self._data:
				print('Distance MAX:{}    Distance MIN:{}\n'.format(
					round(self._data[DIST_TO_PREV].max(), 3),
					round(self._data[DIST_TO_PREV].min(),
						  3)))

			print('\n=========================================================================\n')
		except Exception as e:
			raise e

	def min(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
		print(self.obj._meta.loc)
		return self._data.min(axis, skipna, level, numeric_only, **kwargs)

	def max(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
		return self._data.max(axis, skipna, level, numeric_only, **kwargs)

	def count(self, axis=0, level=None, numeric_only=False):
		return self._data.count(axis, level, numeric_only)

	def groupby(self, by=None, axis=0, level=None, as_index=True, sort=True, group_keys=True, squeeze=False,
				observed=False, **kwargs):
		return self._data.groupby(by, axis, level, as_index, sort, group_keys, squeeze,
				observed, **kwargs)

	def drop_duplicates(self, subset=None, keep='first', inplace=False):
		return self._data.drop_duplicates(subset, keep, inplace)

	def reset_index(self,  level=None, drop=False, inplace=False, col_level=0, col_fill=''):
		return self._data.reset_index(level, drop, inplace, col_level, col_fill)

	#duvida sobre erro quando sem paraetros, perguntar dd
	def plot(self, *args, **kwargs):
		return self._data.plot(*args, **kwargs)

	def select_dtypes(self, include=None, exclude=None):
		return self._data.select_dtypes(include, exclude)

	def sort_values(self, by, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last'):
		return self._data.sort_values(by, axis, ascending, inplace, kind, na_position)

	def astype(self, dtype, copy=True, errors='raise', **kwargs):
		return self._data.astype(dtype, copy, errors, **kwargs)

	def set_index(self, keys, drop=True, append=False, inplace=False, verify_integrity=False):
		return self._data.set_index(keys, drop, append, inplace, verify_integrity)

	def drop(self, labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise'):
		return self._data.drop(labels, axis, index, columns, level, inplace, errors)

	def duplicated(self, subset=None, keep='first'):
		return self._data.duplicated(subset, keep)

	def shift(self, periods=1, freq=None, axis=0, fill_value=None):
		return self._data.shift(periods, freq, axis, fill_value)

	def any(self, axis=0, bool_only=None, skipna=True, level=None, **kwargs):
		return self._data.any(axis, bool_only, skipna, level, **kwargs)

	def dropna(self, axis=0, how='any', thresh=None, subset=None, inplace=False):
		return self._data.dropna(axis, how, thresh, subset, inplace)

	def isin(self, values):
		return self._data.isin(values)

	def append(self, other, ignore_index=False, verify_integrity=False, sort=None):
		return self._data.append(other, ignore_index, verify_integrity, sort)

	def nunique(self, axis=0, dropna=True):
		return self._data.nunique(axis, dropna)

	#erro nao entendi
	def to_csv(self, path_or_buf=None, sep=',', na_rep='', float_format=None, columns=None, header=True, index=True,
			   index_label=None, mode='w', encoding=None, compression='infer', quoting=None, quotechar='"',
			   line_terminator=None, chunksize=None, date_format=None, doublequote=True, escapechar=None, decimal='.'):
		self._data.to_csv(path_or_buf, sep, na_rep, float_format, columns, header, index,
		 index_label, mode, encoding, compression, quoting, quotechar,
		 line_terminator, chunksize, date_format, doublequote, escapechar, decimal)
		# self._data.to_csv("teste3.csv")]


	# @property
	# def loc(self):
	# 	return indexes._loc(self._data)

	@property
	def loc(self):
		return self._data.loc

	@property
	def iloc(self):
		return self._data.iloc

	@property
	def at(self):
		return self._data.at

	@property
	def values(self):
		return self._data.values

	@property
	def columns(self):
		return self._data.columns

	@property
	def index(self):
		return self._data.index

	@property
	def dtypes(self):
		return self._data.dtypes

	@property
	def shape(self):
		return self._data.shape

	@property
	def isin(self):
		return self._data.isin

# # #AJEITAR ESSES 2
	def __setattr__(self, attr, value):
		print("arinaaaa")
		if(attr == "_data"):
			self.__dict__[attr] = value
		else:
			self.__dict__['_data'][attr] = value

	def __getattr__(self, name):
		print("aaaaaaaaaaa")
		try:
			return self.__dict__['_data']
		except Exception as e:
			raise e





