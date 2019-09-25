import pandas as pd
from pymove.utils.traj_utils import format_labels
from pymove.core import MoveDataFrameAbstractModel
from pymove.utils.constants import LATITUDE, LONGITUDE, DATETIME, TRAJ_ID

#TODO: tirar o data do format_labels
#TODO: mover constantes para um arquivo
class PandasMoveDataFrame(pd.DataFrame, MoveDataFrameAbstractModel): # dask sua estrutura de dados
	def __init__(self, data, latitude=LATITUDE, longitude=LONGITUDE, datetime=DATETIME, traj_id = TRAJ_ID):
		# formatar os labels que foram passados pro que usado no pymove -> format_labels
		# renomeia as colunas do dado passado pelo novo dict
		# cria o dataframe
	
		mapping_columns = format_labels(data, traj_id, latitude, longitude, datetime)
		tdf = data.rename(columns=mapping_columns)
		
		if self._has_columns(tdf):
			self._validate_move_data_frame(tdf)
			pd.DataFrame.__init__(self, tdf)
			
			
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
		return self.head(n)
		
	# def read_file(self, filename):
	#     return self.read_file(filename)

  
	# def get_user_number(self):
	#     pass

  
	# def time_interval():
	#     pass  


	# def to_csv():
	#     pass


	# def to_numpy():
	#     pass          

	def write_file(self, file_name, separator = ','):
		self.to_csv(file_name, sep=separator, encoding='utf-8', index=False)


	def len(self):
		return self.shape[0]


	# def to_dict(self):
	#     df = self.copy()
	#     data_dict = df.to_dict() 
	#     return data_dict

	# def get_bbox(self):
		"""
		A bounding box (usually shortened to bbox) is an area defined by two longitudes and two latitudes, where:
			- Latitude is a decimal number between -90.0 and 90.0. 
			- Longitude is a decimal number between -180.0 and 180.0.
		They usually follow the standard format of: 
		- bbox = left, bottom, right, top 
		- bbox = min Longitude , min Latitude , max Longitude , max Latitude 
		Parameters
		----------
		df_ : pandas.core.frame.DataFrame
			Represents the dataset with contains lat, long and datetime.
		
		Returns
		-------
		bbox : tuple
			Represents a bound box, that is a tuple of 4 values with the min and max limits of latitude e longitude.
		Examples
		--------
		>>> from pymove.utils.utils import get_bbox
		>>> get_bbox(df)
		(22.147577, 113.54884299999999, 41.132062, 121.156224)
		"""
#     try:
#         return (self[LATITUDE].min(), self[LONGITUDE].min(), self[LATITUDE].max(),
#                 self[LONGITUDE].max())
#     except Exception as e:
#         raise e


	# def to_grid(self, cell_size, meters_by_degree = lat_meters(-3.8162973555)):
	#     print('\nCreating a virtual grid without polygons')
		
	#     bbox = self.get_bbox()
	#     # Latitude in Fortaleza: -3.8162973555
	#     cell_size_by_degree = cell_size/meters_by_degree
	#     print('...cell size by degree: {}'.format(cell_size_by_degree))

	#     lat_min_y = bbox[0]
	#     lon_min_x = bbox[1]
	#     lat_max_y = bbox[2] 
	#     lon_max_x = bbox[3]

	#     #If cell size does not fit in the grid area, an expansion is made
	#     if math.fmod((lat_max_y - lat_min_y), cell_size_by_degree) != 0:
	#         lat_max_y = lat_min_y + cell_size_by_degree * (math.floor((lat_max_y - lat_min_y) / cell_size_by_degree) + 1)

	#     if math.fmod((lon_max_x - lon_min_x), cell_size_by_degree) != 0:
	#         lon_max_x = lon_min_x + cell_size_by_degree * (math.floor((lon_max_x - lon_min_x) / cell_size_by_degree) + 1)

		
	#     # adjust grid size to lat and lon
	#     grid_size_lat_y = int(round((lat_max_y - lat_min_y) / cell_size_by_degree))
	#     grid_size_lon_x = int(round((lon_max_x - lon_min_x) / cell_size_by_degree))
		
	#     print('...grid_size_lat_y:{}\ngrid_size_lon_x:{}'.format(grid_size_lat_y, grid_size_lon_x))

	#     # Return a dicionary virtual grid 
	#     my_dict = dict()
		
	#     my_dict['lon_min_x'] = lon_min_x
	#     my_dict['lat_min_y'] = lat_min_y
	#     my_dict['grid_size_lat_y'] = grid_size_lat_y
	#     my_dict['grid_size_lon_x'] = grid_size_lon_x
	#     my_dict['cell_size_by_degree'] = cell_size_by_degree
	#     print('\n..A virtual grid was created')
	#     return my_dict    
  


	# def with_move_and_stop_by_radius(self, radius=0, target_label='dist_to_prev'):
	#     if DIST_TO_PREV not in self:
	#         transformations.create_update_dist_features(self)
	#     try:
	#         print('\nCreating or updating features MOVE and STOPS...\n')
	#         conditions = (self[target_label] > radius), (self[target_label] <= radius)
	#         choices = ['move', 'stop']

	#         self["situation"] = np.select(conditions, choices, np.nan)
	#         print('\n....There are {} stops to this parameters\n'.format(self[self["situation"] == 'stop'].shape[0]))
	#     except Exception as e:
	#         raise e

	# def time_interval(self):
	#     time_diff = self[constants.DATETIME].max() - self[constants.DATETIME].min()
	#     return time_diff


	#AJEITAR ESSES 2
	# def __setattr__(atributo, coluna, indice, valor):
	#     atributo.loc[indice, coluna] = valor
	#     # self.__dict__[name] = value
	#
	#
	# def __getattr__(atributo, indice, coluna):
	#     print("entrou aqui")
	#     return atributo.loc[indice, coluna]


	# def plot_all_features(self , figsize=(21, 15), dtype=np.float64, save_fig=True, name='features.png'):
	#     try:
	#         col_float = self.select_dtypes(include=[dtype]).columns
	#         tam = col_float.size
	#         if (tam > 0):
	#             fig, ax = plt.subplots(tam, 1, figsize=figsize)
	#             ax_count = 0
	#             for col in col_float:
	#                 ax[ax_count].set_title(col)
	#                 self[col].plot(subplots=True, ax=ax[ax_count])
	#                 ax_count += 1

	#             if save_fig:
	#                 plt.savefig(fname=name, fig=fig)
	#     except Exception as e:
	#         raise e


	# def plot_trajs(self, figsize=(10,10), return_fig=True, markers= 'o',markersize=20):
	#     fig = plt.figure(figsize=figsize)
	#     ids = self["id"].unique()
		
	#     for id_ in ids:
	#         df_id = self[ self["id"] == id_ ]
	#         plt.plot(df_id[constants.LONGITUDE], df_id[constants.LATITUDE], markers, markersize=markersize)
	#     if return_fig:
	#         return fig

	# def plot_traj_id(self, tid, figsize=(10,10)):
	#     fig = plt.figure(figsize=figsize)
	#     if TID not in self:
	#         transformations.create_update_tid_based_on_id_datatime(self)
	#     df_ = self[ self[TID] == tid ]
	#     plt.plot(df_.iloc[0][constants.LONGITUDE], df_.iloc[0][constants.LATITUDE], 'yo', markersize=20)             # start point
	#     plt.plot(df_.iloc[-1][constants.LONGITUDE], df_.iloc[-1][constants.LATITUDE], 'yX', markersize=20)           # end point
		
	#     if 'isNode'not in df_:
	#         plt.plot(df_[constants.LONGITUDE], df_[constants.LATITUDE])
	#         plt.plot(df_.loc[:, constants.LONGITUDE], df_.loc[:, constants.LATITUDE], 'r.', markersize=8)  # points
	#     else:
	#         filter_ = df_['isNode'] == 1
	#         df_nodes = df_.loc[filter_]
	#         df_points = df_.loc[~filter_]
	#         plt.plot(df_nodes[constants.LONGITUDE], df_nodes[constants.LATITUDE], linewidth=3)
	#         plt.plot(df_points[constants.LONGITUDE], df_points[constants.LATITUDE])
	#         plt.plot(df_nodes[constants.LONGITUDE], df_nodes[constants.LATITUDE], 'go', markersize=10)   # nodes
	#         plt.plot(df_points[constants.LONGITUDE], df_points[constants.LATITUDE], 'r.', markersize=8)  # points
	#     return df_, fig

	#def show_trajectories_info(df_, dic_labels=dic_labels):
		"""
		Show dataset information from dataframe, this is number of rows, datetime interval, and bounding box.

		Parameters
		----------
		df_ : pandas.core.frame.DataFrame
			Represents the dataset with contains lat, long and datetime.

		dic_labels : dict
			Represents mapping of column's header between values passed on params.

		Returns
		-------


		Examples
		--------
		>>> from pymove.utils.utils import show_trajectories_info
		>>> show_trajectories_info(df_, dic_labels)
		======================= INFORMATION ABOUT DATASET =======================

		Number of Points: 217654

		Number of IDs objects: 2

		Start Date:2008-10-23 05:53:05     End Date:2009-03-19 05:46:37

		Bounding Box:(22.147577, 113.54884299999999, 41.132062, 121.156224)


		=========================================================================

		"""
		"""try:
			print('\n======================= INFORMATION ABOUT DATASET =======================\n')
			print('Number of Points: {}\n'.format(df_.shape[0]))
			if constants.TRAJ_ID in df_:
				print('Number of IDs objects: {}\n'.format(df_[constants.TRAJ_ID].nunique()))
			if constants.TID in df_:
				print('Number of TIDs trajectory: {}\n'.format(df_[constants.TID].nunique()))
			if dic_labels['datetime'] in df_:
				print('Start Date:{}     End Date:{}\n'.format(df_[constants.DATETIME].min(),
															   df_[constants.DATETIME].max()))
			if constants.LATITUDE and constants.LONGITUDE in df_:
				print('Bounding Box:{}\n'.format(
					self.get_bbox()))  # bbox return =  Lat_min , Long_min, Lat_max, Long_max)
			if constants.TIME_TO_PREV in df_:
				print(
					'Gap time MAX:{}     Gap time MIN:{}\n'.format(
						round(df_[constants.TIME_TO_PREV].max(), 3),
						round(df_[constants.TIME_TO_PREV].min(), 3)))
			if constants.SPEED_TO_PREV in df_:
				print('Speed MAX:{}    Speed MIN:{}\n'.format(round(df_[constants.SPEED_TO_PREV].max(), 3),
															  round(df_[constants.SPEED_TO_PREV].min(), 3)))
			if constants.DIST_TO_PREV in df_:
				print('Distance MAX:{}    Distance MIN:{}\n'.format(
					round(df_[constants.DIST_TO_PREV].max(), 3),
					round(df_[constants.DIST_TO_PREV].min(),
						  3)))

			print('\n=========================================================================\n')
		except Exception as e:
			raise e"""