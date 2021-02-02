import math
from typing import Callable, Dict, Optional, Text, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
from pandas import DataFrame
from shapely.geometry import Polygon

from pymove.utils.constants import (
    DATETIME,
    INDEX_GRID,
    INDEX_GRID_LAT,
    INDEX_GRID_LON,
    LATITUDE,
    LONGITUDE,
    POLYGON,
    TRAJ_ID,
)
from pymove.utils.conversions import lat_meters
from pymove.utils.log import progress_bar
from pymove.utils.mem import begin_operation, end_operation


class Grid:
    def __init__(
        self,
        data: Union[DataFrame, Dict],
        cell_size: Optional[float] = None,
        meters_by_degree: Optional[float] = lat_meters(-3.8162973555)
    ):
        """
        Creates a virtual grid from the trajectories.

        Parameters
        ----------
        data : DataFrame or dict
            Dataframe containing the trajectories
            Dict with grid information
                'lon_min_x': minimum x of grid,
                'lat_min_y': minimum y of grid,
                'grid_size_lat_y': lat y size of grid,
                'grid_size_lon_x': lon x size of grid,
                'cell_size_by_degree': cell size in radians,
        cell_size : float, optional
            Represents grid cell size, by default None
        meters_by_degree : float, optional
            Represents the corresponding meters of lat by degree,
                by default lat_meters(-3.8162973555)
        """
        self.last_operation = None
        if isinstance(data, dict):
            self._grid_from_dict(data)
        else:
            self._create_virtual_grid(data, cell_size, meters_by_degree)
        self.grid_polygon = None

    def get_grid(self) -> Dict:
        """
        Returns the grid object in a dict format.

        Returns
        -------
        Dict
            Dict with grid information
                'lon_min_x': minimum x of grid,
                'lat_min_y': minimum y of grid,
                'grid_size_lat_y': lat y size of grid,
                'grid_size_lon_x': lon x size of grid,
                'cell_size_by_degree': cell size in radians
        """
        return {
            'lon_min_x': self.lon_min_x,
            'lat_min_y': self.lat_min_y,
            'grid_size_lat_y': self.grid_size_lat_y,
            'grid_size_lon_x': self.grid_size_lon_x,
            'cell_size_by_degree': self.cell_size_by_degree,
        }

    def _grid_from_dict(self, dict_grid: Dict):
        """
        Coverts the dict grid to a Grid object.

        Parameters
        ----------
        dict_grid : dict
            Dictionary with grid information
                'lon_min_x': minimum x of grid,
                'lat_min_y': minimum y of grid,
                'grid_size_lat_y': lat y size of grid,
                'grid_size_lon_x': lon x size of grid,
                'cell_size_by_degree': cell size in radians,
        """
        self.lon_min_x = dict_grid['lon_min_x']
        self.lat_min_y = dict_grid['lat_min_y']
        self.grid_size_lat_y = dict_grid['grid_size_lat_y']
        self.grid_size_lon_x = dict_grid['grid_size_lon_x']
        self.cell_size_by_degree = dict_grid['cell_size_by_degree']

    def _create_virtual_grid(
        self, data: DataFrame, cell_size: float, meters_by_degree: float
    ):
        """
        Create a virtual grid based in dataset bound box.

        Parameters
        ----------
        data : DataFrame
            Represents the dataset with contains lat, long and datetime
        cell_size : float
            Size of grid cell
        meters_by_degree : float
            Represents the meters degree of latitude

        """

        operation = begin_operation('_create_virtual_grid')

        bbox = data.get_bbox()
        print('\nCreating a virtual grid without polygons')

        # Latitude in Fortaleza: -3.8162973555
        cell_size_by_degree = cell_size / meters_by_degree
        print('...cell size by degree: %s' % cell_size_by_degree)

        lat_min_y = bbox[0]
        lon_min_x = bbox[1]
        lat_max_y = bbox[2]
        lon_max_x = bbox[3]

        # If cell size does not fit in the grid area, an expansion is made
        if math.fmod((lat_max_y - lat_min_y), cell_size_by_degree) != 0:
            lat_max_y = lat_min_y + cell_size_by_degree * (
                math.floor((lat_max_y - lat_min_y) / cell_size_by_degree) + 1
            )

        if math.fmod((lon_max_x - lon_min_x), cell_size_by_degree) != 0:
            lon_max_x = lon_min_x + cell_size_by_degree * (
                math.floor((lon_max_x - lon_min_x) / cell_size_by_degree) + 1
            )

        # adjust grid size to lat and lon
        grid_size_lat_y = int(
            round((lat_max_y - lat_min_y) / cell_size_by_degree)
        )
        grid_size_lon_x = int(
            round((lon_max_x - lon_min_x) / cell_size_by_degree)
        )

        print(
            '...grid_size_lat_y:%s\ngrid_size_lon_x:%s'
            % (grid_size_lat_y, grid_size_lon_x)
        )

        self.lon_min_x = lon_min_x
        self.lat_min_y = lat_min_y
        self.grid_size_lat_y = grid_size_lat_y
        self.grid_size_lon_x = grid_size_lon_x
        self.cell_size_by_degree = cell_size_by_degree
        print('\n..A virtual grid was created')

        self.last_operation = end_operation(operation)

    def create_update_index_grid_feature(
        self,
        data: DataFrame,
        unique_index: Optional[bool] = True,
        label_dtype: Optional[Callable] = np.int64,
        sort: Optional[bool] = True
    ):
        """
        Create or update index grid feature. It not necessary pass dic_grid,
        because if don't pass, the function create a dic_grid.

        Parameters
        ----------
        data : DataFrame
            Represents the dataset with contains lat, long and datetime.
        unique_index: bool, optional
            How to index the grid, by default True
        label_dtype : Optional[Callable], optional
            Represents the type of a value of new column in dataframe, by default np.int64
        sort : bool, optional
            Represents if needs to sort the dataframe, by default True

        """

        operation = begin_operation('create_update_index_grid_feature')

        print('\nCreating or updating index of the grid feature..\n')
        try:
            if sort:
                data.sort_values([TRAJ_ID, DATETIME], inplace=True)
            lat_, lon_ = self.point_to_index_grid(
                data[LATITUDE], data[LONGITUDE]
            )
            lat_, lon_ = label_dtype(lat_), label_dtype(lon_)
            dict_grid = self.get_grid()
            if unique_index:
                data[INDEX_GRID] = lon_ * dict_grid['grid_size_lat_y'] + lat_
            else:
                data[INDEX_GRID_LAT] = lat_
                data[INDEX_GRID_LON] = lon_
            self.last_operation = end_operation(operation)
        except Exception as e:
            self.last_operation = end_operation(operation)
            raise e

    def convert_two_index_grid_to_one(
        self,
        data: DataFrame,
        label_grid_lat: Optional[Text] = INDEX_GRID_LAT,
        label_grid_lon: Optional[Text] = INDEX_GRID_LON,
    ):
        """
        Converts grid lat-lon ids to unique values

        Parameters
        ----------
        data : DataFrame
            Dataframe with grid lat-lon ids
        label_grid_lat : str, optional
           grid lat id column, by default INDEX_GRID_LAT
        label_grid_lon : str, optional
            grid lon id column, by default INDEX_GRID_LON
        """
        dict_grid = self.get_grid()
        data[INDEX_GRID] = (
            data[label_grid_lon] * dict_grid['grid_size_lat_y'] + data[label_grid_lat]
        )

    def convert_one_index_grid_to_two(
        self,
        data: DataFrame,
        label_grid_index: Optional[Text] = INDEX_GRID,
    ):
        """
        Converts grid lat-lon ids to unique values

        Parameters
        ----------
        data : DataFrame
            Dataframe with grid lat-lon ids
        label_grid_index : str, optional
            grid unique id column, by default INDEX_GRID
        """
        dict_grid = self.get_grid()
        data[INDEX_GRID_LAT] = data[label_grid_index] % dict_grid['grid_size_lat_y']
        data[INDEX_GRID_LON] = data[label_grid_index] // dict_grid['grid_size_lat_y']

    def create_one_polygon_to_point_on_grid(
        self, index_grid_lat: int, index_grid_lon: int
    ) -> Polygon:
        """
        Create one polygon to point on grid.

        Parameters
        ----------
        index_grid_lat : int
            Represents index of grid that reference latitude.
        index_grid_lon : int
            Represents index of grid that reference longitude.

        Returns
        -------
        Polygon
            Represents a polygon of this cell in a grid.

        """

        operation = begin_operation('create_one_polygon_to_point_on_grid')

        cell_size = self.cell_size_by_degree
        lat_init = self.lat_min_y + cell_size * index_grid_lat
        lon_init = self.lon_min_x + cell_size * index_grid_lon
        polygon = Polygon((
            (lon_init, lat_init),
            (lon_init, lat_init + cell_size),
            (lon_init + cell_size, lat_init + cell_size),
            (lon_init + cell_size, lat_init)
        ))
        self.last_operation = end_operation(operation)

        return polygon

    def create_all_polygons_on_grid(self):
        """
        Create all polygons that are represented in a grid and store them in a
        new dic_grid key .

        """

        operation = begin_operation('create_all_polygons_on_grid')

        print('\nCreating all polygons on virtual grid', flush=True)
        grid_polygon = np.array(
            [
                [None for _ in range(self.grid_size_lon_x)]
                for _ in range(self.grid_size_lat_y)
            ]
        )
        lat_init = self.lat_min_y
        cell_size = self.cell_size_by_degree
        for i in progress_bar(range(self.grid_size_lat_y)):
            lon_init = self.lon_min_x
            for j in range(self.grid_size_lon_x):
                # Cria o polygon da célula
                grid_polygon[i][j] = Polygon((
                    (lon_init, lat_init),
                    (lon_init, lat_init + cell_size),
                    (lon_init + cell_size, lat_init + cell_size),
                    (lon_init + cell_size, lat_init)
                ))
                lon_init += cell_size
            lat_init += cell_size
        self.grid_polygon = grid_polygon
        print('...geometries saved on Grid grid_polygon property')
        self.last_operation = end_operation(operation)

    def create_all_polygons_to_all_point_on_grid(
        self, data: DataFrame
    ) -> DataFrame:
        """
        Create all polygons to all points represented in a grid.

        Parameters
        ----------
        data : DataFrame
            Represents the dataset with contains lat, long and datetime

        Returns
        -------
        DataFrame
            Represents the same dataset with new key 'polygon'
            where polygons were saved.

        """

        operation = begin_operation('create_all_polygons_to_all_point_on_grid')
        if INDEX_GRID_LAT not in data or INDEX_GRID_LON not in data:
            self.create_update_index_grid_feature(data, unique_index=False)

        datapolygons = data[[TRAJ_ID, INDEX_GRID_LAT, INDEX_GRID_LON]].drop_duplicates()

        polygons = datapolygons.apply(
            lambda row: self.create_one_polygon_to_point_on_grid(
                row[INDEX_GRID_LAT], row[INDEX_GRID_LON]
            ), axis=1
        )

        print('...polygons were created')
        datapolygons['polygon'] = polygons
        self.last_operation = end_operation(operation)
        return datapolygons

    def point_to_index_grid(self, event_lat: float, event_lon: float) -> Tuple[int, int]:
        """
        Locate the coordinates x and y in a grid of point (lat, long).

        Parameters
        ----------
        event_lat : float
            Represents the latitude of a point
        event_lon : float
            Represents the longitude of a point

        Returns
        -------
        Tuple[int, int]
            Represents the index y in a grid of a point (lat, long)
            Represents the index x in a grid of a point (lat, long)

        """

        operation = begin_operation('create_all_polygons_to_all_point_on_grid')

        indexes_lat_y = np.floor(
            (np.float64(event_lat) - self.lat_min_y) / self.cell_size_by_degree
        )
        indexes_lon_x = np.floor(
            (np.float64(event_lon) - self.lon_min_x) / self.cell_size_by_degree
        )
        print(
            '...[%s,%s] indexes were created to lat and lon'
            % (indexes_lat_y.size, indexes_lon_x.size)
        )
        self.last_operation = end_operation(operation)

        return indexes_lat_y, indexes_lon_x

    def save_grid_pkl(self, filename: Text):
        """
        Save a grid with new file .pkl.

        Parameters
        ----------
        filename : Text
            Represents the name of a file.

        """

        operation = begin_operation('save_grid_pkl')
        with open(filename, 'wb') as f:
            joblib.dump(self.get_grid(), f)
        self.last_operation = end_operation(operation)

    def read_grid_pkl(self, filename: Text) -> 'Grid':
        """
        Read grid dict from a file .pkl.

        Parameters
        ----------
        filename : str
                Represents the name of a file.

        Returns
        -------
        Grid
            Grid object containing informations about virtual grid

        """
        operation = begin_operation('read_grid_pkl')
        with open(filename, 'rb') as f:
            dict_grid = joblib.load(f)
        grid = Grid(data=dict_grid)
        self.last_operation = end_operation(operation)
        return grid

    def show_grid_polygons(
        self,
        data: DataFrame,
        markersize: Optional[float] = 10,
        linewidth: Optional[float] = 2,
        figsize: Optional[Tuple[int, int]] = (10, 10),
        return_fig: Optional[bool] = True,
        save_fig: Optional[bool] = False,
        name: Optional[Text] = 'grid.png',
    ) -> Optional[figure]:
        """
        Generate a visualization with grid polygons.

        Parameters
        ----------
        data : DataFrame
            Input trajectory data
        markersize : float, optional
            Represents visualization size marker, by default 10
        linewidth : float, optional
            Represents visualization size line, by default 2
        figsize : tuple(int, int), optional
            Represents the size (float: width, float: height) of a figure,
                by default (10, 10)
        return_fig : bool, optional
            Represents whether or not to save the generated picture, by default True
        save_fig : bool, optional
            Wether to save the figure, by default False
        name : str, optional
            Represents name of a file, by default 'grid.png'

        Returns
        -------
        Optional[figure]
            The generated picture or None

        Raises
        ------
            If the dataframe does not contains the POLYGON feature
        IndexError
            If there is no user with the id passed

        """
        if POLYGON not in data:
            raise KeyError('POLYGON feature not in dataframe')

        data.dropna(subset=[POLYGON], inplace=True)

        operation = begin_operation('show_grid_polygons')

        fig = plt.figure(figsize=figsize)

        for _, row in data.iterrows():
            xs, ys = row[POLYGON].exterior.xy
            plt.plot(ys, xs, 'g', linewidth=linewidth, markersize=markersize)
        xs_start, ys_start = data.iloc[0][POLYGON].exterior.xy
        xs_end, ys_end = data.iloc[-1][POLYGON].exterior.xy
        plt.plot(ys_start, xs_start, 'bo', markersize=markersize * 1.5)
        plt.plot(ys_end, xs_end, 'bX', markersize=markersize * 1.5)  # start point

        if save_fig:
            plt.savefig(fname=name, fig=fig)

        self.last_operation = end_operation(operation)

        if return_fig:
            return fig

    def __repr__(self) -> str:
        """
        String representation of grid

        Returns
        -------
        str
            lon_min_x: min longitude
            lat_min_y: min latitude
            grid_size_lat_y: grid latitude size
            grid_size_lon_x: grid longitude size
            cell_size_by_degree: grid cell size
        """
        text = ['{}: {}'.format(k, v) for k, v in self.get_grid().items()]
        return '\n'.join(text)
