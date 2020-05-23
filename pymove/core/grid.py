import math

import joblib
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon

from pymove.utils.constants import (
    DATETIME,
    INDEX_GRID_LAT,
    INDEX_GRID_LON,
    LATITUDE,
    LONGITUDE,
    POLYGON,
    TID,
    TRAJ_ID,
)
from pymove.utils.conversions import lat_meters
from pymove.utils.log import progress_bar
from pymove.utils.mem import begin_operation, end_operation


class Grid:
    def __init__(
        self, data, cell_size, meters_by_degree=lat_meters(-3.8162973555)
    ):
        """
        Creates a virtual grid from the trajectories.

        Parameters
        ----------
        data : dataframe like object.
            Dataframe containing the trajectories.
        cell_size : float.
            Represents grid cell size.
        meters_by_degree : float, optional, default lat_meters(-3.8162973555).
            Represents the corresponding meters of lat by degree.

        """

        self.last_operation = None
        self._create_virtual_grid(data, cell_size, meters_by_degree)
        self.grid_polygon = None

    def get_grid(self):
        """Returns the grid object in a dict format."""
        return {
            'lon_min_x': self.lon_min_x,
            'lat_min_y': self.lat_min_y,
            'grid_size_lat_y': self.grid_size_lat_y,
            'grid_size_lon_x': self.grid_size_lon_x,
            'cell_size_by_degree': self.cell_size_by_degree,
        }

    def _create_virtual_grid(self, data, cell_size, meters_by_degree):
        """
        Create a virtual grid based in dataset'srs bound box.

        Parameters
        ----------
        data : pandas.core.frame.DataFrame
            Represents the dataset with contains lat, long and datetime.
        cell_size : float
            Size of grid'srs cell.
        meters_by_degree : float
            Represents the meters'srs degree of latitude.
            By default the latitude is set in Fortaleza.

        Returns
        -------
        dict
            Contains informations about virtual grid, how
                - lon_min_x: minimum longitude.
                - lat_min_y: minimum latitude.
                - grid_size_lat_y: size of latitude grid.
                - grid_size_lon_x: size of longitude grid.
                - cell_size_by_degree: grid'srs cell size.
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
        self, data, label_dtype=np.int64, sort=True
    ):
        """
        Create or update index grid feature. It'srs not necessary pass dic_grid,
        because if don't pass, the function create a dic_grid.

        Parameters
        ----------
        data : pandas.core.frame.DataFrame
            Represents the dataset with contains lat, long and datetime.
        label_dtype : String
            Represents the type_ of a value of new column in dataframe.
        sort : boolean
            Represents the state of dataframe, if is sorted.

        """

        operation = begin_operation('create_update_index_grid_feature')

        print('\nCreating or updating index of the grid feature..\n')
        try:
            if sort:
                data.sort_values([TRAJ_ID, DATETIME], inplace=True)
            lat_, lon_ = self.point_to_index_grid(
                data[LATITUDE], data[LONGITUDE]
            )
            data[INDEX_GRID_LAT] = label_dtype(lat_)
            data[INDEX_GRID_LON] = label_dtype(lon_)
            self.last_operation = end_operation(operation)
        except Exception as e:
            self.last_operation = end_operation(operation)
            raise e

    def create_one_polygon_to_point_on_grid(
        self, index_grid_lat, index_grid_lon
    ):
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
        shapely.geometry.Polygon
            Represents a polygon of this cell in a grid.

        """

        operation = begin_operation('create_one_polygon_to_point_on_grid')

        cell_size = self.cell_size_by_degree
        lat_init = self.lat_min_y + cell_size * index_grid_lat
        lon_init = self.lon_min_x + cell_size * index_grid_lon
        polygon = Polygon((
            (lat_init, lon_init),
            (lat_init + cell_size, lon_init),
            (lat_init + cell_size, lon_init + cell_size,),
            (lat_init, lon_init + cell_size),
        ))
        self.last_operation = end_operation(operation)

        return polygon

    def create_all_polygons_on_grid(self):
        """
        Create all polygons that are represented in a grid and store them in a
        new dic_grid key .

        """

        operation = begin_operation('create_all_polygons_on_grid')

        try:
            print('\nCreating all polygons on virtual grid', flush=True)
            grid_polygon = np.array(
                [
                    [None for i in range(self.grid_size_lon_x)]
                    for j in range(self.grid_size_lat_y)
                ]
            )
            lat_init = self.lat_min_y
            cell_size = self.cell_size_by_degree
            for i in progress_bar(range(self.grid_size_lat_y)):
                lon_init = self.lon_min_x
                for j in range(self.grid_size_lon_x):
                    # Cria o polygon da célula
                    grid_polygon[i][j] = Polygon((
                        (lat_init, lon_init),
                        (lat_init + cell_size, lon_init),
                        (lat_init + cell_size, lon_init + cell_size),
                        (lat_init, lon_init + cell_size),
                    ))
                    lon_init += cell_size
                lat_init += cell_size
            self.grid_polygon = grid_polygon
            print('...geometries saved on Grid grid_polygon property')
            self.last_operation = end_operation(operation)
        except Exception as e:
            self.last_operation = end_operation(operation)
            raise e

    def create_all_polygons_to_all_point_on_grid(self, data):
        """
        Create all polygons to all points represented in a grid.

        Parameters
        ----------
        data : pandas.core.frame.DataFrame
            Represents the dataset with contains lat, long and datetime.

        Returns
        -------
        pandas.core.frame.DataFrame
            Represents the same dataset with new key 'polygon'
            where polygons were saved.

        """

        operation = begin_operation('create_all_polygons_to_all_point_on_grid')

        try:
            self.create_update_index_grid_feature(data)
            datapolygons = data.loc[
                :, ['id', 'index_grid_lat', 'index_grid_lon']
            ].drop_duplicates()
            size = datapolygons.shape[0]

            # transform series in numpyarray
            index_grid_lat = np.array(data['index_grid_lat'])
            index_grid_lon = np.array(data['index_grid_lon'])

            # transform series in numpyarray
            polygons = np.array([])

            for i in progress_bar(range(size)):
                p = self.create_one_polygon_to_point_on_grid(
                    index_grid_lat[i], index_grid_lon[i]
                )
                polygons = np.append(polygons, p)
            print('...polygons were created')
            datapolygons['polygon'] = polygons
            self.last_operation = end_operation(operation)
            return datapolygons
        except Exception as e:
            self.last_operation = end_operation(operation)
            print('size:{}, i:{}'.format(size, i))
            raise e

    def point_to_index_grid(self, event_lat, event_lon):
        """
        Locate the coordinates x and y in a grid of point (lat, long).

        Parameters
        ----------
        event_lat : float
            Represents the latitude of a point.
        event_lon : float
            Represents the longitude of a point.

        Returns
        -------
        int
            Represents the index y in a grid of a point (lat, long).
        int
            Represents the index x in a grid of a point (lat, long).

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

    def save_grid_pkl(self, filename):
        """
        Save a grid with new file .pkl.

        Parameters
        ----------
        filename : String
            Represents the name of a file.
        dic_grid : dict
            Contains informations about virtual grid, how
                - lon_min_x: longitude mínima.
                - lat_min_y: latitude miníma.
                - grid_size_lat_y: tamanho da grid latitude.
                - grid_size_lon_x: tamanho da longitude da grid.
                - cell_size_by_degree: tamanho da célula da Grid.

        """

        operation = begin_operation('save_grid_pkl')

        try:
            with open(filename, 'wb') as f:
                joblib.dump(self.get_grid(), f)
            print('\nA file was saved')
            self.last_operation = end_operation(operation)
        except Exception as e:
            self.last_operation = end_operation(operation)
            raise e

    def read_grid_pkl(self, filename):
        """
        Read grid dict from a file .pkl.

        Parameters
        ----------
        filename : String
                Represents the name of a file.

        Returns
        -------
        dict
            Contains informations about virtual grid, how
                - lon_min_x: longitude mínima.
                - lat_min_y: latitude miníma.
                - grid_size_lat_y: tamanho da grid latitude.
                - grid_size_lon_x: tamanho da longitude da grid.
                - cell_size_by_degree: tamanho da célula da Grid.

        """
        operation = begin_operation('read_grid_pkl')
        try:
            with open(filename, 'rb') as f:
                dict_grid = joblib.load(f)
            self.last_operation = end_operation(operation)
            return dict_grid
        except Exception as e:
            self.last_operation = end_operation(operation)
            raise e

    def show_grid_polygons(
        self,
        data,
        id_,
        figsize=(10, 10),
        return_fig=True,
        save_fig=False,
        name='grid.png',
    ):
        """
        Generate a visualization with grid polygons.

        Parameters
        ----------
        data : pymove.core.MoveDataFrameAbstract subclass.
            Input trajectory data.
        id_ : String
            Represents the id.
        figsize : tuple
            Represents the size (float: width, float: height) of a figure.
        return_fig : bool, optional, default True.
            Represents whether or not to save the generated picture.
        save_fig : bool, optional, default False.
            Represents whether or not to save the generated picture.
        name : String, optional, default 'grid.png'.
            Represents name of a file.

        Returns
        -------
        matplotlib.pyplot.figure or None
            The generated picture.

        Raises
        ------
        KeyError
            If the dataframe does not contains the POLYGON feature
        IndexError
            If there is no user with the id passed

        """

        print(TRAJ_ID, TID)
        if POLYGON not in data:
            raise KeyError('POLYGON feature not in dataframe')

        df_ = data[data[TRAJ_ID] == id_]

        if not len(df_):
            raise IndexError('No user with id %s in dataframe' % id_)

        operation = begin_operation('show_grid_polygons')

        fig = plt.figure(figsize=figsize)

        xs_start, ys_start = df_.iloc[0][POLYGON].exterior.xy

        plt.plot(ys_start, xs_start, 'bo', markersize=20)  # start point

        for idx in range(df_.shape[0]):
            if not isinstance(df_[POLYGON].iloc[idx], float):
                xs, ys = df_[POLYGON].iloc[idx].exterior.xy
                plt.plot(ys, xs, 'g', linewidth=2, markersize=5)

        if save_fig:
            plt.savefig(fname=name, fig=fig)

        self.last_operation = end_operation(operation)

        if return_fig:
            return fig
