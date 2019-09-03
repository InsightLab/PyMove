# TODO: Modelar como classe
# TODO: Andreza
import math
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm
from shapely.geometry import Polygon
import pickle
import matplotlib.pyplot as plt

from pymove.utils.utils import dic_labels, dic_features_label

def lat_meters(lat):
    """
    Transform latitude degree to meters.

    Parameters
    ----------
    lat : float
        This represent latitude value.

    Returns
    -------
    meters : float
        Represents the corresponding latitude value in meters.

    Examples
    --------
    Example: Latitude in Fortaleza: -3.8162973555
    >>> from pymove.core.grid import lat_meters
    >>> lat_meters(-3.8162973555)
        110826.6722516857

    """
    rlat = float(lat) * math.pi / 180
    # meter per degree Latitude
    meters_lat = 111132.92 - 559.82 * math.cos(2 * rlat) + 1.175 * math.cos(4 * rlat)
    # meter per degree Longitude
    meters_lgn = 111412.84 * math.cos(rlat) - 93.5 * math.cos(3 * rlat)
    meters = (meters_lat + meters_lgn) / 2
    return meters

def create_update_index_grid_feature(df_, dic_grid=None, dic_labels=dic_labels, label_dtype=np.int64, sort=True):
    """
    Create or update index grid feature.
    It's not necessary pass dic_grid, because if don't pass, the function create a dic_grid.
    It's not necessary pass dic_labels, because if don't pass, the function use a dic_labels by default.

    Parameters
    ----------
    df_ : pandas.core.frame.DataFrame
        Represents the dataset with contains lat, long and datetime.

    dic_grid : dict
        Contains informations about virtual grid, how
            - lon_min_x: longitude mínima.
            - lat_min_y: latitude miníma. 
            - grid_size_lat_y: tamanho da grid latitude. 
            - grid_size_lon_x: tamanho da longitude da grid.
            - cell_size_by_degree: tamanho da célula da Grid.
        If value is none, the function ask user by dic_grid.

    dic_labels : dict
        Represents dataframe column value mapping.

    label_dtype : String
        Represents the type of a value of new column in dataframe.

    sort : boolean
        Represents the state of dataframe, if is sorted.

    Returns
    -------
    

    Examples
    --------
    >>> from pymove.core.grid import create_update_index_grid_feature
    >>> create_update_index_grid_feature(df, dic_grid)
    Creating or updating index of the grid feature..
    ...[217654,217654] indexes were created to lat and lon

    """
    print('\nCreating or updating index of the grid feature..\n')
    try:
        if dic_grid is not None:
            if sort:
                df_.sort_values([dic_labels['id'], dic_labels['datetime']], inplace=True)

            lat_, lon_ = point_to_index_grid(df_[dic_labels['lat'] ], df_[dic_labels['lon'] ], dic_grid)
            df_[dic_features_label['index_grid_lat']] = label_dtype(lat_)
            df_[dic_features_label['index_grid_lon']] = label_dtype(lon_)   
        else:
            # TODO fazer com que a própria função chame a create_virtual_grid
            print('... inform a grid virtual dictionary\n')
    except Exception as e:
        raise e

def create_virtual_grid(cell_size, bbox, meters_by_degree = lat_meters(-3.8162973555)):
    """
    Create a virtual grid based in dataset's bound box.

    Parameters
    ----------
    cell_size : float
        Size of grid's cell.

    bbox : tuple
        Represents a bound box, that is a tuple of 4 values with the min and max limits of latitude e longitude.

    meters_by_degree : float
        Represents the meters's degree of latitude. By default the latitude is set in Fortaleza.

    Returns
    -------
    virtual_grid : dict
        Contains informations about virtual grid, how
            - lon_min_x: minimum longitude.
            - lat_min_y: minimum latitude. 
            - grid_size_lat_y: size of latitude grid. 
            - grid_size_lon_x: size of longitude grid.
            - cell_size_by_degree: grid's cell size.

    Examples
    --------
    >>> from pymove.core.grid import create_virtual_grid
    >>> from pymove.utils.utils import get_bbox
    >>> ...
    >>> dic_grid = create_virtual_grid(15, get_bbox(data))
    >>> dic_grid

    {'lon_min_x': 113.54884299999999,
    'lat_min_y': 22.147577,
    'grid_size_lat_y': 140266,
    'grid_size_lon_x': 56207,
    'cell_size_by_degree': 0.0001353464801860623}

    """
    print('\nCreating a virtual grid without polygons')
    
    # Latitude in Fortaleza: -3.8162973555
    cell_size_by_degree = cell_size/meters_by_degree
    print('...cell size by degree: {}'.format(cell_size_by_degree))

    lat_min_y = bbox[0]
    lon_min_x = bbox[1]
    lat_max_y = bbox[2] 
    lon_max_x = bbox[3]

    #If cell size does not fit in the grid area, an expansion is made
    if math.fmod((lat_max_y - lat_min_y), cell_size_by_degree) != 0:
        lat_max_y = lat_min_y + cell_size_by_degree * (math.floor((lat_max_y - lat_min_y) / cell_size_by_degree) + 1)

    if math.fmod((lon_max_x - lon_min_x), cell_size_by_degree) != 0:
        lon_max_x = lon_min_x + cell_size_by_degree * (math.floor((lon_max_x - lon_min_x) / cell_size_by_degree) + 1)

    
    # adjust grid size to lat and lon
    grid_size_lat_y = int(round((lat_max_y - lat_min_y) / cell_size_by_degree))
    grid_size_lon_x = int(round((lon_max_x - lon_min_x) / cell_size_by_degree))
    
    print('...grid_size_lat_y:{}\ngrid_size_lon_x:{}'.format(grid_size_lat_y, grid_size_lon_x))

    # Return a dicionary virtual grid 
    virtual_grid = dict()
    
    virtual_grid['lon_min_x'] = lon_min_x
    virtual_grid['lat_min_y'] = lat_min_y
    virtual_grid['grid_size_lat_y'] = grid_size_lat_y
    virtual_grid['grid_size_lon_x'] = grid_size_lon_x
    virtual_grid['cell_size_by_degree'] = cell_size_by_degree
    print('\n..A virtual grid was created')
    return virtual_grid

def create_one_polygon_to_point_on_grid(dic_grid, index_grid_lat, index_grid_lon):
    """
    Create one polygon to point on grid. 

    Parameters
    ----------
    dic_grid : dict
        Contains informations about virtual grid, how
            - lon_min_x: longitude mínima.
            - lat_min_y: latitude miníma. 
            - grid_size_lat_y: tamanho da grid latitude. 
            - grid_size_lon_x: tamanho da longitude da grid.
            - cell_size_by_degree: tamanho da célula da Grid.

    index_grid_lat : int
        Represents index of grid that reference latitude.

    index_grid_lon : int
        Represents index of grid that reference longitude.

    Returns
    -------
    polygon: Polygon
        Represents a polygon of this cell in a grid.

    Examples
    --------
    >>> from pymove.core.grid import create_one_polygon_to_point_on_grid
    >>> create_one_polygon_to_point_on_grid(dic_grid, 10, 12)

    """
    
    lat_init = dic_grid['lat_min_y'] + dic_grid['cell_size_by_degree'] * index_grid_lat
    lon_init = dic_grid['lon_min_x'] + dic_grid['cell_size_by_degree'] * index_grid_lon
    polygon = Polygon(((lat_init, lon_init),
         (lat_init + dic_grid['cell_size_by_degree'], lon_init),
         (lat_init + dic_grid['cell_size_by_degree'], lon_init + dic_grid['cell_size_by_degree']),
         (lat_init, lon_init + dic_grid['cell_size_by_degree'])
                                            ))
    return polygon

def create_all_polygons_on_grid(dic_grid):
    """
    Create all polygons that are represented in a grid and store them in a new dic_grid key . 

    Parameters
    ----------
    dic_grid : dict
        Contains informations about virtual grid, how
            - lon_min_x: longitude mínima.
            - lat_min_y: latitude miníma. 
            - grid_size_lat_y: tamanho da grid latitude. 
            - grid_size_lon_x: tamanho da longitude da grid.
            - cell_size_by_degree: tamanho da célula da Grid.

    Examples
    --------
    >>> from pymove.core.grid import create_all_polygons_on_grid
    >>> create_all_polygons_on_grid(dic_grid)

    """
    # Cria o vetor vazio de gometrias da grid
    try:
        print('\nCreating all polygons on virtual grid')
        grid_polygon = np.array([[None for i in range(dic_grid['grid_size_lon_x'])] for j in range(dic_grid['grid_size_lat_y'])])
        lat_init = dic_grid['lat_min_y']    
        for i in tqdm(range(dic_grid['grid_size_lat_y'])):
            lon_init = dic_grid['lon_min_x']
            for j in range(dic_grid['grid_size_lon_x']):
                # Cria o polygon da célula
                grid_polygon[i][j] = Polygon(((lat_init, lon_init),
                                            (lat_init + dic_grid['cell_size_by_degree'], lon_init),
                                            (lat_init + dic_grid['cell_size_by_degree'], lon_init + dic_grid['cell_size_by_degree']),
                                            (lat_init, lon_init + dic_grid['cell_size_by_degree'])
                                            ))
                lon_init += dic_grid['cell_size_by_degree']
            lat_init += dic_grid['cell_size_by_degree']
        dic_grid['grid_polygon'] = grid_polygon
        print('...geometry was created to a virtual grid')
    except Exception as e:
        raise e

def create_all_polygons_to_all_point_on_grid(df_, dic_grid):
    """
    Create all polygons to all points represented in a grid. 

    Parameters
    ----------
    df_ : pandas.core.frame.DataFrame
        Represents the dataset with contains lat, long and datetime.

    dic_grid : dict
        Contains informations about virtual grid, how
            - lon_min_x: longitude mínima.
            - lat_min_y: latitude miníma. 
            - grid_size_lat_y: tamanho da grid latitude. 
            - grid_size_lon_x: tamanho da longitude da grid.
            - cell_size_by_degree: tamanho da célula da Grid.

    Returns
    -------
    df_polygons: pandas.core.frame.DataFrame
        Represents the same dataset with new key 'polygon' where polygons were saved. 

    Examples
    --------
    >>> from pymove.core.grid import create_all_polygons_on_grid
    >>> df = create_all_polygons_on_grid(df, dic_grid)

    """
    try:
        create_update_index_grid_feature(df_, dic_grid)
        df_polygons = df_.loc[:,['index_grid_lat', 'index_grid_lon']].drop_duplicates()
        size = df_polygons.shape[0]
        
        """transform series in numpyarray"""
        index_grid_lat = np.array(df_['index_grid_lat'])
        index_grid_lon = np.array(df_['index_grid_lon'])

        """transform series in numpyarray"""
        polygons = np.array([])

        for i in tqdm(range(size)):
            p = create_one_polygon_to_point_on_grid(dic_grid, index_grid_lat[i], index_grid_lon[i])
            polygons = np.append(polygons, p)
        print('...polygons were created')
        df_polygons['polygon'] = polygons
        return df_polygons
    except Exception as e:
        print('size:{}, i:{}'.format(size, i))
        raise e  

def point_to_index_grid(event_lat, event_lon, dic_grid):
    """
    Locate the coordinates x and y in a grid of point (lat, long). 

    Parameters
    ----------
    event_lat : float
        Represents the latitude of a point.

    event_lon : float 
        Represents the longitude of a point.

    dic_grid : dict
        Contains informations about virtual grid, how
            - lon_min_x: longitude mínima.
            - lat_min_y: latitude miníma. 
            - grid_size_lat_y: tamanho da grid latitude. 
            - grid_size_lon_x: tamanho da longitude da grid.
            - cell_size_by_degree: tamanho da célula da Grid.

    Returns
    -------
    indexes_lat_y : int
        Represents the index y in a grid of a point (lat, long). 

    indexes_lon_x : int
        Represents the index x in a grid of a point (lat, long).

    Examples
    --------
    >>> from pymove.core.grid import point_to_index_grid
    >>> dic_grid = {'lon_min_x': 113.54884299999999,
                    'lat_min_y': 22.147577,
                    'grid_size_lat_y': 140266,
                    'grid_size_lon_x': 56207,
                    'cell_size_by_degree': 0.0001353464801860623}
    >>> y, x  = point_to_index_grid(39.984094, 116.319236, dic_grid)
    >>> y
    131784.0

    >>> x
    20468.0

    """
    indexes_lat_y = np.floor((np.float64(event_lat) - dic_grid['lat_min_y'])/ dic_grid['cell_size_by_degree'])
    indexes_lon_x = np.floor((np.float64(event_lon) - dic_grid['lon_min_x'])/ dic_grid['cell_size_by_degree'])
    print('...[{},{}] indexes were created to lat and lon'.format(indexes_lat_y.size, indexes_lon_x.size))
    return indexes_lat_y, indexes_lon_x

def save_grid_pkl(filename, dic_grid):
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

    Returns
    -------
    

    Examples
    --------
    >>> from pymove.core.grid import save_grid
    >>> dic_grid = {'lon_min_x': 113.54884299999999,
                    'lat_min_y': 22.147577,
                    'grid_size_lat_y': 140266,
                    'grid_size_lon_x': 56207,
                    'cell_size_by_degree': 0.0001353464801860623}
    >>> grid_file = 'grid.pkl'
    
    >>> save_grid(grid_file, dict_grid)
    
    """
    try:
        f = open(filename,"wb")
        pickle.dump(dic_grid,f)
        f.close()
        print('\nA file was saved')
    except Exception as e:
        raise e

def read_grid_pkl(filename):
    """
    Save a grid with new file .pkl. 

    Parameters
    ----------
    filename : String
               Represents the name of a file.

    Returns
    -------
     dic_grid : dict
        Contains informations about virtual grid, how
            - lon_min_x: longitude mínima.
            - lat_min_y: latitude miníma. 
            - grid_size_lat_y: tamanho da grid latitude. 
            - grid_size_lon_x: tamanho da longitude da grid.
            - cell_size_by_degree: tamanho da célula da Grid.

    Examples
    --------
    >>> from pymove.core.grid import read_grid_pkl
    >>> grid_file = 'grid.pkl'
    >>> read_grid_pkl(grid_file)

    {'lon_min_x': 113.54884299999999,
    'lat_min_y': 22.147577,
    'grid_size_lat_y': 140266,
    'grid_size_lon_x': 56207,
    'cell_size_by_degree': 0.0001353464801860623}

    """
    try:
        with open(filename, 'rb') as f:
            dic_grid = pickle.load(f)
            f.close()
            return dic_grid
    except Exception as e:
        raise e

# TODO: ajeitar que tá dando erro + finalizar comentários
def show_grid_polygons(df_, id_, label_id = dic_labels['id'], label_polygon='polygon', figsize=(10,10)):   
    """
    Save a grid with new file .pkl. 

    Parameters
    ----------
    df_ : pandas.core.frame.DataFrame
        Represents the dataset with contains lat, long and datetime.
    
    id_ : String
        Represents the id.
    
    label_id : -
        ----
    
    label_polygon : -
        -------

    figsize : tuple
        Represents the size (float: width, float: height) of a figure.

    Returns
    -------
    df_ : pandas.core.frame.DataFrame
     
    fig : 


    Examples
    --------
    >>> from pymove.core.grid import show_grid_polygons
    >>> show_grid_polygons()

    """
    fig = plt.figure(figsize=figsize)
    
    #filter dataframe by id
    df_ = df_[ df_[label_id] == id_]
    
    xs_start, ys_start = df_.iloc[0][label_polygon].exterior.xy
    #xs_end, ys_end = df_.iloc[1][label_polygon].exterior.xy
    
    plt.plot(ys_start,xs_start, 'bo', markersize=20)             # start point
    #plt.plot(ys_end, xs_end, 'rX', markersize=20)           # end point
   

    for idx in range(df_.shape[0]):
        xs, ys = df_[label_polygon].iloc[idx].exterior.xy
        plt.plot(ys,xs, 'g', linewidth=2, markersize=5) 

    return df_, fig