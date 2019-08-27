# TODO: Modelar como classe
# TODO: Andreza
import math
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt

from tqdm import tqdm_notebook as tqdm
from geojson import Feature, FeatureCollection
from shapely.geometry import Polygon
from geojson import Polygon as jsonPolygon
import pickle

from pymove import trajutils

def lat_meters(Lat):
    """
    Transform latitude degree to meters
        Example: Latitude in Fortaleza: -3.8162973555
    
    """
    rlat = float(Lat) * math.pi / 180
    # meter per degree Latitude
    meters_lat = 111132.92 - 559.82 * math.cos(2 * rlat) + 1.175 * math.cos(4 * rlat)
    # meter per degree Longitude
    meters_lgn = 111412.84 * math.cos(rlat) - 93.5 * math.cos(3 * rlat)
    meters = (meters_lat + meters_lgn) / 2
    return meters

def create_virtual_grid(cell_size, bbox, meters_by_degree = lat_meters(-3.8162973555)):
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
    my_dict = dict()
    
    my_dict['lon_min_x'] = lon_min_x
    my_dict['lat_min_y'] = lat_min_y
    my_dict['grid_size_lat_y'] = grid_size_lat_y
    my_dict['grid_size_lon_x'] = grid_size_lon_x
    my_dict['cell_size_by_degree'] = cell_size_by_degree
    print('\n..A virtual grid was created')
    return my_dict

def create_all_polygons_on_grid(dic_grid):
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
    try:
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

def create_one_polygon_to_point_on_grid(dic_grid, index_grid_lat, index_grid_lon):
    lat_init = dic_grid['lat_min_y'] + dic_grid['cell_size_by_degree'] * index_grid_lat
    lon_init = dic_grid['lon_min_x'] + dic_grid['cell_size_by_degree'] * index_grid_lon
    polygon = Polygon(((lat_init, lon_init),
         (lat_init + dic_grid['cell_size_by_degree'], lon_init),
         (lat_init + dic_grid['cell_size_by_degree'], lon_init + dic_grid['cell_size_by_degree']),
         (lat_init, lon_init + dic_grid['cell_size_by_degree'])
                                            ))
    return polygon

def point_to_index_grid(event_lat, event_lon, dic_grid):
    indexes_lat_y = np.floor((np.float64(event_lat) - dic_grid['lat_min_y'])/ dic_grid['cell_size_by_degree'])
    indexes_lon_x = np.floor((np.float64(event_lon) - dic_grid['lon_min_x'])/ dic_grid['cell_size_by_degree'])
    print('...[{},{}] indexes were created to lat and lon'.format(indexes_lat_y.size, indexes_lon_x.size))
    return indexes_lat_y, indexes_lon_x

def save_grid_pkl(filename, dic_grid):
    """ex: save_grid(grid_file, my_dict_grid)"""
    try:
        f = open(filename,"wb")
        pickle.dump(dic_grid,f)
        f.close()
        print('\nA file was saved')
    except Exception as e:
        raise e

def read_grid_pkl(filename):
    """ex: read_grid(grid_file)"""
    try:
        with open(filename, 'rb') as f:
            dic_grid = pickle.load(f)
            f.close()
            return dic_grid
    except Exception as e:
        raise e

def show_grid_polygons(df_, id_, label_id = trajutils.dic_labels['id'], label_polygon='polygon', figsize=(10,10)):   
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