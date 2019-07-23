import matplotlib
import matplotlib.pyplot as plt
import colorsys
import numpy as np
import folium
from matplotlib.colors import LinearSegmentedColormap

from utils import trajutils

# http://www.color-hex.com/color/

def rgb(RGBcolors):
    """ Return a tuple of integers, as used in AWT/Java plots. """
    blue  = RGBcolors[0]
    red   = RGBcolors[1]
    green = RGBcolors[2]
    return int(red*255), int(green*255), int(blue*255)

def hexRgb(RGBcolors):
    """ Return a hex string, as used in Tk plots. """
    return "#%02X%02X%02X" % rgb(RGBcolors)

def cmap_hex_color(cmap, i):
    return matplotlib.colors.rgb2hex(cmap(i))

def save_map(df, file, tiles='OpenStreetMap', label_id=trajutils.dic_labels['id'], dic_labels = trajutils.dic_labels, cmap='tab20'):
    #df['lat'] = df['lat'].astype('float64')
    #df['lon'] = df['lon'].astype('float64')
    m = folium.Map(tiles=tiles)
    m.fit_bounds([ [df[trajutils.dic_labels['lat']].min(), df[trajutils.dic_labels['lon']].min()], [df[trajutils.dic_labels['lat']].max(), df[trajutils.dic_labels['lon']].max()] ])
    
    ids = df[label_id].unique()
    
    #vals = np.linspace(0,1,256)
    #np.random.seed(color_seed)
    #np.random.shuffle(vals)
    #cmap_ = plt.cm.colors.ListedColormap(plt.cm.hsv(vals))
    # cmap_ = plt.cm.get_cmap( base_cmap, lut=ids.shape[0] )
    cmap_ = plt.cm.get_cmap( cmap )
    N = cmap_.N
    
    for id_ in ids:
        id_index = np.where(ids==id_)[0][0]
        df_ = df[ df[label_id] == id_ ]
        #print('id:{}, shape:{}'.format(id_, df_.shape))
        points_ = [ (point[0], point[1]) for point in df_[[trajutils.dic_labels['lat'], trajutils.dic_labels['lon']]].values]
        color_ = cmap_hex_color(cmap_, (id_index % N))
        #print( 'id_index:{}, color:{}'.format(id_index, color_) )
        folium.PolyLine(points_, weight=3, color=color_).add_to(m)
    m.save(file) 
  
def save_wkt(df, file_str, label_id=trajutils.dic_labels['id'], dic_labels=trajutils.dic_labels):
    str_ = '{};linestring\n'.format(label_id)
    ids = df[label_id].unique()
    for id_ in ids:
        df_ = df[ df[label_id] == id_ ]
        str_ += '{};LINESTRING('.format(id_)
        str_ += ','.join('{} {}'.format(x[0],x[1]) for x in df_[dic_labels['lon'], dic_labels['lat']].values)
        str_ += ')\n'
    open(file_str, "w").write(str_)    
    
def invert_map(map_):
    inv_map = {}
    items_ = map_.items()
    for k, v in items_:
        inv_map[v] = inv_map.get(v, [])
        inv_map[v].append(k)
    return inv_map
    
def show_traj(df, label_tid=trajutils.dic_features_label['tid'], dic_labels=trajutils.dic_labels, figsize=(10,10), return_fig=False):
    fig = plt.figure(figsize=figsize)
    ids = df[label_tid].unique()
    for id_ in ids:
        df_id = df[ df[label_tid] == id_ ]
        plt.plot(df_id[dic_labels['lon']], df_id[dic_labels['lat']], 'rX',markersize=20)
    if return_fig:
        return fig

def show_traj_id(df, tid, label_tid=trajutils.dic_features_label['tid'], dic_labels=trajutils.dic_labels, figsize=(10,10)):
    fig = plt.figure(figsize=figsize)
    df_ = df[ df[label_tid] == tid ]
    plt.plot(df_.iloc[0][dic_labels['lon']], df_.iloc[0][dic_labels['lat']], 'yo', markersize=20)             # start point
    plt.plot(df_.iloc[-1][dic_labels['lon']], df_.iloc[-1][dic_labels['lat']], 'yX', markersize=20)           # end point
    
    if 'isNode'not in df_:
        plt.plot(df_[dic_labels['lon']], df_[dic_labels['lat']])
        plt.plot(df_.loc[:, dic_labels['lon']], df_.loc[:, dic_labels['lat']], 'r.', markersize=8)  # points
    else:
        filter_ = df_['isNode'] == 1
        df_nodes = df_.loc[filter_]
        df_points = df_.loc[~filter_]
        plt.plot(df_nodes[dic_labels['lon']], df_nodes[dic_labels['lat']], linewidth=3)
        plt.plot(df_points[dic_labels['lon']], df_points[dic_labels['lat']])
        plt.plot(df_nodes[dic_labels['lon']], df_nodes[dic_labels['lat']], 'go', markersize=10)   # nodes
        plt.plot(df_points[dic_labels['lon']], df_points[dic_labels['lat']], 'r.', markersize=8)  # points  
    return df_, fig

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
    
def save_bbox(bbox_tuple, file, tiles='OpenStreetMap', color='red'):
    m = folium.Map(tiles=tiles)
    m.fit_bounds([ [bbox_tuple[0], bbox_tuple[1]], [bbox_tuple[2], bbox_tuple[3]] ])
    points_ = [ (bbox_tuple[0], bbox_tuple[1]), (bbox_tuple[0], bbox_tuple[3]), 
                (bbox_tuple[2], bbox_tuple[3]), (bbox_tuple[2], bbox_tuple[1]),
                (bbox_tuple[0], bbox_tuple[1]) ]
    folium.PolyLine(points_, weight=3, color=color).add_to(m)
    m.save(file) 