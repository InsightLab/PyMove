from matplotlib.colors import LinearSegmentedColormap
import matplotlib
import matplotlib.pyplot as plt
import colorsys
import numpy as np
import folium

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


def save_map(df, file, tiles='OpenStreetMap', label_id='id', label_lat='lat', label_lon='long', cmap='tab20'):
    m = folium.Map(tiles=tiles)
    m.fit_bounds([ [df[label_lat].min(), df[label_lon].min()], [df[label_lat].max(), df[label_lon].max()] ])
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
        points_ = [ (point[0], point[1]) for point in df_[[label_lat, label_lon]].values]
        color_ = cmap_hex_color(cmap_, (id_index % N))
        #print( 'id_index:{}, color:{}'.format(id_index, color_) )
        folium.PolyLine(points_, weight=3, color=color_).add_to(m)
    m.save(file) 

    
def save_wkt(df, file_str, label_id='id', label_lat='lat', label_lon='lon'):
    str_ = '{};linestring\n'.format(label_id)
    ids = df[label_id].unique()
    for id_ in ids:
        df_ = df[ df[label_id] == id_ ]
        str_ += '{};LINESTRING('.format(id_)
        str_ += ','.join('{} {}'.format(x[0],x[1]) for x in df_[[label_lon, label_lat]].values)
        str_ += ')\n'
    open(file_str, "w").write(str_)    
    
def save_bbox(bbox_tuple, file, tiles='OpenStreetMap'):
    m = folium.Map(tiles=tiles)
    m.fit_bounds([ [bbox_tuple[0], bbox_tuple[1]], [bbox_tuple[2], bbox_tuple[3]] ])
    points_ = [ (bbox_tuple[0], bbox_tuple[1]), (bbox_tuple[0], bbox_tuple[3]), 
                (bbox_tuple[2], bbox_tuple[3]), (bbox_tuple[2], bbox_tuple[1]),
                (bbox_tuple[0], bbox_tuple[1]) ]
    folium.PolyLine(points_, weight=3, color='red').add_to(m)
    m.save(file) 

    
def show_traj(df, label_id='id', label_lat='lat', label_lon='lon', figsize=(10,10), return_fig=False):
    fig = plt.figure(figsize=figsize)
    ids = df[label_id].unique()
    for id_ in ids:
        df_id = df[ df[label_id] == id_ ]
        plt.plot(df_id[label_lon], df_id[label_lat])
    if return_fig:
        return fig

    
def show_traj_id(df, tid, label_id='id', label_lat='lat', label_lon='lon', figsize=(10,10)):
    fig = plt.figure(figsize=figsize)
    df_ = df[ df[label_id] == tid ]
    plt.plot(df_.iloc[0][label_lon], df_.iloc[0][label_lat], 'yo', markersize=20)             # start point
    plt.plot(df_.iloc[-1][label_lon], df_.iloc[-1][label_lat], 'yX', markersize=20)           # end point
    
    if 'isNode' in df_:
        filter_ = df_['isNode'] == 1
        df_nodes = df_.loc[filter_]
        df_points = df_.loc[~filter_]
        plt.plot(df_nodes[label_lon], df_nodes[label_lat], linewidth=3)
        plt.plot(df_points[label_lon], df_points[label_lat])
        plt.plot(df_nodes[label_lon], df_nodes[label_lat], 'go', markersize=10)   # nodes
        plt.plot(df_points[label_lon], df_points[label_lat], 'r.', markersize=8)  # points
    else:
        plt.plot(df_[label_lon], df_[label_lat])
        plt.plot(df_.loc[:, label_lon], df_.loc[:, label_lat], 'r.', markersize=8)  # points
        
    return df_, fig

