# TODO: Andreza
import folium
import colorsys
import numpy as np
# import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pymove.utils.utils import dic_features_label, dic_labels
from pymove.utils.transformations import create_update_date_features, create_update_hour_features, create_update_day_of_the_week_features, create_update_time_of_day_features

def rgb(rgb_colors):
    """
    Return a tuple of integers, as used in AWT/Java plots.

    Parameters
    ----------
    rgb_colors : list
        Represents a list with three positions that correspond to the percentage red, green and blue colors.

    Returns
    -------
    tuple_rgb : tuple
        Represents a tuple of integers that correspond the colors values.

    Examples
    --------
    >>> from pymove.visualization.visualization import rgb
    >>> rgb([0.6,0.2,0.2])
        (51, 51, 153)

    """
    blue  = rgb_colors[0]
    red   = rgb_colors[1]
    green = rgb_colors[2]
    tuple_rgb = int(red*255), int(green*255), int(blue*255)
    return tuple_rgb

def hex_rgb(rgb_colors):
    """
    Return a hex string, as used in Tk plots.

    Parameters
    ----------
    rgb_colors : list
        Represents a list with three positions that correspond to the percentage red, green and blue colors.

    Returns
    -------
    hex_colors : String
        Represents a color in hexadecimal format.

    Examples
    --------
    >>> from pymove.visualization.visualization import hex_rgb
    >>> hex_rgb([0.6,0.2,0.2])
    '#333399'

    """
    hex_colors = "#%02X%02X%02X" % rgb(rgb_colors)
    return hex_colors

def cmap_hex_color(cmap, i):
    return matplotlib.colors.rgb2hex(cmap(i))

def save_map(df, file, tiles='OpenStreetMap', label_id=dic_labels['id'], dic_labels = dic_labels, cmap='tab20'):
    m = folium.Map(tiles=tiles)
    m.fit_bounds([ [df[dic_labels['lat']].min(), df[dic_labels['lon']].min()], [df[dic_labels['lat']].max(), df[dic_labels['lon']].max()] ])
    
    ids = df[label_id].unique()
    cmap_ = plt.cm.get_cmap( cmap )
    N = cmap_.N
    
    for id_ in ids:
        id_index = np.where(ids==id_)[0][0]
        df_ = df[ df[label_id] == id_ ]
        points_ = [ (point[0], point[1]) for point in df_[[dic_labels['lat'], dic_labels['lon']]].values]
        color_ = cmap_hex_color(cmap_, (id_index % N))
        folium.PolyLine(points_, weight=3, color=color_).add_to(m)
    m.save(file) 
  
def save_wkt(df, file_str, label_id=dic_labels['id'], dic_labels=dic_labels):
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
    
""" https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.plot.html """

def show_object_id_by_date(df_, create_features=True, figsize=(21,9), save_fig=True, name='shot_points_by_date.png', low_memory=True):
    fig, ax = plt.subplots(2,2,figsize=figsize)
    create_update_date_features(df_)
    df_.loc[:,['date', 'id']].groupby(['id', 'date']).count().reset_index().groupby('date').count().plot(subplots=True, kind = 'line', grid=True, ax=ax[1][0], rot=45, fontsize=12)
    create_update_hour_features(df_)
    df_.loc[:,['hour', 'id']].groupby(['hour', 'id']).count().reset_index().groupby('hour').count().plot(subplots=True, kind = 'line', grid=True, ax=ax[1][1], fontsize=12)
    del df_['date']    
    del df_['hour']
    create_update_day_of_the_week_features(df_)
    df_.loc[:,['period', 'id']].groupby(['period', 'id']).count().reset_index().groupby('period').count().plot(subplots=True, kind = 'bar', rot=0, ax=ax[0][0], fontsize=12)
    del df_['period']
    create_update_time_of_day_features(df_)  
    df_.loc[:,['day', 'id']].groupby(['day', 'id']).count().reset_index().groupby('day').count().plot(subplots=True,  kind = 'bar', ax=ax[0][1], rot=0, fontsize=12)
    del df_['day']

    if save_fig:
        plt.savefig(fname=name, fig=fig)

def show_lat_lon_GPS(df_, dic_labels=dic_labels, kind='scatter', figsize=(21,9), save_fig=False, name='show_gps_points.png'):
    try:
        if dic_labels['lat'] in df_ and dic_labels['lon'] in df_:
            df_.drop_duplicates([dic_labels['lat'], dic_labels['lon']]).plot(kind=kind, x=dic_labels['lon'], y=dic_labels['lat'], figsize=figsize)
            plt.plot(df_.iloc[0][dic_labels['lon']], df_.iloc[0][dic_labels['lat']], 'yo', markersize=10)             # start point
            plt.plot(df_.iloc[-1][dic_labels['lon']], df_.iloc[-1][dic_labels['lat']], 'yX', markersize=10)           # end point
            
            if save_fig == True:
                plt.savefig(name)   
    except Exception as e:
        raise e

def show_all_features(df_, figsize=(21,15), dtype=np.float64, save_fig=True, name='features.png'):
    try:
        col_float = df_.select_dtypes(include=[dtype]).columns
        tam = col_float.size
        if(tam > 0):
            fig, ax = plt.subplots(tam,1, figsize=figsize)
            ax_count = 0
            for col in col_float:
                ax[ax_count].set_title(col)
                df_[col].plot(subplots=True, ax=ax[ax_count])
                ax_count+=1
            
            if save_fig:
                plt.savefig(fname=name, fig=fig)
    except Exception as e:
        raise e

def show_traj(df, label_tid=dic_labels['id'], dic_labels=dic_labels, figsize=(10,10), return_fig=True, markers= 'o',markersize=20):
    fig = plt.figure(figsize=figsize)
    ids = df[label_tid].unique()
    
    for id_ in ids:
        df_id = df[ df[label_tid] == id_ ]
        plt.plot(df_id[dic_labels['lon']], df_id[dic_labels['lat']], markers, markersize=markersize)
    if return_fig:
        return fig

def show_traj_id(df, tid, label_tid=dic_features_label['tid'], dic_labels=dic_labels, figsize=(10,10)):
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
