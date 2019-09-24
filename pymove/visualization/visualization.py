# TODO: Andreza
import folium
import colorsys
import numpy as np
# import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
# from pymove.utils.utils import dic_features_label, dic_labels
from pymove.utils.transformations import create_update_date_features, create_update_hour_features, create_update_day_of_the_week_features, create_update_time_of_day_features
from pymove.utils.constants import LATITUDE, LONGITUDE, DATETIME, TRAJ_ID, TID, PERIOD, DATE, HOUR, DAY

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

#TODO: Perguntar o que faz, pq eu não sei
def cmap_hex_color(cmap, i): 
    """
    ?

    Parameters
    ----------
    cmap : ?
        ?

    i : ?
        ?


    Returns
    -------
    

    Examples
    --------
    >>> from pymove.visualization.visualization import cmap_hex_color
    >>> cmap_hex_color(?, ?)

    """
    return matplotlib.colors.rgb2hex(cmap(i))

#TODO finalizar doc
def save_map(df, filename, tiles='OpenStreetMap', label_id=TRAJ_ID, cmap='tab20'):
    """
    Save a visualization in a map in a new file.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Represents the dataset with contains lat, long and datetime.

    filename : String
        Represents the filename.

    tiles : String
        ?

    label_id : String
        ?
    
    cmap: String
        ?

    Returns
    -------
    

    Examples
    --------
    >>> from pymove.visualization.visualization import save_map
    >>> save_map(df, 'saida')
    -
    """
    m = folium.Map(tiles=tiles)
    m.fit_bounds([ [df[LATITUDE].min(), df[LONGITUDE].min()], [df[LATITUDE].max(), df[LONGITUDE].max()] ])
    
    ids = df[label_id].unique()
    cmap_ = plt.cm.get_cmap( cmap )
    N = cmap_.N
    
    for id_ in ids:
        id_index = np.where(ids==id_)[0][0]
        df_ = df[ df[label_id] == id_ ]
        points_ = [ (point[0], point[1]) for point in df_[[LATITUDE, LONGITUDE]].values]
        color_ = cmap_hex_color(cmap_, (id_index % N))
        folium.PolyLine(points_, weight=3, color=color_).add_to(m)
    m.save(filename) 
  
#TODO: Finalizar comentário e testar
def save_wkt(df, filename, label_id=TRAJ_ID):
    """
    ?

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Represents the dataset with contains lat, long and datetime.

    filename : String
        Represents the filename.

    label_id : String
        ?
    
    Returns
    -------
    
    Examples
    --------
    >>> from pymove.visualization.visualization import save_wkt
    >>> save_wkt(df, 'saida')
    -
    """
    str_ = '{};linestring\n'.format(label_id)
    ids = df[label_id].unique()
    for id_ in ids:
        df_ = df[ df[label_id] == id_ ]
        str_ += '{};LINESTRING('.format(id_)
        str_ += ','.join('{} {}'.format(x[0],x[1]) for x in df_[LONGITUDE, LATITUDE].values)
        str_ += ')\n'
    open(filename, "w").write(str_)    
    
#TODO: Finalizar comentário e testar
def invert_map(map_):
    """
    ?

    Parameters
    ----------
    map_ : ?
        ?
    
    Returns
    -------
    
    Examples
    --------
    >>> from pymove.visualization.visualization import invert_map
    >>> invert_map(?)
    -

    """
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
    df_.loc[:,[DATE, TRAJ_ID]].groupby([TRAJ_ID, DATE]).count().reset_index().groupby(DATE).count().plot(subplots=True, kind = 'line', grid=True, ax=ax[1][0], rot=45, fontsize=12)
    create_update_hour_features(df_)
    df_.loc[:,[HOUR, TRAJ_ID]].groupby([HOUR, TRAJ_ID]).count().reset_index().groupby(HOUR).count().plot(subplots=True, kind = 'line', grid=True, ax=ax[1][1], fontsize=12)
    del df_[DATE]    
    del df_[HOUR]
    create_update_day_of_the_week_features(df_)
    df_.loc[:,[PERIOD, TRAJ_ID]].groupby([PERIOD, TRAJ_ID]).count().reset_index().groupby(PERIOD).count().plot(subplots=True, kind = 'bar', rot=0, ax=ax[0][0], fontsize=12)
    del df_[PERIOD]
    create_update_time_of_day_features(df_)  
    df_.loc[:,[DAY, TRAJ_ID]].groupby([DAY, TRAJ_ID]).count().reset_index().groupby(DAY).count().plot(subplots=True,  kind = 'bar', ax=ax[0][1], rot=0, fontsize=12)
    del df_[DAY]

    if save_fig:
        plt.savefig(fname=name, fig=fig)

def show_lat_lon_GPS(df_, kind='scatter', figsize=(21,9), save_fig=False, name='show_gps_points.png'):
    try:
        if LATITUDE in df_ and LONGITUDE in df_:
            df_.drop_duplicates([LATITUDE, LONGITUDE]).plot(kind=kind, x=LONGITUDE, y=LATITUDE, figsize=figsize)
            plt.plot(df_.iloc[0][LONGITUDE], df_.iloc[0][LATITUDE], 'yo', markersize=10)             # start point
            plt.plot(df_.iloc[-1][LONGITUDE], df_.iloc[-1][LATITUDE], 'yX', markersize=10)           # end point
            
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

def show_traj(df, label_tid=TRAJ_ID, figsize=(10,10), return_fig=True, markers= 'o',markersize=20):
    fig = plt.figure(figsize=figsize)
    ids = df[label_tid].unique()
    
    for id_ in ids:
        df_id = df[ df[label_tid] == id_ ]
        plt.plot(df_id[LONGITUDE], df_id[LATITUDE], markers, markersize=markersize)
    if return_fig:
        return fig

def show_traj_id(df, tid, label_tid=TID,  figsize=(10,10)):
    fig = plt.figure(figsize=figsize)
    df_ = df[ df[label_tid] == tid ]
    plt.plot(df_.iloc[0][LONGITUDE], df_.iloc[0][LATITUDE], 'yo', markersize=20)             # start point
    plt.plot(df_.iloc[-1][LONGITUDE], df_.iloc[-1][LATITUDE], 'yX', markersize=20)           # end point
    
    if 'isNode'not in df_:
        plt.plot(df_[LONGITUDE], df_[LATITUDE])
        plt.plot(df_.loc[:, LONGITUDE], df_.loc[:, LATITUDE], 'r.', markersize=8)  # points
    else:
        filter_ = df_['isNode'] == 1
        df_nodes = df_.loc[filter_]
        df_points = df_.loc[~filter_]
        plt.plot(df_nodes[LONGITUDE], df_nodes[LATITUDE], linewidth=3)
        plt.plot(df_points[LONGITUDE], df_points[LATITUDE])
        plt.plot(df_nodes[LONGITUDE], df_nodes[LATITUDE], 'go', markersize=10)   # nodes
        plt.plot(df_points[LONGITUDE], df_points[LATITUDE], 'r.', markersize=8)  # points  
    return df_, fig
