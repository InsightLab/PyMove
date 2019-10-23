import folium
import numpy as np
import matplotlib.pyplot as plt
from folium.plugins import HeatMap, HeatMapWithTime, MarkerCluster, FastMarkerCluster
from pymove.utils.constants import LATITUDE, LONGITUDE, TRAJ_ID, TID, PERIOD, DATE, HOUR, DAY, COUNT

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
    df_.generate_date_features()
    df_.loc[:,[DATE, TRAJ_ID]].groupby([TRAJ_ID, DATE]).count().reset_index().groupby(DATE).count().plot(subplots=True, kind = 'line', grid=True, ax=ax[1][0], rot=45, fontsize=12)
    df_.generate_hour_features(df_)
    df_.loc[:,[HOUR, TRAJ_ID]].groupby([HOUR, TRAJ_ID]).count().reset_index().groupby(HOUR).count().plot(subplots=True, kind = 'line', grid=True, ax=ax[1][1], fontsize=12)
    del df_[DATE]    
    del df_[HOUR]
    df_.generate_day_of_the_week_features()
    df_.loc[:,[PERIOD, TRAJ_ID]].groupby([PERIOD, TRAJ_ID]).count().reset_index().groupby(PERIOD).count().plot(subplots=True, kind = 'bar', rot=0, ax=ax[0][0], fontsize=12)
    del df_[PERIOD]
    df_.generate_time_of_day_features(df_)
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

def generateBaseMap(default_location, default_zoom_start=12):
    base_map = folium.Map(location=default_location, control_scale=True, zoom_start=default_zoom_start)
    return base_map

def heatmap(df_, n_rows = None, lat_origin=None, lon_origin=None, zoom_start=12, radius = 8, max_zoom = 13, base_map=None, save_as_html=False, filename='heatmap.html'):
    if base_map is None:
        if lat_origin is None and lon_origin is None:
            lat_origin = df_.loc[0][LATITUDE]
            lon_origin = df_.loc[0][LONGITUDE]
        base_map = generateBaseMap(default_location=[lat_origin, lon_origin], default_zoom_start=zoom_start)

    if n_rows is None:
        n_rows = df_.shape[0]

    df_[COUNT] = 1
    HeatMap(data=df_.loc[:n_rows, [LATITUDE, LONGITUDE, COUNT]].groupby([LATITUDE, LONGITUDE]).sum().reset_index().values.tolist(),
        radius=radius, max_zoom=max_zoom).add_to(base_map)
    # base_map.add_child(folium.ClickForMarker(popup='Potential Location')) # habilita marcações no mapa
    df_.drop(columns=[COUNT], inplace=True)
    if save_as_html:
        base_map.save(outfile=filename)
    else:
        return base_map


def heatmap_with_time(df_, n_rows = None, lat_origin=None, lon_origin=None, zoom_start=12, radius = 5, min_opacity = 0.5, max_opacity = 0.8, base_map=None, save_as_html=False,
                      filename='heatmap_with_time.html'):
    if base_map is None:
        if lat_origin is None and lon_origin is None:
            lat_origin = df_.loc[0][LATITUDE]
            lon_origin = df_.loc[0][LONGITUDE]
        base_map = generateBaseMap(default_location=[lat_origin, lon_origin], default_zoom_start=zoom_start)

    if n_rows is None:
        n_rows = df_.shape[0]

    df_[HOUR] = df_.datetime.dt.hour
    df_[COUNT] = 1
    df_hour_list = []
    for hour in df_.hour.sort_values().unique():
        df_hour_list.append(df_.loc[df_.hour == hour, [LATITUDE, LONGITUDE, COUNT]].groupby(
            [LATITUDE, LONGITUDE]).sum().reset_index().values.tolist())

    HeatMapWithTime(df_hour_list[:n_rows], radius=radius, gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red'},
                    min_opacity=min_opacity, max_opacity=max_opacity, use_local_extrema=True).add_to(base_map)

    df_.drop(columns=[COUNT, HOUR], inplace=True)
    if save_as_html:
        base_map.save(outfile=filename)
    else:
        return base_map

def cluster(df_, n_rows = None, lat_origin=None, lon_origin=None, zoom_start=12,  base_map=None, save_as_html=False, filename='cluster.html'):
    if base_map is None:
        if lat_origin is None and lon_origin is None:
            lat_origin = df_.loc[0][LATITUDE]
            lon_origin = df_.loc[0][LONGITUDE]
        base_map = generateBaseMap(default_location=[lat_origin, lon_origin], default_zoom_start=zoom_start)

    if n_rows is None:
        n_rows = df_.shape[0]

    mc = MarkerCluster()
    for row in df_[:n_rows].iterrows():
        pop = "<b>Latitude:</b> " + str(row[1].lat) + "\n<b>Longitude:</b> " + str(row[1].lon) + "\n<b>Datetime:</b> " + str(row[1].datetime)
        mc.add_child(folium.Marker(location=[row[1].lat, row[1].lon], popup=pop))
    base_map.add_child(mc)

    if save_as_html:
        base_map.save(outfile=filename)
    else:
        return base_map

def faster_cluster(df_, n_rows = None, lat_origin=None, lon_origin=None, zoom_start=12,  base_map=None, save_as_html=False, filename='faster_cluster.html'):
    if base_map is None:
        if lat_origin is None and lon_origin is None:
            lat_origin = df_.loc[0][LATITUDE]
            lon_origin = df_.loc[0][LONGITUDE]
        base_map = generateBaseMap(default_location=[lat_origin, lon_origin], default_zoom_start=zoom_start)

    if n_rows is None:
        n_rows = df_.shape[0]

    callback = """\
    function (row) {
        var marker;
        marker = L.circle(new L.LatLng(row[0], row[1]), {color:'red'});
        return marker;
    };
    """
    FastMarkerCluster(df_.loc[:n_rows, [LATITUDE, LONGITUDE]].values.tolist(), callback=callback).add_to(base_map)

    if save_as_html:
        base_map.save(outfile=filename)
    else:
        return base_map

def plot_trejectory_with_folium(df_, n_rows = None, lat_origin=None, lon_origin=None, zoom_start=12,  base_map=None, save_as_html=False, filename='plot_trejectory_with_folium.html'):
    if base_map is None:
        if lat_origin is None and lon_origin is None:
            lat_origin = df_.loc[0][LATITUDE]
            lon_origin = df_.loc[0][LONGITUDE]
        base_map = generateBaseMap(default_location=[lat_origin, lon_origin], default_zoom_start=zoom_start)

    if n_rows is None:
        n_rows = df_.shape[0]

    for each in df_[:n_rows].iterrows():
        pop = "<b>Latitude:</b> " + str(each[1].lat) + "\n<b>Longitude:</b> " + str(
            each[1].lon) + "\n<b>Datetime:</b> " + str(each[1].datetime)
        folium.Marker(location=[each[1]['lat'], each[1]['lon']], clustered_marker=True, popup=pop).add_to(base_map)

    if save_as_html:
        base_map.save(outfile=filename)
    else:
        return base_map