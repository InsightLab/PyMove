import folium
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from folium.plugins import HeatMap, HeatMapWithTime, MarkerCluster, FastMarkerCluster
from pymove.utils.constants import LATITUDE, LONGITUDE, TRAJ_ID, PERIOD, DATE, HOUR, DAY, COUNT


def rgb(rgb_colors):
    """
    Return a tuple of integers, as used in AWT/Java plots.

    Parameters
    ----------
    rgb_colors : list
        Represents a list with three positions that correspond to the percentage red, green and blue colors.

    Returns
    -------
    tuple
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
    return int(red*255), int(green*255), int(blue*255)


def hex_rgb(rgb_colors):
    """
    Return a hex string, as used in Tk plots.

    Parameters
    ----------
    rgb_colors : list
        Represents a list with three positions that correspond to the percentage red, green and blue colors.

    Returns
    -------
    String
        Represents a color in hexadecimal format.

    Examples
    --------
    >>> from pymove.visualization.visualization import hex_rgb
    >>> hex_rgb([0.6,0.2,0.2])
    '#333399'

    """
    return "#%02X%02X%02X" % rgb(rgb_colors)


def cmap_hex_color(cmap, i): 
    """
    Convert a Colormap to hex color.

    Parameters
    ----------
    cmap : matplotlib.colors.ListedColormap
        Represents the Colormap.

    i : int
        List color index.

    Returns
    -------
    String
        Represents corresponding hex string.

    """
    return matplotlib.colors.rgb2hex(cmap(i))


def save_map(
    move_data,
    filename,
    tiles='OpenStreetMap',
    label_id=TRAJ_ID,
    cmap='tab20'
):
    """
    Save a visualization in a map in a new file.

    Parameters
    ----------
    move_data : pymove.core.MoveDataFrameAbstract subclass.
        Input trajectory data.

    filename : String
        Represents the filename.

    tiles : String
        Represents the type of tile that will be used on the map.

    label_id : String
        Represents column name of trajectory id.
    
    cmap: String
        Represents the Colormap.

    Returns
    -------

    """
    m = folium.Map(tiles=tiles)
    m.fit_bounds([[move_data[LATITUDE].min(), move_data[LONGITUDE].min()],
                  [move_data[LATITUDE].max(), move_data[LONGITUDE].max()]])
    
    ids = move_data[label_id].unique()
    cmap_ = plt.cm.get_cmap(cmap)
    n = cmap_.N
    
    for id_ in ids:
        id_index = np.where(ids == id_)[0][0]
        move_data = move_data[move_data[label_id] == id_]
        points_ = [(point[0], point[1]) for point in move_data[[LATITUDE, LONGITUDE]].values]
        color_ = cmap_hex_color(cmap_, (id_index % n))
        folium.PolyLine(points_, weight=3, color=color_).add_to(m)
    m.save(filename) 
  

def save_wkt(move_data, filename, label_id=TRAJ_ID):
    """
    Save a visualization in a map in a new file .wkt.

    Parameters
    ----------
    move_data : pymove.core.MoveDataFrameAbstract subclass.
        Input trajectory data.

    filename : String
        Represents the filename.

    label_id : String
        Represents column name of trajectory id.

    Returns
    -------

    """
    str_ = '{};linestring\n'.format(label_id)
    ids = move_data[label_id].unique()
    for id_ in ids:
        move_data = move_data[move_data[label_id] == id_]
        str_ += '{};LINESTRING('.format(id_)
        str_ += ','.join('{} {}'.format(x[0],x[1]) for x in move_data[LONGITUDE, LATITUDE].values)
        str_ += ')\n'
    open(filename, "w").write(str_)    
    

#TODO: não sei o que faz
def invert_map(map_):
    """
    ?

    Parameters
    ----------
    map_ : ?
        ?
    
    Returns
    -------

    """
    inv_map = {}
    items_ = map_.items()
    for k, v in items_:
        inv_map[v] = inv_map.get(v, [])
        inv_map[v].append(k)
    return inv_map


def show_object_id_by_date(
    move_data,
    figsize=(21, 9),
    save_fig=True,
    name='shot_points_by_date.png',
):
    """
    Generates four visualizations based on datetime feature:
        - Bar chart trajectories by day periods
        - Bar chart trajectories day of the week
        - Line chart trajectory by date
        - Line chart of trajectory byhours of the day.

    Parameters
    ----------
    move_data : pymove.core.MoveDataFrameAbstract subclass.
        Input trajectory data.

    figsize : tuple, optional, default (21,9).
        Represents dimensions of figure.

    save_fig : bool, optional, default True.
        Represents whether or not to save the generated picture.

    name : String, optional, default 'shot_points_by_date.png'.
        Represents name of a file.

    Returns
    -------

    References
    ----------
    https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.plot.html
    """
    fig, ax = plt.subplots(2, 2, figsize=figsize)

    move_data.generate_date_features()
    move_data.loc[:, [DATE, TRAJ_ID]].groupby([TRAJ_ID, DATE]).count().reset_index().groupby(DATE).count() \
        .plot(subplots=True, kind='line', grid=True, ax=ax[1][0], rot=45, fontsize=12)

    move_data.generate_hour_features()
    move_data.loc[:, [HOUR, TRAJ_ID]].groupby([HOUR, TRAJ_ID]).count().reset_index().groupby(HOUR).count() \
        .plot(subplots=True, kind = 'line', grid=True, ax=ax[1][1], fontsize=12)

    move_data.generate_time_of_day_features()
    move_data.loc[:, [PERIOD, TRAJ_ID]].groupby([PERIOD, TRAJ_ID]).count().reset_index().groupby(PERIOD).count() \
        .plot(subplots=True, kind = 'bar', rot=0, ax=ax[0][0], fontsize=12)

    move_data.generate_day_of_the_week_features()
    move_data.loc[:, [DAY, TRAJ_ID]].groupby([DAY, TRAJ_ID]).count().reset_index().groupby(DAY).count() \
        .plot(subplots=True,  kind = 'bar', ax=ax[0][1], rot=0, fontsize=12)

    move_data.drop(columns=[DATE, HOUR, PERIOD, DAY], inplace=True)

    if save_fig:
        plt.savefig(fname=name, fig=fig)


def show_lat_lon_GPS(
    move_data,
    kind='scatter',
    figsize=(21, 9),
    save_fig=False,
    name='show_gps_points.png'
):
    """
    Generate a visualization with points [lat, lon] of dataset.

    Parameters
    ----------
    move_data : pymove.core.MoveDataFrameAbstract subclass.
        Input trajectory data.

    kind : String, optional, default 'scatter'.
        Represents chart type.

    figsize : tuple, optional, default (21,9).
        Represents dimensions of figure.

    save_fig : bool, optional, default True.
        Represents whether or not to save the generated picture.

    name : String, optional, default 'show_gps_points.png'.
        Represents name of a file.

    Returns
    -------

    """
    try:
        if LATITUDE in move_data and LONGITUDE in move_data:
            move_data.drop_duplicates([LATITUDE, LONGITUDE]).plot(kind=kind, x=LONGITUDE, y=LATITUDE, figsize=figsize)
            # start point
            plt.plot(move_data.iloc[0][LONGITUDE], move_data.iloc[0][LATITUDE], 'yo', markersize=10)
            # end point
            plt.plot(move_data.iloc[-1][LONGITUDE], move_data.iloc[-1][LATITUDE], 'yX', markersize=10)
            
            if save_fig:
                plt.savefig(name)   
    except Exception as e:
        raise e


def generate_base_map(default_location, default_zoom_start=12):
    """
    Generate a folium map.

    Parameters
    ----------
    default_location : tuple.
        Represents coordinates lat, lon which will be the center of the map.

    default_zoom_start : int, optional, default 12.
        Represents the zoom which will be the center of the map.

    Returns
    -------
    base_map : folium.folium.Map.
        Represents a folium map.

    """
    base_map = folium.Map(location=default_location, control_scale=True, zoom_start=default_zoom_start)
    return base_map


def heatmap(
    move_data,
    n_rows=None,
    lat_origin=None,
    lon_origin=None,
    zoom_start=12,
    radius=8,
    max_zoom=13,
    base_map=None,
    save_as_html=False,
    filename='heatmap.html'
):
    """
    Generate visualization of Heat Map using folium plugin.

    Parameters
    ----------
    move_data : pymove.core.MoveDataFrameAbstract subclass.
        Input trajectory data.

    n_rows : int, optional, default None.
        Represents number of data rows that are will plot.

    lat_origin : float, optional, default None.
        Represents the latitude which will be the center of the map. If not entered, the first data from the dataset is
        used.

    lon_origin : float, optional, default None.
        Represents the longitude which will be the center of the map. If not entered, the first data from the dataset is
        used.

    zoom_start : int, optional, default 12.
        Represents the trajectory id.

    radius : float, optional, default 8.
        Radius of each “point” of the heatmap.

    max_zoom : int, optional, default 13.
        Zoom level where the points reach maximum intensity (as intensity scales with zoom), equals maxZoom of the
        map by default.

    base_map : folium.folium.Map, optional, default None.
        Represents the folium map. If not informed, a new map is generated using the function generate_base_map(), with
        the lat_origin, lon_origin and zoom_start.

    save_as_html : bool, optional, default False.
        Represents if want save this visualization in a new file .html.

    filename : String, optional, default 'heatmap.html'.
        Represents the file name of new file .html.

    Returns
    -------
    base_map : folium.folium.Map.
        Represents a folium map with visualization.

    """
    if base_map is None:
        if lat_origin is None and lon_origin is None:
            lat_origin = move_data.loc[0][LATITUDE]
            lon_origin = move_data.loc[0][LONGITUDE]
        base_map = generate_base_map(default_location=[lat_origin, lon_origin], default_zoom_start=zoom_start)

    if n_rows is None:
        n_rows = move_data.shape[0]

    move_data[COUNT] = 1
    HeatMap(data=move_data.loc[:n_rows, [LATITUDE, LONGITUDE, COUNT]]
            .groupby([LATITUDE, LONGITUDE]).sum().reset_index().values.tolist(), radius=radius, max_zoom=max_zoom)\
        .add_to(base_map)
    # base_map.add_child(folium.ClickForMarker(popup='Potential Location')) # habilita marcações no mapa
    move_data.drop(columns=[COUNT], inplace=True)
    if save_as_html:
        base_map.save(outfile=filename)
    else:
        return base_map


def heatmap_with_time(
    move_data,
    n_rows=None,
    lat_origin=None,
    lon_origin=None,
    zoom_start=12,
    radius=5,
    min_opacity=0.5,
    max_opacity=0.8,
    base_map=None,
    save_as_html=False,
    filename='heatmap_with_time.html'
):
    """
    Generate visualization of Heat Map with time using folium plugin.

    Parameters
    ----------
    move_data : pymove.core.MoveDataFrameAbstract subclass.
        Input trajectory data.

    n_rows : int, optional, default None.
        Represents number of data rows that are will plot.

    lat_origin : float, optional, default None.
        Represents the latitude which will be the center of the map. If not entered, the first data from the dataset is
        used.

    lon_origin : float, optional, default None.
        Represents the longitude which will be the center of the map. If not entered, the first data from the dataset is
        used.

    zoom_start : int, optional, default 12.
        Represents the trajectory id.

    radius : float, optional, default 8.
        Radius of each “point” of the heatmap

    min_opacity : float, optional, default 0.5.
        The minimum opacity for the heatmap.

    max_opacity : float, optional, default 0.8.
        The maximum opacity for the heatmap.

    base_map : folium.folium.Map, optional, default None.
        Represents the folium map. If not informed, a new map is generated using the function generate_base_map(), with
        the lat_origin, lon_origin and zoom_start.

    save_as_html : bool, optional, default False.
        Represents if want save this visualization in a new file .html.

    filename : String, optional, default 'heatmap_with_time.html'.
        Represents the file name of new file .html.

    Returns
    -------
    base_map : folium.folium.Map.
        Represents a folium map with visualization.

    """
    if base_map is None:
        if lat_origin is None and lon_origin is None:
            lat_origin = move_data.loc[0][LATITUDE]
            lon_origin = move_data.loc[0][LONGITUDE]
        base_map = generate_base_map(default_location=[lat_origin, lon_origin], default_zoom_start=zoom_start)

    if n_rows is None:
        n_rows = move_data.shape[0]

    move_data[HOUR] = move_data.datetime.dt.hour
    move_data[COUNT] = 1
    move_hour_list = []
    for hour in move_data.hour.sort_values().unique():
        move_hour_list.append(move_data.loc[move_data.hour == hour, [LATITUDE, LONGITUDE, COUNT]]
                              .groupby([LATITUDE, LONGITUDE]).sum().reset_index().values.tolist())

    HeatMapWithTime(move_hour_list[:n_rows], radius=radius,
                    gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red'},
                    min_opacity=min_opacity, max_opacity=max_opacity, use_local_extrema=True).add_to(base_map)

    move_data.drop(columns=[COUNT, HOUR], inplace=True)
    if save_as_html:
        base_map.save(outfile=filename)
    else:
        return base_map


def cluster(
    move_data,
    n_rows=None,
    lat_origin=None,
    lon_origin=None,
    zoom_start=12,
    base_map=None,
    save_as_html=False,
    filename='cluster.html'
):
    """
    Generate visualization of Marker Cluster using folium plugin.

    Parameters
    ----------
    move_data : pymove.core.MoveDataFrameAbstract subclass.
        Input trajectory data.

    n_rows : int, optional, default None.
        Represents number of data rows that are will plot.

    lat_origin : float, optional, default None.
        Represents the latitude which will be the center of the map. If not entered, the first data from the dataset is
        used.

    lon_origin : float, optional, default None.
        Represents the longitude which will be the center of the map. If not entered, the first data from the dataset is
        used.

    zoom_start : int, optional, default 12.
        Represents the trajectory id.

    base_map : folium.folium.Map, optional, default None.
        Represents the folium map. If not informed, a new map is generated using the function generate_base_map(), with
        the lat_origin, lon_origin and zoom_start.

    save_as_html : bool, optional, default False.
        Represents if want save this visualization in a new file .html.

    filename : String, optional, default 'cluster.html'.
        Represents the file name of new file .html.

    Returns
    -------
    base_map : folium.folium.Map.
        Represents a folium map with visualization.

    """
    if base_map is None:
        if lat_origin is None and lon_origin is None:
            lat_origin = move_data.loc[0][LATITUDE]
            lon_origin = move_data.loc[0][LONGITUDE]
        base_map = generate_base_map(default_location=[lat_origin, lon_origin], default_zoom_start=zoom_start)

    if n_rows is None:
        n_rows = move_data.shape[0]

    mc = MarkerCluster()
    for row in move_data[:n_rows].iterrows():
        pop = "<b>Latitude:</b> " + str(row[1].lat) \
              + "\n<b>Longitude:</b> " + str(row[1].lon) \
              + "\n<b>Datetime:</b> " + str(row[1].datetime)
        mc.add_child(folium.Marker(location=[row[1].lat, row[1].lon], popup=pop))
    base_map.add_child(mc)

    if save_as_html:
        base_map.save(outfile=filename)
    else:
        return base_map


def faster_cluster(
    move_data,
    n_rows=None,
    lat_origin=None,
    lon_origin=None,
    zoom_start=12,
    base_map=None,
    save_as_html=False,
    filename='faster_cluster.html'
):
    """
    Generate visualization of Faster Cluster using folium plugin.

    Parameters
    ----------
    move_data : pymove.core.MoveDataFrameAbstract subclass.
        Input trajectory data.

    n_rows : int, optional, default None.
        Represents number of data rows that are will plot.

    lat_origin : float, optional, default None.
        Represents the latitude which will be the center of the map. If not entered, the first data from the dataset is
        used.

    lon_origin : float, optional, default None.
        Represents the longitude which will be the center of the map. If not entered, the first data from the dataset is
        used.

    zoom_start : int, optional, default 12.
        Represents the trajectory id.

    base_map : folium.folium.Map, optional, default None.
        Represents the folium map. If not informed, a new map is generated using the function generate_base_map(), with
        the lat_origin, lon_origin and zoom_start.

    save_as_html : bool, optional, default False.
        Represents if want save this visualization in a new file .html.

    filename : String, optional, default 'faster_cluster.html'.
        Represents the file name of new file .html.

    Returns
    -------
    base_map : folium.folium.Map.
        Represents a folium map with visualization.

    """
    if base_map is None:
        if lat_origin is None and lon_origin is None:
            lat_origin = move_data.loc[0][LATITUDE]
            lon_origin = move_data.loc[0][LONGITUDE]
        base_map = generate_base_map(default_location=[lat_origin, lon_origin], default_zoom_start=zoom_start)

    if n_rows is None:
        n_rows = move_data.shape[0]

    callback = """\
    function (row) {
        var marker;
        marker = L.circle(new L.LatLng(row[0], row[1]), {color:'red'});
        return marker;
    };
    """
    FastMarkerCluster(move_data.loc[:n_rows, [LATITUDE, LONGITUDE]].values.tolist(), callback=callback).add_to(base_map)

    if save_as_html:
        base_map.save(outfile=filename)
    else:
        return base_map


def plot_trejectory_with_folium(
    move_data,
    n_rows=None,
    lat_origin=None,
    lon_origin=None,
    zoom_start=12,
    base_map=None,
    save_as_html=False,
    filename='plot_trejectory_with_folium.html'
):
    """
    Generate visualization of trajectory with folium.

    Parameters
    ----------
    move_data : pymove.core.MoveDataFrameAbstract subclass.
        Input trajectory data.

    n_rows : int, optional, default None.
        Represents number of data rows that are will plot.

    lat_origin : float, optional, default None.
        Represents the latitude which will be the center of the map. If not entered, the first data from the dataset is
        used.

    lon_origin : float, optional, default None.
        Represents the longitude which will be the center of the map. If not entered, the first data from the dataset is
        used.

    zoom_start : int, optional, default 12.
        Represents the trajectory id.

    base_map : folium.folium.Map, optional, default None.
        Represents the folium map. If not informed, a new map is generated using the function generate_base_map(), with
        the lat_origin, lon_origin and zoom_start.

    save_as_html : bool, optional, default False.
        Represents if want save this visualization in a new file .html.

    filename : String, optional, default 'plot_trejectory_with_folium.html'.
        Represents the file name of new file .html.

    Returns
    -------
    base_map : folium.folium.Map.
        Represents a folium map with visualization.

    """
    if base_map is None:
        if lat_origin is None and lon_origin is None:
            lat_origin = move_data.loc[0][LATITUDE]
            lon_origin = move_data.loc[0][LONGITUDE]
        base_map = generate_base_map(default_location=[lat_origin, lon_origin], default_zoom_start=zoom_start)

    if n_rows is None:
        n_rows = move_data.shape[0]

    for each in move_data[:n_rows].iterrows():
        pop = "<b>Latitude:</b> " + str(each[1].lat) \
              + "\n<b>Longitude:</b> " + str(each[1].lon) \
              + "\n<b>Datetime:</b> " + str(each[1].datetime)
        folium.Marker(location=[each[1]['lat'], each[1]['lon']], clustered_marker=True, popup=pop).add_to(base_map)

    if save_as_html:
        base_map.save(outfile=filename)
    else:
        return base_map