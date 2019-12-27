import folium
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from folium.plugins import HeatMap, MarkerCluster, FastMarkerCluster
from pymove.utils.datetime import str_to_datetime
from pymove.utils.constants import (
    LATITUDE,
    LONGITUDE,
    TRAJ_ID,
    PERIOD,
    DATE,
    HOUR,
    DAY,
    COUNT,
    COLORS,
    SITUATION,
    STOP,
    DATETIME,
    TILES
)

from pymove.preprocessing.stay_point_detection import (
    create_update_move_and_stop_by_radius
)


def generate_color():
    return COLORS[np.random.randint(0, len(COLORS))]


def rgb(rgb_colors):
    """
    Return a tuple of integers, as used in AWT/Java plots.

    Parameters
    ----------
    rgb_colors : list
        Represents a list with three positions that correspond to the percentage red, green and
        blue colors.

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
    blue = rgb_colors[0]
    red = rgb_colors[1]
    green = rgb_colors[2]
    return int(red*255), int(green*255), int(blue*255)


def hex_rgb(rgb_colors):
    """
    Return a hex string, as used in Tk plots.

    Parameters
    ----------
    rgb_colors : list
        Represents a list with three positions that correspond to the percentage red, green and
        blue colors.

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
    map = folium.Map(tiles=tiles)
    map.fit_bounds([[move_data[LATITUDE].min(), move_data[LONGITUDE].min()],
                  [move_data[LATITUDE].max(), move_data[LONGITUDE].max()]])
    
    ids = move_data[label_id].unique()
    cmap_ = plt.cm.get_cmap(cmap)
    num = cmap_.N
    
    for id_ in ids:
        id_index = np.where(ids == id_)[0][0]
        move_data = move_data[move_data[label_id] == id_]
        points_ = [(point[0], point[1]) for point in move_data[[LATITUDE, LONGITUDE]].values]
        color_ = cmap_hex_color(cmap_, (id_index % num))
        folium.PolyLine(points_, weight=3, color=color_).add_to(map)
    map.save(filename)
  

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
        str_ += ','.join('{} {}'.format(x[0], x[1]) for x in move_data[LONGITUDE, LATITUDE].values)
        str_ += ')\n'
    open(filename, "w").write(str_)    


def show_object_id_by_date(
    move_data,
    create_features=True,
    kind=['bar', 'bar', 'line', 'line'],
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
    create_features : bool, optional, default True.
        Represents whether or not to delete features created for viewing.
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
    move_data.generate_hour_features()
    move_data.generate_time_of_day_features()
    move_data.generate_day_of_the_week_features()

    move_data.groupby([PERIOD])[TRAJ_ID].nunique()\
        .plot(subplots=True, kind=kind[0], rot=0, ax=ax[0][0], fontsize=12)
    move_data.groupby([DAY])[TRAJ_ID].nunique()\
        .plot(subplots=True, kind=kind[1], ax=ax[0][1], rot=0, fontsize=12)
    move_data.groupby([DATE])[TRAJ_ID].nunique()\
        .plot(subplots=True, kind=kind[2], grid=True, ax=ax[1][0], rot=90, fontsize=12)
    move_data.groupby([HOUR])[TRAJ_ID].nunique()\
        .plot(subplots=True, kind=kind[3], grid=True, ax=ax[1][1], fontsize=12)

    if create_features == False:
        move_data.drop(columns=[DATE, HOUR, PERIOD, DAY], inplace=True)

    if save_fig:
        plt.savefig(fname=name, fig=fig)


def show_lat_lon_gps(
    move_data,
    kind='scatter',
    figsize=(21, 9),
    save_fig=False,
    plot_start_and_end=True,
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
            move_data.drop_duplicates([LATITUDE, LONGITUDE])\
                .plot(kind=kind, x=LONGITUDE, y=LATITUDE, figsize=figsize)

            if plot_start_and_end:
                plt.plot(move_data.iloc[0][LONGITUDE], move_data.iloc[0][LATITUDE], 'yo',
                         markersize=10)  # start point
                plt.plot(move_data.iloc[-1][LONGITUDE], move_data.iloc[-1][LATITUDE], 'yX',
                         markersize=10)  # end point
            if save_fig:
                plt.savefig(name)   
    except Exception as exception:
        raise exception


def create_base_map(default_location, tile, default_zoom_start=12):
    """
    Generate a folium map.

    Parameters
    ----------
    default_location : tuple.
        Represents coordinates lat, lon which will be the center of the map.
    default_zoom_start : int, optional, default 12.
        Represents the zoom which will be the center of the map.
    tile : String.
        Represents the map's tiles.
    Returns
    -------
    base_map : folium.folium.Map.
        Represents a folium map.

    """
    base_map = folium.Map(location=default_location, control_scale=True,
                          zoom_start=default_zoom_start, tiles=tile)
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
    tile=TILES[0],
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
        Represents the latitude which will be the center of the map.
        If not entered, the first data from the dataset is used.
    lon_origin : float, optional, default None.
        Represents the longitude which will be the center of the map.
        If not entered, the first data from the dataset is used.
    zoom_start : int, optional, default 12.
        Represents the trajectory id.
    radius : float, optional, default 8.
        Radius of each “point” of the heatmap.
    max_zoom : int, optional, default 13.
        Zoom level where the points reach maximum intensity (as intensity
        scales with zoom), equals maxZoom of the map by default.
    base_map : folium.folium.Map, optional, default None.
        Represents the folium map. If not informed, a new map is generated
        using the function create_base_map(), with the lat_origin, lon_origin
        and zoom_start.
    save_as_html : bool, optional, default False.
        Represents if want save this visualization in a new file .html.
    filename : String, optional, default 'heatmap.html'.
        Represents the file name of new file .html.
    tile : String, optional, default 'OpenStreetMap'.
        Represents the map's tiles.

    Returns
    -------
    base_map : folium.folium.Map.
        Represents a folium map with visualization.

    """
    if base_map is None:
        if lat_origin is None and lon_origin is None:
            lat_origin = move_data.loc[0][LATITUDE]
            lon_origin = move_data.loc[0][LONGITUDE]
        base_map = create_base_map(default_location=[lat_origin, lon_origin], tile=tile,
                                   default_zoom_start=zoom_start)

    if n_rows is None:
        n_rows = move_data.shape[0]

    move_data[COUNT] = 1
    HeatMap(data=move_data.loc[:n_rows, [LATITUDE, LONGITUDE, COUNT]]
            .groupby([LATITUDE, LONGITUDE]).sum().reset_index().values.tolist(),
            radius=radius, max_zoom=max_zoom)\
        .add_to(base_map)

    move_data.drop(columns=[COUNT], inplace=True)
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
    tile=TILES[0],
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
        Represents the latitude which will be the center of the map.
        If not entered, the first data from the dataset is used.
    lon_origin : float, optional, default None.
        Represents the longitude which will be the center of the map.
        If not entered, the first data from the dataset is used.
    zoom_start : int, optional, default 12.
        Represents the trajectory id.
    base_map : folium.folium.Map, optional, default None.
        Represents the folium map. If not informed, a new map is generated
        using the function create_base_map(), with the lat_origin, lon_origin
        and zoom_start.
    tile : String, optional, default 'OpenStreetMap'.
        Represents the map's tiles.
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
        base_map = create_base_map(default_location=[lat_origin, lon_origin], tile=tile,
                                   default_zoom_start=zoom_start)

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
    tile=TILES[0],
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
        Represents the latitude which will be the center of the map.
        If not entered, the first data from the dataset is used.
    lon_origin : float, optional, default None.
        Represents the longitude which will be the center of the map.
        If not entered, the first data from the dataset is used.
    zoom_start : int, optional, default 12.
        Represents the trajectory id.
    base_map : folium.folium.Map, optional, default None.
        Represents the folium map. If not informed, a new map is generated
        using the function create_base_map(), with the lat_origin, lon_origin
        and zoom_start.
    tile : String, optional, default 'OpenStreetMap'.
        Represents the map's tiles.
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
        base_map = create_base_map(default_location=[lat_origin, lon_origin], tile=tile,
                                   default_zoom_start=zoom_start)

    if n_rows is None:
        n_rows = move_data.shape[0]

    callback = """\
    function (row) {
        var marker;
        marker = L.circle(new L.LatLng(row[0], row[1]), {color:'red'});
        return marker;
    };
    """
    FastMarkerCluster(move_data.loc[:n_rows, [LATITUDE, LONGITUDE]].values.tolist(),
                      callback=callback).add_to(base_map)

    if save_as_html:
        base_map.save(outfile=filename)
    else:
        return base_map


def plot_markers(
    move_data,
    n_rows=None,
    lat_origin=None,
    lon_origin=None,
    zoom_start=12,
    base_map=None,
    tile=TILES[0],
    save_as_html=False,
    filename='plot_markers.html'
):
    """
    Plot markers of Folium on map.

    Parameters
    ----------
    move_data : pymove.core.MoveDataFrameAbstract subclass.
        Input trajectory data.
    n_rows : int, optional, default None.
        Represents number of data rows that are will plot.
    lat_origin : float, optional, default None.
        Represents the latitude which will be the center of the map.
        If not entered, the first data from the dataset is used.
    lon_origin : float, optional, default None.
        Represents the longitude which will be the center of the map.
        If not entered, the first data from the dataset is used.
    zoom_start : int, optional, default 12.
        Represents the trajectory id.
    base_map : folium.folium.Map, optional, default None.
        Represents the folium map. If not informed, a new map is generated
        using the function create_base_map(), with the lat_origin,
        lon_origin and zoom_start.
    tile : String, optional, default 'OpenStreetMap'.
        Represents the map's tiles.
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
        base_map = create_base_map(default_location=[lat_origin, lon_origin], tile=tile,
                                   default_zoom_start=zoom_start)

    if n_rows is None:
        n_rows = move_data.shape[0]

    folium.Marker(location=[move_data.iloc[0][LATITUDE], move_data.iloc[0][LONGITUDE]],
                  color="green", clustered_marker=True,
                  popup="Início", icon=folium.Icon(color='green', icon='info-sign')).add_to(base_map)

    folium.Marker(location=[move_data.iloc[-1][LATITUDE], move_data.iloc[-1][LONGITUDE]],
                  color="red", clustered_marker=True,
                  popup="Fim", icon=folium.Icon(color='red', icon='info-sign')).add_to(base_map)

    for each in move_data[:n_rows-1].iterrows():
        pop = "<b>Latitude:</b> " + str(each[1].lat) \
              + "\n<b>Longitude:</b> " + str(each[1].lon) \
              + "\n<b>Datetime:</b> " + str(each[1].datetime)
        folium.Marker(location=[each[1]['lat'], each[1]['lon']], clustered_marker=True, popup=pop)\
            .add_to(base_map)

    if save_as_html:
        base_map.save(outfile=filename)
    else:
        return base_map


def plot_trajectories_with_folium(
    move_data,
    n_rows=None,
    lat_origin=None,
    lon_origin=None,
    zoom_start=12,
    base_map=None,
    tile=TILES[0],
    save_as_html=False,
    color='black',
    filename='plot_trajectories_with_folium.html'
):
    """
    Generate visualization of all trajectories with folium.

    Parameters
    ----------
    move_data : pymove.core.MoveDataFrameAbstract subclass.
        Input trajectory data.
    n_rows : int, optional, default None.
        Represents number of data rows that are will plot.
    lat_origin : float, optional, default None.
        Represents the latitude which will be the center of the map.
        If not entered, the first data from the dataset is used.
    lon_origin : float, optional, default None.
        Represents the longitude which will be the center of the map.
        If not entered, the first data from the dataset is used.
    zoom_start : int, optional, default 12.
        Represents the trajectory id.
    base_map : folium.folium.Map, optional, default None.
        Represents the folium map. If not informed, a new map is generated
        using the function create_base_map(), with the lat_origin, lon_origin
         and zoom_start.
    tile : String, optional, default 'OpenStreetMap'.
        Represents the map's tiles.
    save_as_html : bool, optional, default False.
        Represents if want save this visualization in a new file .html.
    color : String, optional, default 'black'.
        Represents line's color of visualization.
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
        base_map = create_base_map(default_location=[lat_origin, lon_origin], tile=tile,
                                   default_zoom_start=zoom_start)

    if n_rows is None:
        n_rows = move_data.shape[0]

    folium.Marker(location=[move_data.iloc[0][LATITUDE], move_data.iloc[0][LONGITUDE]],
                  color="green", clustered_marker=True, popup="Início",
                  icon=folium.Icon(color='green', icon='info-sign'))\
        .add_to(base_map)

    folium.Marker(location=[move_data.iloc[-1][LATITUDE], move_data.iloc[-1][LONGITUDE]],
                  color="red", clustered_marker=True, popup="Fim",
                  icon=folium.Icon(color='red', icon='info-sign')).add_to(base_map)

    folium.PolyLine(move_data.loc[:n_rows, [LATITUDE, LONGITUDE]], color=color,
                    weight=2.5, opacity=1).add_to(base_map)
    # for each in move_data[:n_rows].iterrows():
    #     pop = "<b>Latitude:</b> " + str(each[1].lat) \
    #           + "\n<b>Longitude:</b> " + str(each[1].lon) \
    #           + "\n<b>Datetime:</b> " + str(each[1].datetime)
    #     folium.Marker(location=[each[1]['lat'], each[1]['lon']],
    #     clustered_marker=True, popup=pop).add_to(base_map)

    if save_as_html:
        base_map.save(outfile=filename)
    else:
        return base_map


def plot_trajectory_by_id_with_folium(
    move_data,
    id_,
    n_rows=None,
    lat_origin=None,
    lon_origin=None,
    zoom_start=12,
    base_map=None,
    tile=TILES[0],
    save_as_html=False,
    color='black',
    filename='plot_trajectory_by_id_with_folium.html'
):
    """
    Generate visualization of trajectory with the id provided by user.

    Parameters
    ----------
    move_data : pymove.core.MoveDataFrameAbstract subclass.
        Input trajectory data.
    id_: int
        Represents trajectory ID.
    n_rows : int, optional, default None.
        Represents number of data rows that are will plot.

    lat_origin : float, optional, default None.
        Represents the latitude which will be the center of the map.
        If not entered, the first data from the dataset is used.
    lon_origin : float, optional, default None.
        Represents the longitude which will be the center of the map.
        If not entered, the first data from the dataset is used.
    zoom_start : int, optional, default 12.
        Represents the trajectory id.
    base_map : folium.folium.Map, optional, default None.
        Represents the folium map. If not informed, a new map is
        generated using the function create_base_map(), with the
        lat_origin, lon_origin and zoom_start.
    tile : String, optional, default 'OpenStreetMap'.
        Represents the map's tiles.
    save_as_html : bool, optional, default False.
        Represents if want save this visualization in a new file .html.
    color : String, optional, default 'black'.
        Represents line's color of visualization.
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
        base_map = create_base_map(default_location=[lat_origin, lon_origin],
                                   tile=tile, default_zoom_start=zoom_start)

    if n_rows is None:
        n_rows = move_data.shape[0]

    mv_df = move_data[move_data[TRAJ_ID] == id_].reset_index().loc[:n_rows, [LATITUDE, LONGITUDE]]

    folium.Marker(location=[mv_df.iloc[0][LATITUDE], mv_df.iloc[0][LONGITUDE]],
                  color="green", clustered_marker=True, popup="Início",
                  icon=folium.Icon(color='green', icon='info-sign')).add_to(base_map)

    folium.Marker(location=[mv_df.iloc[-1][LATITUDE], mv_df.iloc[-1][LONGITUDE]],
                  color="red", clustered_marker=True, popup="Fim",
                  icon=folium.Icon(color='red', icon='info-sign')).add_to(base_map)

    folium.PolyLine(mv_df, color=color, weight=2.5, opacity=1).add_to(base_map)

    if save_as_html:
        base_map.save(outfile=filename)
    else:
        return base_map


def plot_trajectory_by_period(
    move_data,
    period,
    n_rows=None,
    lat_origin=None,
    lon_origin=None,
    zoom_start=12,
    base_map=None,
    tile=TILES[0],
    save_as_html=False,
    color='black',
    filename='plot_trajectory_by_period_with_folium.html'
):
    """
    Generate trajectory view by period of day provided by user.

    Parameters
    ----------
    move_data : pymove.core.MoveDataFrameAbstract subclass.
        Input trajectory data.
    period: String
        Represents period of day.
    n_rows : int, optional, default None.
        Represents number of data rows that are will plot.
    lat_origin : float, optional, default None.
        Represents the latitude which will be the center of the map.
        If not entered, the first data from the dataset is used.
    lon_origin : float, optional, default None.
        Represents the longitude which will be the center of the map.
        If not entered, the first data from the dataset is used.
    zoom_start : int, optional, default 12.
        Represents the trajectory id.
    base_map : folium.folium.Map, optional, default None.
        Represents the folium map. If not informed, a new map is
        generated using the function create_base_map(), with
        the lat_origin, lon_origin and zoom_start.
    tile : String, optional, default 'OpenStreetMap'.
        Represents the map's tiles.
    color : String, optional, default 'black'.
        Represents line's color of visualization.
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
        base_map = create_base_map(default_location=[lat_origin, lon_origin],
                                   tile=tile, default_zoom_start=zoom_start)

    if n_rows is None:
        n_rows = move_data.shape[0]

    if PERIOD not in move_data:
        move_data.generate_time_of_day_features()

    mv_df = move_data[move_data[PERIOD] == period].reset_index() \
            .loc[:n_rows, [LATITUDE, LONGITUDE]]

    folium.Marker(location=[mv_df.iloc[0][LATITUDE], mv_df.iloc[0][LONGITUDE]],
                  color="green", clustered_marker=True, popup="Início",
                  icon=folium.Icon(color='green', icon='info-sign')).add_to(base_map)

    folium.Marker(location=[mv_df.iloc[-1][LATITUDE], mv_df.iloc[-1][LONGITUDE]],
                  color="red", clustered_marker=True, popup="Fim",
                  icon=folium.Icon(color='red', icon='info-sign')).add_to(base_map)

    folium.PolyLine(mv_df, color=color, weight=2.5, opacity=1).add_to(base_map)

    if save_as_html:
        base_map.save(outfile=filename)
    else:
        return base_map


def plot_trajectory_by_day_week(
    move_data,
    day_week,
    n_rows=None,
    lat_origin=None,
    lon_origin=None,
    zoom_start=12,
    base_map=None,
    tile=TILES[0],
    save_as_html=False,
    color='black',
    filename='plot_trajectory_by_day_week.html'
):
    """
    Generate trajectory view by day week provided by user.

    Parameters
    ----------
    move_data : pymove.core.MoveDataFrameAbstract subclass.
        Input trajectory data.
    day_week: String
        Represents day week.
    n_rows : int, optional, default None.
        Represents number of data rows that are will plot.
    lat_origin : float, optional, default None.
        Represents the latitude which will be the center of the map.
        If not entered, the first data from the dataset is used.
    lon_origin : float, optional, default None.
        Represents the longitude which will be the center of the map.
        If not entered, the first data from the dataset is used.
    zoom_start : int, optional, default 12.
        Represents the trajectory id.
    base_map : folium.folium.Map, optional, default None.
        Represents the folium map. If not informed, a new map is
        generated using the function create_base_map(), with
        the lat_origin, lon_origin and zoom_start.
    tile : String, optional, default 'OpenStreetMap'.
        Represents the map's tiles.
    save_as_html : bool, optional, default False.
        Represents if want save this visualization in a new file .html.
    color : String, optional, default 'black'.
        Represents line's color of visualization.
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
        base_map = create_base_map(default_location=[lat_origin, lon_origin],
                                   tile=tile, default_zoom_start=zoom_start)

    if n_rows is None:
        n_rows = move_data.shape[0]

    if DAY not in move_data:
        move_data.generate_day_of_the_week_features()

    mv_df = move_data[move_data[DAY] == day_week].reset_index().loc[:n_rows, [LATITUDE, LONGITUDE]]

    folium.Marker(location=[mv_df.iloc[0][LATITUDE], mv_df.iloc[0][LONGITUDE]],
                  color="green", clustered_marker=True, popup="Início",
                  icon=folium.Icon(color='green', icon='info-sign')).add_to(base_map)

    folium.Marker(location=[mv_df.iloc[-1][LATITUDE], mv_df.iloc[-1][LONGITUDE]],
                  color="red", clustered_marker=True, popup="Fim",
                  icon=folium.Icon(color='red', icon='info-sign')).add_to(base_map)

    folium.PolyLine(mv_df, color=color, weight=2.5, opacity=1).add_to(base_map)

    if save_as_html:
        base_map.save(outfile=filename)
    else:
        return base_map


def plot_trajectory_by_date(
    move_data,
    start_date,
    end_date,
    n_rows=None,
    lat_origin=None,
    lon_origin=None,
    zoom_start=12,
    base_map=None,
    tile=TILES[0],
    save_as_html=False,
    color='black',
    filename='plot_trajectory_by_date.html'
):
    """
    Generate trajectory view by period of time provided by user.

    Parameters
    ----------
    move_data : pymove.core.MoveDataFrameAbstract subclass.
        Input trajectory data.
    start_date : String
        Represents start date of time period.
    end_date : String
        Represents end date of time period.
    n_rows : int, optional, default None.
        Represents number of data rows that are will plot.
    lat_origin : float, optional, default None.
        Represents the latitude which will be the center of the map.
        If not entered, the first data from the dataset is used.
    lon_origin : float, optional, default None.
        Represents the longitude which will be the center of the map.
        If not entered, the first data from the dataset is used.
    zoom_start : int, optional, default 12.
        Represents the trajectory id.
    base_map : folium.folium.Map, optional, default None.
        Represents the folium map. If not informed, a new map is generated
        using the function create_base_map(), with the lat_origin,
        lon_origin and zoom_start.
    tile : String, optional, default 'OpenStreetMap'.
        Represents the map's tiles.
    save_as_html : bool, optional, default False.
        Represents if want save this visualization in a new file .html.
    color : String, optional, default 'black'.
        Represents line's color of visualization.
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
        base_map = create_base_map(default_location=[lat_origin, lon_origin],
                                   tile=tile, default_zoom_start=zoom_start)

    if n_rows is None:
        n_rows = move_data.shape[0]

    if type(start_date) == str:
        start_date = str_to_datetime(start_date).date()

    if type(end_date) == str:
        end_date = str_to_datetime(end_date).date()

    if DATE not in move_data:
        move_data.generate_date_features()

    mv_df = move_data[(move_data[DATE] <= end_date) & (move_data[DATE] >= start_date)] \
            .reset_index().loc[:n_rows, [LATITUDE, LONGITUDE]]

    folium.Marker(location=[mv_df.iloc[0][LATITUDE], mv_df.iloc[0][LONGITUDE]],
                  color="green", clustered_marker=True, popup="Início",
                  icon=folium.Icon(color='green', icon='info-sign')).add_to(base_map)

    folium.Marker(location=[mv_df.iloc[-1][LATITUDE], mv_df.iloc[-1][LONGITUDE]],
                  color="red", clustered_marker=True, popup="Fim",
                  icon=folium.Icon(color='red', icon='info-sign')).add_to(base_map)

    folium.PolyLine(mv_df, color=color, weight=2.5, opacity=1).add_to(base_map)

    if save_as_html:
        base_map.save(outfile=filename)
    else:
        return base_map


def plot_trajectory_by_hour(
    move_data,
    start_hour,
    end_hour,
    n_rows=None,
    lat_origin=None,
    lon_origin=None,
    zoom_start=12,
    base_map=None,
    tile=TILES[0],
    save_as_html=False,
    color='black',
    filename='plot_trajectory_by_hour.html'
):
    """
    Generate trajectory view by period of time provided by user.

    Parameters
    ----------
    move_data : pymove.core.MoveDataFrameAbstract subclass.
        Input trajectory data.
    start_hour : int
        Represents start hour of time period.
    end_hour : int
        Represents end hour of time period.
    n_rows : int, optional, default None.
        Represents number of data rows that are will plot.
    lat_origin : float, optional, default None.
        Represents the latitude which will be the center of the map.
        If not entered, the first data from the dataset is used.
    lon_origin : float, optional, default None.
        Represents the longitude which will be the center of the map.
        If not entered, the first data from the dataset is used.
    zoom_start : int, optional, default 12.
        Represents the trajectory id.
    base_map : folium.folium.Map, optional, default None.
        Represents the folium map. If not informed, a new map is
        generated using the function create_base_map(), with
        the lat_origin, lon_origin and zoom_start.
    tile : String, optional, default 'OpenStreetMap'.
        Represents the map's tiles.
    save_as_html : bool, optional, default False.
        Represents if want save this visualization in a new file .html.
    color : String, optional, default 'black'.
        Represents line's color of visualization.
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
        base_map = create_base_map(default_location=[lat_origin, lon_origin],
                                   tile=tile, default_zoom_start=zoom_start)

    if n_rows is None:
        n_rows = move_data.shape[0]

    if HOUR not in move_data:
        move_data.generate_hour_features()

    mv_df = move_data[(move_data[HOUR] <= end_hour) & (move_data[HOUR] >= start_hour)] \
            .reset_index().loc[:n_rows, [LATITUDE, LONGITUDE]]

    folium.Marker(location=[mv_df.iloc[0][LATITUDE], mv_df.iloc[0][LONGITUDE]],
                  color="green", clustered_marker=True, popup="Início",
                  icon=folium.Icon(color='green', icon='info-sign')).add_to(base_map)

    folium.Marker(location=[mv_df.iloc[-1][LATITUDE], mv_df.iloc[-1][LONGITUDE]],
                  color="red", clustered_marker=True, popup="Fim ",
                  icon=folium.Icon(color='red', icon='info-sign')).add_to(base_map)

    folium.PolyLine(mv_df, color=color, weight=2.5, opacity=1).add_to(base_map)

    if save_as_html:
        base_map.save(outfile=filename)
    else:
        return base_map


def plot_stops(
    move_data,
    radius=0,
    weight=3,
    n_rows=None,
    lat_origin=None,
    lon_origin=None,
    zoom_start=12,
    base_map=None,
    tile=TILES[0],
    save_as_html=False,
    color='black',
    filename='plot_stops.html'
):
    """
    Generate points on map that represents stops points with folium.

    Parameters
    ----------
    move_data : pymove.core.MoveDataFrameAbstract subclass.
        Input trajectory data.
    radius :  Double, optional(900 by default)
        The radius value is used to determine if a segment is a stop.
        If the value of the point in target_label is greater than
        radius, the segment is a stop, otherwise it's a move.
    n_rows : int, optional, default None.
        Represents number of data rows that are will plot.
    lat_origin : float, optional, default None.
        Represents the latitude which will be the center of the map.
        If not entered, the first data from the dataset is used.
    lon_origin : float, optional, default None.
        Represents the longitude which will be the center of the map.
        If not entered, the first data from the dataset is used.
    zoom_start : int, optional, default 12.
        Represents the trajectory id.
    base_map : folium.folium.Map, optional, default None.
        Represents the folium map. If not informed, a new map is generated
        using the function create_base_map(), with the lat_origin,
        lon_origin and zoom_start.
    tile : String, optional, default 'OpenStreetMap'.
        Represents the map's tiles.
    save_as_html : bool, optional, default False.
        Represents if want save this visualization in a new file .html.
    color : String, optional, default 'black'.
        Represents line's color of visualization.
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
        base_map = create_base_map(default_location=[lat_origin, lon_origin],
                                   tile=tile, default_zoom_start=zoom_start)

    if n_rows is None:
        n_rows = move_data.shape[0]

    create_update_move_and_stop_by_radius(move_data, radius=radius)
    stops = move_data[move_data[SITUATION] == STOP].loc[:n_rows, [LATITUDE, LONGITUDE, DATETIME]]

    for stop in stops.iterrows():
        base_map.add_child(
            folium.Circle([stop[1][LATITUDE], stop[1][LONGITUDE]],
                          color=color, weight=weight, radius=20, opacity=0.5,
                          popup=stop[1][DATETIME], fill_color=color, fill_opacity=0.5))

    if save_as_html:
        base_map.save(outfile=filename)
    else:
        return base_map
