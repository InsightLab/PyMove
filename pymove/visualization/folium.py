import matplotlib
import matplotlib.pyplot as plt
import folium
import numpy as np
import pandas as pd
from folium.plugins import FastMarkerCluster, HeatMap, MarkerCluster
from pymove.utils.mapfolium import add_map_legend
from pymove.utils import constants
from pymove.preprocessing import filters
from pymove.utils import distances
from pymove.utils.constants import (
    COLORS,
    COUNT,
    DATE,
    DATETIME,
    DAY,
    HOUR,
    LATITUDE,
    LONGITUDE,
    PERIOD,
    SITUATION,
    STOP,
    TILES,
    TRAJ_ID,
)

from pymove.utils.datetime import str_to_datetime

def generate_color():
    """
    Generates a random color.

    Returns
    -------
    Random HEX color
    """
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
    return int(red * 255), int(green * 255), int(blue * 255)


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
    move_data, filename, tiles="OpenStreetMap", label_id=TRAJ_ID, cmap="tab20"
):
    """
    Save a visualization of a map in a new file.

    Parameters
    ----------
    move_data : pymove.core.MoveDataFrameAbstract subclass.
        Input trajectory data.
    filename : String
        Represents the filename.
    tiles : String
        Represents the type_ of tile that will be used on the map.
    label_id : String
        Represents column name of trajectory id.
    cmap: String
        Represents the Colormap.

    Returns
    -------
    """
    map_ = folium.Map(tiles=tiles)
    map_.fit_bounds(
        [
            [move_data[LATITUDE].min(), move_data[LONGITUDE].min()],
            [move_data[LATITUDE].max(), move_data[LONGITUDE].max()],
        ]
    )

    ids = move_data[label_id].unique()
    cmap_ = plt.cm.get_cmap(cmap)
    num = cmap_.N

    for id_ in ids:
        id_index = np.where(ids == id_)[0][0]
        move_df = move_data[move_data[label_id] == id_]
        points_ = [
            (point[0], point[1])
            for point in move_df[[LATITUDE, LONGITUDE]].values
        ]
        color_ = cmap_hex_color(cmap_, (id_index % num))
        folium.PolyLine(points_, weight=3, color=color_).add_to(map_)
    map_.save(filename)

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
    str_ = "{};linestring\n".format(label_id)
    ids = move_data[label_id].unique()
    for id_ in ids:
        move_df = move_data[move_data[label_id] == id_]
        curr_str = "{};LINESTRING(".format(id_)
        curr_str += ",".join(
            "{} {}".format(x[0], x[1])
            for x in move_df[[LONGITUDE, LATITUDE]].values
        )
        curr_str += ")\n"
        str_ += curr_str
    with open(filename, "w") as f:
        f.write(str_)

def create_base_map(default_location, tile=TILES[0], default_zoom_start=12):
    """
    Generate a folium map.

    Parameters
    ----------
    default_location : tuple.
        Represents coordinates lat, lon which will be the center of the map.
    default_zoom_start : int, optional, default 12.
        Represents the zoom which will be the center of the map.
    tile : String, optional, default 'OpenStreetMap'.
        Represents the map's tiles.
    Returns
    -------
    base_map : folium.folium.Map.
        Represents a folium map.
    """
    base_map = folium.Map(
        location=default_location,
        control_scale=True,
        zoom_start=default_zoom_start,
        tiles=tile,
    )
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
    filename="heatmap.html",
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
        Initial zoom level for the map
    radius : float, optional, default 8.
        Radius of each “point” of the heatmap.
    max_zoom : int, optional, default 13.
        Zoom level where the points reach maximum intensity (as intensity
        scales with zoom), equals maxZoom of the map by default.
    base_map : folium.folium.Map, optional, default None.
        Represents the folium map. If not informed, a new map is generated
        using the function create_base_map(), with the lat_origin, lon_origin
        and zoom_start.
    tile : String, optional, default 'OpenStreetMap'.
        Represents the map's tiles.
    save_as_html : bool, optional, default False.
        Represents if want save this visualization in a new file .html.
    filename : String, optional, default 'heatmap.html'.
        Represents the file name of new file .html.

    Returns
    -------
    base_map : folium.folium.Map.
        Represents a folium map with visualization.
    """
    move_df = move_data.reset_index()
    if base_map is None:
        if lat_origin is None and lon_origin is None:
            lat_origin = move_df.loc[0][LATITUDE]
            lon_origin = move_df.loc[0][LONGITUDE]
        base_map = create_base_map(
            default_location=[lat_origin, lon_origin],
            tile=tile,
            default_zoom_start=zoom_start,
        )

    if n_rows is None:
        n_rows = move_df.shape[0]

    move_df[COUNT] = 1
    HeatMap(
        data=move_df.loc[:n_rows, [LATITUDE, LONGITUDE, COUNT]]
        .groupby([LATITUDE, LONGITUDE])
        .sum()
        .reset_index()
        .values.tolist(),
        radius=radius,
        max_zoom=max_zoom,
    ).add_to(base_map)

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
    filename="cluster.html",
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
        Initial zoom level for the map
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
    move_df = move_data.reset_index()

    if base_map is None:
        if lat_origin is None and lon_origin is None:
            lat_origin = move_df.loc[0][LATITUDE]
            lon_origin = move_df.loc[0][LONGITUDE]
        base_map = create_base_map(
            default_location=[lat_origin, lon_origin],
            tile=tile,
            default_zoom_start=zoom_start,
        )

    if n_rows is None:
        n_rows = move_df.shape[0]

    mc = MarkerCluster()
    for row in move_df[:n_rows].iterrows():
        pop = (
            "<b>Latitude:</b> "
            + str(row[1].lat)
            + "\n<b>Longitude:</b> "
            + str(row[1].lon)
            + "\n<b>Datetime:</b> "
            + str(row[1].datetime)
        )
        mc.add_child(
            folium.Marker(location=[row[1].lat, row[1].lon], popup=pop)
        )
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
    filename="faster_cluster.html",
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
        Initial zoom level for the map
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
    move_df = move_data.reset_index()

    if base_map is None:
        if lat_origin is None and lon_origin is None:
            lat_origin = move_df.loc[0][LATITUDE]
            lon_origin = move_df.loc[0][LONGITUDE]
        base_map = create_base_map(
            default_location=[lat_origin, lon_origin],
            tile=tile,
            default_zoom_start=zoom_start,
        )

    if n_rows is None:
        n_rows = move_df.shape[0]

    callback = """\
    function (row) {
        var marker;
        marker = L.circle(new L.LatLng(row[0], row[1]), {color:'red'});
        return marker;
    };
    """
    FastMarkerCluster(
        move_df.loc[:n_rows, [LATITUDE, LONGITUDE]].values.tolist(),
        callback=callback,
    ).add_to(base_map)

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
    filename="plot_markers.html",
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
        Initial zoom level for the map
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
    move_df = move_data.reset_index()

    if base_map is None:
        if lat_origin is None and lon_origin is None:
            lat_origin = move_df.loc[0][LATITUDE]
            lon_origin = move_df.loc[0][LONGITUDE]
        base_map = create_base_map(
            default_location=[lat_origin, lon_origin],
            tile=tile,
            default_zoom_start=zoom_start,
        )

    if n_rows is None:
        n_rows = move_df.shape[0]

    folium.Marker(
        location=[move_df.iloc[0][LATITUDE], move_df.iloc[0][LONGITUDE]],
        color="green",
        clustered_marker=True,
        popup="Início",
        icon=folium.Icon(color="green", icon="info-sign"),
    ).add_to(base_map)

    folium.Marker(
        location=[move_df.iloc[-1][LATITUDE], move_df.iloc[-1][LONGITUDE]],
        color="red",
        clustered_marker=True,
        popup="Fim",
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(base_map)

    for each in move_df[: n_rows - 1].iterrows():
        pop = (
            "<b>Latitude:</b> "
            + str(each[1].lat)
            + "\n<b>Longitude:</b> "
            + str(each[1].lon)
            + "\n<b>Datetime:</b> "
            + str(each[1].datetime)
        )
        folium.Marker(
            location=[each[1]["lat"], each[1]["lon"]],
            clustered_marker=True,
            popup=pop,
        ).add_to(base_map)

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
    legend=False,
    base_map=None,
    tile=TILES[0],
    save_as_html=False,
    color="black",
    filename="plot_trajectories_with_folium.html",
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
        Initial zoom level for the map
    legend: boolean
        Whether to add a legend to the map
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
    filename : String, optional, default 'plot_trajectory_with_folium.html'.
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
        base_map = create_base_map(
            default_location=[lat_origin, lon_origin],
            tile=tile,
            default_zoom_start=zoom_start,
        )

    if n_rows is None:
        n_rows = move_data.shape[0]

    mv_df = move_data.loc[
        :n_rows, [LATITUDE, LONGITUDE, TRAJ_ID]
    ].reset_index()

    ids = mv_df[TRAJ_ID].unique()
    if isinstance(color, str):
        colors = [generate_color() for _ in ids]
    else:
        colors = color[:]
    items = list(zip(ids, colors))

    for _id, color in items:
        mv = mv_df[mv_df[TRAJ_ID] == _id]
        folium.Marker(
            location=[mv.iloc[0][LATITUDE], mv.iloc[0][LONGITUDE]],
            color="green",
            clustered_marker=True,
            popup="Início",
            icon=folium.Icon(color="green", icon="info-sign"),
        ).add_to(base_map)

        folium.Marker(
            location=[mv.iloc[-1][LATITUDE], mv.iloc[-1][LONGITUDE]],
            color="red",
            clustered_marker=True,
            popup="Fim",
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(base_map)

        folium.PolyLine(
            mv[[LATITUDE, LONGITUDE]], color=color, weight=2.5, opacity=1
        ).add_to(base_map)

    if legend:
        add_map_legend(base_map, "Color by user ID", items)

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
    color="black",
    filename="plot_trajectory_by_id_with_folium.html",
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
        Initial zoom level for the map
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
    filename : String, optional, default 'plot_trajectory_by_id_with_folium.html'.
        Represents the file name of new file .html.

    Returns
    -------
    base_map : folium.folium.Map.
        Represents a folium map with visualization.

    Raises
    ------
        IndexError if there is no user with the id passed
    """
    if base_map is None:
        if lat_origin is None and lon_origin is None:
            lat_origin = move_data.loc[0][LATITUDE]
            lon_origin = move_data.loc[0][LONGITUDE]
        base_map = create_base_map(
            default_location=[lat_origin, lon_origin],
            tile=tile,
            default_zoom_start=zoom_start,
        )

    if n_rows is None:
        n_rows = move_data.shape[0]

    mv_df = move_data[move_data[TRAJ_ID] == id_]
    if not len(mv_df):
        raise IndexError(f"No user with id {id_} in dataframe")
    mv = mv_df.reset_index().loc[:n_rows, [LATITUDE, LONGITUDE]]
    folium.Marker(
        location=[mv.iloc[0][LATITUDE], mv.iloc[0][LONGITUDE]],
        color="green",
        clustered_marker=True,
        popup="Início",
        icon=folium.Icon(color="green", icon="info-sign"),
    ).add_to(base_map)

    folium.Marker(
        location=[mv.iloc[-1][LATITUDE], mv.iloc[-1][LONGITUDE]],
        color="red",
        clustered_marker=True,
        popup="Fim",
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(base_map)

    folium.PolyLine(
        mv[[LATITUDE, LONGITUDE]], color=color, weight=2.5, opacity=1
    ).add_to(base_map)

    if save_as_html:
        base_map.save(outfile=filename)
    else:
        return base_map

def plot_trajectory_by_period(
    move_data,
    period,
    id_=None,
    legend=True,
    n_rows=None,
    lat_origin=None,
    lon_origin=None,
    zoom_start=12,
    base_map=None,
    tile=TILES[0],
    save_as_html=False,
    color="black",
    filename="plot_trajectory_by_period_with_folium.html",
):
    """
    Generate trajectory view by period of day provided by user.

    Parameters
    ----------
    move_data : pymove.core.MoveDataFrameAbstract subclass.
        Input trajectory data.
    period: String
        Represents period of day.
    id_: int or None
        If int, plots trajectory of the user, else plot for all users
    legend: boolean
        Whether to add a legend to the map
    n_rows : int, optional, default None.
        Represents number of data rows that are will plot.
    lat_origin : float, optional, default None.
        Represents the latitude which will be the center of the map.
        If not entered, the first data from the dataset is used.
    lon_origin : float, optional, default None.
        Represents the longitude which will be the center of the map.
        If not entered, the first data from the dataset is used.
    zoom_start : int, optional, default 12.
        Initial zoom level for the map
    base_map : folium.folium.Map, optional, default None.
        Represents the folium map. If not informed, a new map is generated
        using the function create_base_map(), with the lat_origin,
        lon_origin and zoom_start.
    tile : String, optional, default 'OpenStreetMap'.
        Represents the map's tiles.
    save_as_html : bool, optional, default False.
        Represents if want save this visualization in a new file .html.
    color : String or List, optional, default 'black'.
        Represents line's color of visualization.
        Pass a list if ploting for many users. Else colors will be chosen at random
    filename : String, optional, default 'plot_trajectory_by_period.html'.
        Represents the file name of new file .html.

    Returns
    -------
    base_map : folium.folium.Map.
        Represents a folium map with visualization.

    Raises
    ------
        KeyError period not found in dataframe
        IndexError if there is no user with the id passed
    """
    if base_map is None:
        if lat_origin is None and lon_origin is None:
            lat_origin = move_data.loc[0][LATITUDE]
            lon_origin = move_data.loc[0][LONGITUDE]
        base_map = create_base_map(
            default_location=[lat_origin, lon_origin],
            tile=tile,
            default_zoom_start=zoom_start,
        )

    if PERIOD not in move_data:
        move_data.generate_time_of_day_features()

    mv_df = move_data[move_data[PERIOD] == period].reset_index()
    if not len(mv_df):
        raise KeyError(f"No PERIOD found in dataframe")

    if n_rows is None:
        n_rows = mv_df.shape[0]

    if id_ is not None:
        mv_df = mv_df[mv_df[TRAJ_ID] == id_].loc[
            :n_rows, [LATITUDE, LONGITUDE, TRAJ_ID]
        ]
        if not len(mv_df):
            raise IndexError(f"No user with id {id_} in dataframe")
    else:
        mv_df = mv_df.loc[:n_rows, [LATITUDE, LONGITUDE, TRAJ_ID]]

    if id_ is not None:
        items = list(zip([id_], [color]))
    else:
        ids = mv_df[TRAJ_ID].unique()
        if isinstance(color, str):
            colors = [generate_color() for _ in ids]
        else:
            colors = color[:]
        items = list(zip(ids, colors))

    for _id, color in items:
        mv = mv_df[mv_df[TRAJ_ID] == _id]
        folium.Marker(
            location=[mv.iloc[0][LATITUDE], mv.iloc[0][LONGITUDE]],
            color="green",
            clustered_marker=True,
            popup="Início",
            icon=folium.Icon(color="green", icon="info-sign"),
        ).add_to(base_map)

        folium.Marker(
            location=[mv.iloc[-1][LATITUDE], mv.iloc[-1][LONGITUDE]],
            color="red",
            clustered_marker=True,
            popup="Fim",
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(base_map)

        folium.PolyLine(
            mv[[LATITUDE, LONGITUDE]], color=color, weight=2.5, opacity=1
        ).add_to(base_map)

    if id_ is None and legend:
        add_map_legend(base_map, "Color by user ID", items)

    if save_as_html:
        base_map.save(outfile=filename)
    else:
        return base_map

def plot_trajectory_by_day_week(
    move_data,
    day_week,
    id_=None,
    legend=True,
    n_rows=None,
    lat_origin=None,
    lon_origin=None,
    zoom_start=12,
    base_map=None,
    tile=TILES[0],
    save_as_html=False,
    color="black",
    filename="plot_trajectory_by_day_week.html",
):
    """
    Generate trajectory view by day week provided by user.

    Parameters
    ----------
    move_data : pymove.core.MoveDataFrameAbstract subclass.
        Input trajectory data.
    day_week: String
        Represents day week.
    id_: int or None
        If int, plots trajectory of the user, else plot for all users
    legend: boolean
        Whether to add a legend to the map
    n_rows : int, optional, default None.
        Represents number of data rows that are will plot.
    lat_origin : float, optional, default None.
        Represents the latitude which will be the center of the map.
        If not entered, the first data from the dataset is used.
    lon_origin : float, optional, default None.
        Represents the longitude which will be the center of the map.
        If not entered, the first data from the dataset is used.
    zoom_start : int, optional, default 12.
        Initial zoom level for the map
    base_map : folium.folium.Map, optional, default None.
        Represents the folium map. If not informed, a new map is generated
        using the function create_base_map(), with the lat_origin,
        lon_origin and zoom_start.
    tile : String, optional, default 'OpenStreetMap'.
        Represents the map's tiles.
    save_as_html : bool, optional, default False.
        Represents if want save this visualization in a new file .html.
    color : String or List, optional, default 'black'.
        Represents line's color of visualization.
        Pass a list if ploting for many users. Else colors will be chosen at random
    filename : String, optional, default 'plot_trajectory_by_day_week.html'.
        Represents the file name of new file .html.

    Returns
    -------
    base_map : folium.folium.Map.
        Represents a folium map with visualization.

    Raises
    ------
        KeyError day_week not found in dataframe
        IndexError if there is no user with the id passed
    """
    if base_map is None:
        if lat_origin is None and lon_origin is None:
            lat_origin = move_data.loc[0][LATITUDE]
            lon_origin = move_data.loc[0][LONGITUDE]
        base_map = create_base_map(
            default_location=[lat_origin, lon_origin],
            tile=tile,
            default_zoom_start=zoom_start,
        )

    if DAY not in move_data:
        move_data.generate_day_of_the_week_features()

    mv_df = move_data[move_data[DAY] == day_week].reset_index()
    if not len(mv_df):
        raise KeyError(f"No DAY found in dataframe")

    if n_rows is None:
        n_rows = mv_df.shape[0]

    if id_ is not None:
        mv_df = mv_df[mv_df[TRAJ_ID] == id_].loc[
            :n_rows, [LATITUDE, LONGITUDE, TRAJ_ID]
        ]
        if not len(mv_df):
            raise IndexError(f"No user with id {id_} in dataframe")
    else:
        mv_df = mv_df.loc[:n_rows, [LATITUDE, LONGITUDE, TRAJ_ID]]

    if id_ is not None:
        items = list(zip([id_], [color]))
    else:
        ids = mv_df[TRAJ_ID].unique()
        if isinstance(color, str):
            colors = [generate_color() for _ in ids]
        else:
            colors = color[:]
        items = list(zip(ids, colors))

    for _id, color in items:
        mv = mv_df[mv_df[TRAJ_ID] == _id]
        folium.Marker(
            location=[mv.iloc[0][LATITUDE], mv.iloc[0][LONGITUDE]],
            color="green",
            clustered_marker=True,
            popup="Início",
            icon=folium.Icon(color="green", icon="info-sign"),
        ).add_to(base_map)

        folium.Marker(
            location=[mv.iloc[-1][LATITUDE], mv.iloc[-1][LONGITUDE]],
            color="red",
            clustered_marker=True,
            popup="Fim",
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(base_map)

        folium.PolyLine(
            mv[[LATITUDE, LONGITUDE]], color=color, weight=2.5, opacity=1
        ).add_to(base_map)

    if id_ is None and legend:
        add_map_legend(base_map, "Color by user ID", items)

    if save_as_html:
        base_map.save(outfile=filename)
    else:
        return base_map

def plot_trajectory_by_date(
    move_data,
    start_date,
    end_date,
    id_=None,
    legend=True,
    n_rows=None,
    lat_origin=None,
    lon_origin=None,
    zoom_start=12,
    base_map=None,
    tile=TILES[0],
    save_as_html=False,
    color="black",
    filename="plot_trajectory_by_date.html",
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
    id_: int or None
        If int, plots trajectory of the user, else plot for all users
    legend: boolean
        Whether to add a legend to the map
    n_rows : int, optional, default None.
        Represents number of data rows that are will plot.
    lat_origin : float, optional, default None.
        Represents the latitude which will be the center of the map.
        If not entered, the first data from the dataset is used.
    lon_origin : float, optional, default None.
        Represents the longitude which will be the center of the map.
        If not entered, the first data from the dataset is used.
    zoom_start : int, optional, default 12.
        Initial zoom level for the map
    base_map : folium.folium.Map, optional, default None.
        Represents the folium map. If not informed, a new map is generated
        using the function create_base_map(), with the lat_origin,
        lon_origin and zoom_start.
    tile : String, optional, default 'OpenStreetMap'.
        Represents the map's tiles.
    save_as_html : bool, optional, default False.
        Represents if want save this visualization in a new file .html.
    color : String or List, optional, default 'black'.
        Represents line's color of visualization.
        Pass a list if ploting for many users. Else colors will be chosen at random
    filename : String, optional, default 'plot_trejectory_with_folium.html'.
        Represents the file name of new file .html.

    Returns
    -------
    base_map : folium.folium.Map.
        Represents a folium map with visualization.

    Raises
    ------
        KeyError start or end date range not found in dataframe
        IndexError if there is no user with the id passed
    """

    if base_map is None:
        if lat_origin is None and lon_origin is None:
            lat_origin = move_data.loc[0][LATITUDE]
            lon_origin = move_data.loc[0][LONGITUDE]
        base_map = create_base_map(
            default_location=[lat_origin, lon_origin],
            tile=tile,
            default_zoom_start=zoom_start,
        )

    if isinstance(start_date, str):
        start_date = str_to_datetime(start_date).date()

    if isinstance(end_date, str):
        end_date = str_to_datetime(end_date).date()

    if DATE not in move_data:
        move_data.generate_date_features()

    mv_df = move_data[
        (move_data[DATE] <= end_date) & (move_data[DATE] >= start_date)
    ].reset_index()
    if not len(mv_df):
        raise KeyError(f"No DATE in range found in dataframe")

    if n_rows is None:
        n_rows = mv_df.shape[0]

    if id_ is not None:
        mv_df = mv_df[mv_df[TRAJ_ID] == id_].loc[
            :n_rows, [LATITUDE, LONGITUDE, TRAJ_ID]
        ]
        if not len(mv_df):
            raise IndexError(f"No user with id {id_} in dataframe")
    else:
        mv_df = mv_df.loc[:n_rows, [LATITUDE, LONGITUDE, TRAJ_ID]]

    if id_ is not None:
        items = list(zip([id_], [color]))
    else:
        ids = mv_df[TRAJ_ID].unique()
        if isinstance(color, str):
            colors = [generate_color() for _ in ids]
        else:
            colors = color[:]
        items = list(zip(ids, colors))

    for _id, color in items:
        mv = mv_df[mv_df[TRAJ_ID] == _id]
        folium.Marker(
            location=[mv.iloc[0][LATITUDE], mv.iloc[0][LONGITUDE]],
            color="green",
            clustered_marker=True,
            popup="Início",
            icon=folium.Icon(color="green", icon="info-sign"),
        ).add_to(base_map)

        folium.Marker(
            location=[mv.iloc[-1][LATITUDE], mv.iloc[-1][LONGITUDE]],
            color="red",
            clustered_marker=True,
            popup="Fim",
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(base_map)

        folium.PolyLine(
            mv[[LATITUDE, LONGITUDE]], color=color, weight=2.5, opacity=1
        ).add_to(base_map)

    if id_ is None and legend:
        add_map_legend(base_map, "Color by user ID", items)

    if save_as_html:
        base_map.save(outfile=filename)
    else:
        return base_map

def plot_trajectory_by_hour(
    move_data,
    start_hour,
    end_hour,
    id_=None,
    legend=True,
    n_rows=None,
    lat_origin=None,
    lon_origin=None,
    zoom_start=12,
    base_map=None,
    tile=TILES[0],
    save_as_html=False,
    color="black",
    filename="plot_trajectory_by_hour.html",
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
    id_: int or None
        If int, plots trajectory of the user, else plot for all users
    legend: boolean
        Whether to add a legend to the map
    n_rows : int, optional, default None.
        Represents number of data rows that are will plot.
    lat_origin : float, optional, default None.
        Represents the latitude which will be the center of the map.
        If not entered, the first data from the dataset is used.
    lon_origin : float, optional, default None.
        Represents the longitude which will be the center of the map.
        If not entered, the first data from the dataset is used.
    zoom_start : int, optional, default 12.
        Initial zoom level for the map
    base_map : folium.folium.Map, optional, default None.
        Represents the folium map. If not informed, a new map is generated
        using the function create_base_map(), with the lat_origin,
        lon_origin and zoom_start.
    tile : String, optional, default 'OpenStreetMap'.
        Represents the map's tiles.
    save_as_html : bool, optional, default False.
        Represents if want save this visualization in a new file .html.
    color : String or List, optional, default 'black'.
        Represents line's color of visualization.
        Pass a list if ploting for many users. Else colors will be chosen at random
    filename : String, optional, default 'plot_trajectory_by_hour.html'.
        Represents the file name of new file .html.

    Returns
    -------
    base_map : folium.folium.Map.
        Represents a folium map with visualization.

    Raises
    ------
        KeyError if start to end hour range not found in dataframe
        IndexError if there is no user with the id passed
    """
    if base_map is None:
        if lat_origin is None and lon_origin is None:
            lat_origin = move_data.loc[0][LATITUDE]
            lon_origin = move_data.loc[0][LONGITUDE]
        base_map = create_base_map(
            default_location=[lat_origin, lon_origin],
            tile=tile,
            default_zoom_start=zoom_start,
        )

    if HOUR not in move_data:
        move_data.generate_hour_features()

    mv_df = move_data[
        (move_data[HOUR] <= end_hour) & (move_data[HOUR] >= start_hour)
    ].reset_index()
    if not len(mv_df):
        raise KeyError(f"No HOUR in range found in dataframe")

    if n_rows is None:
        n_rows = mv_df.shape[0]

    if id_ is not None:
        mv_df = mv_df[mv_df[TRAJ_ID] == id_].loc[
            :n_rows, [LATITUDE, LONGITUDE, TRAJ_ID]
        ]
        if not len(mv_df):
            raise IndexError(f"No user with id {id_} in dataframe")
    else:
        mv_df = mv_df.loc[:n_rows, [LATITUDE, LONGITUDE, TRAJ_ID]]

    if id_ is not None:
        items = list(zip([id_], [color]))
    else:
        ids = mv_df[TRAJ_ID].unique()
        if isinstance(color, str):
            colors = [generate_color() for _ in ids]
        else:
            colors = color[:]
        items = list(zip(ids, colors))

    for _id, color in items:
        mv = mv_df[mv_df[TRAJ_ID] == _id]
        folium.Marker(
            location=[mv.iloc[0][LATITUDE], mv.iloc[0][LONGITUDE]],
            color="green",
            clustered_marker=True,
            popup="Início",
            icon=folium.Icon(color="green", icon="info-sign"),
        ).add_to(base_map)

        folium.Marker(
            location=[mv.iloc[-1][LATITUDE], mv.iloc[-1][LONGITUDE]],
            color="red",
            clustered_marker=True,
            popup="Fim ",
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(base_map)

        folium.PolyLine(
            mv[[LATITUDE, LONGITUDE]], color=color, weight=2.5, opacity=1
        ).add_to(base_map)

    if id_ is None and legend:
        add_map_legend(base_map, "Color by user ID", items)

    if save_as_html:
        base_map.save(outfile=filename)
    else:
        return base_map

def plot_stops(
    move_data,
    radius=900,
    weight=3,
    id_=None,
    legend=True,
    n_rows=None,
    lat_origin=None,
    lon_origin=None,
    zoom_start=12,
    base_map=None,
    tile=TILES[0],
    save_as_html=False,
    color="black",
    filename="plot_stops.html",
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
    weight: int or None
        Stroke width in pixels
    id_: int or None
        If int, plots trajectory of the user, else plot for all users
    legend: boolean
        Whether to add a legend to the map
    n_rows : int, optional, default None.
        Represents number of data rows that are will plot.
    lat_origin : float, optional, default None.
        Represents the latitude which will be the center of the map.
        If not entered, the first data from the dataset is used.
    lon_origin : float, optional, default None.
        Represents the longitude which will be the center of the map.
        If not entered, the first data from the dataset is used.
    zoom_start : int, optional, default 12.
        Initial zoom level for the map
    base_map : folium.folium.Map, optional, default None.
        Represents the folium map. If not informed, a new map is generated
        using the function create_base_map(), with the lat_origin,
        lon_origin and zoom_start.
    tile : String, optional, default 'OpenStreetMap'.
        Represents the map's tiles.
    save_as_html : bool, optional, default False.
        Represents if want save this visualization in a new file .html.
    color : String or List, optional, default 'black'.
        Represents line's color of visualization.
        Pass a list if ploting for many users. Else colors will be chosen at random
    filename : String, optional, default 'plot_stops.html'.
        Represents the file name of new file .html.

    Returns
    -------
    base_map : folium.folium.Map.
        Represents a folium map with visualization.

    Raises
        ------
        KeyError if no STOPs found
        IndexError if there is no user with the id passed
    """
    if base_map is None:
        if lat_origin is None and lon_origin is None:
            lat_origin = move_data.loc[0][LATITUDE]
            lon_origin = move_data.loc[0][LONGITUDE]
        base_map = create_base_map(
            default_location=[lat_origin, lon_origin],
            tile=tile,
            default_zoom_start=zoom_start,
        )

    if SITUATION not in move_data:
        move_data.generate_move_and_stop_by_radius(radius=radius)

    stops = move_data[move_data[SITUATION] == STOP].reset_index()
    if not len(stops):
        raise KeyError(f"No STOPS found in dataframe")

    if n_rows is None:
        n_rows = stops.shape[0]

    if id_ is not None:
        stops = stops[stops[TRAJ_ID] == id_].loc[
            :n_rows, [LATITUDE, LONGITUDE, DATETIME, TRAJ_ID]
        ]
        if not len(stops):
            raise IndexError(f"No user with id {id_} in dataframe")
    else:
        stops = stops.loc[:n_rows, [LATITUDE, LONGITUDE, DATETIME, TRAJ_ID]]

    if id_ is not None:
        items = list(zip([id_], [color]))
    else:
        ids = stops[TRAJ_ID].unique()
        if isinstance(color, str):
            colors = [generate_color() for _ in ids]
        else:
            colors = color[:]
        items = list(zip(ids, colors))

    for _id, color in items:
        for stop in stops[stops[TRAJ_ID] == _id].iterrows():
            base_map.add_child(
                folium.Circle(
                    (stop[1][LATITUDE], stop[1][LONGITUDE]),
                    color=color,
                    weight=weight,
                    radius=30,
                    opacity=0.5,
                    popup=stop[1][DATETIME],
                    fill_color=color,
                    fill_opacity=0.5,
                )
            )
    if id_ is None and legend:
        add_map_legend(base_map, "Color by user ID", items)

    if save_as_html:
        base_map.save(outfile=filename)
    else:
        return base_map
#------------------------------
def formate_tags(line, slice_):
    
    map_formated_tags = map(lambda tag: '{}: {}'.format(tag, line[tag]), slice_)

    return  '<br/>'.join(map_formated_tags)


def invert_map(map_):
    inv_map = {}
    items_ = map_.items()
    for k, v in items_:
        inv_map[v] = inv_map.get(v, [])
        inv_map[v].append(k)
    return inv_map

def add_traj_folium(df,
              dict_plot = constants.dict_plot,
              dict_labels = constants.dict_labels,
              sort=False,
              folium_map = None,
              slice_tags = None,
              tiles='OpenStreetMap'):
        """
        Receivies a MoveDataFrame and returns a folium map with the trajectories plots.
    
        Parameters:
        ----------
        df: Dataframe
            Trajectory data.
        dict_plot: dictionary, optional, default pymove.utils.constants.dict_plot
            folium plot paramenters.
        dict_labels: dictionary, optional, default pymove.utils.constants.dict_labels
        sort:Boolean, optional, default False.
            If True the data will be sorted.
        folium_map: Folium map, optional, default None.
            A folium map to plot the trajectories. If None a map will be created.
        slice_tags: optional, default None.
        tiles: string, optional, default 'OpenStreetMap'.
            The map type.
            
        Returns:
        -------
        A folium map.
        """
              
        def circle_maker(iter_tuple, map_):
            
            _,line = iter_tuple


            x = line[dict_labels['tnz_lat']]
            y = line[dict_labels['tnz_lon']]
            
            tags_formated = formate_tags(line,slice_tags)
            
            folium.Circle(
            radius=1,
            location=[x, y],
            popup=tags_formated,
            color=dict_plot['tnz_point'],
            fill=False
            ).add_to(map_)
        
        def plot_incial_end_points(list_rowns, map_):
            
            #plot the start tnz_point
            line = list_rowns[0][1]
            #tags_formated = str(line[slice_tags]).replace('\n','<br/>').replace('\s', ':\s')
            tags_formated = formate_tags(line,slice_tags)
            
            x = line[dict_labels['tnz_lat']]
            y = line[dict_labels['tnz_lon']]
            

            folium.Marker(
                location=[x, y],
                popup='START<br/>'+tags_formated,
                icon=folium.Icon(color='green')
            ).add_to(folium_map)

            # folium.Circle(
            #             radius=1,
            #             location=[x, y],
            #             tooltip='START<br/>'+tags_formated,
            #             color=dict_plot['start'],
            #             fill=False,
            #         ).add_to(folium_map)

            #plot the last tnz_point
            line = list_rowns[-1][1]
            #tags_formated = str(line[slice_tags]).replace('\n','<br/>').replace('\s', ':\s')
            tags_formated = formate_tags(line,slice_tags)

            x = line[dict_labels['tnz_lat']]
            y = line[dict_labels['tnz_lon']]
            
            # folium.Circle(
            #             radius=1,
            #             location=[x, y],
            #             tooltip='END<br/>'+tags_formated,
            #             color=dict_plot['end'],
            #             fill=False,
            #         ).add_to(folium_map)
            folium.Marker(
                location=[x, y],
                popup='END<br/>'+tags_formated,
                icon=folium.Icon(color='red', icon='info-sign')
                ).add_to(folium_map)
        

        if not slice_tags:
            slice_tags = df.columns
    
        # If not have a map a map is create with mean to lat and lon
        if not folium_map:
            initial_lat = df[ dict_labels['tnz_lat'] ].mean()
            initial_lon = df[ dict_labels['tnz_lon'] ].mean()
            folium_map = create_base_map([initial_lat, initial_lon], tile=tiles)
        
        #if needs sort the data
        if sort:       
            df.sort_values(dict_labels['tnz_datetime'], inplace=True)

        #plot the lines
        tnz_points = list(zip(df[dict_labels['tnz_lat']],df[dict_labels['tnz_lon']]))
        
        folium.PolyLine(tnz_points, 
                        color= dict_plot['line'],weight=2).add_to(folium_map)
        
            
        list(map(lambda x: circle_maker(x,folium_map), df.iterrows()))   
        
        plot_incial_end_points(list(df.iterrows()), folium_map)

        return folium_map
def add_point_folium(df,
              dict_plot = constants.dict_plot,
              dict_labels = constants.dict_labels,
              folium_map = None,
              slice_tags = None,
              tiles='cartodbpositron'):
        """
        Receivies a MoveDataFrame and returns a folium map with the trajectories plots and a point.
    
        Parameters:
        ----------
        df: Dataframe
            Trajectory data.
        dict_plot: dictionary, optional, default pymove.utils.constants.dict_plot
            folium plot paramenters.
        dict_labels: dictionary, optional, default pymove.utils.constants.dict_labels
        sort:Boolean, optional, default False.
            If True the data will be sorted.
        folium_map: Folium map, optional, default None.
            A folium map to plot the trajectories. If None a map will be created.
        slice_tags: optional, default None.
        tiles: string, optional, default 'OpenStreetMap'.
            The map type.
            
        Returns:
        -------
        A folium map.
        """

        if not slice_tags:
            slice_tags = df.columns
    
        # If not have a map a map is create with mean to lat and lon
        if not folium_map:
            initial_lat = df[ dict_labels['lat'] ].mean()
            initial_lon = df[ dict_labels['lon'] ].mean()
            folium_map = create_base_map([initial_lat, initial_lon], tile=tiles)
        
        def circle_maker(iter_tuple, map_):
            
            _,line = iter_tuple


            x = line[dict_labels['lat']]
            y = line[dict_labels['lon']]
            
            #tags_formated = str(line[slice_tags]).replace('\n','<br/>').replace('\s', ':\s')
            tags_formated = formate_tags(line,slice_tags)
            
            folium.Circle(
            radius=1,
            location=[x, y],
            popup=tags_formated,
            tooltip=tags_formated,
            color=dict_plot['poi_point'],
            fill=False
            ).add_to(map_)
        

        list(map(lambda x: circle_maker(x,folium_map), df.iterrows()))   

        return folium_map

def add_poi_folium(df,
              dict_plot = constants.dict_plot,
              dict_labels = constants.dict_labels,
              folium_map = None,
              slice_tags = None):

        """
        Receivies a MoveDataFrame and returns a folium map with poi points.

        Parameters:
        ----------
        df: DataFrame
            Trajectory input data
        dict_plot: dictionary, optional, default pymove.utils.constants.dict_plot
            folium plot paramenters.
        dict_labels: dictionary, optional, default pymove.utils.constants.dict_labels
        folium_map: Folium map, optional, default None.
            A folium map to plot. If None a map. If None a map will be created.
        """

        if not slice_tags:
            slice_tags = df.columns
    
        # If not have a map a map is create with mean to lat and lon
        if not folium_map:
            initial_lat = df[ dict_labels['poi_lat'] ].mean()
            initial_lon = df[ dict_labels['poi_lon'] ].mean()
            folium_map = create_base_map([initial_lat, initial_lon])
        
        def circle_maker(iter_tuple, map_):
            
            _,line = iter_tuple


            x = line[dict_labels['poi_lat']]
            y = line[dict_labels['poi_lon']]
            
            #tags_formated = str(line[slice_tags]).replace('\n','<br/>').replace('\s', ':\s')
            tags_formated = formate_tags(line,slice_tags)
            
            folium.Circle(
            radius=1,
            location=[x, y],
            popup=tags_formated,
            tooltip=tags_formated,
            color=dict_plot['poi_point'],
            fill=False
            ).add_to(map_)
        

        list(map(lambda x: circle_maker(x,folium_map), df.iterrows()))   

        return folium_map

def add_event_folium(df,
              dict_plot = constants.dict_plot,
              dict_labels = constants.dict_labels,
              folium_map = None,
              slice_tags = None,
              tiles='OpenStreetMap'):

        """
        Receivies a MoveDataFrame and returns a folium map with events.

        Parameters:
        ----------
        df: DataFrame
            Trajectory input data
        dict_plot: dictionary, optional, default pymove.utils.constants.dict_plot
            folium plot paramenters.
        dict_labels: dictionary, optional, default pymove.utils.constants.dict_labels
        folium_map: Folium map, optional, default None.
            A folium map to plot. If None a map. If None a map will be created.
        tiles: string, optional, default 'OpenStreetMap'

        """
        if not slice_tags:
            slice_tags = df.columns
    
        # If not have a map a map is create with mean to lat and lon
        if not folium_map:
            initial_lat = df[ dict_labels['event_lat'] ].mean()
            initial_lon = df[ dict_labels['event_lon'] ].mean()
            folium_map = create_base_map([initial_lat, initial_lon], tile=tiles)
        
        def circle_maker(iter_tuple, map_):
            
            _,line = iter_tuple


            x = line[dict_labels['event_lat']]
            y = line[dict_labels['event_lon']]
            
            #tags_formated = str(line[slice_tags]).replace('\n','<br/>').replace('\s', ':\s')
            tags_formated = formate_tags(line,slice_tags)
            
            folium.Circle(
            radius=1,
            location=[x, y],
            tooltip=tags_formated,
            color=dict_plot['event_point'],
            fill=False
            ).add_to(map_)

            folium.Circle(
            radius=dict_plot['radius'],
            location=[x, y],
            tooltip=tags_formated,
            color=dict_plot['event_point'],
            fill=False,
            fill_color=dict_plot['event_point']
            ).add_to(map_)


        

        list(map(lambda x: circle_maker(x,folium_map), df.iterrows()))   

        return folium_map

def show_trajs_with_event(df_tnz, 
                        window_time_tnz,
                        df_event,
                       window_time_event,
                       radius,
                      dict_plot = constants.dict_plot,
                      dict_labels = constants.dict_labels,
                    slice_event_show = None, 
                    slice_tnz_show = None):
    """
        Plot a trajectory, incluiding your tnz_points lat lon and your tags, and CVP
    """
        
    #building structure for deltas
    delta_event = pd.to_timedelta(window_time_event, unit='s')
    delta_tnz = pd.to_timedelta(window_time_tnz, unit='s')
    
    #length of df_tnz
    len_df_tnz = df_tnz.shape[0]
        
    #building structure for lat and lon array
    lat_arr = np.zeros(len_df_tnz)
    lon_arr = np.zeros(len_df_tnz)
    
    #folium map list
    folium_maps = []
    
    #for each cvp in df_cvp
    for _, line in df_event.iterrows():
        
        event_lat = line[dict_labels['event_lat']]
        
        event_lon = line[dict_labels['event_lon']]
        
        event_datetime = line[dict_labels['event_datetime']]
        
        event_id = line[dict_labels['event_id']]
                
        #building time window for cvp search
        start_time = pd.to_datetime(event_datetime - delta_event)
        end_time = pd.to_datetime(event_datetime + delta_event)
        
        #filtering df_tnz for time window
        df_filted=pd.DataFrame(filters.by_datetime(df_tnz,
                                     start_datetime = start_time,
                                     end_datetime=end_time))
        
        #length of df_temp
        len_df_temp = df_filted.shape[0]
        
        #using the util part of the array for haversine function
        lat_arr[:len_df_temp] = event_lat
        lon_arr[:len_df_temp] = event_lon
        
        #building distances to cvp column
        df_filted['distances'] = distances.haversine(lat_arr[:len_df_temp],
                                         lon_arr[:len_df_temp], 
                                         df_filted[dict_labels['tnz_lat']],
                                         df_filted[dict_labels['tnz_lon']])
        
        #building nearby column
        df_filted['nearby'] = df_filted['distances'].map(lambda x: (x<=radius))
        
        #if any data for df_tnz in cvp time window is True
        if df_filted['nearby'].any():
            
            #buildng the df for the first tnz_points of tnz in nearby cvp
            df_begin = df_filted[df_filted['nearby'] == True].sort_values(dict_labels['tnz_datetime'])
            
            #making the folium map
            #folium_map = create_folium_map([df_begin[dict_labels['tnz_lat']].mean(),
            #                                   df_begin[dict_labels['tnz_lon']].mean()])
            
            #plot cvp in map

            df_ = df_event[df_event[dict_labels['event_id']] == event_id]
            
            folium_map = add_event_folium(df_, 
            dict_plot=dict_plot, 
            dict_labels=dict_labels, 
            slice_tags=slice_event_show)
        
            #keep only the first tnz_point nearby to cvp for each tnz
            df_begin.drop_duplicates(subset=[dict_labels['tnz_id'],'nearby'], inplace=True)
            
            #for each tnz nearby to cvp
            tnzs = []
            for time_tnz, id_tnz in zip(df_begin[dict_labels['tnz_datetime']], 
                                            df_begin[dict_labels['tnz_id']]):
                
                #making the time window for tnz
                start_time = pd.to_datetime(time_tnz - delta_tnz)
                end_time = pd.to_datetime(time_tnz + delta_tnz)

                #building the df for one id
                df_id = df_tnz[df_tnz[dict_labels['tnz_id']] == id_tnz]

                #filtering df_id for time window
                df_temp = pd.DataFrame(trajutils.filter_by_datetime(df_id,
                                         dic_labels = dict_labels,
                                         startDatetime = start_time,
                                         endDatetime=end_time))
                
                tnzs.append(df_temp)
                #add to folium map created
                add_traj_folium(df_temp, 
                          dict_labels=dict_labels, 
                          dict_plot=dict_plot,
                        folium_map=folium_map,
                        slice_tags=slice_tnz_show, 
                        sort=True)
            
            #add to folium maps list: (id cvp, folium map, quantity of tnz in map, df)  
            folium_maps.append( (folium_map, pd.concat(tnzs)) )

    return folium_maps

