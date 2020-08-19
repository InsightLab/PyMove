import folium
import numpy as np
import pandas as pd
from folium import plugins
from folium.plugins import FastMarkerCluster, HeatMap, HeatMapWithTime, MarkerCluster

from pymove.preprocessing import filters
from pymove.utils import distances
from pymove.utils.constants import (
    COLORS,
    COUNT,
    DATE,
    DATETIME,
    DAY,
    EVENT_ID,
    EVENT_POINT,
    HOUR,
    LATITUDE,
    LINE_COLOR,
    LONGITUDE,
    PERIOD,
    POI_POINT,
    SITUATION,
    STOP,
    TILES,
    TRAJ_ID,
    UID,
    USER_POINT,
)
from pymove.utils.datetime import str_to_datetime
from pymove.utils.log import progress_bar
from pymove.utils.visual import add_map_legend, cmap_hex_color, get_cmap


def save_map(
    move_data,
    filename,
    tiles=TILES[0],
    label_id=TRAJ_ID,
    cmap='Set1',
    return_map=False
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
        Represents the type_ of tile that will be used on the map.
    label_id : String
        Represents column name of trajectory id.
    cmap: String
        Represents the Colormap.

    """

    map_ = folium.Map(tiles=tiles)
    map_.fit_bounds(
        [
            [move_data[LATITUDE].min(), move_data[LONGITUDE].min()],
            [move_data[LATITUDE].max(), move_data[LONGITUDE].max()],
        ]
    )

    ids = move_data[label_id].unique()
    cmap_ = get_cmap(cmap)
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

    if return_map:
        return map_


def create_base_map(
    move_data,
    lat_origin=None,
    lon_origin=None,
    tile=TILES[0],
    default_zoom_start=12,
):
    """
    Generate a folium map.

    Parameters
    ----------
    move_data : pymove.core.MoveDataFrameAbstract subclass.
        Input trajectory data.
    lat_origin : float, optional, default None.
        Represents the latitude which will be the center of the map.
        If not entered, the first data from the dataset is used.
    lon_origin : float, optional, default None.
        Represents the longitude which will be the center of the map.
        If not entered, the first data from the dataset is used.
    default_zoom_start : int, optional, default 12.
        Represents the zoom which will be the center of the map.
    tile : String, optional, default 'CartoDB positron'.
        Represents the map'srs tiles.

    Returns
    -------
    folium.folium.Map.
        Represents a folium map.

    """

    if lat_origin is None and lon_origin is None:
        lat_origin = move_data.iloc[0][LATITUDE]
        lon_origin = move_data.iloc[0][LONGITUDE]
    base_map = folium.Map(
        location=[lat_origin, lon_origin],
        control_scale=True,
        zoom_start=default_zoom_start,
        tiles=tile
    )
    return base_map


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
    base_map = folium.Map(
        location=default_location,
        control_scale=True,
        zoom_start=default_zoom_start
    )
    return base_map


def heatmap(
    move_data,
    n_rows=None,
    lat_origin=None,
    lon_origin=None,
    zoom_start=12,
    radius=8,
    base_map=None,
    tile=TILES[0],
    save_as_html=False,
    filename='heatmap.html',
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
    base_map : folium.folium.Map, optional, default None.
        Represents the folium map. If not informed, a new map is generated
        using the function create_base_map(), with the lat_origin, lon_origin
        and zoom_start.
    tile : String, optional, default 'CartoDB positron'.
        Represents the map'srs tiles.
    save_as_html : bool, optional, default False.
        Represents if want save this visualization in a new file .html.
    filename : String, optional, default 'heatmap.html'.
        Represents the file name of new file .html.

    Returns
    -------
    folium.folium.Map.
        Represents a folium map with visualization.

    """
    if base_map is None:
        base_map = create_base_map(
            move_data,
            lat_origin,
            lon_origin,
            tile=tile,
            default_zoom_start=zoom_start,
        )

    if n_rows is None:
        n_rows = move_data.shape[0]

    move_data[COUNT] = 1
    HeatMap(
        data=move_data.iloc[:n_rows][[LATITUDE, LONGITUDE, COUNT]]
        .groupby([LATITUDE, LONGITUDE])
        .sum()
        .reset_index()
        .values.tolist(),
        radius=radius
    ).add_to(base_map)
    move_data.drop(columns=COUNT, inplace=True)

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
    radius=8,
    min_opacity=0.5,
    max_opacity=0.8,
    base_map=None,
    tile=TILES[0],
    save_as_html=False,
    filename='heatmap_with_time.html'
):
    """
    Generate a heatmap with time.

    Parameters
    -----------
    move_data: DataFrame.
        Trajectories input data.
    n_rows: int, optional, default None.
        the number of rows to use.
    lat_origin: float, optional, default None.
        The latitude coordinate to use as origin.
    lon_origin: float, optional, default None.
       The longitude coordinate to use as origin.
    zoom_start: int, optional, default 12.
        The zoom to start whit.
    radius: float, optional, default 5.
        Radius to form heatmap cluster
    min_opacity: float, optional, default 0.5.
        Minimum heat opacity
    max_opacity: float, optional, default 0.8.
        Maximum heat opacity
    base_map : folium.folium.Map, optional, default None.
        Represents the folium map. If not informed, a new map is generated
        using the function create_base_map(), with the lat_origin, lon_origin
        and zoom_start.
    tile : String, optional, default 'CartoDB positron'.
        Represents the map'srs tiles.
    save_as_html : bool, optional, default False.
        Represents if want save this visualization in a new file .html.
    filename : String, optional, default 'heatmap_with_time.html'.
        Represents the file name of new file .html.

    Returns
    -------
    folium.folium.Map.
        Represents a folium map with visualization.

    """

    if base_map is None:
        base_map = create_base_map(
            move_data,
            lat_origin,
            lon_origin,
            tile=tile,
            default_zoom_start=zoom_start,
        )

    if n_rows is None:
        n_rows = move_data.shape[0]

    move_data = move_data.iloc[:n_rows].copy()

    move_data[COUNT] = 1
    move_data[HOUR] = move_data[DATETIME].apply(lambda x: x.hour)
    move_data_hour_list = []
    for hour in move_data[HOUR].sort_values().unique():
        move_data_hour_list.append(
            move_data.loc[move_data.hour == hour, [LATITUDE, LONGITUDE, COUNT]]
            .groupby([LATITUDE, LONGITUDE])
            .sum()
            .reset_index()
            .values.tolist()
        )

    HeatMapWithTime(
        move_data_hour_list,
        radius=radius,
        gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red'},
        min_opacity=min_opacity,
        max_opacity=max_opacity,
        use_local_extrema=True
    ).add_to(base_map)
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
    tile=TILES[0],
    save_as_html=False,
    filename='cluster.html',
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
    tile : String, optional, default 'CartoDB positron'.
        Represents the map'srs tiles.
    save_as_html : bool, optional, default False.
        Represents if want save this visualization in a new file .html.
    filename : String, optional, default 'cluster.html'.
        Represents the file name of new file .html.

    Returns
    -------
    folium.folium.Map.
        Represents a folium map with visualization.

    """

    if base_map is None:
        base_map = create_base_map(
            move_data,
            lat_origin,
            lon_origin,
            tile=tile,
            default_zoom_start=zoom_start,
        )

    if n_rows is None:
        n_rows = move_data.shape[0]

    mc = MarkerCluster()
    for row in move_data.iloc[:n_rows].iterrows():
        pop = (
            '<b>Latitude:</b> '
            + str(row[1][LATITUDE])
            + '\n<b>Longitude:</b> '
            + str(row[1][LONGITUDE])
            + '\n<b>Datetime:</b> '
            + str(row[1][DATETIME])
        )
        mc.add_child(
            folium.Marker(
                location=[row[1][LATITUDE], row[1][LONGITUDE]], popup=pop
            )
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
    filename='faster_cluster.html',
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
    tile : String, optional, default 'CartoDB positron'.
        Represents the map'srs tiles.
    save_as_html : bool, optional, default False.
        Represents if want save this visualization in a new file .html.
    filename : String, optional, default 'faster_cluster.html'.
        Represents the file name of new file .html.

    Returns
    -------
    folium.folium.Map.
        Represents a folium map with visualization.

    """

    if base_map is None:
        base_map = create_base_map(
            move_data,
            lat_origin,
            lon_origin,
            tile=tile,
            default_zoom_start=zoom_start,
        )

    if n_rows is None:
        n_rows = move_data.shape[0]

    callback = """\
    function (row) {
        var marker;
        marker = L.circle(new L.LatLng(row[0], row[1]), {color:'red'});
        return marker;
    };
    """
    FastMarkerCluster(
        move_data.iloc[:n_rows][[LATITUDE, LONGITUDE]].values.tolist(),
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
    filename='plot_markers.html',
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
    tile : String, optional, default 'CartoDB positron'.
        Represents the map'srs tiles.
    save_as_html : bool, optional, default False.
        Represents if want save this visualization in a new file .html.
    filename : String, optional, default 'plot_trejectory_with_folium.html'.
        Represents the file name of new file .html.

    Returns
    -------
    folium.folium.Map.
        Represents a folium map with visualization.

    """

    if base_map is None:
        base_map = create_base_map(
            move_data,
            lat_origin,
            lon_origin,
            tile=tile,
            default_zoom_start=zoom_start,
        )

    if n_rows is None:
        n_rows = move_data.shape[0]

    _add_begin_end_markers_to_folium_map(move_data.iloc[:n_rows], base_map)

    for row in move_data.iloc[1: n_rows - 1].iterrows():
        pop = (
            '<b>Latitude:</b> '
            + str(row[1][LATITUDE])
            + '\n<b>Longitude:</b> '
            + str(row[1][LONGITUDE])
            + '\n<b>Datetime:</b> '
            + str(row[1][DATETIME])
        )
        folium.Marker(
            location=[row[1][LATITUDE], row[1][LONGITUDE]],
            clustered_marker=True,
            popup=pop,
        ).add_to(base_map)

    if save_as_html:
        base_map.save(outfile=filename)
    else:
        return base_map


def _filter_and_generate_colors(
    move_data, id_=None, n_rows=None, color=None
):
    """
    Filters the dataframe and generate colors for folium map.

    Parameters
    ----------
    move_data : pymove.core.MoveDataFrameAbstract subclass.
        Input trajectory data.
    id_: int or None.
        The TRAJ_ID's to be plotted
    n_rows : int, optional, default None.
        Represents number of data rows that are will plot.
    color: string or None.
        The color of the trajectory, of each trajectory or a colormap

    Returns
    -------
    pymove.core.MoveDataFrameAbstract subclass.
        Filtered trajectories
    list of tuples
        list containing a combination of id and color

    """

    if n_rows is None:
        n_rows = move_data.shape[0]

    if id_ is not None:
        mv_df = move_data[move_data[TRAJ_ID] == id_].iloc[:n_rows][
            [LATITUDE, LONGITUDE, DATETIME, TRAJ_ID]
        ]
        if not len(mv_df):
            raise IndexError('No user with id %s in dataframe' % id_)
    else:
        mv_df = move_data.iloc[:n_rows][
            [LATITUDE, LONGITUDE, DATETIME, TRAJ_ID]
        ]

    if id_ is not None:
        if color is None:
            color = 'black'
        items = [(id_, color)]
    else:
        if color is None:
            color = 'Set1'
        ids = mv_df[TRAJ_ID].unique()

        if isinstance(color, str):
            cmap_ = get_cmap(color)
            num = cmap_.N
            colors = [
                cmap_hex_color(cmap_, (i % num))
                for i, _ in enumerate(ids)
            ]
            diff = (len(ids) // len(colors)) + 1
            colors *= diff
        else:
            colors = color[:]
        items = [*zip(ids, colors)]

    return mv_df, items


def _filter_generated_feature(move_data, feature, values):
    """
    Filters the values from the dataframe.

    Parameters
    __________
    move_data : pymove.core.MoveDataFrameAbstract subclass.
        Input trajectory data.
    feature: string
        Name of the feature
    value:
        value of the feature

    Returns
    -------
    dataframe
        filtered dataframe

    """
    if len(values) == 1:
        mv_df = move_data[move_data[feature] == values[0]]
    else:
        mv_df = move_data[
            (move_data[feature] >= values[0])
            & (move_data[feature] <= values[1])
        ]
    if not len(mv_df):
        raise KeyError('No %s found in dataframe' % feature)
    return mv_df


def _add_begin_end_markers_to_folium_map(move_data, base_map):
    """
    Adds a green marker to beginning of the trajectory and a red marker to the
    end of the trajectory.

    Parameters
    ----------
    move_data : pymove.core.MoveDataFrameAbstract subclass.
        Input trajectory data.
    base_map : folium.folium.Map, optional, default None.
        Represents the folium map. If not informed, a new map is generated.

    """

    folium.Marker(
        location=[move_data.iloc[0][LATITUDE], move_data.iloc[0][LONGITUDE]],
        color='green',
        clustered_marker=True,
        popup='Início',
        icon=folium.Icon(color='green', icon='info-sign'),
    ).add_to(base_map)

    folium.Marker(
        location=[move_data.iloc[-1][LATITUDE], move_data.iloc[-1][LONGITUDE]],
        color='red',
        clustered_marker=True,
        popup='Fim',
        icon=folium.Icon(color='red', icon='info-sign'),
    ).add_to(base_map)


def _add_trajectories_to_folium_map(
    move_data,
    items,
    base_map,
    legend=True,
    save_as_html=True,
    filename='map.html',
):
    """
    Adds a trajectory to a folium map with begin and end markers.

    Parameters
    ----------
    move_data : pymove.core.MoveDataFrameAbstract subclass.
        Input trajectory data.
    legend: boolean, default True
        Whether to add a legend to the map
    base_map : folium.folium.Map, optional, default None.
        Represents the folium map. If not informed, a new map is generated.
    save_as_html : bool, optional, default False.
        Represents if want save this visualization in a new file .html.
    filename : String, optional, default 'plot_trajectory_by_period.html'.
        Represents the file name of new file .html.

    """

    for _id, color in items:
        mv = move_data[move_data[TRAJ_ID] == _id]

        _add_begin_end_markers_to_folium_map(move_data, base_map)

        folium.PolyLine(
            mv[[LATITUDE, LONGITUDE]], color=color, weight=2.5, opacity=1
        ).add_to(base_map)

    if legend:
        add_map_legend(base_map, 'Color by user ID', items)

    if save_as_html:
        base_map.save(outfile=filename)


def plot_trajectories_with_folium(
    move_data,
    n_rows=None,
    lat_origin=None,
    lon_origin=None,
    zoom_start=12,
    legend=True,
    base_map=None,
    tile=TILES[0],
    save_as_html=False,
    color=None,
    filename='plot_trajectories_with_folium.html',
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
    legend: boolean, default True
        Whether to add a legend to the map
    base_map : folium.folium.Map, optional, default None.
        Represents the folium map. If not informed, a new map is generated
        using the function create_base_map(), with the lat_origin, lon_origin
         and zoom_start.
    tile : String, optional, default 'CartoDB positron'.
        Represents the map'srs tiles.
    save_as_html : bool, optional, default False.
        Represents if want save this visualization in a new file .html.
    color : String, list, optional, default None.
        Represents line colors of visualization.
        Can be a single color name, a list of colors or a colormap name.
    filename : String, optional, default 'plot_trajectory_with_folium.html'.
        Represents the file name of new file .html.

    Returns
    -------
    folium.folium.Map.
        Represents a folium map with visualization.

    """

    if base_map is None:
        base_map = create_base_map(
            move_data,
            lat_origin,
            lon_origin,
            tile=tile,
            default_zoom_start=zoom_start,
        )

    mv_df, items = _filter_and_generate_colors(
        move_data, n_rows=n_rows, color=color
    )
    _add_trajectories_to_folium_map(
        mv_df, items, base_map, legend, save_as_html, filename
    )

    return base_map


def plot_trajectory_by_id_folium(
    move_data,
    id_,
    n_rows=None,
    lat_origin=None,
    lon_origin=None,
    zoom_start=12,
    legend=True,
    base_map=None,
    tile=TILES[0],
    save_as_html=False,
    color='black',
    filename='plot_trajectory_by_id_folium.html',
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
    legend: boolean, default True
        Whether to add a legend to the map
    base_map : folium.folium.Map, optional, default None.
        Represents the folium map. If not informed, a new map is
        generated using the function create_base_map(), with the
        lat_origin, lon_origin and zoom_start.
    tile : String, optional, default 'CartoDB positron'.
        Represents the map'srs tiles.
    save_as_html : bool, optional, default False.
        Represents if want save this visualization in a new file .html.
    color : String, optional, default 'black'.
        Represents line colors of visualization.
        Can be a single color name, a list of colors or a colormap.
    filename : String, optional, default 'plot_trajectory_by_id_folium.html'.
        Represents the file name of new file .html.

    Returns
    -------
    folium.folium.Map.
        Represents a folium map with visualization.

    Raises
    ------
    IndexError
        If there is no user with the id passed

    """
    if base_map is None:
        base_map = create_base_map(
            move_data,
            lat_origin,
            lon_origin,
            tile=tile,
            default_zoom_start=zoom_start,
        )

    mv_df, items = _filter_and_generate_colors(move_data, id_, n_rows, color)
    _add_trajectories_to_folium_map(
        mv_df, items, base_map, legend, save_as_html, filename
    )

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
    color=None,
    filename='plot_trajectory_by_period_with_folium.html',
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
    legend: boolean, default True
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
    tile : String, optional, default 'CartoDB positron'.
        Represents the map'srs tiles.
    save_as_html : bool, optional, default False.
        Represents if want save this visualization in a new file .html.
    color : String or List, optional, default None.
        Represents line colors of visualization.
        Can be a single color name, a list of colors or a colormap.
    filename : String, optional, default 'plot_trajectory_by_period.html'.
        Represents the file name of new file .html.

    Returns
    -------
    folium.folium.Map.
        Represents a folium map with visualization.

    Raises
    ------
    KeyError
        If period value is not found in dataframe
    IndexError
        If there is no user with the id passed

    """
    if base_map is None:
        base_map = create_base_map(
            move_data,
            lat_origin,
            lon_origin,
            tile=tile,
            default_zoom_start=zoom_start,
        )

    if PERIOD not in move_data:
        move_data.generate_time_of_day_features()

    mv_df = _filter_generated_feature(move_data, PERIOD, [period])
    mv_df, items = _filter_and_generate_colors(mv_df, id_, n_rows, color)
    _add_trajectories_to_folium_map(
        mv_df, items, base_map, legend, save_as_html, filename
    )

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
    color=None,
    filename='plot_trajectory_by_day_week.html',
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
    legend: boolean, default True
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
    tile : String, optional, default 'CartoDB positron'.
        Represents the map'srs tiles.
    save_as_html : bool, optional, default False.
        Represents if want save this visualization in a new file .html.
    color : String or List, optional, default None.
        Represents line colors of visualization.
        Can be a single color name, a list of colors or a colormap.
    filename : String, optional, default 'plot_trajectory_by_day_week.html'.
        Represents the file name of new file .html.

    Returns
    -------
    folium.folium.Map.
        Represents a folium map with visualization.

    Raises
    ------
    KeyError
        If day_week value is not found in dataframe
    IndexError
        If there is no user with the id passed

    """
    if base_map is None:
        base_map = create_base_map(
            move_data,
            lat_origin,
            lon_origin,
            tile=tile,
            default_zoom_start=zoom_start,
        )

    if DAY not in move_data:
        move_data.generate_day_of_the_week_features()

    mv_df = _filter_generated_feature(move_data, DAY, [day_week])
    mv_df, items = _filter_and_generate_colors(mv_df, id_, n_rows, color)
    _add_trajectories_to_folium_map(
        mv_df, items, base_map, legend, save_as_html, filename
    )

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
    color=None,
    filename='plot_trajectory_by_date.html',
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
    legend: boolean, default True
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
    tile : String, optional, default 'CartoDB positron'.
        Represents the map'srs tiles.
    save_as_html : bool, optional, default False.
        Represents if want save this visualization in a new file .html.
    color : String or List, optional, default None.
        Represents line colors of visualization.
        Can be a single color name, a list of colors or a colormap.
    filename : String, optional, default 'plot_trejectory_with_folium.html'.
        Represents the file name of new file .html.

    Returns
    -------
    folium.folium.Map.
        Represents a folium map with visualization.

    Raises
    ------
    KeyError
        If start to end date range not found in dataframe
    IndexError
        If there is no user with the id passed

    """

    if base_map is None:
        base_map = create_base_map(
            move_data,
            lat_origin,
            lon_origin,
            tile=tile,
            default_zoom_start=zoom_start,
        )

    if isinstance(start_date, str):
        start_date = str_to_datetime(start_date).date()

    if isinstance(end_date, str):
        end_date = str_to_datetime(end_date).date()

    if DATE not in move_data:
        move_data.generate_date_features()

    mv_df = _filter_generated_feature(move_data, DATE, [start_date, end_date])
    mv_df, items = _filter_and_generate_colors(mv_df, id_, n_rows, color)
    _add_trajectories_to_folium_map(
        mv_df, items, base_map, legend, save_as_html, filename
    )

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
    color=None,
    filename='plot_trajectory_by_hour.html',
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
    legend: boolean, default True
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
    tile : String, optional, default 'CartoDB positron'.
        Represents the map'srs tiles.
    save_as_html : bool, optional, default False.
        Represents if want save this visualization in a new file .html.
    color : String or List, optional, default None.
        Represents line colors of visualization.
        Can be a single color name, a list of colors or a colormap.
    filename : String, optional, default 'plot_trajectory_by_hour.html'.
        Represents the file name of new file .html.

    Returns
    -------
    folium.folium.Map.
        Represents a folium map with visualization.

    Raises
    ------
    KeyError
        If start to end hour range not found in dataframe
    IndexError
        If there is no user with the id passed

    """
    if base_map is None:
        base_map = create_base_map(
            move_data,
            lat_origin,
            lon_origin,
            tile=tile,
            default_zoom_start=zoom_start,
        )

    if HOUR not in move_data:
        move_data.generate_hour_features()

    mv_df = _filter_generated_feature(move_data, HOUR, [start_hour, end_hour])
    mv_df, items = _filter_and_generate_colors(mv_df, id_, n_rows, color)
    _add_trajectories_to_folium_map(
        mv_df, items, base_map, legend, save_as_html, filename
    )

    return base_map


def plot_stops(
    move_data,
    radius=0,
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
    color=None,
    filename='plot_stops.html',
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
        radius, the segment is a stop, otherwise it'srs a move.
    weight: int or None
        Stroke width in pixels
    id_: int or None
        If int, plots trajectory of the user, else plot for all users
    legend: boolean, default True
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
    tile : String, optional, default 'CartoDB positron'.
        Represents the map'srs tiles.
    save_as_html : bool, optional, default False.
        Represents if want save this visualization in a new file .html.
    color : String or List, optional, default None.
        Represents line colors of visualization.
        Can be a single color name, a list of colors or a colormap.
    filename : String, optional, default 'plot_stops.html'.
        Represents the file name of new file .html.

    Returns
    -------
    folium.folium.Map.
        Represents a folium map with visualization.

    Raises
    ------
    KeyError
        If no STOPs found
    IndexError
        If there is no user with the id passed

    """

    if base_map is None:
        base_map = create_base_map(
            move_data,
            lat_origin,
            lon_origin,
            tile=tile,
            default_zoom_start=zoom_start,
        )

    if SITUATION not in move_data:
        move_data.generate_move_and_stop_by_radius(radius=radius)

    mv_df = _filter_generated_feature(move_data, SITUATION, STOP)
    mv_df, items = _filter_and_generate_colors(mv_df, id_, n_rows, color)

    for _id, color in items:
        for stop in mv_df[mv_df[TRAJ_ID] == _id].iterrows():
            base_map.add_child(
                folium.Circle(
                    (stop[1][LATITUDE], stop[1][LONGITUDE]),
                    color=color,
                    weight=weight,
                    radius=40,
                    opacity=0.5,
                    popup=stop[1][DATETIME],
                    fill_color=color,
                    fill_opacity=0.5,
                )
            )

    if legend:
        add_map_legend(base_map, 'Color by user ID', items)

    if save_as_html:
        base_map.save(outfile=filename)
    else:
        return base_map


def _format_tags(line, slice_):
    """
    Create or format tags.

    Parameters
    -----------
    line: Line to add a tag.

    slice_: Tag interval.

    Returns
    -------
    str: formatted html tag

    """
    map_formated_tags = map(lambda tag: '{}: {}'.format(tag, line[tag]), slice_)

    return '<br/>'.join(map_formated_tags)


def _circle_maker(
    iter_tuple,
    user_lat,
    user_lon,
    slice_tags,
    user_point,
    map_
):
    """
    Return a circle.

    Parameters
    -----------
    iter_tuple: DataFrame iter_tuple.
    user_lat: String.
        Latitude column name.
    user_lon: String.
        Longitude column name.
    slice_tags:

    user_point: String.
        Point color.
    map_: Folium map.
    """

    _, line = iter_tuple

    x = line[user_lat]
    y = line[user_lon]

    tags_formated = _format_tags(line, slice_tags)

    folium.Circle(
        radius=1,
        location=[x, y],
        popup=tags_formated,
        color=user_point,
        fill=False
    ).add_to(map_)


def plot_incial_end_points(
    list_rows,
    user_lat,
    user_lon,
    slice_tags,
    base_map,
    map_
):
    """
    Returns incial and end points.

    Parameters
    -----------
    list_rows: List of DataFrame iter_tuple.
    user_lat: String.
        Latitude column name.
    user_lon: String.
        Longitude column name.
    slice_tags:

    user_point: String.
        Point color.
    map_: Folium map.
    """

    # plot the start user_point
    line = list_rows[0][1]

    tags_formated = _format_tags(line, slice_tags)

    x = line[user_lat]
    y = line[user_lon]

    folium.Marker(
        location=[x, y],
        popup='START<br/>' + tags_formated,
        icon=folium.Icon(color='green')
    ).add_to(base_map)

    line = list_rows[-1][1]

    tags_formated = _format_tags(line, slice_tags)

    x = line[user_lat]
    y = line[user_lon]

    folium.Marker(
        location=[x, y],
        popup='END<br/>' + tags_formated,
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(base_map)


def add_traj_folium(
    move_data,
    user_lat=LATITUDE,
    user_lon=LONGITUDE,
    user_point=USER_POINT,
    line_color=LINE_COLOR,
    user_datetime=DATETIME,
    sort=False,
    base_map=None,
    slice_tags=None,
    tiles=TILES[0]
):
    """
    Receives a MoveDataFrame and returns a folium map with the trajectories plots.

    Parameters
    ----------
    move_data: Dataframe
        Trajectory data.
    user_lat: String, optional, default 'lat'.
        Latitude column name.
    user_lon: String, optional, default 'lon'.
        Longitude column name.
    user_point: String, optional, default 'purple'.
        User point color.
    line_color: String, optional, default 'blue'.
        Line color.
    user_datetime: String, optional, default 'datetime'.
        Datetime column name.
    sort:Boolean, optional, default False.
        If True the data will be sorted.
    base_map: Folium map, optional, default None.
        A folium map to plot the trajectories. If None a map will be created.
    slice_tags: optional, default None.
    tiles: string, optional, default 'OpenStreetMap'.
        The map type.

    Returns
    -------
        A folium map.
    """

    if not slice_tags:
        slice_tags = move_data.columns

    # If not have a map a map is create with mean to lat and lon
    if not base_map:
        initial_lat = move_data[user_lat].mean()
        initial_lon = move_data[user_lon].mean()
        base_map = create_base_map(
            move_data=move_data,
            lat_origin=initial_lat,
            lon_origin=initial_lon,
            tile=tiles
        )

    # if needs sort the data
    if sort:
        move_data.sort_values(user_datetime, inplace=True)

    # plot the lines
    user_points = list(zip(move_data[user_lat], move_data[user_lon]))

    folium.PolyLine(
        user_points,
        color=line_color,
        weight=2
    ).add_to(base_map)

    list(
        map(
            lambda x: _circle_maker(
                x,
                user_lat,
                user_lon,
                slice_tags,
                user_point,
                base_map
            ),
            move_data.iterrows()
        )
    )

    plot_incial_end_points(
        list(move_data.iterrows()),
        user_lat,
        user_lon,
        slice_tags,
        base_map,
        base_map
    )

    return base_map


def add_point_folium(
    move_data,
    user_lat=LATITUDE,
    user_lon=LONGITUDE,
    user_point=USER_POINT,
    poi_point=POI_POINT,
    base_map=None,
    slice_tags=None,
    tiles=TILES[0]
):
    """
    Receives a MoveDataFrame and returns a folium map with the trajectories plots
    and a point.

    Parameters
    ----------
    move_data: Dataframe
        Trajectory data.
    user_lat: String, optional, default 'lat'.
        Latitude column name.
    user_lon: String, optional, default 'lon'.
        Longitude column name.
    user_point: String, optional, default 'orange'.
        The point color.
    poi_point: String, optional, default 'red'.
        Poi point color.
    sort:Boolean, optional, default False.
        If True the data will be sorted.
    base_map: Folium map, optional, default None.
        A folium map to plot the trajectories. If None a map will be created.
    slice_tags: optional, default None.
    tiles: string, optional, default 'OpenStreetMap'.
        The map type.

    Returns
    -------
    A folium map.
    """

    if not slice_tags:
        slice_tags = move_data.columns

    # If not have a map a map is create with mean to lat and lon
    if not base_map:
        initial_lat = move_data[user_lat].mean()
        initial_lon = move_data[user_lon].mean()
        base_map = create_base_map(
            move_data=move_data,
            lat_origin=initial_lat,
            lon_origin=initial_lon,
            tile=tiles
        )

    list(
        map(
            lambda x: _circle_maker(
                x,
                user_lat,
                user_lon,
                slice_tags,
                user_point,
                base_map
            ),
            move_data.iterrows()
        )
    )

    return base_map


def add_poi_folium(
    move_data,
    poi_lat=LATITUDE,
    poi_lon=LONGITUDE,
    poi_point=POI_POINT,
    base_map=None,
    slice_tags=None
):

    """
    Receives a MoveDataFrame and returns a folium map with poi points.

    Parameters
    ----------
    move_data: DataFrame
        Trajectory input data
    poi_lat: String, optional, default 'lat'.
        Latitude column name.
    poi_lon: String, optional, default 'lon'.
        Longitude column name.
    poi_point: String, optional, default 'red'.
        Poi point color.
    base_map: Folium map, optional, default None.
        A folium map to plot. If None a map. If None a map will be created.

    Returns
    -------
    folium.folium.Map.
        Represents a folium map with visualization.
    """

    if not slice_tags:
        slice_tags = move_data.columns

    # If not have a map a map is create with mean to lat and lon
    if not base_map:
        initial_lat = move_data[poi_lat].mean()
        initial_lon = move_data[poi_lon].mean()
        base_map = create_base_map(
            move_data=move_data,
            lat_origin=initial_lat,
            lon_origin=initial_lon
        )

    list(
        map(
            lambda x: _circle_maker(
                x,
                poi_lat,
                poi_lon,
                slice_tags,
                poi_point,
                base_map
            ),
            move_data.iterrows()
        )
    )

    return base_map


def add_event_folium(
    move_data,
    event_lat=LATITUDE,
    event_lon=LONGITUDE,
    event_point=EVENT_POINT,
    radius=150,
    base_map=None,
    slice_tags=None,
    tiles=TILES[0]
):

    """
    Receives a MoveDataFrame and returns a folium map with events.

    Parameters
    ----------
    move_data: DataFrame
        Trajectory input data
    event_lat: String, optional, default 'lat'.
        Latitude column name.
    event_lon: String, optional, default 'lon'.
        Longitude column name.
    event_point: String, optional, default 'red'.
        Event color.
    radius: Float, optional, default 150.
        radius size.
    base_map: Folium map, optional, default None.
        A folium map to plot. If None a map. If None a map will be created.
    tiles: string, optional, default 'OpenStreetMap'

    Returns
    -------
    A folium map.
    """
    if not slice_tags:
        slice_tags = move_data.columns

    # If not have a map a map is create with mean to lat and lon
    if not base_map:
        initial_lat = move_data[event_lat].mean()
        initial_lon = move_data[event_lon].mean()
        base_map = create_base_map(
            move_data=move_data,
            lat_origin=initial_lat,
            lon_origin=initial_lon,
            tile=tiles
        )

    list(
        map(
            lambda x: _circle_maker(
                x,
                event_lat,
                event_lon,
                slice_tags,
                event_point,
                base_map
            ),
            move_data.iterrows()
        )
    )

    return base_map


def show_trajs_with_event(
    move_data,
    window_time_subject,
    df_event,
    window_time_event,
    radius,
    event_lat_=LATITUDE,
    event_lon_=LONGITUDE,
    event_datetime_=DATETIME,
    user_lat=LATITUDE,
    user_lon=LONGITUDE,
    user_datetime=DATETIME,
    event_id_=EVENT_ID,
    event_point=EVENT_POINT,
    user_id=UID,
    user_point=USER_POINT,
    line_color=LINE_COLOR,
    slice_event_show=None,
    slice_subject_show=None,
):
    """
    Plot a trajectory, including your user_points lat lon and your tags.

    Parameters
    -----------
    move_data: DataFrame.
        Trajectory input data.
    window_time_subject: float.
        The subject time window.
    Window_time_event: float.
        The event time window.
    radius: float.
        The radius to use.
    event_lat_: String, optional, default 'lat'.
        Event latitude column name.
    event_lon_: String, optional, default 'lon'.
        Event longitude column name.
    event_datetime_: String, optional, default 'datetime'.
        Event datetime column name.
    user_lat: String, optional, default 'lat'.
        User latitude column name.
    user_lon: String, optional, default 'lon'.
        User longitude column name.
    user_datetime: String, optional, default 'datetime'.
        User datetime column name.
    event_id_: String, optional, default 'id'.
        Event id column name.
    event_point: String, optional, default 'red'.
        Event color.
    user_id: String, optional, default 'id'.
        User id column name.
    user_point: String, optional, default 'orange'.
        User point color.
    line_color: String, optional, default 'blue'.
        Line color.
    slice_event_show: int, optional, default None.
    slice_subject_show: int, optional, default

    Returns
    -------
    A list of folium maps.
    """

    # building structure for deltas
    delta_event = pd.to_timedelta(window_time_event, unit='s')
    delta_user = pd.to_timedelta(window_time_subject, unit='s')

    # length of df_user
    len_df_user = move_data.shape[0]

    # building structure for lat and lon array
    lat_arr = np.zeros(len_df_user)
    lon_arr = np.zeros(len_df_user)

    # folium map list
    folium_maps = []

    # for each event in df_event
    for _, line in df_event.iterrows():

        event_lat = line[event_lat_]

        event_lon = line[event_lon_]

        event_datetime = line[event_datetime_]

        event_id = line[event_id_]

        # building time window for event search
        start_time = pd.to_datetime(event_datetime - delta_event)
        end_time = pd.to_datetime(event_datetime + delta_event)

        # filtering df_ for time window
        df_filtered = filters.by_datetime(
            move_data,
            start_datetime=start_time,
            end_datetime=end_time
        )

        # length of df_temp
        len_df_temp = df_filtered.shape[0]

        # using the util part of the array for haversine function
        lat_arr[:len_df_temp] = event_lat
        lon_arr[:len_df_temp] = event_lon

        # building distances to event column
        df_filtered['distances'] = distances.haversine(
            lat_arr[:len_df_temp],
            lon_arr[:len_df_temp],
            df_filtered[user_lat].values,
            df_filtered[user_lon].values
        )

        # building nearby column
        df_filtered['nearby'] = df_filtered['distances'].map(lambda x: (x <= radius))

        # if any data for df_ in event time window is True
        if df_filtered['nearby'].any():

            # building the df for the first user_points of user in nearby event
            df_begin = df_filtered[df_filtered['nearby']].sort_values(
                user_datetime
            )

            move_data = df_event[df_event[event_id_] == event_id]

            base_map = add_event_folium(
                move_data,
                event_lat=event_lat_,
                event_lon=event_lon_,
                event_point=event_point,
                slice_tags=slice_event_show
            )

            # keep only the first user_point nearby to event for each user
            df_begin.drop_duplicates(
                subset=[user_id, 'nearby'],
                inplace=True
            )

            # for each user nearby to event
            users = []

            for time_user, id_user in zip(
                df_begin[user_datetime],
                df_begin[user_id]
            ):

                # making the time window for user
                start_time = pd.to_datetime(time_user - delta_user)
                end_time = pd.to_datetime(time_user + delta_user)

                # building the df for one id
                df_id = move_data[move_data[user_id] == id_user]

                # filtering df_id for time window
                df_temp = filters.by_datetime(
                    df_id,
                    start_datetime=start_time,
                    end_datetime=end_time
                )

                users.append(df_temp)
                # add to folium map created
                add_traj_folium(
                    df_temp,
                    user_lat=user_lat,
                    user_lon=user_lon,
                    user_point=user_point,
                    line_color=line_color,
                    base_map=base_map,
                    slice_tags=slice_subject_show,
                    sort=True
                )

            # add to folium maps list: (id event, folium map, quantity of user in map, df)
            folium_maps.append((base_map, pd.concat(users)))

    return folium_maps


def show_traj_id_with_event(
    move_data,
    window_time_subject,
    df_event,
    window_time_event,
    radius,
    subject_id,
    event_lat_=LATITUDE,
    event_lon_=LONGITUDE,
    event_datetime_=DATETIME,
    user_lat=LATITUDE,
    user_lon=LONGITUDE,
    user_datetime=DATETIME,
    event_id_=EVENT_ID,
    event_point=EVENT_POINT,
    user_id=UID,
    user_point=USER_POINT,
    line_color=LINE_COLOR,
    slice_event_show=None,
    slice_subject_show=None
):
    """
    Plot a trajectory, including your user_points lat lon and your tags.

    Parameters
    -----------
    move_data: DataFrame.
        Trajectory input data.
    window_time_subject: float.
        The subject time window.
    Window_time_event: float.
        The event time window.
    radius: float.
        The radius to use.
    event_lat_: String, optional, default 'lat'.
        Event latitude column name.
    event_lon_: String, optional, default 'lon'.
        Event longitude column name.
    event_datetime_: String, optional, default 'datetime'.
        Event datetime column name.
    user_lat: String, optional, default 'lat'.
        User latitude column name.
    user_lon: String, optional, default 'lon'.
        User longitude column name.
    user_datetime: String, optional, default 'datetime'.
        User datetime column name.
    event_id_: String, optional, default 'id'.
        Event id column name.
    event_point: String, optional, default 'red'.
        Event color.
    user_id: String, optional, default 'id'.
        User id column name.
    user_point: String, optional, default 'orange'.
        User point color.
    line_color: String, optional, default 'blue'.
        Line color.
    slice_envet_show: int, optional, default None.

    slice_subject_show: int, optional, default.

    Returns
    -------
    A folium map.
    """

    df_id = move_data[move_data[user_id] == subject_id]

    return show_trajs_with_event(
        df_id,
        window_time_subject,
        df_event,
        window_time_event,
        radius,
        event_lat_=event_lat_,
        event_lon_=event_lon_,
        event_datetime_=event_datetime_,
        user_lat=user_lat,
        user_lon=user_lon,
        user_datetime=user_datetime,
        event_id_=event_id_,
        event_point=event_point,
        user_id=user_id,
        user_point=user_point,
        line_color=line_color,
        slice_event_show=slice_event_show,
        slice_subject_show=slice_subject_show
    )[0]


def _create_geojson_features_line(
    move_data,
    label_lat=LATITUDE,
    label_lon=LONGITUDE,
    label_datetime=DATETIME
):
    """
    Create geojson features.

    Parameters
    -----------
    move_data: DataFrame.
        Input trajectory data.
    label_datetime: string, optional, default 'datetime'.
        date_time colum label.
    label_lat: string, optional, default 'lat'.
        latitude column label.
    label_lon: string, optional, default 'long'.
        longitude column label.

    Returns
    -------
    GeoJSON features.
    """
    print('> Creating GeoJSON features...')
    features = []

    row_iterator = move_data.iterrows()
    _, last = next(row_iterator)
    columns = move_data.columns

    for i, row in progress_bar(row_iterator, total=move_data.shape[0] - 1) :
        last_time = last[label_datetime].strftime('%Y-%m-%dT%H:%M:%S')
        next_time = row[label_datetime].strftime('%Y-%m-%dT%H:%M:%S')

        popup_list = [i + ': ' + str(last[i]) for i in columns]
        popup1 = '<br>'.join(popup_list)

        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'LineString',
                'coordinates': [
                    [last['lon'], last['lat']],
                    [row['lon'], row['lat']]
                ]

            },
            'properties': {
                'times': [last_time, next_time],
                'popup': popup1,
                'style': {
                    'color': 'red',
                    'icon': 'circle',
                    'iconstyle': {
                        'color': 'red',
                        'weight': 4
                    }
                }
            }
        }
        _, last = i, row

        features.append(feature)

    return features


def plot_traj_timestamp_geo_json(
    move_data,
    label_datetime=DATETIME,
    label_lat=LATITUDE,
    label_lon=LONGITUDE,
    tiles=TILES[0]
):
    """
    Plot trajectories wit geo_json.

    Parameters
    -----------
    move_data: DataFrame.
        Input trajectory data.
    label_datetime: string, optional, default 'datetime'.
        date_time colunm label.
    label_lat: string, optional, default 'lat'.
        latitude column label.
    label_lon: string, optional, default 'long'.
        longitude column label.
    tiles: string, optional, default 'cartodbpositron'.
        folium tiles.

    Returns
    -------
    A folium map.
    """
    features = _create_geojson_features_line(move_data, label_datetime)
    print('creating folium map')
    map_ = create_base_map(
        move_data=move_data,
        lat_origin=move_data[label_lat].mean(),
        lon_origin=move_data[label_lon].mean(),
        tile=tiles
    )
    print('Genering timestamp map')
    plugins.TimestampedGeoJson(
        {
            'type': 'FeatureCollection',
            'features': features,
        },
        period='PT1M',
        add_last_point=True
    ).add_to(map_)
    return map_
