from typing import Any, Dict, List, Optional, Text, Tuple, Union

import folium
import numpy as np
import pandas as pd
from folium import Map, plugins
from folium.plugins import FastMarkerCluster, HeatMap, HeatMapWithTime, MarkerCluster
from pandas import DataFrame

from pymove.preprocessing import filters
from pymove.utils import distances
from pymove.utils.constants import (
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
    move_data: DataFrame,
    filename: Text,
    tiles: Optional[Text] = TILES[0],
    label_id: Optional[Text] = TRAJ_ID,
    cmap: Optional[Text] = 'Set1',
    return_map: Optional[bool] = False
) -> Optional[Map]:
    """
    Save a visualization in a map in a new file

    Parameters
    ----------
    move_data : DataFrame
        Input trajectory data
    filename : Text
        Represents the filename path
    tiles : str, optional
        Represents the type_ of tile that will be used on the map, by default TILES[0]
    label_id : str, optional
        Represents column name of trajectory id, by default TRAJ_ID
    cmap : str, optional
        Color map to use, by default 'Set1'
    return_map : bool, optional
        Represents the Colormap, by default False

    Returns
    -------
    Map
        folium map or None
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
    move_data: DataFrame,
    lat_origin: Optional[float] = None,
    lon_origin: Optional[float] = None,
    tile: Optional[Text] = TILES[0],
    default_zoom_start: Optional[float] = 12,
) -> Map:
    """
    Generate a folium map

    Parameters
    ----------
    move_data : DataFrame
        Input trajectory data
    lat_origin : float, optional
        Represents the latitude which will be the center of the map, by default None
    lon_origin : float, optional
        Represents the longitude which will be the center of the map, by default None
    tile : str, optional
        Represents the map tiles, by default TILES[0]
    default_zoom_start : float, optional
        Represents the zoom which will be the center of the map, by default 12

    Returns
    -------
    Map
        a folium map
    """

    if lat_origin is None and lon_origin is None:
        lat_origin = move_data[LATITUDE].median()
        lon_origin = move_data[LONGITUDE].median()
    base_map = folium.Map(
        location=[lat_origin, lon_origin],
        control_scale=True,
        zoom_start=default_zoom_start,
        tiles=tile
    )
    return base_map


def heatmap(
    move_data: DataFrame,
    n_rows: Optional[int] = None,
    lat_origin: Optional[float] = None,
    lon_origin: Optional[float] = None,
    zoom_start: Optional[float] = 12,
    radius: Optional[float] = 8,
    base_map: Optional[Map] = None,
    tile: Optional[Text] = TILES[0],
    save_as_html: Optional[bool] = False,
    filename: Optional[Text] = 'heatmap.html',
) -> Map:
    """
    Generate visualization of Heat Map using folium plugin.

    Parameters
    ----------
    move_data : DataFrame
        Input trajectory data
    n_rows : int, optional
        Represents number of data rows that are will plot, by default None
    lat_origin : float, optional
        Represents the latitude which will be the center of the map, by default None
    lon_origin : float, optional
        Represents the longitude which will be the center of the map, by default None
    zoom_start : float, optional
        Initial zoom level for the map, by default 12
    radius : float, optional
        Radius of each “point” of the heatmap, by default 8
    base_map : Map, optional
        Represents the folium map. If not informed, a new map is generated
        using the function create_base_map(), with the lat_origin, lon_origin
        and zoom_start, by default None
    tile : str, optional
        Represents the map tiles, by default TILES[0]
    save_as_html : bool, optional
        Represents if want save this visualization in a new file .html, by default False
    filename : str, optional
        Represents the file name of new file .html, by default 'heatmap.html'

    Returns
    -------
    Map
        folium Map
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
    return base_map


def heatmap_with_time(
    move_data: DataFrame,
    n_rows: Optional[int] = None,
    lat_origin: Optional[float] = None,
    lon_origin: Optional[float] = None,
    zoom_start: Optional[float] = 12,
    radius: Optional[float] = 8,
    min_opacity: Optional[float] = 0.5,
    max_opacity: Optional[float] = 0.8,
    base_map: Optional[Map] = None,
    tile: Optional[Text] = TILES[0],
    save_as_html: Optional[bool] = False,
    filename: Optional[Text] = 'heatmap_time.html',
) -> Map:
    """
    Generate visualization of Heat Map using folium plugin.

    Parameters
    ----------
    move_data : DataFrame
        Input trajectory data
    n_rows : int, optional
        Represents number of data rows that are will plot, by default None
    lat_origin : float, optional
        Represents the latitude which will be the center of the map, by default None
    lon_origin : float, optional
        Represents the longitude which will be the center of the map, by default None
    zoom_start : float, optional
        Initial zoom level for the map, by default 12
    radius : float, optional
        Radius of each “point” of the heatmap, by default 8
    min_opacity: float, optional
        Minimum heat opacity, by default 0.5.
    max_opacity: float, optional
        Maximum heat opacity, by default 0.8.
    base_map : Map, optional
        Represents the folium map. If not informed, a new map is generated
        using the function create_base_map(), with the lat_origin, lon_origin
        and zoom_start, by default None
    tile : str, optional
        Represents the map tiles, by default TILES[0]
    save_as_html : bool, optional
        Represents if want save this visualization in a new file .html, by default False
    filename : str, optional
        Represents the file name of new file .html, by default 'heatmap_time.html'

    Returns
    -------
    Map
        folium Map
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
        gradient={0.2: 'blue', 0.4: 'lime', 0.6: USER_POINT, 1: 'red'},
        min_opacity=min_opacity,
        max_opacity=max_opacity,
        use_local_extrema=True
    ).add_to(base_map)
    move_data.drop(columns=[COUNT, HOUR], inplace=True)

    if save_as_html:
        base_map.save(outfile=filename)
    return base_map


def cluster(
    move_data: DataFrame,
    n_rows: Optional[int] = None,
    lat_origin: Optional[float] = None,
    lon_origin: Optional[float] = None,
    zoom_start: Optional[float] = 12,
    base_map: Optional[Map] = None,
    tile: Optional[Text] = TILES[0],
    save_as_html: Optional[bool] = False,
    filename: Optional[Text] = 'cluster.html',
) -> Map:
    """
    Generate visualization of Heat Map using folium plugin.

    Parameters
    ----------
    move_data : DataFrame
        Input trajectory data
    n_rows : int, optional
        Represents number of data rows that are will plot, by default None
    lat_origin : float, optional
        Represents the latitude which will be the center of the map, by default None
    lon_origin : float, optional
        Represents the longitude which will be the center of the map, by default None
    zoom_start : float, optional
        Initial zoom level for the map, by default 12
    radius : float, optional
        Radius of each “point” of the heatmap, by default 8
    base_map : Map, optional
        Represents the folium map. If not informed, a new map is generated
        using the function create_base_map(), with the lat_origin, lon_origin
        and zoom_start, by default None
    tile : str, optional
        Represents the map tiles, by default TILES[0]
    save_as_html : bool, optional
        Represents if want save this visualization in a new file .html, by default False
    filename : str, optional
        Represents the file name of new file .html, by default 'cluster.html'

    Returns
    -------
    Map
        folium Map
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
    return base_map


def faster_cluster(
    move_data: DataFrame,
    n_rows: Optional[int] = None,
    lat_origin: Optional[float] = None,
    lon_origin: Optional[float] = None,
    zoom_start: Optional[float] = 12,
    base_map: Optional[Map] = None,
    tile: Optional[Text] = TILES[0],
    save_as_html: Optional[bool] = False,
    filename: Optional[Text] = 'faster_cluster.html',
) -> Map:
    """
    Generate visualization of Heat Map using folium plugin.

    Parameters
    ----------
    move_data : DataFrame
        Input trajectory data
    n_rows : int, optional
        Represents number of data rows that are will plot, by default None
    lat_origin : float, optional
        Represents the latitude which will be the center of the map, by default None
    lon_origin : float, optional
        Represents the longitude which will be the center of the map, by default None
    zoom_start : float, optional
        Initial zoom level for the map, by default 12
    radius : float, optional
        Radius of each “point” of the heatmap, by default 8
    base_map : Map, optional
        Represents the folium map. If not informed, a new map is generated
        using the function create_base_map(), with the lat_origin, lon_origin
        and zoom_start, by default None
    tile : str, optional
        Represents the map tiles, by default TILES[0]
    save_as_html : bool, optional
        Represents if want save this visualization in a new file .html, by default False
    filename : str, optional
        Represents the file name of new file .html, by default 'faster_cluster.html'

    Returns
    -------
    Map
        folium Map
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
    return base_map


def plot_markers(
    move_data: DataFrame,
    n_rows: Optional[int] = None,
    lat_origin: Optional[float] = None,
    lon_origin: Optional[float] = None,
    zoom_start: Optional[float] = 12,
    base_map: Optional[Map] = None,
    tile: Optional[Text] = TILES[0],
    save_as_html: Optional[bool] = False,
    filename: Optional[Text] = 'markers.html',
) -> Map:
    """
    Generate visualization of Heat Map using folium plugin.

    Parameters
    ----------
    move_data : DataFrame
        Input trajectory data
    n_rows : int, optional
        Represents number of data rows that are will plot, by default None
    lat_origin : float, optional
        Represents the latitude which will be the center of the map, by default None
    lon_origin : float, optional
        Represents the longitude which will be the center of the map, by default None
    zoom_start : float, optional
        Initial zoom level for the map, by default 12
    radius : float, optional
        Radius of each “point” of the heatmap, by default 8
    base_map : Map, optional
        Represents the folium map. If not informed, a new map is generated
        using the function create_base_map(), with the lat_origin, lon_origin
        and zoom_start, by default None
    tile : str, optional
        Represents the map tiles, by default TILES[0]
    save_as_html : bool, optional
        Represents if want save this visualization in a new file .html, by default False
    filename : str, optional
        Represents the file name of new file .html, by default 'markers.html'

    Returns
    -------
    Map
        folium Map
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

    for i, row in enumerate(move_data.iloc[:n_rows].iterrows()):
        if i == 0:
            se = '<b>START</b>\n'
            color = 'green'
        elif i == n_rows - 1:
            se = '<b>END\ns</b>\n'
            color = 'red'
        else:
            se = ''
            color = 'blue'
        pop = (
            se
            + '<b>Latitude:</b> '
            + str(row[1][LATITUDE])
            + '\n<b>Longitude:</b> '
            + str(row[1][LONGITUDE])
            + '\n<b>Datetime:</b> '
            + str(row[1][DATETIME])
        )
        folium.Marker(
            location=[row[1][LATITUDE], row[1][LONGITUDE]],
            color=color,
            clustered_marker=True,
            popup=pop,
            icon=folium.Icon(color=color)
        ).add_to(base_map)

    if save_as_html:
        base_map.save(outfile=filename)
    return base_map


def _filter_and_generate_colors(
    move_data: DataFrame,
    id_: Optional[int] = None,
    n_rows: Optional[int] = None,
    color: Optional[Text] = None,
    color_by_id: Optional[Dict] = None
) -> Tuple[DataFrame, List[Tuple]]:
    """
    Filters the dataframe and generate colors for folium map.

    Parameters
    ----------
    move_data : DataFrame
        Input trajectory data.
    id_: int, optional
        The TRAJ_ID's to be plotted, by default None
    n_rows : int, optional
        Represents number of data rows that are will plot, by default None.
    color: str, optional
        The color of the trajectory, of each trajectory or a colormap, by default None
    color_by_id: dict, optional
        A dictionary where the key is the trajectory id and value is a color(str),
        by default None.

    Returns
    -------
    DataFrame
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
            try:
                cmap_ = get_cmap(color)
                num = cmap_.N
                colors = [
                    cmap_hex_color(cmap_, (i % num))
                    for i, _ in enumerate(ids)
                ]
                diff = (len(ids) // len(colors)) + 1
                colors *= diff
            except ValueError:
                colors = [color]
        else:
            colors = color[:]
        items = [*zip(ids, colors)]
        if color_by_id is not None:
            keys = color_by_id.keys()
            for key in keys:
                for count, item in enumerate(items):
                    if str(key) == str(item[0]):
                        items[count] = (item[0], color_by_id[key])
    return mv_df, items


def _filter_generated_feature(
    move_data: DataFrame, feature: Text, values: Any
) -> DataFrame:
    """
    Filters the values from the dataframe.

    Parameters
    __________
    move_data : DataFrame
        Input trajectory data.
    feature: str
        Name of the feature
    value: any
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


def _add_begin_end_markers_to_folium_map(
    move_data: DataFrame,
    base_map: Map,
    color: Optional[Text] = None,
    _id: Optional[int] = None
):
    """
    Adds a green marker to beginning of the trajectory and a red marker to the
    end of the trajectory.

    Parameters
    ----------
    move_data : DataFrane
        Input trajectory data.
    base_map : Map, optional
        Represents the folium map. If not informed, a new map is generated.
    color : str, optional
        Color of the markers, by default None
    id: int, optional
        Id of the trajectory, by default None
    """

    points = folium.map.FeatureGroup(
        'The start and end points of trajectory {}'.format(_id or '')
    )

    folium.Marker(
        location=[move_data.iloc[0][LATITUDE], move_data.iloc[0][LONGITUDE]],
        color='green',
        clustered_marker=True,
        popup='Início',
        icon=plugins.BeautifyIcon(
            icon='play', icon_shape='marker', background_color=color or 'green'
        )
    ).add_to(points)

    folium.Marker(
        location=[move_data.iloc[-1][LATITUDE], move_data.iloc[-1][LONGITUDE]],
        color='red',
        clustered_marker=True,
        popup='Fim',
        icon=plugins.BeautifyIcon(
            icon='times-circle', icon_shape='marker', background_color=color or 'red'
        )
    ).add_to(points)

    base_map.add_child(points)


def _add_trajectories_to_folium_map(
    move_data: DataFrame,
    items: Tuple,
    base_map: Map,
    legend: Optional[bool] = True,
    save_as_html: Optional[bool] = True,
    filename: Optional[Text] = 'map.html',
):
    """
    Adds a trajectory to a folium map with begin and end markers.

    Parameters
    ----------
    move_data : DataFrame
        Input trajectory data.
    base_map : Map
        Represents the folium map. If not informed, a new map is generated.
    legend: bool
        Whether to add a legend to the map, by default True
    save_as_html : bool, optional
        Represents if want save this visualization in a new file .html, by default False.
    filename : str, optional
        Represents the file name of new file .html, by default 'map.html'.

    """

    for _id, color in items:
        mv = move_data[move_data[TRAJ_ID] == _id]

        _add_begin_end_markers_to_folium_map(mv, base_map, color, _id)

        folium.PolyLine(
            mv[[LATITUDE, LONGITUDE]], color=color, weight=2.5, opacity=1
        ).add_to(base_map)

    if legend:
        add_map_legend(base_map, 'Color by user ID', items)

    folium.map.LayerControl().add_to(base_map)

    if save_as_html:
        base_map.save(outfile=filename)


def plot_trajectories_with_folium(
    move_data: DataFrame,
    n_rows: Optional[int] = None,
    lat_origin: Optional[float] = None,
    lon_origin: Optional[float] = None,
    zoom_start: Optional[float] = 12,
    legend: Optional[bool] = True,
    base_map: Optional[Map] = None,
    tile: Optional[Text] = TILES[0],
    save_as_html: Optional[bool] = False,
    color: Optional[Union[Text, List[Text]]] = None,
    color_by_id: Optional[Dict] = None,
    filename: Optional[Text] = 'plot_trajectories_with_folium.html',
) -> Map:
    """
    Generate visualization of all trajectories with folium.

    Parameters
    ----------
    move_data : DataFrame
        Input trajectory data.
    n_rows : int, optional
        Represents number of data rows that are will plot, by default None.
    lat_origin : float, optional
        Represents the latitude which will be the center of the map.
        If not entered, the first data from the dataset is used, by default None.
    lon_origin : float, optional
        Represents the longitude which will be the center of the map.
        If not entered, the first data from the dataset is used, by default None.
    zoom_start : int, optional
        Initial zoom level for the map, by default 12.
    legend: boolean
        Whether to add a legend to the map, by default True
    base_map : folium.folium.Map, optional
        Represents the folium map. If not informed, a new map is generated
        using the function create_base_map(), with the lat_origin, lon_origin
         and zoom_start, by default None.
    tile : str, optional
        Represents the map tiles, by default TILES[0]
    save_as_html : bool, optional
        Represents if want save this visualization in a new file .html, by default False.
    color : str, list, optional
        Represents line colors of visualization.
        Can be a single color name, a list of colors or a colormap name, by default None.
    color_by_id: dict, optional
        A dictionary where the key is the trajectory id and value is a color(str),
        by default None.
    filename : str, optional
        Represents the file name of new file .html,
        by default 'plot_trajectory_with_folium.html'.

    Returns
    -------
    Map
        a folium map with visualization.

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
        move_data, n_rows=n_rows, color=color, color_by_id=color_by_id
    )

    _add_trajectories_to_folium_map(
        mv_df, items, base_map, legend, save_as_html, filename
    )

    return base_map


def plot_trajectory_by_id_folium(
    move_data: DataFrame,
    id_: int,
    n_rows: Optional[int] = None,
    lat_origin: Optional[float] = None,
    lon_origin: Optional[float] = None,
    zoom_start: Optional[float] = 12,
    legend: Optional[bool] = True,
    base_map: Optional[Map] = None,
    tile: Optional[Text] = TILES[0],
    save_as_html: Optional[bool] = False,
    color: Optional[Union[Text, List[Text]]] = None,
    filename: Optional[Text] = 'plot_trajectories_with_folium.html',
) -> Map:
    """
    Generate visualization of all trajectories with folium.

    Parameters
    ----------
    move_data : DataFrame
        Input trajectory data
    id_: int
        Trajectory id to plot
    n_rows : int, optional
        Represents number of data rows that are will plot, by default None.
    lat_origin : float, optional
        Represents the latitude which will be the center of the map.
        If not entered, the first data from the dataset is used, by default None.
    lon_origin : float, optional
        Represents the longitude which will be the center of the map.
        If not entered, the first data from the dataset is used, by default None.
    zoom_start : int, optional
        Initial zoom level for the map, by default 12.
    legend: boolean
        Whether to add a legend to the map, by default True
    base_map : folium.folium.Map, optional
        Represents the folium map. If not informed, a new map is generated
        using the function create_base_map(), with the lat_origin, lon_origin
         and zoom_start, by default None.
    tile : str, optional
        Represents the map tiles, by default TILES[0]
    save_as_html : bool, optional
        Represents if want save this visualization in a new file .html, by default False.
    color : str, list, optional
        Represents line colors of visualization.
        Can be a single color name, a list of colors or a colormap name, by default None.
    filename : str, optional
        Represents the file name of new file .html,
        by default 'plot_trajectory_by_id_with_folium.html'.

    Returns
    -------
    Map
        a folium map with visualization

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
    move_data: DataFrame,
    period: Text,
    id_: Optional[int] = None,
    n_rows: Optional[int] = None,
    lat_origin: Optional[float] = None,
    lon_origin: Optional[float] = None,
    zoom_start: Optional[float] = 12,
    legend: Optional[bool] = True,
    base_map: Optional[Map] = None,
    tile: Optional[Text] = TILES[0],
    save_as_html: Optional[bool] = False,
    color: Optional[Union[Text, List[Text]]] = None,
    color_by_id: Optional[Dict] = None,
    filename: Optional[Text] = 'plot_trajectories_by_period.html',
) -> Map:
    """
    Generate visualization of all trajectories with folium.

    Parameters
    ----------
    move_data : DataFrame
        Input trajectory data
    period: str
        Period of the day
    id_: int
        Trajectory id to plot, by default None
    n_rows : int, optional
        Represents number of data rows that are will plot, by default None.
    lat_origin : float, optional
        Represents the latitude which will be the center of the map.
        If not entered, the first data from the dataset is used, by default None.
    lon_origin : float, optional
        Represents the longitude which will be the center of the map.
        If not entered, the first data from the dataset is used, by default None.
    zoom_start : int, optional
        Initial zoom level for the map, by default 12.
    legend: boolean
        Whether to add a legend to the map, by default True
    base_map : folium.folium.Map, optional
        Represents the folium map. If not informed, a new map is generated
        using the function create_base_map(), with the lat_origin, lon_origin
         and zoom_start, by default None.
    tile : str, optional
        Represents the map tiles, by default TILES[0]
    save_as_html : bool, optional
        Represents if want save this visualization in a new file .html, by default False.
    color : str, list, optional
        Represents line colors of visualization.
        Can be a single color name, a list of colors or a colormap name, by default None.
    color_by_id: dict, optional
        A dictionary where the key is the trajectory id and value is a color,
        by default None.
    filename : str, optional
        Represents the file name of new file .html,
        by default 'plot_trajectories_by_period.html'.

    Returns
    -------
    Map
        a folium map with visualization

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
    mv_df, items = _filter_and_generate_colors(mv_df, id_, n_rows, color, color_by_id)
    _add_trajectories_to_folium_map(
        mv_df, items, base_map, legend, save_as_html, filename
    )

    return base_map


def plot_trajectory_by_day_week(
    move_data: DataFrame,
    day_week: Text,
    id_: Optional[int] = None,
    n_rows: Optional[int] = None,
    lat_origin: Optional[float] = None,
    lon_origin: Optional[float] = None,
    zoom_start: Optional[float] = 12,
    legend: Optional[bool] = True,
    base_map: Optional[Map] = None,
    tile: Optional[Text] = TILES[0],
    save_as_html: Optional[bool] = False,
    color: Optional[Union[Text, List[Text]]] = None,
    color_by_id: Optional[Dict] = None,
    filename: Optional[Text] = 'plot_trajectories_by_day_week.html',
) -> Map:
    """
    Generate visualization of all trajectories with folium.

    Parameters
    ----------
    move_data : DataFrame
        Input trajectory data
    day_week: str
        Day of the week
    id_: int
        Trajectory id to plot, by default None
    n_rows : int, optional
        Represents number of data rows that are will plot, by default None.
    lat_origin : float, optional
        Represents the latitude which will be the center of the map.
        If not entered, the first data from the dataset is used, by default None.
    lon_origin : float, optional
        Represents the longitude which will be the center of the map.
        If not entered, the first data from the dataset is used, by default None.
    zoom_start : int, optional
        Initial zoom level for the map, by default 12.
    legend: boolean
        Whether to add a legend to the map, by default True
    base_map : folium.folium.Map, optional
        Represents the folium map. If not informed, a new map is generated
        using the function create_base_map(), with the lat_origin, lon_origin
         and zoom_start, by default None.
    tile : str, optional
        Represents the map tiles, by default TILES[0]
    save_as_html : bool, optional
        Represents if want save this visualization in a new file .html, by default False.
    color : str, list, optional
        Represents line colors of visualization.
        Can be a single color name, a list of colors or a colormap name, by default None.
    color_by_id: dict, optional
        A dictionary where the key is the trajectory id and value is a color,
        by default None.
    filename : str, optional
        Represents the file name of new file .html,
        by default 'plot_trajectories_by_day_week.html'.

    Returns
    -------
    Map
        a folium map with visualization

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

    if DAY not in move_data:
        move_data.generate_day_of_the_week_features()

    mv_df = _filter_generated_feature(move_data, DAY, [day_week])
    mv_df, items = _filter_and_generate_colors(mv_df, id_, n_rows, color, color_by_id)
    _add_trajectories_to_folium_map(
        mv_df, items, base_map, legend, save_as_html, filename
    )

    return base_map


def plot_trajectory_by_date(
    move_data: DataFrame,
    start_date: Text,
    end_date: Text,
    id_: Optional[int] = None,
    n_rows: Optional[int] = None,
    lat_origin: Optional[float] = None,
    lon_origin: Optional[float] = None,
    zoom_start: Optional[float] = 12,
    legend: Optional[bool] = True,
    base_map: Optional[Map] = None,
    tile: Optional[Text] = TILES[0],
    save_as_html: Optional[bool] = False,
    color: Optional[Union[Text, List[Text]]] = None,
    color_by_id: Optional[Dict] = None,
    filename: Optional[Text] = 'plot_trajectories_by_date.html',
) -> Map:
    """
    Generate visualization of all trajectories with folium.

    Parameters
    ----------
    move_data : DataFrame
        Input trajectory data
    start_date : str
        Represents start date of time period.
    end_date : str
        Represents end date of time period.
    id_: int, optional
        Trajectory id to plot, by default None
    n_rows : int, optional
        Represents number of data rows that are will plot, by default None.
    lat_origin : float, optional
        Represents the latitude which will be the center of the map.
        If not entered, the first data from the dataset is used, by default None.
    lon_origin : float, optional
        Represents the longitude which will be the center of the map.
        If not entered, the first data from the dataset is used, by default None.
    zoom_start : int, optional
        Initial zoom level for the map, by default 12.
    legend: boolean
        Whether to add a legend to the map, by default True
    base_map : folium.folium.Map, optional
        Represents the folium map. If not informed, a new map is generated
        using the function create_base_map(), with the lat_origin, lon_origin
         and zoom_start, by default None.
    tile : str, optional
        Represents the map tiles, by default TILES[0]
    save_as_html : bool, optional
        Represents if want save this visualization in a new file .html, by default False.
    color : str, list, optional
        Represents line colors of visualization.
        Can be a single color name, a list of colors or a colormap name, by default None.
    color_by_id: dict, optional
        A dictionary where the key is the trajectory id and value is a color,
        by default None.
    filename : str, optional
        Represents the file name of new file .html,
        by default 'plot_trajectories_by_date.html'.

    Returns
    -------
    Map
        a folium map with visualization

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

    if isinstance(start_date, str):
        start_date = str_to_datetime(start_date).date()

    if isinstance(end_date, str):
        end_date = str_to_datetime(end_date).date()

    if DATE not in move_data:
        move_data.generate_date_features()

    mv_df = _filter_generated_feature(move_data, DATE, [start_date, end_date])
    mv_df, items = _filter_and_generate_colors(mv_df, id_, n_rows, color, color_by_id)
    _add_trajectories_to_folium_map(
        mv_df, items, base_map, legend, save_as_html, filename
    )

    return base_map


def plot_trajectory_by_hour(
    move_data: DataFrame,
    start_hour: Text,
    end_hour: Text,
    id_: Optional[int] = None,
    n_rows: Optional[int] = None,
    lat_origin: Optional[float] = None,
    lon_origin: Optional[float] = None,
    zoom_start: Optional[float] = 12,
    legend: Optional[bool] = True,
    base_map: Optional[Map] = None,
    tile: Optional[Text] = TILES[0],
    save_as_html: Optional[bool] = False,
    color: Optional[Union[Text, List[Text]]] = None,
    color_by_id: Optional[Dict] = None,
    filename: Optional[Text] = 'plot_trajectories_by_hour.html',
) -> Map:
    """
    Generate visualization of all trajectories with folium.

    Parameters
    ----------
    move_data : DataFrame
        Input trajectory data
    start_hour : str
        Represents start hour of time period.
    end_hour : str
        Represents end hour of time period.
    id_: int, optional
        Trajectory id to plot, by default None
    n_rows : int, optional
        Represents number of data rows that are will plot, by default None.
    lat_origin : float, optional
        Represents the latitude which will be the center of the map.
        If not entered, the first data from the dataset is used, by default None.
    lon_origin : float, optional
        Represents the longitude which will be the center of the map.
        If not entered, the first data from the dataset is used, by default None.
    zoom_start : int, optional
        Initial zoom level for the map, by default 12.
    legend: boolean
        Whether to add a legend to the map, by default True
    base_map : folium.folium.Map, optional
        Represents the folium map. If not informed, a new map is generated
        using the function create_base_map(), with the lat_origin, lon_origin
         and zoom_start, by default None.
    tile : str, optional
        Represents the map tiles, by default TILES[0]
    save_as_html : bool, optional
        Represents if want save this visualization in a new file .html, by default False.
    color : str, list, optional
        Represents line colors of visualization.
        Can be a single color name, a list of colors or a colormap name, by default None.
    color_by_id: dict, optional
        A dictionary where the key is the trajectory id and value is a color,
        by default None.
    filename : str, optional
        Represents the file name of new file .html,
        by default 'plot_trajectories_by_hour.html'.

    Returns
    -------
    Map
        a folium map with visualization

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

    if HOUR not in move_data:
        move_data.generate_hour_features()

    mv_df = _filter_generated_feature(move_data, HOUR, [start_hour, end_hour])
    mv_df, items = _filter_and_generate_colors(mv_df, id_, n_rows, color, color_by_id)
    _add_trajectories_to_folium_map(
        mv_df, items, base_map, legend, save_as_html, filename
    )

    return base_map


def plot_stops(
    move_data: DataFrame,
    radius: Optional[float] = 0,
    weight: Optional[float] = 3,
    id_: Optional[int] = None,
    n_rows: Optional[int] = None,
    lat_origin: Optional[float] = None,
    lon_origin: Optional[float] = None,
    zoom_start: Optional[float] = 12,
    legend: Optional[bool] = True,
    base_map: Optional[Map] = None,
    tile: Optional[Text] = TILES[0],
    save_as_html: Optional[bool] = False,
    color: Optional[Union[Text, List[Text]]] = None,
    filename: Optional[Text] = 'plot_stops.html',
) -> Map:
    """
    Generate visualization of all trajectories with folium.

    Parameters
    ----------
    move_data : DataFrame
        Input trajectory data
    radius : float, optional
        The radius value is used to determine if a segment is a stop.
        If the value of the point in target_label is greater than
        radius, the segment is a stop, by default 0
    weight: float, optional
        Stroke width in pixels, by default 3
    id_: int, optional
        Trajectory id to plot, by default None
    n_rows : int, optional
        Represents number of data rows that are will plot, by default None.
    lat_origin : float, optional
        Represents the latitude which will be the center of the map.
        If not entered, the first data from the dataset is used, by default None.
    lon_origin : float, optional
        Represents the longitude which will be the center of the map.
        If not entered, the first data from the dataset is used, by default None.
    zoom_start : int, optional
        Initial zoom level for the map, by default 12.
    legend: boolean
        Whether to add a legend to the map, by default True
    base_map : folium.folium.Map, optional
        Represents the folium map. If not informed, a new map is generated
        using the function create_base_map(), with the lat_origin, lon_origin
         and zoom_start, by default None.
    tile : str, optional
        Represents the map tiles, by default TILES[0]
    save_as_html : bool, optional
        Represents if want save this visualization in a new file .html, by default False.
    color : str, list, optional
        Represents line colors of visualization.
        Can be a single color name, a list of colors or a colormap name, by default None.
    filename : str, optional
        Represents the file name of new file .html, by default 'plot_stops.html'.

    Returns
    -------
    Map
        a folium map with visualization

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

    return base_map


def plot_bbox(
    bbox_tuple: Tuple[float, float, float, float],
    base_map: Optional[Map] = None,
    tiles: Optional[Text] = TILES[0],
    color: Optional[Text] = 'red',
    save_as_html: Optional[bool] = False,
    filename: Optional[Text] = 'bbox.html'
) -> Map:
    """
    Plots a bbox using Folium.

    Parameters
    ----------
    bbox_tuple : tuple.
        Represents a bound box, that is a tuple of 4 values with the
        min and max limits of latitude e longitude.
    base_map: Folium map, optional
        A folium map to plot the trajectories. If None a map will be created,
        by default None.
    tiles : str, optional
        by default TILES[0]
    color : str, optional
        Represents color of lines on map, by default 'red'.
    file : str, optional
        Represents filename, by default 'bbox.html'.
    save_map: Boolean, optional
        Wether to save the bbox folium map, by default False.

    Returns
    --------
    Map
        folium map with bounding box

    """
    if base_map is None:
        base_map = folium.Map(tiles=tiles)
    base_map.fit_bounds(
        [[bbox_tuple[0], bbox_tuple[1]], [bbox_tuple[2], bbox_tuple[3]]]
    )
    points_ = [
        (bbox_tuple[0], bbox_tuple[1]),
        (bbox_tuple[0], bbox_tuple[3]),
        (bbox_tuple[2], bbox_tuple[3]),
        (bbox_tuple[2], bbox_tuple[1]),
        (bbox_tuple[0], bbox_tuple[1]),
    ]
    polygon = folium.PolyLine(points_, weight=3, color=color)
    polygon.add_to(base_map)

    if save_as_html:
        base_map.save(filename)

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
    radius,
    map_
):
    """
    Return a circle.

    Parameters
    -----------
    iter_tuple: DataFrame iter_tuple.
    user_lat: str.
        Latitude column name.
    user_lon: str.
        Longitude column name.
    slice_tags:

    user_point: str.
        Point color.
    radius: float.
        radius size.
    map_: Folium map.
    """

    _, line = iter_tuple

    x = line[user_lat]
    y = line[user_lon]

    tags_formated = _format_tags(line, slice_tags)

    folium.Circle(
        radius=radius,
        location=[x, y],
        popup=tags_formated,
        color=user_point,
        fill=False
    ).add_to(map_)


def plot_points_folium(
    move_data: DataFrame,
    user_lat: Optional[Text] = LATITUDE,
    user_lon: Optional[Text] = LONGITUDE,
    user_point: Optional[Text] = USER_POINT,
    radius: Optional[float] = 2,
    base_map: Optional[Map] = None,
    slice_tags: Optional[List] = None,
    tiles: Optional[Text] = TILES[0],
    save_as_html: Optional[bool] = False,
    filename: Optional[Text] = 'points.html'
) -> Map:
    """
    Receives a MoveDataFrame and returns a folium map with the trajectories plots
    and a point.

    Parameters
    ----------
    move_data: Dataframe
        Trajectory data.
    user_lat: str, optional
        Latitude column name, by default LATITUDE.
    user_lon: str, optional
        Longitude column name, by default LONGITUDE.
    user_point: str, optional
        The point color, by default USER_POINT.
    radius: float, optional
        radius size, by default 2.
    sort:Boolean, optional
        If True the data will be sorted, by default False.
    base_map: Folium map, optional
        A folium map to plot the trajectories. If None a map will be created,
        by default None.
    slice_tags: optional, by default None.
    tiles: str, optional, by default TILES[0]
        The map type.
    save_as_html : bool, optional
        Represents if want save this visualization in a new file .html, by default False.
    filename : str, optional
        Represents the file name of new file .html, by default 'points.html'.

    Returns
    -------
    Map
        A folium map
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
                radius,
                base_map
            ),
            move_data.iterrows()
        )
    )

    if save_as_html:
        base_map.save(outfile=filename)
    return base_map


def plot_poi_folium(
    move_data,
    poi_lat=LATITUDE,
    poi_lon=LONGITUDE,
    poi_point=POI_POINT,
    radius=2,
    base_map=None,
    slice_tags=None,
    tiles=TILES[0],
    save_as_html: Optional[bool] = False,
    filename: Optional[Text] = 'pois.html'
) -> Map:

    """
    Receives a MoveDataFrame and returns a folium map with poi points.

    Parameters
    ----------
    move_data: DataFrame
        Trajectory input data
    poi_lat: str, optional
        Latitude column name, by default LATITUDE.
    poi_lon: str, optional
        Longitude column name, by default LONGITUDE.
    poi_point: str, optional
        Poi point color, by default POI_POINT.
    radius: float, optional
        radius size, by default 2.
    base_map: Folium map, optional
        A folium map to plot. If None a map. If None a map will be created,
        by default None.
    slice_tags: optional, by default None.
    tiles: str, optional, by default TILES[0]
        The map type.
    save_as_html : bool, optional
        Represents if want save this visualization in a new file .html, by default False.
    filename : str, optional
        Represents the file name of new file .html, by default 'pois.html'.

    Returns
    -------
    folium.folium.Map.
        Represents a folium map with visualization.
    """
    return plot_points_folium(
        move_data,
        user_lat=poi_lat,
        user_lon=poi_lon,
        user_point=poi_point,
        radius=radius,
        base_map=base_map,
        slice_tags=slice_tags,
        tiles=tiles,
        save_as_html=save_as_html,
        filename=filename
    )


def plot_event_folium(
    move_data,
    event_lat=LATITUDE,
    event_lon=LONGITUDE,
    event_point=EVENT_POINT,
    radius=2,
    base_map=None,
    slice_tags=None,
    tiles=TILES[0],
    save_as_html: Optional[bool] = False,
    filename: Optional[Text] = 'events.html'
) -> Map:

    """
    Receives a MoveDataFrame and returns a folium map with events.

    Parameters
    ----------
    move_data: DataFrame
        Trajectory input data
    event_lat: str, optional
        Latitude column name, by default LATITUDE.
    event_lon: str, optional
        Longitude column name, by default LONGITUDE.
    event_point: str, optional
        Event color, by default EVENT_POI
    radius: float, optional
        radius size, by default 2.
    base_map: Folium map, optional
        A folium map to plot. If None a map. If None a map will be created,
        by default None.
    tiles: str, optional, by default TILES[0]
    save_as_html : bool, optional
        Represents if want save this visualization in a new file .html, by default False.
    filename : str, optional
        Represents the file name of new file .html, by default 'events.html'.

    Returns
    -------
    A folium map.
    """
    return plot_points_folium(
        move_data,
        user_lat=event_lat,
        user_lon=event_lon,
        user_point=event_point,
        radius=radius,
        base_map=base_map,
        slice_tags=slice_tags,
        tiles=tiles,
        save_as_html=save_as_html,
        filename=filename
    )


def show_trajs_with_event(
    move_data: DataFrame,
    window_time_subject: float,
    df_event: DataFrame,
    window_time_event: float,
    radius: float,
    event_lat: Optional[Text] = LATITUDE,
    event_lon: Optional[Text] = LONGITUDE,
    event_datetime: Optional[Text] = DATETIME,
    user_lat: Optional[Text] = LATITUDE,
    user_lon: Optional[Text] = LONGITUDE,
    user_datetime: Optional[Text] = DATETIME,
    event_id: Optional[Text] = EVENT_ID,
    event_point: Optional[Text] = EVENT_POINT,
    user_id: Optional[Text] = UID,
    user_point: Optional[Text] = USER_POINT,
    line_color: Optional[Text] = LINE_COLOR,
    slice_event_show: Optional[int] = None,
    slice_subject_show: Optional[int] = None,
) -> List[Map]:
    """
    Plot a trajectory, including your user_points lat lon and your tags.

    Parameters
    -----------
    move_data: DataFrame.
        Trajectory input data.
    window_time_subject: float.
        The subject time window.
    window_time_event: float.
        The event time window.
    radius: float.
        The radius to use.
    event_lat: str, optional
        Event latitude column name, by default LATITUDE.
    event_lon: str, optional
        Event longitude column name, by default LONGITUDE.
    event_datetime: str, optional
        Event datetime column name, by default DATETIME.
    user_lat: str, optional
        User latitude column name, by default LATITUDE.
    user_lon: str, optional
        User longitude column name, by default LONGITUDE.
    user_datetime: str, optional
        User datetime column name, by default DATETIME.
    event_id_: str, optional
        Event id column name, by default TRAJ_ID.
    event_point: str, optional
        Event color, by default EVENT_POI.
    user_id: str, optional
        User id column name, by default TRAJ_ID.
    user_point: str, optional
        User point color, by default USER_POINT.
    line_color: str, optional
        Line color, by default 'blue'.
    slice_event_show: int, optional
        by default None.
    slice_subject_show: int, optional
        by default None.

    Returns
    -------
    list of Map
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

        e_lat = line[event_lat]
        e_lon = line[event_lon]
        e_datetime = line[event_datetime]
        e_id = line[event_id]

        # building time window for event search
        start_time = pd.to_datetime(e_datetime - delta_event)
        end_time = pd.to_datetime(e_datetime + delta_event)

        # filtering df_ for time window
        df_filtered = filters.by_datetime(
            move_data,
            start_datetime=start_time,
            end_datetime=end_time
        )

        # length of df_temp
        len_df_temp = df_filtered.shape[0]

        # using the util part of the array for haversine function
        lat_arr[:len_df_temp] = e_lat
        lon_arr[:len_df_temp] = e_lon

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

            move_data = df_event[df_event[event_id] == e_id]

            base_map = plot_event_folium(
                move_data,
                event_lat=event_lat,
                event_lon=event_lon,
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
                base_map = plot_trajectories_with_folium(
                    df_temp,
                    color=[line_color],
                    base_map=base_map
                )
                base_map = plot_points_folium(
                    df_temp,
                    user_lat=user_lat,
                    user_lon=user_lon,
                    user_point=user_point,
                    base_map=base_map,
                    slice_tags=slice_subject_show
                )
            # add to folium maps list: (id event, folium map, quantity of user in map, df)
            folium_maps.append((base_map, pd.concat(users)))

    return folium_maps


def show_traj_id_with_event(
    move_data: DataFrame,
    window_time_subject: float,
    df_event: DataFrame,
    window_time_event: float,
    radius: float,
    subject_id: int,
    event_lat: Optional[Text] = LATITUDE,
    event_lon: Optional[Text] = LONGITUDE,
    event_datetime: Optional[Text] = DATETIME,
    user_lat: Optional[Text] = LATITUDE,
    user_lon: Optional[Text] = LONGITUDE,
    user_datetime: Optional[Text] = DATETIME,
    event_id: Optional[Text] = EVENT_ID,
    event_point: Optional[Text] = EVENT_POINT,
    user_id: Optional[Text] = UID,
    user_point: Optional[Text] = USER_POINT,
    line_color: Optional[Text] = LINE_COLOR,
    slice_event_show: Optional[int] = None,
    slice_subject_show: Optional[int] = None,
) -> Map:
    """
    Plot a trajectory, including your user_points lat lon and your tags.

    Parameters
    -----------
    move_data: DataFrame.
        Trajectory input data.
    window_time_subject: float.
        The subject time window.
    window_time_event: float.
        The event time window.
    radius: float.
        The radius to use.
    subject_id: int
        Id of the trajectory
    event_lat: str, optional
        Event latitude column name, by default LATITUDE.
    event_lon: str, optional
        Event longitude column name, by default LONGITUDE.
    event_datetime: str, optional
        Event datetime column name, by default DATETIME.
    user_lat: str, optional
        User latitude column name, by default LATITUDE.
    user_lon: str, optional
        User longitude column name, by default LONGITUDE.
    user_datetime: str, optional
        User datetime column name, by default DATETIME.
    event_id_: str, optional
        Event id column name, by default TRAJ_ID.
    event_point: str, optional
        Event color, by default EVENT_POINT.
    user_id: str, optional
        User id column name, by default TRAJ_ID.
    user_point: str, optional
        User point color, by default USER_POINT.
    line_color: str, optional
        Line color, by default 'blue'.
    slice_event_show: int, optional
        by default None.
    slice_subject_show: int, optional
        by default None.

    Returns
    -------
    Map
        A list of folium maps.
    """

    df_id = move_data[move_data[user_id] == subject_id]

    return show_trajs_with_event(
        df_id,
        window_time_subject,
        df_event,
        window_time_event,
        radius,
        event_lat=event_lat,
        event_lon=event_lon,
        event_datetime=event_datetime,
        user_lat=user_lat,
        user_lon=user_lon,
        user_datetime=user_datetime,
        event_id=event_id,
        event_point=event_point,
        user_id=user_id,
        user_point=user_point,
        line_color=line_color,
        slice_event_show=slice_event_show,
        slice_subject_show=slice_subject_show
    )[0]


def _create_geojson_features_line(
    move_data: DataFrame,
    label_lat: Optional[Text] = LATITUDE,
    label_lon: Optional[Text] = LONGITUDE,
    label_datetime: Optional[Text] = DATETIME
) -> List:
    """
    Create geojson features.

    Parameters
    -----------
    move_data: DataFrame.
        Input trajectory data.
    label_datetime: str, optional
        date_time colum label, by default DATETIME.
    label_lat: str, optional
        latitude column label, by default LATITUDE.
    label_lon: str, optional
        longitude column label, by default LONGITUDE.

    Returns
    -------
    list
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
                'type': 'Linestr',
                'coordinates': [
                    [last[label_lon], last[label_lat]],
                    [row[label_lon], row[label_lat]]
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
    label_lat=LATITUDE,
    label_lon=LONGITUDE,
    label_datetime=DATETIME,
    tiles=TILES[0],
    save_as_html: Optional[bool] = False,
    filename: Optional[Text] = 'events.html'
) -> Map:
    """
    Plot trajectories wit geo_json.

    Parameters
    -----------
    move_data: DataFrame.
        Input trajectory data.
    label_datetime: str, optional, by default DATETIME.
        date_time column label.
    label_lat: str, optional, by default LATITUDE.
        latitude column label.
    label_lon: str, optional, by default LONGITUDE.
        longitude column label.
    tiles: str, optional
        map tiles, by default TILES[0]
    save_as_html : bool, optional
        Represents if want save this visualization in a new file .html, by default False.
    filename : str, optional
        Represents the file name of new file .html, by default 'events.html'.

    Returns
    -------
    Map
        A folium map.
    """
    features = _create_geojson_features_line(
        move_data,
        label_lat=label_lat,
        label_lon=label_lon,
        label_datetime=label_datetime
    )
    print('creating folium map')
    base_map = create_base_map(
        move_data=move_data,
        lat_origin=move_data[label_lat].mean(),
        lon_origin=move_data[label_lon].mean(),
        tile=tiles
    )
    print('Genering timestamp map')
    print(features)
    plugins.TimestampedGeoJson(
        {
            'type': 'FeatureCollection',
            'features': features,
        },
        period='PT1M',
        add_last_point=True
    ).add_to(base_map)
    if save_as_html:
        base_map.save(filename)
    return base_map
