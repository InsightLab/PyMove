"""
Folium operations.

save_map,
create_base_map,
heatmap,
heatmap_with_time,
cluster,
faster_cluster,
plot_markers,
plot_trajectories_with_folium,
plot_trajectory_by_id_folium,
plot_trajectory_by_period,
plot_trajectory_by_day_week,
plot_trajectory_by_date,
plot_trajectory_by_hour,
plot_stops,
plot_bbox,
plot_points_folium,
plot_poi_folium,
plot_event_folium,
show_trajs_with_event,
show_traj_id_with_event,
plot_traj_timestamp_geo_json

"""

from datetime import date
from typing import Any, Dict, List, Optional, Sequence, Text, Tuple, Union

import folium
import numpy as np
from folium import Map, plugins
from folium.plugins import FastMarkerCluster, HeatMap, HeatMapWithTime, MarkerCluster
from pandas import DataFrame

from pymove import PandasMoveDataFrame
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
    tiles: Text = TILES[0],
    label_id: Text = TRAJ_ID,
    cmap: Text = 'Set1',
    return_map: bool = False
) -> Optional[Map]:
    """
    Save a visualization in a map in a new file.

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

    Examples
    --------
    >>> from pymove.visualization.folium import save_map
    >>> move_df.head()
              lat          lon              datetime   id
    0   39.984094   116.319236   2008-10-23 05:53:05    1
    1   39.984198   116.319322   2008-10-23 05:53:06    1
    2   39.984224   116.319402   2008-10-23 05:53:11    1
    3   39.984211   116.319389   2008-10-23 05:53:16    1
    4   39.984217   116.319422   2008-10-23 05:53:21    1
    >>> save_map(df, filename='test.map')
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
    tile: Text = TILES[0],
    default_zoom_start: float = 12,
) -> Map:
    """
    Generates a folium map.

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

    Examples
    --------
    >>> from pymove.visualization.folium import create_base_map
    >>> move_df.head()
              lat          lon              datetime   id
    0   39.984094   116.319236   2008-10-23 05:53:05    1
    1   39.984198   116.319322   2008-10-23 05:53:06    1
    2   39.984224   116.319402   2008-10-23 05:53:11    1
    3   39.984211   116.319389   2008-10-23 05:53:16    1
    4   39.984217   116.319422   2008-10-23 05:53:21    1
    >>> create_base_map(move_df)
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
    zoom_start: float = 12,
    radius: float = 8,
    base_map: Optional[Map] = None,
    tile: Text = TILES[0],
    save_as_html: bool = False,
    filename: Text = 'heatmap.html',
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

    Examples
    --------
    >>> from pymove.visualization.folium import heatmap
    >>> move_df.head()
              lat          lon              datetime   id
    0   39.984094   116.319236   2008-10-23 05:53:05    1
    1   39.984198   116.319322   2008-10-23 05:53:06    1
    2   39.984224   116.319402   2008-10-23 05:53:11    1
    3   39.984211   116.319389   2008-10-23 05:53:16    1
    4   39.984217   116.319422   2008-10-23 05:53:21    1
    >>> heatmap(move_df)
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
    zoom_start: float = 12,
    radius: float = 8,
    min_opacity: float = 0.5,
    max_opacity: float = 0.8,
    base_map: Optional[Map] = None,
    tile: Text = TILES[0],
    save_as_html: bool = False,
    filename: Text = 'heatmap_time.html',
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

    Examples
    --------
    >>> from pymove.visualization.folium import heatmap_with_time
    >>> move_df.head()
              lat          lon              datetime   id
    0   39.984094   116.319236   2008-10-23 05:53:05    1
    1   39.984198   116.319322   2008-10-23 05:53:06    1
    2   39.984224   116.319402   2008-10-23 05:53:11    1
    3   39.984211   116.319389   2008-10-23 05:53:16    1
    4   39.984217   116.319422   2008-10-23 05:53:21    1
    >>> heatmap_with_time(move_df)
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
    zoom_start: float = 12,
    base_map: Optional[Map] = None,
    tile: Text = TILES[0],
    save_as_html: bool = False,
    filename: Text = 'cluster.html',
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

    Examples
    --------
    >>> from pymove.visualization.folium import cluster
    >>> move_df.head()
              lat          lon              datetime   id
    0   39.984094   116.319236   2008-10-23 05:53:05    1
    1   39.984198   116.319322   2008-10-23 05:53:06    1
    2   39.984224   116.319402   2008-10-23 05:53:11    1
    3   39.984211   116.319389   2008-10-23 05:53:16    1
    4   39.984217   116.319422   2008-10-23 05:53:21    1
    >>> cluster(move_df)
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
    zoom_start: float = 12,
    base_map: Optional[Map] = None,
    tile: Text = TILES[0],
    save_as_html: bool = False,
    filename: Text = 'faster_cluster.html',
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

    Examples
    --------
    >>> from pymove.visualization.folium import faster_cluster
    >>> move_df.head()
              lat          lon              datetime   id
    0   39.984094   116.319236   2008-10-23 05:53:05    1
    1   39.984198   116.319322   2008-10-23 05:53:06    1
    2   39.984224   116.319402   2008-10-23 05:53:11    1
    3   39.984211   116.319389   2008-10-23 05:53:16    1
    4   39.984217   116.319422   2008-10-23 05:53:21    1
    >>> faster_cluster(move_df)
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
    zoom_start: float = 12,
    base_map: Optional[Map] = None,
    tile: Text = TILES[0],
    save_as_html: bool = False,
    filename: Text = 'markers.html',
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

    Examples
    --------
    >>> from pymove.visualization.folium import plot_markers
    >>> move_df.head()
              lat          lon              datetime   id
    0   39.984094   116.319236   2008-10-23 05:53:05    1
    1   39.984198   116.319322   2008-10-23 05:53:06    1
    2   39.984224   116.319402   2008-10-23 05:53:11    1
    3   39.984211   116.319389   2008-10-23 05:53:16    1
    4   39.984217   116.319422   2008-10-23 05:53:21    1
    >>> plot_markers(move_df)
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
    color: Optional[Union[Text, List[Text]]] = None,
    color_by_id: Optional[Dict] = None
) -> Tuple[DataFrame, List[Tuple[Any, Any]]]:
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

    Examples
    --------
    >>> from pymove.visualization.folium import _filter_and_generate_colors
    >>> move_df.head()
              lat          lon              datetime   id
    0   39.984094   116.319236   2008-10-23 05:53:05    1
    1   39.984198   116.319322   2008-10-23 05:53:06    1
    2   39.984224   116.319402   2008-10-23 05:53:11    1
    3   39.984211   116.319389   2008-10-23 05:53:16    2
    4   39.984217   116.319422   2008-10-23 05:53:21    2
    >>> df, colors = _filter_and_generate_colors(move_df)
    >>> df
              lat          lon              datetime   id
    0   39.984094   116.319236   2008-10-23 05:53:05    1
    1   39.984198   116.319322   2008-10-23 05:53:06    1
    2   39.984224   116.319402   2008-10-23 05:53:11    1
    3   39.984211   116.319389   2008-10-23 05:53:16    2
    4   39.984217   116.319422   2008-10-23 05:53:21    2
    >>> colors
    [(1, '#e41a1c'), (2, '#377eb8')]
    """
    if n_rows is None:
        n_rows = move_data.shape[0]

    if id_ is not None:
        mv_df = move_data[move_data[TRAJ_ID] == id_].head(n_rows)[
            [LATITUDE, LONGITUDE, DATETIME, TRAJ_ID]
        ]
        if not len(mv_df):
            raise IndexError('No user with id %s in dataframe' % id_)
    else:
        mv_df = move_data.head(n_rows)[
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
    ----------
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

    Examples
    --------
    >>> from pymove.visualization.folium import _filter_generated_feature
    >>> move_df.head()
              lat          lon              datetime   id
    0   39.984094   116.319236   2008-10-23 05:53:05    1
    1   39.984198   116.319322   2008-10-23 05:53:06    1
    2   39.984224   116.319402   2008-10-23 05:53:11    1
    3   39.984211   116.319389   2008-10-23 05:53:16    1
    4   39.984217   116.319422   2008-10-23 05:53:21    1
    >>> _filter_generated_feature(move_df, feature='lat', values=[39.984198])
              lat          lon              datetime   id
    1   39.984198   116.319322   2008-10-23 05:53:06    1
    >>> _filter_generated_feature(move_df, feature='lon', values=[116.319236])
              lat          lon              datetime   id
    0   39.984094   116.319236   2008-10-23 05:53:05    1
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
    Adds markers to the beggining and end of a trajectory.

    Adds a green marker to beginning of the trajectory
    and a red marker to the end of the trajectory.

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

    Examples
    --------
    >>> from pymove.visualization.folium import _add_begin_end_markers_to_folium_map
    >>> move_df.head()
              lat          lon              datetime   id
    0   39.984094   116.319236   2008-10-23 05:53:05    1
    1   39.984198   116.319322   2008-10-23 05:53:06    1
    2   39.984224   116.319402   2008-10-23 05:53:11    1
    3   39.984211   116.319389   2008-10-23 05:53:16    1
    4   39.984217   116.319422   2008-10-23 05:53:21    1
    >>> map = create_base_map(move_df)
    >>> _add_begin_end_markers_to_folium_map(move_df, map)
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
    items: Sequence[Tuple],
    base_map: Map,
    legend: bool = True,
    save_as_html: bool = True,
    filename: Text = 'map.html',
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

    Examples
    --------
    >>> from pymove.visualization.folium import _add_trajectories_to_folium_map
    >>> move_df
                   lat          lon              datetime   id
    0        39.984094   116.319236   2008-10-23 05:53:05    1
    1        39.984198   116.319322   2008-10-23 05:53:06    1
    3        39.988118   116.326672   2008-10-25 14:39:19    5
    4        39.987965   116.326675   2008-10-25 14:39:24    5
    >>> _add_trajectories_to_folium_map(
    >>>    move_data=move_df,
    >>>    base_map=map1,
    >>>    items=[(1, 'red'), [5, 'green']]
    >>> )
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
    zoom_start: float = 12,
    legend: bool = True,
    base_map: Optional[Map] = None,
    tile: Text = TILES[0],
    save_as_html: bool = False,
    color: Optional[Union[Text, List[Text]]] = None,
    color_by_id: Optional[Dict] = None,
    filename: Text = 'plot_trajectories_with_folium.html',
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

    Examples
    --------
    >>>  from pymove.visualization.folium import plot_trajectories_with_folium
    >>> move_df.head()
              lat          lon              datetime   id
    0   39.984094   116.319236   2008-10-23 05:53:05    1
    1   39.984198   116.319322   2008-10-23 05:53:06    1
    2   39.984224   116.319402   2008-10-23 05:53:11    1
    3   39.984211   116.319389   2008-10-23 05:53:16    1
    4   39.984217   116.319422   2008-10-23 05:53:21    1
    >>> plot_trajectories_with_folium(move_df)
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
    zoom_start: float = 12,
    legend: bool = True,
    base_map: Optional[Map] = None,
    tile: Text = TILES[0],
    save_as_html: bool = False,
    color: Optional[Union[Text, List[Text]]] = None,
    filename: Text = 'plot_trajectories_with_folium.html',
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

    Examples
    --------
    >>> from pymove.visualization.folium import plot_trajectory_by_id_folium
    >>> move_df.head()
              lat          lon              datetime   id
    0   39.984094   116.319236   2008-10-23 05:53:05    1
    1   39.984198   116.319322   2008-10-23 05:53:06    1
    2   39.984224   116.319402   2008-10-23 05:53:11    1
    3   39.984211   116.319389   2008-10-23 05:53:16    2
    4   39.984217   116.319422   2008-10-23 05:53:21    2
    >>> plot_trajectory_by_id_folium(move_df, id_=1)
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
    move_data: PandasMoveDataFrame,
    period: Text,
    id_: Optional[int] = None,
    n_rows: Optional[int] = None,
    lat_origin: Optional[float] = None,
    lon_origin: Optional[float] = None,
    zoom_start: float = 12,
    legend: bool = True,
    base_map: Optional[Map] = None,
    tile: Text = TILES[0],
    save_as_html: bool = False,
    color: Optional[Union[Text, List[Text]]] = None,
    color_by_id: Optional[Dict] = None,
    filename: Text = 'plot_trajectories_by_period.html',
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

    Examples
    --------
    >>> from pymove.visualization.folium import plot_trajectory_by_period
    >>> move_df.head()
              lat          lon              datetime   id
    0   39.984094   116.319236   2008-10-23 05:53:05    1
    1   39.984198   116.319322   2008-10-23 05:53:06    1
    2   39.984224   116.319402   2008-10-23 05:53:11    1
    3   39.984211   116.319389   2008-10-23 05:53:16    1
    4   39.984217   116.319422   2008-10-23 05:53:21    1
    >>> plot_trajectory_by_period(move_df, period='Early morning')
    >>> move_df.head()
              lat          lon              datetime   id          period
    0   39.984094   116.319236   2008-10-23 05:53:05    1   Early morning
    1   39.984198   116.319322   2008-10-23 05:53:06    1   Early morning
    2   39.984224   116.319402   2008-10-23 05:53:11    1   Early morning
    3   39.984211   116.319389   2008-10-23 05:53:16    1   Early morning
    4   39.984217   116.319422   2008-10-23 05:53:21    1   Early morning
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
    move_data: PandasMoveDataFrame,
    day_week: Text,
    id_: Optional[int] = None,
    n_rows: Optional[int] = None,
    lat_origin: Optional[float] = None,
    lon_origin: Optional[float] = None,
    zoom_start: float = 12,
    legend: bool = True,
    base_map: Optional[Map] = None,
    tile: Text = TILES[0],
    save_as_html: bool = False,
    color: Optional[Union[Text, List[Text]]] = None,
    color_by_id: Optional[Dict] = None,
    filename: Text = 'plot_trajectories_by_day_week.html',
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

    Examples
    --------
    >>> from pymove.visualization.folium import plot_trajectory_by_day_week
    >>> move_df.head()
              lat          lon              datetime   id
    0   39.984094   116.319236   2008-10-23 05:53:05    1
    1   39.984198   116.319322   2008-10-23 05:53:06    1
    2   39.984224   116.319402   2008-10-23 05:53:11    1
    3   39.984211   116.319389   2008-10-23 05:53:16    1
    4   39.984217   116.319422   2008-10-23 05:53:21    1
    >>> plot_trajectory_by_day_week(move_df, day_week='Friday')
    >>> move_df.head()
              lat          lon              datetime   id        day
    0   39.984094   116.319236   2008-10-23 05:53:05    1   Thursday
    1   39.984198   116.319322   2008-10-23 05:53:06    1   Thursday
    2   39.984224   116.319402   2008-10-23 05:53:11    1   Thursday
    3   39.984211   116.319389   2008-10-23 05:53:16    1   Thursday
    4   39.984217   116.319422   2008-10-23 05:53:21    1   Thursday
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
    move_data: PandasMoveDataFrame,
    start_date: Union[Text, date],
    end_date: Union[Text, date],
    id_: Optional[int] = None,
    n_rows: Optional[int] = None,
    lat_origin: Optional[float] = None,
    lon_origin: Optional[float] = None,
    zoom_start: float = 12,
    legend: bool = True,
    base_map: Optional[Map] = None,
    tile: Text = TILES[0],
    save_as_html: bool = False,
    color: Optional[Union[Text, List[Text]]] = None,
    color_by_id: Optional[Dict] = None,
    filename: Text = 'plot_trajectories_by_date.html',
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

    Examples
    --------
    >>> from pymove.visualization.folium import plot_trajectory_by_date
    >>> move_df.head()
              lat          lon              datetime   id
    0   39.984094   116.319236   2008-10-23 05:53:05    1
    1   39.984198   116.319322   2008-10-23 05:53:06    1
    2   39.984224   116.319402   2008-10-23 05:53:11    1
    3   39.984211   116.319389   2008-10-23 05:53:16    1
    4   39.984217   116.319422   2008-10-23 05:53:21    1
    >>> plot_trajectory_by_date(
    >>>     move_df,
    >>>     start_date='2008-10-23 05:53:05',
    >>>     end_date='2008-10-23 23:43:56'
    >>> )
    >>> move_df.head()
              lat          lon              datetime   id         date
    0   39.984094   116.319236   2008-10-23 05:53:05    1   2008-10-23
    1   39.984198   116.319322   2008-10-23 05:53:06    1   2008-10-23
    2   39.984224   116.319402   2008-10-23 05:53:11    1   2008-10-23
    3   39.984211   116.319389   2008-10-23 05:53:16    1   2008-10-23
    4   39.984217   116.319422   2008-10-23 05:53:21    1   2008-10-23
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
    move_data: PandasMoveDataFrame,
    start_hour: Text,
    end_hour: Text,
    id_: Optional[int] = None,
    n_rows: Optional[int] = None,
    lat_origin: Optional[float] = None,
    lon_origin: Optional[float] = None,
    zoom_start: float = 12,
    legend: bool = True,
    base_map: Optional[Map] = None,
    tile: Text = TILES[0],
    save_as_html: bool = False,
    color: Optional[Union[Text, List[Text]]] = None,
    color_by_id: Optional[Dict] = None,
    filename: Text = 'plot_trajectories_by_hour.html',
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

    Examples
    --------
    >>> from pymove.visualization.folium import plot_trajectory_by_hour
    >>> move_df.head()
              lat          lon              datetime   id
    0   39.984094   116.319236   2008-10-23 05:53:05    1
    1   39.984198   116.319322   2008-10-23 05:53:06    1
    2   39.984224   116.319402   2008-10-23 05:53:11    1
    3   39.984211   116.319389   2008-10-23 05:53:16    1
    4   39.984217   116.319422   2008-10-23 05:53:21    1
    >>> plot_trajectory_by_hour(move_df, start_hour=4,end_hour=6)
              lat          lon              datetime   id   hour
    0   39.984094   116.319236   2008-10-23 05:53:05    1      5
    1   39.984198   116.319322   2008-10-23 05:53:06    1      5
    2   39.984224   116.319402   2008-10-23 05:53:11    1      5
    3   39.984211   116.319389   2008-10-23 05:53:16    1      5
    4   39.984217   116.319422   2008-10-23 05:53:21    1      5
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
    move_data: PandasMoveDataFrame,
    radius: float = 0,
    weight: float = 3,
    id_: Optional[int] = None,
    n_rows: Optional[int] = None,
    lat_origin: Optional[float] = None,
    lon_origin: Optional[float] = None,
    zoom_start: float = 12,
    legend: bool = True,
    base_map: Optional[Map] = None,
    tile: Text = TILES[0],
    save_as_html: bool = False,
    color: Optional[Union[Text, List[Text]]] = None,
    filename: Text = 'plot_stops.html',
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

    Examples
    --------
    >>> from pymove.visualization.folium import plot_stops
    >>> move_df.head()
              lat          lon              datetime   id
    0   39.984094   116.319236   2008-10-23 05:53:05    1
    1   39.984198   116.319322   2008-10-23 05:53:06    1
    2   39.984224   116.319402   2008-10-23 05:53:11    1
    3   39.984211   116.319389   2008-10-23 05:53:16    1
    4   39.984217   116.319422   2008-10-23 05:53:21    1
    >>> plot_stops(move_df)
    >>> move_df.head()
              lat          lon              datetime   id \
    dist_to_prev   dist_to_next   dist_prev_to_next   situation
    0   39.984094   116.319236   2008-10-23 05:53:05    1 \
             NaN      13.690153                 NaN         nan
    1   39.984198   116.319322   2008-10-23 05:53:06    1 \
       13.690153       7.403788           20.223428        move
    2   39.984224   116.319402   2008-10-23 05:53:11    1 \
        7.403788       1.821083            5.888579        move
    3   39.984211   116.319389   2008-10-23 05:53:16    1 \
        1.821083       2.889671            1.873356        move
    4   39.984217   116.319422   2008-10-23 05:53:21    1 \
        2.889671      66.555997           68.727260        move
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
    tiles: Text = TILES[0],
    color: Text = 'red',
    save_as_html: bool = False,
    filename: Text = 'bbox.html'
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
    -------
    Map
        folium map with bounding box

    Examples
    --------
    >>> from pymove.visualization.folium import plot_bbox
    >>> plot_bbox((39.984094,116.319236,39.997535,116.196345))
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


def _format_tags(line: Union[List, Dict], slice_: List) -> Text:
    """
    Create or format tags.

    Parameters
    ----------
    line: Line to add a tag.

    slice_: Tag interval.

    Returns
    -------
    str: formatted html tag

    Examples
    --------
    >>> from pymove.visualization.folium import _format_tags, plot_points_folium
    >>> move_df.head()
              lat          lon              datetime   id
    0   39.984094   116.319236   2008-10-23 05:53:05    1
    1   39.984198   116.319322   2008-10-23 05:53:06    1
    2   39.984224   116.319402   2008-10-23 05:53:11    1
    3   39.984211   116.319389   2008-10-23 05:53:16    1
    4   39.984217   116.319422   2008-10-23 05:53:21    1
    >>> _format_tags(
    >>>    line={
    >>>        'lat': 39.984094,
    >>>        'lon': 116.319236,
    >>>        'datetime': '2008-10-23 05:53:05',
    >>>        'id': 1
    >>>    },
    >>>    slice_=['lat', 'lon', 'datetime', 'id']
    >>> )
    lat: 39.984094<br/>lon: 116.319236<br/>datetime: 2008-10-23 05:53:05<br/>id: 1
    """
    map_formated_tags = map(lambda tag: '{}: {}'.format(tag, line[tag]), slice_)

    return '<br/>'.join(map_formated_tags)


def _circle_maker(
    iter_tuple: DataFrame,
    user_lat: Text,
    user_lon: Text,
    slice_tags: List,
    user_point: Text,
    radius: float,
    map_: Map
):
    """
    Return a circle.

    Parameters
    ----------
    iter_tuple: DataFrame iter_tuple.
    user_lat: str.
        Latitude column name.
    user_lon: str.
        Longitude column name.
    slice_tags: list or iterable

    user_point: str.
        Point color.
    radius: float.
        radius size.
    map_: Folium map.

    Examples
    --------
    >>> from pymove.visualization.folium import _circle_maker
    >>> move_df.head()
              lat          lon              datetime   id
    0   39.984094   116.319236   2008-10-23 05:53:05    1
    1   39.984198   116.319322   2008-10-23 05:53:06    1
    2   39.984224   116.319402   2008-10-23 05:53:11    1
    3   39.984211   116.319389   2008-10-23 05:53:16    1
    4   39.984217   116.319422   2008-10-23 05:53:21    1
    >>> row = move_df.iloc[0]
    >>> iter_tuple = (0, row)
    >>> user_lat = 'lat'
    >>> user_lon = 'lon'
    >>> slice_tags = row.keys()
    >>> user_point = 'pink'
    >>> radius = 10
    >>> map_ = create_base_map(move_df)
    >>> _circle_maker(
    >>>    iter_tuple, user_lat, user_lon,
    >>>    slice_tags, user_point, radius, map_
    >>> )
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
    user_lat: Text = LATITUDE,
    user_lon: Text = LONGITUDE,
    user_point: Text = USER_POINT,
    radius: float = 2,
    base_map: Optional[Map] = None,
    slice_tags: Optional[List] = None,
    tiles: Text = TILES[0],
    save_as_html: bool = False,
    filename: Text = 'points.html'
) -> Map:
    """
    Generates a folium map with the trajectories plots and a point.

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

    Examples
    --------
    >>> from pymove.visualization.folium import plot_points_folium
    >>> move_df.head()
              lat          lon              datetime   id
    0   39.984094   116.319236   2008-10-23 05:53:05    1
    1   39.984198   116.319322   2008-10-23 05:53:06    1
    2   39.984224   116.319402   2008-10-23 05:53:11    1
    3   39.984211   116.319389   2008-10-23 05:53:16    1
    4   39.984217   116.319422   2008-10-23 05:53:21    1
    >>> plot_points_folium(move_df)
    """
    if slice_tags is None:
        slice_tags = list(move_data.columns)

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

    for row in move_data.iterrows():
        _circle_maker(
            row,
            user_lat,
            user_lon,
            slice_tags,
            user_point,
            radius,
            base_map
        )

    if save_as_html:
        base_map.save(outfile=filename)
    return base_map


def plot_poi_folium(
    move_data: DataFrame,
    poi_lat: Text = LATITUDE,
    poi_lon: Text = LONGITUDE,
    poi_point: Text = POI_POINT,
    radius: float = 2,
    base_map: Optional[Map] = None,
    slice_tags: Optional[List] = None,
    tiles: Text = TILES[0],
    save_as_html: bool = False,
    filename: Text = 'pois.html'
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

    Examples
    --------
    >>> from pymove.visualization.folium import plot_poi_folium
    >>> move_df.head()
              lat          lon              datetime   id
    0   39.984094   116.319236   2008-10-23 05:53:05    1
    1   39.984198   116.319322   2008-10-23 05:53:06    1
    2   39.984224   116.319402   2008-10-23 05:53:11    1
    3   39.984211   116.319389   2008-10-23 05:53:16    1
    4   39.984217   116.319422   2008-10-23 05:53:21    1
    >>> plot_poi_folium(move_df)
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
    move_data: DataFrame,
    event_lat: Text = LATITUDE,
    event_lon: Text = LONGITUDE,
    event_point: Text = EVENT_POINT,
    radius: float = 2,
    base_map: Optional[Map] = None,
    slice_tags: Optional[List] = None,
    tiles: Text = TILES[0],
    save_as_html: bool = False,
    filename: Text = 'events.html'
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

    Examples
    --------
    >>> from pymove.visualization.folium import plot_event_folium
    >>> move_df.head()
              lat          lon              datetime   id
    0   39.984094   116.319236   2008-10-23 05:53:05    1
    1   39.984198   116.319322   2008-10-23 05:53:06    1
    2   39.984224   116.319402   2008-10-23 05:53:11    1
    3   39.984211   116.319389   2008-10-23 05:53:16    1
    4   39.984217   116.319422   2008-10-23 05:53:21    1
    >>> plot_event_folium(move_df)
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


def _create_geojson_features_line(
    move_data: DataFrame,
    label_lat: Text = LATITUDE,
    label_lon: Text = LONGITUDE,
    label_datetime: Text = DATETIME
) -> List:
    """
    Create geojson features.

    Parameters
    ----------
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

    Examples
    --------
    >>> from pymove.visualization.folium import _create_geojson_features_line
    >>> move_df.head()
              lat          lon              datetime   id
    0   39.984094   116.319236   2008-10-23 05:53:05    1
    1   39.984198   116.319322   2008-10-23 05:53:06    1
    2   39.984224   116.319402   2008-10-23 05:53:11    1
    3   39.984211   116.319389   2008-10-23 05:53:16    1
    4   39.984217   116.319422   2008-10-23 05:53:21    1
    >>> _create_geojson_features_line(move_df)
    [
    {
        "type":"Feature",
        "geometry":{
            "type":"Linestr",
            "coordinates":[
                [
                116.319236,
                39.984094
                ],
                [
                116.319322,
                39.984198
                ]
            ]
        },
        "properties":{
            "times":[
                "2008-10-23T05:53:05",
                "2008-10-23T05:53:06"
            ],
            "popup":"lat: 39.984094<br>lon: 116.319236<br> \
                datetime: 2008-10-23 05:53:05<br>id: 1",
            "style":{
                "color":"red",
                "icon":"circle",
                "iconstyle":{
                "color":"red",
                "weight":4
                }
            }
        }
    },
    ...
    ]
    """
    features = []

    row_iterator = move_data.iterrows()
    _, last = next(row_iterator)
    columns = move_data.columns

    for i, row in progress_bar(
        row_iterator,
        total=move_data.shape[0],
        desc='Generating GeoJSon'
    ):
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
    move_data: DataFrame,
    label_lat: Text = LATITUDE,
    label_lon: Text = LONGITUDE,
    label_datetime: Text = DATETIME,
    tiles: Text = TILES[0],
    save_as_html: bool = False,
    filename: Text = 'events.html'
) -> Map:
    """
    Plot trajectories wit geo_json.

    Parameters
    ----------
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

    Examples
    --------
    >>> from pymove.visualization.folium import plot_traj_timestamp_geo_json
    >>> move_df.head()
              lat          lon              datetime   id
    0   39.984094   116.319236   2008-10-23 05:53:05    1
    1   39.984198   116.319322   2008-10-23 05:53:06    1
    2   39.984224   116.319402   2008-10-23 05:53:11    1
    3   39.984211   116.319389   2008-10-23 05:53:16    1
    4   39.984217   116.319422   2008-10-23 05:53:21    1
    >>> plot_traj_timestamp_geo_json(move_df)
    """
    features = _create_geojson_features_line(
        move_data,
        label_lat=label_lat,
        label_lon=label_lon,
        label_datetime=label_datetime
    )
    base_map = create_base_map(
        move_data=move_data,
        lat_origin=move_data[label_lat].mean(),
        lon_origin=move_data[label_lon].mean(),
        tile=tiles
    )
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
