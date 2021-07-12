"""
Integration operations.

union_poi_bank,
union_poi_bus_station,
union_poi_bar_restaurant,
union_poi_parks,
union_poi_police,
join_collective_areas,
join_with_pois,
join_with_pois_by_category,
join_with_events,
join_with_event_by_dist_and_time,
join_with_home_by_id,
merge_home_with_poi

"""

from collections import namedtuple
from typing import List, Optional, Text, Tuple

import numpy as np
from numpy import ndarray
from pandas import DataFrame, Timedelta
from pandas.core.series import Series

from pymove.preprocessing import filters
from pymove.utils.constants import (
    ADDRESS,
    CITY,
    DATETIME,
    DIST_EVENT,
    DIST_HOME,
    DIST_POI,
    EVENT_ID,
    EVENT_TYPE,
    GEOMETRY,
    HOME,
    ID_POI,
    LATITUDE,
    LONGITUDE,
    NAME_POI,
    TRAJ_ID,
    TYPE_POI,
    VIOLATING,
)
from pymove.utils.distances import haversine
from pymove.utils.log import logger, progress_bar


def union_poi_bank(
    data: DataFrame,
    label_poi: Text = TYPE_POI,
    banks: Optional[List[Text]] = None,
    inplace: bool = False
) -> Optional[DataFrame]:
    """
    Performs the union between the different bank categories.

    For Points of Interest in a single category named 'banks'.

    Parameters
    ----------
    data : DataFrame
        Input points of interest data
    label_poi : str, optional
        Label referring to the Point of Interest category, by default TYPE_POI
    banks : list of str, optional
        Names of poi refering to banks, by default
            banks = [
            'bancos_filiais',
            'bancos_agencias',
            'bancos_postos',
            'bancos_PAE',
            'bank',
        ]
    inplace : boolean, optional
        if set to true the original dataframe will be altered to contain
        the result of the filtering, otherwise a copy will be returned, by default False

    Returns
    -------
    DataFrame
        data with poi or None

    Examples
    --------
    >>> from pymove.utils.integration import union_poi_bank
    >>> pois_df
              lat          lon   id          type_poi
    0   39.984094   116.319236    1              bank
    1   39.984198   116.319322    2       randomvalue
    2   39.984224   116.319402    3     bancos_postos
    3   39.984211   116.319389    4       randomvalue
    4   39.984217   116.319422    5        bancos_PAE
    5   39.984710   116.319865    6     bancos_postos
    6   39.984674   116.319810    7   bancos_agencias
    7   39.984623   116.319773    8    bancos_filiais
    8   39.984606   116.319732    9             banks
    9   39.984555   116.319728   10             banks
    >>> union_poi_bank(pois_df)
              lat          lon   id      type_poi
    0   39.984094   116.319236    1         banks
    1   39.984198   116.319322    2   randomvalue
    2   39.984224   116.319402    3         banks
    3   39.984211   116.319389    4   randomvalue
    4   39.984217   116.319422    5         banks
    5   39.984710   116.319865    6         banks
    6   39.984674   116.319810    7         banks
    7   39.984623   116.319773    8         banks
    8   39.984606   116.319732    9         banks
    9   39.984555   116.319728   10         banks
    """
    if not inplace:
        data = data.copy()
    logger.debug('union bank categories to one category')
    logger.debug('... There are {} -- {}'.format(data[label_poi].nunique(), label_poi))
    if banks is None:
        banks = [
            'bancos_filiais',
            'bancos_agencias',
            'bancos_postos',
            'bancos_PAE',
            'bank',
        ]
    filter_bank = data[label_poi].isin(banks)
    data.at[data[filter_bank].index, label_poi] = 'banks'
    if not inplace:
        return data


def union_poi_bus_station(
    data: DataFrame,
    label_poi: Text = TYPE_POI,
    bus_stations: Optional[List[Text]] = None,
    inplace: bool = False
) -> Optional[DataFrame]:
    """
    Performs the union between the different bus station categories.

    For Points of Interest in a single category named 'bus_station'.

    Parameters
    ----------
    data : DataFrame
        Input points of interest data
    label_poi : str, optional
        Label referring to the Point of Interest category, by default TYPE_POI
    bus_stations : list of str, optional
        Names of poi refering to bus_stations, by default
            bus_stations = [
                'transit_station',
                'pontos_de_onibus'
            ]
    inplace : boolean, optional
        if set to true the original dataframe will be altered to contain
        the result of the filtering, otherwise a copy will be returned, by default False

    Returns
    -------
    DataFrame
        data with poi or None

    Examples
    --------
    >>> from pymove.utils.integration import union_poi_bus_station
    >>> pois_df
              lat          lon  id           type_poi
    0   39.984094   116.319236   1    transit_station
    1   39.984198   116.319322   2        randomvalue
    2   39.984224   116.319402   3    transit_station
    3   39.984211   116.319389   4   pontos_de_onibus
    4   39.984217   116.319422   5    transit_station
    5   39.984710   116.319865   6        randomvalue
    6   39.984674   116.319810   7        bus_station
    7   39.984623   116.319773   8        bus_station
    >>> union_poi_bus_station(pois_df)
              lat          lon  id           type_poi
    0   39.984094   116.319236   1        bus_station
    1   39.984198   116.319322   2        randomvalue
    2   39.984224   116.319402   3        bus_station
    3   39.984211   116.319389   4        bus_station
    4   39.984217   116.319422   5        bus_station
    5   39.984710   116.319865   6        randomvalue
    6   39.984674   116.319810   7        bus_station
    7   39.984623   116.319773   8        bus_station
    """
    if not inplace:
        data = data.copy()
    logger.debug('union bus station categories to one category')
    if bus_stations is None:
        bus_stations = [
            'transit_station',
            'pontos_de_onibus'
        ]
    filter_bus_station = data[label_poi].isin(
        bus_stations
    )
    data.at[data[filter_bus_station].index, label_poi] = 'bus_station'
    if not inplace:
        return data


def union_poi_bar_restaurant(
    data: DataFrame,
    label_poi: Text = TYPE_POI,
    bar_restaurant: Optional[List[Text]] = None,
    inplace: bool = False
) -> Optional[DataFrame]:
    """
    Performs the union between bar and restaurant categories.

    For Points of Interest in a single category named 'bar-restaurant'.

    Parameters
    ----------
    data : DataFrame
        Input points of interest data
    label_poi : str, optional
        Label referring to the Point of Interest category, by default TYPE_POI
    bar_restaurant : list of str, optional
        Names of poi refering to bars or restaurants, by default
         bar_restaurant = [
            'restaurant',
            'bar'
        ]
    inplace : boolean, optional
        if set to true the original dataframe will be altered to contain
        the result of the filtering, otherwise a copy will be returned, by default False

    Returns
    -------
    DataFrame
        data with poi or None

    Examples
    --------
    >>> from pymove.utils.integration import union_poi_bar_restaurant
    >>> pois_df
              lat          lon   id         type_poi
    0   39.984094   116.319236    1       restaurant
    1   39.984198   116.319322    2       restaurant
    2   39.984224   116.319402    3      randomvalue
    3   39.984211   116.319389    4              bar
    4   39.984217   116.319422    5              bar
    5   39.984710   116.319865    6   bar-restaurant
    6   39.984674   116.319810    7        random123
    7   39.984623   116.319773    8              123
    >>> union_poi_bar_restaurant(pois_df)
              lat          lon   id         type_poi
    0   39.984094   116.319236    1   bar-restaurant
    1   39.984198   116.319322    2   bar-restaurant
    2   39.984224   116.319402    3      randomvalue
    3   39.984211   116.319389    4   bar-restaurant
    4   39.984217   116.319422    5   bar-restaurant
    5   39.984710   116.319865    6   bar-restaurant
    6   39.984674   116.319810    7        random123
    7   39.984623   116.319773    8              123
    """
    if not inplace:
        data = data.copy()
    logger.debug('union restaurant and bar categories to one category')
    if bar_restaurant is None:
        bar_restaurant = ['restaurant', 'bar']
    filter_bar_restaurant = data[label_poi].isin(bar_restaurant)
    data.at[data[filter_bar_restaurant].index, label_poi] = 'bar-restaurant'
    if not inplace:
        return data


def union_poi_parks(
    data: DataFrame,
    label_poi: Text = TYPE_POI,
    parks: Optional[List[Text]] = None,
    inplace: bool = False
) -> Optional[DataFrame]:
    """
    Performs the union between park categories.

    For Points of Interest in a single category named 'parks'.

    Parameters
    ----------
    data : DataFrame
        Input points of interest data
    label_poi : str, optional
        Label referring to the Point of Interest category, by default TYPE_POI
    parks : list of str, optional
        Names of poi refering to parks, by default
            parks = [
                'pracas_e_parques',
                'park'
            ]
    inplace : boolean, optional
        if set to true the original dataframe will be altered to contain
        the result of the filtering, otherwise a copy will be returned, by default False

    Returns
    -------
    DataFrame
        data with poi or None

    Examples
    --------
    >>> from pymove.utils.integration import union_poi_parks
    >>> pois_df
              lat          lon   id           type_poi
    0   39.984094   116.319236    1   pracas_e_parques
    1   39.984198   116.319322    2               park
    2   39.984224   116.319402    3              parks
    3   39.984211   116.319389    4             random
    4   39.984217   116.319422    5                123
    5   39.984710   116.319865    6               park
    6   39.984674   116.319810    7              parks
    7   39.984623   116.319773    8   pracas_e_parques
    >>> union_poi_parks(pois_df)
              lat          lon   id           type_poi
    0   39.984094   116.319236    1              parks
    1   39.984198   116.319322    2              parks
    2   39.984224   116.319402    3              parks
    3   39.984211   116.319389    4             random
    4   39.984217   116.319422    5                123
    5   39.984710   116.319865    6              parks
    6   39.984674   116.319810    7              parks
    7   39.984623   116.319773    8              parks
    """
    if not inplace:
        data = data.copy()
    logger.debug('union parks categories to one category')
    if parks is None:
        parks = ['pracas_e_parques', 'park']
    filter_parks = data[label_poi].isin(parks)
    data.at[data[filter_parks].index, label_poi] = 'parks'
    if not inplace:
        return data


def union_poi_police(
    data: DataFrame,
    label_poi: Text = TYPE_POI,
    police: Optional[List[Text]] = None,
    inplace: bool = False
) -> Optional[DataFrame]:
    """
    Performs the union between police categories.

    For Points of Interest in a single category named 'police'.

    Parameters
    ----------
    data : DataFrame
        Input points of interest data
    label_poi : str, optional
        Label referring to the Point of Interest category, by default TYPE_POI
    police : list of str, optional
        Names of poi refering to police stations, by default
            police = [
                'distritos_policiais',
                'delegacia'
            ]
    inplace : boolean, optional
        if set to true the original dataframe will be altered to contain
        the result of the filtering, otherwise a copy will be returned, by default False

    Returns
    -------
    DataFrame
        data with poi or None

    Examples
    --------
    >>> from pymove.utils.integration import union_poi_police
    >>> pois_df
              lat          lon   id              type_poi
    0   39.984094   116.319236    1   distritos_policiais
    1   39.984198   116.319322    2                police
    2   39.984224   116.319402    3                police
    3   39.984211   116.319389    4   distritos_policiais
    4   39.984217   116.319422    5                random
    5   39.984710   116.319865    6           randomvalue
    6   39.984674   116.319810    7                   123
    7   39.984623   116.319773    8           bus_station
    >>> union_poi_police(pois_df)
              lat          lon   id              type_poi
    0   39.984094   116.319236    1                police
    1   39.984198   116.319322    2                police
    2   39.984224   116.319402    3                police
    3   39.984211   116.319389    4                police
    4   39.984217   116.319422    5                random
    5   39.984710   116.319865    6           randomvalue
    6   39.984674   116.319810    7                   123
    7   39.984623   116.319773    8           bus_station
    """
    if not inplace:
        data = data.copy()
    logger.debug('union distritos policies and police categories')
    if police is None:
        police = ['distritos_policiais', 'delegacia']
    filter_police = data[label_poi].isin(police)
    data.at[data[filter_police].index, label_poi] = 'police'
    if not inplace:
        return data


def join_collective_areas(
    data: DataFrame,
    areas: DataFrame,
    label_geometry: Text = GEOMETRY,
    inplace: bool = False
) -> Optional[DataFrame]:
    """
    Performs the integration between trajectories and collective areas.

    Generating a new column that informs if the point of the
    trajectory is inserted in a collective area.

    Parameters
    ----------
    data : geopandas.GeoDataFrame
        The input trajectory data
    areas : geopandas.GeoDataFrame
        The input coletive areas data
    label_geometry : str, optional
        Label referring to the Point of Interest category, by default GEOMETRY
    inplace : boolean, optional
        if set to true the original dataframe will be altered to contain
        the result of the filtering, otherwise a copy will be returned, by default False

    Returns
    -------
    DataFrame
        data with joined geometries or None

    Examples
    --------
    >>> from pymove.utils.integration import join_collective_areas
    >>> data
              lat          lon              datetime   id                      geometry
    0   39.984094   116.319236   2008-10-23 05:53:05    1    POINT (116.31924 39.98409)
    1   39.984198   116.319322   2008-10-23 05:53:06    1    POINT (116.31932 39.98420)
    2   39.984224   116.319402   2008-10-23 05:53:11    1    POINT (116.31940 39.98422)
    3   39.984211   116.319389   2008-10-23 05:53:16    1    POINT (116.31939 39.98421)
    4   39.984217   116.319422   2008-10-23 05:53:21    1    POINT (116.31942 39.98422)
    >>> area_c
             lat         lon               datetime  id                         geometry
    0  39.984094  116.319236    2008-10-23 05:53:05   1     POINT (116.319236 39.984094)
    1  40.006436  116.317701    2008-10-23 10:53:31   1     POINT (116.317701 40.006436)
    2  40.014125  116.306159    2008-10-23 23:43:56   1     POINT (116.306159 40.014125)
    3  39.984211  116.319389    2008-10-23 05:53:16   1     POINT (116.319389 39.984211)
        POINT (116.32687 39.97901)
    >>> join_collective_areas(gdf, area_c)
    >>> gdf.head()
             lat         lon                datetime  id \
                            geometry    violating
    0  39.984094  116.319236    2008-10-23 05:53:05   1 \
        POINT (116.319236 39.984094)         True
    1  39.984198  116.319322    2008-10-23 05:53:06   1 \
        POINT (116.319322 39.984198)        False
    2  39.984224  116.319402    2008-10-23 05:53:11   1 \
        POINT (116.319402 39.984224)        False
    3  39.984211  116.319389    2008-10-23 05:53:16   1 \
        POINT (116.319389 39.984211)         True
    4  39.984217  116.319422    2008-10-23 05:53:21   1 \
        POINT (116.319422 39.984217)        False

    """
    if not inplace:
        data = data.copy()
    logger.debug('Integration between trajectories and collectives areas')
    Geometry = namedtuple('Geometry', 'geom coordinates')

    polygons = areas[label_geometry].apply(
        lambda g: Geometry(g.__class__, g.__geo_interface__.get('coordinates'))
    ).unique()
    polygons = [p.geom(p.coordinates) for p in polygons]
    data[VIOLATING] = False
    for p in progress_bar(polygons, desc='Joining trajectories and areas'):
        intersects = data[label_geometry].apply(lambda x: x.intersects(p))
        index = data[intersects].index
        data.at[index, VIOLATING] = True
    if not inplace:
        return data


def _reset_and_creates_id_and_lat_lon(
    data: DataFrame,
    df_pois: DataFrame,
    lat_lon_poi: bool = True,
    reset_index: bool = True
) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
    """
    Resets the indexes of the dataframes.

    Returns the minimum distance
    between the two dataframes, and return their respective variables
    (id, tags, latitude and longitude).

    Parameters
    ----------
    data : DataFrame
        The input trajectory data.
    df_pois : DataFrame
        The input point of interest data.
    lat_lon_poi : bool, optional
        Flag to determine if the ids and tags is of size equivalent to df_pois,
        by default True
    reset_index : bool, optional
        Flag for reset index of the df_pois and data dataframes before the join,
        by default True

    Returns
    -------
    distances, ids, tags, lat, lon: arrays with default values for join operation

    Examples
    --------
    >>> from pymove.utils.integration import _reset_and_creates_id_and_lat_lon
    >>> move_df.head()
              lat          lon              datetime   id
    0   39.984094   116.319236   2008-10-23 05:53:05    1
    >>> pois.head()
              lat          lon   id     type_poi                  name_poi
    0   39.984094   116.319236    1      policia            distrito_pol_1
    >>> _reset_and_creates_id_and_lat_lon(move_df, pois)
    (
        array([inf]),
        array([''], dtype=object),
        array([''], dtype=object),
        array([inf]),
        array([inf])
    )
    """
    if reset_index:
        logger.debug('... Resetting index to operation...')
        data.reset_index(drop=True, inplace=True)
        df_pois.reset_index(drop=True, inplace=True)

    # create numpy array to store new column to DataFrame of movement objects
    distances = np.full(
        data.shape[0], np.Infinity, dtype=np.float64
    )

    ids = np.full(data.shape[0], '', dtype='object_')
    tags = np.full(data.shape[0], '', dtype='object_')

    # creating lat and lon array to operation
    if lat_lon_poi:
        lat = np.full(df_pois.shape[0], np.Infinity, dtype=np.float64)
        lon = np.full(df_pois.shape[0], np.Infinity, dtype=np.float64)
    else:
        lat = np.full(data.shape[0], np.Infinity, dtype=np.float64)
        lon = np.full(data.shape[0], np.Infinity, dtype=np.float64)

    return distances, ids, tags, lat, lon


def _reset_set_window__and_creates_event_id_type(
    data: DataFrame, df_events: DataFrame, time_window: float, label_date: Text = DATETIME
) -> Tuple[Series, Series, ndarray, ndarray, ndarray]:
    """
    Resets the indexes of the dataframes.

    Set time window, and returns
    the current distance between the two dataframes, and return their
    respective variables (event_id, event_type).

    Parameters
    ----------
    data : DataFrame
        The input trajectory data.
    df_events : DataFrame
        The input event point of interest data.
    time_window : float
        Number of seconds of the time window.
    label_date : str, optional
        Label of data referring to the datetime, by default DATETIME

    Returns
    -------
    window_starts, window_ends, current_distances, event_id, event_type

    Examples
    --------
    >>> from pymove.utils.integration import
    _reset_set_window__and_creates_event_id_type
    >>> move_df.head()
              lat          lon              datetime   id
    0   39.984094   116.319236   2008-10-23 05:53:05    1
    >>> pois_df
              lat          lon   event_id              datetime             event_type
    0   39.984094   116.319236          1   2008-10-24 01:57:57     show do tropykalia
    >>> _reset_set_window__and_creates_event_id_type(move_df, pois, 600)
    (
        0   2008-10-23 05:43:05
        Name: datetime, dtype: datetime64[ns],
        0   2008-10-23 06:03:05
        Name: datetime, dtype: datetime64[ns],
        array([inf]),
        array([''], dtype=object),
        array([''], dtype=object)
    )
    """
    # get a vector with windows time to each point
    data.reset_index(drop=True, inplace=True)
    df_events.reset_index(drop=True, inplace=True)

    # compute windows time
    window_starts = data[label_date] - Timedelta(seconds=time_window)
    window_ends = data[label_date] + Timedelta(seconds=time_window)

    # create vector to store distances
    current_distances = np.full(
        data.shape[0], np.Infinity, dtype=np.float64
    )
    event_type = np.full(data.shape[0], '', dtype='object_')
    event_id = np.full(data.shape[0], '', dtype='object_')

    return window_starts, window_ends, current_distances, event_id, event_type


def _reset_set_window_and_creates_event_id_type_all(
    data: DataFrame, df_events: DataFrame, time_window: float, label_date: Text = DATETIME
) -> Tuple[Series, Series, ndarray, ndarray, ndarray]:
    """
    Resets the indexes of the dataframes.

    Set time window, and returns
    the current distance between the two dataframes, and return their
    respective variables (event_id, event_type).

    Parameters
    ----------
    data : DataFrame
        The input trajectory data.
    df_events : DataFrame
        The input event point of interest data.
    time_window : float
        Number of seconds of the time window.
    label_date : str
        Label of data referring to the datetime.

    Returns
    -------
    window_starts, window_ends, current_distances, event_id, event_type
        arrays with default values for join operation

    Examples
    --------
    >>> from pymove.utils.integration import _reset_set_window_and_creates_event_id_type_all  # noqa
    >>> move_df.head()
              lat          lon              datetime   id
    0   39.984094   116.319236   2008-10-23 05:53:05    1
    >>> pois_df
              lat          lon   event_id              datetime             event_type
    0   39.984094   116.319236          1   2008-10-24 01:57:57     show do tropykalia
    >>> _reset_set_window_and_creates_event_id_type_all(move_df, pois, 600)
    (
        0   2008-10-23 05:43:05
        Name: datetime, dtype: datetime64[ns],
        0   2008-10-23 06:03:05
        Name: datetime, dtype: datetime64[ns],
        array([None], dtype=object),
        array([None], dtype=object),
        array([None], dtype=object)
    )
    """
    # get a vector with windows time to each point
    data.reset_index(drop=True, inplace=True)
    df_events.reset_index(drop=True, inplace=True)

    # compute windows time
    window_starts = data[label_date] - Timedelta(seconds=time_window)
    window_ends = data[label_date] + Timedelta(seconds=time_window)

    # create vector to store distances
    current_distances = np.full(
        data.shape[0], None, dtype=np.ndarray
    )
    event_type = np.full(data.shape[0], None, dtype=np.ndarray)
    event_id = np.full(data.shape[0], None, dtype=np.ndarray)

    return window_starts, window_ends, current_distances, event_id, event_type


def join_with_pois(
    data: DataFrame,
    df_pois: DataFrame,
    label_id: Text = TRAJ_ID,
    label_poi_name: Text = NAME_POI,
    reset_index: bool = True,
    inplace: bool = False
):
    """
    Performs the integration between trajectories and the closest point of interest.

    Generating two new columns referring to the
    name and the distance from the point of interest closest
    to each point of the trajectory.

    Parameters
    ----------
    data : DataFrame
        The input trajectory data.
    df_pois : DataFrame
        The input point of interest data.
    label_id : str, optional
        Label of df_pois referring to the Point of Interest id, by default TRAJ_ID
    label_poi_name : str, optional
        Label of df_pois referring to the Point of Interest name, by default NAME_POI
    reset_index : bool, optional
        Flag for reset index of the df_pois and data dataframes before the join,
        by default True
    inplace : boolean, optional
        if set to true the original dataframe will be altered to contain
        the result of the filtering, otherwise a copy will be returned, by default False

    Examples
    --------
    >>> from pymove.utils.integration import join_with_pois
    >>>  move_df
              lat          lon              datetime   id
    0   39.984094   116.319236   2008-10-23 05:53:05    1
    1   39.984559   116.326696   2008-10-23 10:37:26    1
    2   40.002899   116.321520   2008-10-23 10:50:16    1
    3   40.016238   116.307691   2008-10-23 11:03:06    1
    4   40.013814   116.306525   2008-10-23 11:58:33    2
    5   40.009735   116.315069   2008-10-23 23:50:45    2
    >>> pois
              lat          lon   id   type_poi              name_poi
    0   39.984094   116.319236    1    policia        distrito_pol_1
    1   39.991013   116.326384    2    policia       policia_federal
    2   40.010000   116.312615    3   comercio   supermercado_aroldo
    >>> join_with_pois(move_df, pois)
              lat          lon              datetime   id   id_poi \
           dist_poi              name_poi
    0   39.984094   116.319236   2008-10-23 05:53:05    1        1 \
           0.000000        distrito_pol_1
    1   39.984559   116.326696   2008-10-23 10:37:26    1        1 \
         637.690216        distrito_pol_1
    2   40.002899   116.321520   2008-10-23 10:50:16    1        3 \
        1094.860663   supermercado_aroldo
    3   40.016238   116.307691   2008-10-23 11:03:06    1        3 \
         810.542998   supermercado_aroldo
    4   40.013814   116.306525   2008-10-23 11:58:33    2        3 \
         669.973155   supermercado_aroldo
    5   40.009735   116.315069   2008-10-23 23:50:45    2        3 \
         211.069129   supermercado_aroldo
    """
    if not inplace:
        data = data.copy()
        df_pois = df_pois.copy()

    values = _reset_and_creates_id_and_lat_lon(data, df_pois, False, reset_index)
    minimum_distances, ids_pois, tag_pois, lat_poi, lon_poi = values

    df_pois.rename(
        columns={label_id: TRAJ_ID, label_poi_name: NAME_POI},
        inplace=True
    )

    for idx, row in progress_bar(
        df_pois.iterrows(), total=len(df_pois), desc='Optimized integration with POIs'
    ):
        # update lat and lon of current index
        lat_poi.fill(row[LATITUDE])
        lon_poi.fill(row[LONGITUDE])

        # First iteration is minimum distances
        if idx == 0:
            minimum_distances = np.array(
                haversine(
                    lat_poi,
                    lon_poi,
                    data[LATITUDE].values,
                    data[LONGITUDE].values
                )
            )
            ids_pois.fill(row[label_id])
            tag_pois.fill(row[label_poi_name])
        else:
            # compute dist between a POI and ALL
            current_distances = np.float64(
                haversine(
                    lat_poi,
                    lon_poi,
                    data[LATITUDE].values,
                    data[LONGITUDE].values
                )
            )
            compare = current_distances < minimum_distances
            minimum_distances = np.minimum(
                current_distances, minimum_distances, dtype=np.float64
            )
            ids_pois[compare] = row[label_id]
            tag_pois[compare] = row[label_poi_name]

    data[ID_POI] = ids_pois
    data[DIST_POI] = minimum_distances
    data[NAME_POI] = tag_pois
    logger.debug('Integration with POI was finalized')

    if not inplace:
        return data


def join_with_pois_by_category(
    data: DataFrame,
    df_pois: DataFrame,
    label_category: Text = TYPE_POI,
    label_id: Text = TRAJ_ID,
    inplace: bool = False
):
    """
    Performs the integration between trajectories and each type of points of interest.

    Generating new columns referring to the
    category and distance from the nearest point of interest
    that has this category at each point of the trajectory.

    Parameters
    ----------
    data : DataFrame
        The input trajectory data.
    df_pois : DataFrame
        The input point of interest data.
    label_category : str, optional
        Label of df_pois referring to the point of interest category, by default TYPE_POI
    label_id : str, optional
        Label of df_pois referring to the point of interest id, by default TRAJ_ID
    inplace : boolean, optional
        if set to true the original dataframe will be altered to contain
        the result of the filtering, otherwise a copy will be returned, by default False

    Examples
    --------
    >>> from pymove.utils.integration import join_with_pois_by_category
    >>>  move_df
              lat          lon              datetime   id
    0   39.984094   116.319236   2008-10-23 05:53:05    1
    1   39.984559   116.326696   2008-10-23 10:37:26    1
    2   40.002899   116.321520   2008-10-23 10:50:16    1
    3   40.016238   116.307691   2008-10-23 11:03:06    1
    4   40.013814   116.306525   2008-10-23 11:58:33    2
    5   40.009735   116.315069   2008-10-23 23:50:45    2
    >>> pois
              lat          lon   id   type_poi              name_poi
    0   39.984094   116.319236    1    policia        distrito_pol_1
    1   39.991013   116.326384    2    policia       policia_federal
    2   40.010000   116.312615    3   comercio   supermercado_aroldo
    >>> join_with_pois_by_category(move_df, pois)
              lat          lon              datetime   id \
            id_policia   dist_policia   id_comercio   dist_comercio
    0   39.984094   116.319236   2008-10-23 05:53:05    1 \
                     1       0.000000             3     2935.310277
    1   39.984559   116.326696   2008-10-23 10:37:26    1 \
                     1     637.690216             3     3072.696379
    2   40.002899   116.321520   2008-10-23 10:50:16    1 \
                     2    1385.087181             3     1094.860663
    3   40.016238   116.307691   2008-10-23 11:03:06    1 \
                     2    3225.288831             3      810.542998
    4   40.013814   116.306525   2008-10-23 11:58:33    2 \
                     2    3047.838222             3      669.973155
    5   40.009735   116.315069   2008-10-23 23:50:45    2 \
                     2    2294.075820             3      211.069129
    """
    if not inplace:
        data = data.copy()
        df_pois = df_pois.copy()

    logger.debug('Integration with POIs...')

    # get a vector with windows time to each point
    data.reset_index(drop=True, inplace=True)
    df_pois.reset_index(drop=True, inplace=True)

    # create numpy array to store new column to DataFrame of movement objects
    current_distances = np.full(
        data.shape[0], np.Infinity, dtype=np.float64
    )
    ids_pois = np.full(data.shape[0], np.NAN, dtype='object_')

    unique_categories = df_pois[label_category].unique()
    size_categories = len(unique_categories)
    logger.debug('There are %s categories' % size_categories)

    for i, c in enumerate(unique_categories, start=1):
        # creating lat and lon array to operation
        df_category = df_pois[df_pois[label_category] == c]
        df_category.reset_index(drop=True, inplace=True)

        desc = 'computing dist to {} category ({}/{})'.format(c, i, size_categories)
        for idx, row in progress_bar(data.iterrows(), total=len(data), desc=desc):
            lat_user = np.full(
                df_category.shape[0], row[LATITUDE], dtype=np.float64
            )
            lon_user = np.full(
                df_category.shape[0], row[LONGITUDE], dtype=np.float64
            )

            # computing distances to
            distances = haversine(
                lat_user,
                lon_user,
                df_category[LATITUDE].values,
                df_category[LONGITUDE].values,
            )

            # get index to arg_min and min distance
            index_min = np.argmin(distances)

            # setting data for a single object movement
            current_distances[idx] = np.min(distances)
            ids_pois[idx] = df_category.at[index_min, label_id]

        data['id_%s' % c] = ids_pois
        data['dist_%s' % c] = current_distances
    logger.debug('Integration with POI was finalized')

    if not inplace:
        return data


def join_with_events(
    data: DataFrame,
    df_events: DataFrame,
    label_date: Text = DATETIME,
    time_window: int = 900,
    label_event_id: Text = EVENT_ID,
    label_event_type: Text = EVENT_TYPE,
    inplace: bool = False
):
    """
    Performs the integration between trajectories and the closest event in time window.

    Generating new columns referring to the
    category of the point of interest, the distance from the
    nearest point of interest based on time of each point of
    the trajectories.

    Parameters
    ----------
    data : DataFrame
        The input trajectory data.
    df_events : DataFrame
        The input events points of interest data.
    label_date : str, optional
        Label of data referring to the datetime of the input trajectory data,
        by default DATETIME
    time_window : float, optional
        tolerable length of time range in `seconds` for assigning the event's
        point of interest to the trajectory point, by default 900
    label_event_id : str, optional
        Label of df_events referring to the id of the event, by default EVENT_ID
    label_event_type : str, optional
        Label of df_events referring to the type of the event, by default EVENT_TYPE
    inplace : boolean, optional
        if set to true the original dataframe will be altered to contain
        the result of the filtering, otherwise a copy will be returned, by default False

    Examples
    --------
    >>> from pymove.utils.integration import join_with_events
    >>>  move_df
             lat         lon            datetime  id
    0  39.984094  116.319236 2008-10-23 05:53:05   1
    1  39.984559  116.326696 2008-10-23 10:37:26   1
    2  39.993527  116.326483 2008-10-24 00:02:14   2
    3  39.978575  116.326975 2008-10-24 00:22:01   3
    4  39.981668  116.310769 2008-10-24 01:57:57   3
    >>> events
             lat         lon  id            datetime  event_type             event_id
    0  39.984094  116.319236   1 2008-10-23 05:53:05        show     forro_tropykalia
    1  39.991013  116.326384   2 2008-10-23 10:37:26        show     dia_do_municipio
    2  40.010000  116.312615   3 2008-10-24 01:57:57        feira   adocao_de_animais
    >>> join_with_events(move_df, events)
             lat         lon            datetime  id \
               event_type   dist_event             event_id
    0  39.984094  116.319236 2008-10-23 05:53:05   1 \
                     show     0.000000     forro_tropykalia
    1  39.984559  116.326696 2008-10-23 10:37:26   1 \
                     show   718.144152     dia_do_municipio
    2  39.993527  116.326483 2008-10-24 00:02:14   2 \
                                   inf
    3  39.978575  116.326975 2008-10-24 00:22:01   3 \
                                   inf
    4  39.981668  116.310769 2008-10-24 01:57:57   3 \
                    feira  3154.296880    adocao_de_animais

    Raises
    ------
    ValueError
        If feature generation fails

    """
    if not inplace:
        data = data.copy()
        df_events = df_events.copy()

    values = _reset_set_window__and_creates_event_id_type(
        data, df_events, time_window, label_date
    )
    *_, current_distances, event_id, event_type = values
    window_starts, window_ends, *_ = _reset_set_window__and_creates_event_id_type(
        df_events, data, time_window, label_date
    )

    minimum_distances = np.full(
        data.shape[0], np.Infinity, dtype=np.float64
    )

    # Rename for access columns of each row directly
    df_events.rename(
        columns={label_event_id: label_event_id, label_event_type: label_event_type},
        inplace=True
    )

    for idx, row in progress_bar(
        df_events.iterrows(), total=len(df_events), desc='Integration with Events'
    ):
        df_filtered = filters.by_datetime(
            data, window_starts[idx], window_ends[idx]
        )
        if df_filtered is None:
            raise ValueError('Filtering datetime failed!')

        size_filter = df_filtered.shape[0]

        if size_filter > 0:
            indexes = df_filtered.index
            lat_event = np.full(
                df_filtered.shape[0], row[LATITUDE], dtype=np.float64
            )
            lon_event = np.full(
                df_filtered.shape[0], row[LONGITUDE], dtype=np.float64
            )

            # First iteration is minimum distances
            if idx == 0:
                minimum_distances[indexes] = haversine(
                    lat_event,
                    lon_event,
                    df_filtered[LATITUDE].values,
                    df_filtered[LONGITUDE].values,
                )
                event_id[indexes] = row[label_event_id]
                event_type[indexes] = row[label_event_type]
            else:
                current_distances[indexes] = haversine(
                    lat_event,
                    lon_event,
                    df_filtered[LATITUDE].values,
                    df_filtered[LONGITUDE].values,
                )
                compare = current_distances < minimum_distances
                minimum_distances = np.minimum(
                    current_distances, minimum_distances
                )
                event_id[compare] = row[label_event_id]
                event_type[compare] = row[label_event_type]

    data[label_event_id] = event_id
    data[DIST_EVENT] = minimum_distances
    data[label_event_type] = event_type
    logger.debug('Integration with events was completed')

    if not inplace:
        return data


def join_with_event_by_dist_and_time(
    data: DataFrame,
    df_events: DataFrame,
    label_date: Text = DATETIME,
    label_event_id: Text = EVENT_ID,
    label_event_type: Text = EVENT_TYPE,
    time_window: float = 3600,
    radius: float = 1000,
    inplace: bool = False
):
    """
    Performs the integration between trajectories and events on windows.

    Generating new columns referring to the category of the point of interest,
    the distance between the location of the user and location of the poi
    based on the distance and on time of each point of the trajectories.

    Parameters
    ----------
    data : DataFrame
        The input trajectory data.
    df_pois : DataFrame
        The input events points of interest data.
    label_date : str, optional
        Label of data referring to the datetime of the input trajectory data,
        by default DATETIME
    label_event_id : str, optional
        Label of df_events referring to the id of the event, by default EVENT_ID
    label_event_type : str, optional
        Label of df_events referring to the type of the event, by default EVENT_TYPE
    time_window : float, optional
        tolerable length of time range in `seconds`for assigning the event's
        point of interest to the trajectory point, by default 3600
    radius: float, optional
        maximum radius of pois in `meters`, by default 1000
    inplace : boolean, optional
        if set to true the original dataframe will be altered to contain
        the result of the filtering, otherwise a copy will be returned, by default False

    Examples
    --------
    >>> from pymove.utils.integration import join_with_pois_by_dist_and_datetime
    >>>  move_df
             lat         lon            datetime  id
    0  39.984094  116.319236 2008-10-23 05:53:05   1
    1  39.984559  116.326696 2008-10-23 10:37:26   1
    2  39.993527  116.326483 2008-10-24 00:02:14   2
    3  39.978575  116.326975 2008-10-24 00:22:01   3
    4  39.981668  116.310769 2008-10-24 01:57:57   3
    >>> events
             lat         lon  id            datetime type_poi           name_poi
    0  39.984094  116.319236   1 2008-10-23 05:53:05     show   forro_tropykalia
    1  39.991013  116.326384   2 2008-10-23 10:27:26  corrida   racha_de_jumento
    2  39.990013  116.316384   2 2008-10-23 10:37:26     show   dia_do_municipio
    3  40.010000  116.312615   3 2008-10-24 01:57:57    feira  adocao_de_animais
    >>> join_with_pois_by_dist_and_datetime(move_df, pois)
    >>> move_df
             lat         lon            datetime  id \
               type_poi          dist_event                              name_poi
    0  39.984094  116.319236 2008-10-23 05:53:05   1 \
                 [show]               [0.0]                    [forro_tropykalia]
    1  39.984559  116.326696 2008-10-23 10:37:26   1 \
        [corrida, show]  [718.144, 1067.53]  [racha_de_jumento, dia_do_municipio]
    2  39.993527  116.326483 2008-10-24 00:02:14   2 \
                  None                 None                                  None
    3  39.978575  116.326975 2008-10-24 00:22:01   3 \
                  None                 None                                  None
    4  39.981668  116.310769 2008-10-24 01:57:57   3 \
                  None                 None                                  None

    Raises
    ------
    ValueError
        If feature generation fails

    """
    if label_date not in df_events:
        raise KeyError("POI's DataFrame must contain a %s column" % label_date)

    if not inplace:
        data = data.copy()
        df_events = df_events.copy()

    values = _reset_set_window_and_creates_event_id_type_all(
        data, df_events, time_window, label_date
    )

    window_start, window_end, current_distances, event_id, event_type = values

    for idx, row in progress_bar(
        data.iterrows(), total=len(data), desc='Integration with Events'
    ):
        # set min and max of coordinates by radius
        bbox = filters.get_bbox_by_radius(
            (row[LATITUDE], row[LONGITUDE]), radius
        )

        # filter event by radius
        df_filtered = filters.by_bbox(
            df_events, bbox, inplace=False
        )

        if df_filtered is None:
            raise ValueError('Filtering bbox failed')

        # filter event by datetime
        filters.by_datetime(
            df_filtered,
            start_datetime=window_start[idx],
            end_datetime=window_end[idx],
            inplace=True
        )

        # get df_filtered size
        size_filter = df_filtered.shape[0]

        if size_filter > 0:
            # reseting index of data frame
            df_filtered.reset_index(drop=True, inplace=True)

            # create lat and lon array to operation
            lat_user = np.full(
                size_filter, row[LATITUDE], dtype=np.float64
            )
            lon_user = np.full(
                size_filter, row[LONGITUDE], dtype=np.float64
            )

            # calculate of distances between points
            distances = haversine(
                lat_user,
                lon_user,
                df_filtered[LATITUDE].to_numpy(),
                df_filtered[LONGITUDE].to_numpy()
            )

            current_distances[idx] = distances
            event_type[idx] = df_filtered[label_event_type].to_numpy(dtype=np.ndarray)
            event_id[idx] = df_filtered[label_event_id].to_numpy(dtype=np.ndarray)

    data[label_event_id] = event_id
    data[DIST_EVENT] = current_distances
    data[label_event_type] = event_type
    logger.debug('Integration with event was completed')

    if not inplace:
        return data


def join_with_home_by_id(
    data: DataFrame,
    df_home: DataFrame,
    label_id: Text = TRAJ_ID,
    label_address: Text = ADDRESS,
    label_city: Text = CITY,
    drop_id_without_home: bool = False,
    inplace: bool = False
):
    """
    Performs the integration between trajectories and home points.

    Generating new columns referring to the distance of the nearest
    home point, address and city of each trajectory point.

    Parameters
    ----------
    data : DataFrame
        The input trajectory data.
    df_home : DataFrame
        The input home points data.
    label_id : str, optional
        Label of df_home referring to the home point id, by default TRAJ_ID
    label_address : str, optional
        Label of df_home referring to the home point address, by default ADDRESS
    label_city : str, optional
        Label of df_home referring to the point city, by default CITY
    drop_id_without_home : bool, optional
        flag as an option to drop id's that don't have houses, by default False
    inplace : boolean, optional
        if set to true the original dataframe will be altered to contain
        the result of the filtering, otherwise a copy will be returned, by default False

    Examples
    --------
    >>> from pymove.utils.integration import join_with_home_by_id
    >>> move_df
              lat          lon              datetime   id
    0   39.984094   116.319236   2008-10-23 05:53:05    1
    1   39.984559   116.326696   2008-10-23 10:37:26    1
    2   40.002899   116.321520   2008-10-23 10:50:16    1
    3   40.016238   116.307691   2008-10-23 11:03:06    1
    4   40.013814   116.306525   2008-10-23 11:58:33    2
    5   40.009735   116.315069   2008-10-23 23:50:45    2
    >>> home_df
              lat          lon   id   formatted_address            city
    0   39.984094   116.319236    1          rua da mae       quixiling
    1   40.013821   116.306531    2      rua da familia   quixeramoling
    >>> join_with_home_by_id(move_df, home_df)
    >>> move_df
        id         lat          lon              datetime     dist_home \
                  home                city
    0    1   39.984094   116.319236   2008-10-23 05:53:05      0.000000 \
            rua da mae           quixiling
    1    1   39.984559   116.326696   2008-10-23 10:37:26    637.690216 \
            rua da mae           quixiling
    2    1   40.002899   116.321520   2008-10-23 10:50:16   2100.053501 \
            rua da mae           quixiling
    3    1   40.016238   116.307691   2008-10-23 11:03:06   3707.066732 \
            rua da mae           quixiling
    4    2   40.013814   116.306525   2008-10-23 11:58:33      0.931101 \
        rua da familia       quixeramoling
    5    2   40.009735   116.315069   2008-10-23 23:50:45    857.417540 \
        rua da familia       quixeramoling
    """
    if not inplace:
        data = data.copy()
        df_home = df_home.copy()

    ids_without_home = []

    if data.index.name is None:
        logger.debug('...setting {} as index'.format(label_id))
        data.set_index(label_id, inplace=True)

    for idx in progress_bar(
        data.index.unique(), total=len(data.index.unique()), desc='Integration with Home'
    ):
        filter_home = df_home[label_id] == idx

        if df_home[filter_home].shape[0] == 0:
            logger.debug('...id: {} has not HOME'.format(idx))
            ids_without_home.append(idx)
        else:
            home = df_home[filter_home].iloc[0]
            lat_user = data.at[idx, LATITUDE].values
            lon_user = data.at[idx, LONGITUDE].values

            # if user has a single tuple
            if not isinstance(lat_user, np.ndarray):
                lat_home = home[LATITUDE].values
                lon_home = home[LONGITUDE].values
                data.at[idx, DIST_HOME] = haversine(
                    lat_user, lon_user, lat_home, lon_home
                )
                data.at[idx, HOME] = home[label_address]
                data.at[idx, label_city] = home[label_city]
            else:
                lat_home = np.full(
                    data.loc[idx].shape[0], home[LATITUDE], dtype=np.float64
                )
                lon_home = np.full(
                    data.loc[idx].shape[0], home[LONGITUDE], dtype=np.float64
                )
                data.at[idx, DIST_HOME] = haversine(
                    lat_user, lon_user, lat_home, lon_home
                )
                data.at[idx, HOME] = np.array(home[label_address])
                data.at[idx, label_city] = np.array(home[label_city])

    data.reset_index(inplace=True)
    logger.debug('... Resetting index')

    if drop_id_without_home:
        data.drop(data.loc[data[TRAJ_ID].isin(ids_without_home)].index, inplace=True)

    if not inplace:
        return data


def merge_home_with_poi(
    data: DataFrame,
    label_dist_poi: Text = DIST_POI,
    label_name_poi: Text = NAME_POI,
    label_id_poi: Text = ID_POI,
    label_home: Text = HOME,
    label_dist_home: Text = DIST_HOME,
    drop_columns: bool = True,
    inplace: bool = False
):
    """
    Performs or merges the points of interest and the trajectories.

    Considering the starting points as other points of interest,
    generating a new DataFrame.

    Parameters
    ----------
    data : DataFrame
        The input trajectory data, with join_with_pois and join_with_home_by_id applied.
    label_dist_poi : str, optional
        Label of data referring to the distance from the nearest point of interest,
        by default DIST_POI
    label_name_poi : str, optional
        Label of data referring to the name from the nearest point of interest,
        by default NAME_POI
    label_id_poi : str, optional
        Label of data referring to the id from the nearest point of interest,
        by default ID_POI
    label_home : str, optional
        Label of df_home referring to the home point, by default HOME
    label_dist_home: str, optional
        Label of df_home referring to the distance to the home point,
        by default DIST_HOME
    drop_columns : bool, optional
        Flag that controls the deletion of the columns referring to the
        id and the distance from the home point, by default
    inplace : boolean, optional
        if set to true the original dataframe will be altered to contain
        the result of the filtering, otherwise a copy will be returned, by default False

    Examples
    --------
    >>> from pymove.utils.integration import (
    >>>    merge_home_with_poi,
    >>>    join_with_home_by_id
    >>> )
    >>> move_df
              lat          lon              datetime   id \
                   id_poi       dist_poi          name_poi
    0   39.984094   116.319236   2008-10-23 05:53:05    1 \
                        1       0.000000    distrito_pol_1
    1   39.984559   116.326696   2008-10-23 10:37:26    1 \
                        1     637.690216    distrito_pol_1
    2   40.002899   116.321520   2008-10-23 10:50:16    1 \
                        2    1385.087181   policia_federal
    3   40.016238   116.307691   2008-10-23 11:03:06    1 \
                        2    3225.288831   policia_federal
    4   40.013814   116.306525   2008-10-23 11:58:33    2 \
                        2    3047.838222   policia_federal
    5   40.009735   116.315069   2008-10-23 23:50:45    2 \
                        2    2294.075820   policia_federal
    >>> home_df
               lat          lon   id   formatted_address            city
    0   39.984094   116.319236    1          rua da mae       quixiling
    1   40.013821   116.306531    2      rua da familia   quixeramoling
    >>> join_with_home_by_id(move, home_df, inplace=True)
    >>> move_df
        id         lat          lon              datetime   id_poi      dist_poi \
               name_poi    dist_home         home                city
    0    1   39.984094   116.319236   2008-10-23 05:53:05        1      0.000000 \
         distrito_pol_1     0.000000        rua da mae       quixiling
    1    1   39.984559   116.326696   2008-10-23 10:37:26        1    637.690216 \
         distrito_pol_1   637.690216        rua da mae       quixiling
    2    1   40.002899   116.321520   2008-10-23 10:50:16        2   1385.087181 \
        policia_federal  2100.053501        rua da mae       quixiling
    3    1   40.016238    16.307691   2008-10-23 11:03:06        2   3225.288831 \
        policia_federal  3707.066732        rua da mae       quixiling
    4    2   40.013814   116.306525   2008-10-23 11:58:33        2   3047.838222 \
        policia_federal     0.931101    rua da familia   quixeramoling
    5    2   40.009735   116.315069   2008-10-23 23:50:45        2   2294.075820 \
        policia_federal   857.417540    rua da familia   quixeramoling
    >>> merge_home_with_poi(move_df)
        id         lat          lon              datetime           id_poi \
          dist_poi           name_poi            city
    0    1   39.984094   116.319236   2008-10-23 05:53:05       rua da mae \
          0.000000               home       quixiling
    1    1   39.984559   116.326696   2008-10-23 10:37:26       rua da mae \
        637.690216               home       quixiling
    2    1   40.002899   116.321520   2008-10-23 10:50:16                2 \
       1385.087181    policia_federal       quixiling
    3    1   40.016238   116.307691   2008-10-23 11:03:06                2 \
       3225.288831    policia_federal       quixiling
    4    2   40.013814   116.306525   2008-10-23 11:58:33   rua da familia \
          0.931101               home   quixeramoling
    5    2   40.009735   116.315069   2008-10-23 23:50:45   rua da familia \
        857.417540               home   quixeramoling
    """
    if not inplace:
        data = data.copy()

    logger.debug('merge home with POI using shortest distance')
    idx = data[data[label_dist_home] <= data[label_dist_poi]].index

    data.loc[idx, label_name_poi] = label_home
    data.loc[idx, label_dist_poi] = data.loc[idx, label_dist_home]
    data.loc[idx, label_id_poi] = data.loc[idx, label_home]

    if(drop_columns):
        data.drop(columns=[label_dist_home, label_home], inplace=True)

    if not inplace:
        return data
