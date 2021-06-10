"""
Integration operations.

union_poi_bank,
union_poi_bus_station,
union_poi_bar_restaurant,
union_poi_parks,
union_poi_police,
join_collective_areas,
join_with_pois,
join_with_pois_optimizer,
join_with_pois_by_category,
join_with_poi_datetime,
join_with_poi_datetime_optimizer,
join_with_pois_by_dist_and_datetime,
join_with_home_by_id,
merge_home_with_poi

"""

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


def union_poi_bank(data: DataFrame, label_poi: Optional[Text] = TYPE_POI):
    """
    Performs the union between the different bank categories.

    For Points of Interest in a single category named 'banks'.

    Parameters
    ----------
    data : DataFrame
        Input points of interest data
    label_poi : str, optional
        Label referring to the Point of Interest category, by default TYPE_POI

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
    logger.debug('union bank categories to one category')
    logger.debug('... There are {} -- {}'.format(data[label_poi].nunique(), label_poi))
    banks = [
        'bancos_filiais',
        'bancos_agencias',
        'bancos_postos',
        'bancos_PAE',
        'bank',
    ]
    filter_bank = data[label_poi].isin(banks)
    data.at[data[filter_bank].index, label_poi] = 'banks'


def union_poi_bus_station(data: DataFrame, label_poi: Optional[Text] = TYPE_POI):
    """
    Performs the union between the different bus station categories.

    For Points of Interest in a single category named 'bus_station'.

    Parameters
    ----------
    data : DataFrame
        Input points of interest data
    label_poi : str, optional
        Label referring to the Point of Interest category, by default TYPE_POI

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
    >>> pois_df
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
    logger.debug('union bus station categories to one category')
    filter_bus_station = data[label_poi].isin(
        ['transit_station', 'pontos_de_onibus']
    )
    data.at[data[filter_bus_station].index, label_poi] = 'bus_station'


def union_poi_bar_restaurant(data: DataFrame, label_poi: Optional[Text] = TYPE_POI):
    """
    Performs the union between bar and restaurant categories.

    For Points of Interest in a single category named 'bar-restaurant'.

    Parameters
    ----------
    data : DataFrame
        Input points of interest data
    label_poi : str, optional
        Label referring to the Point of Interest category, by default TYPE_POI

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
    >>> pois_df
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
    logger.debug('union restaurant and bar categories to one category')
    filter_bar_restaurant = data[label_poi].isin(['restaurant', 'bar'])
    data.at[data[filter_bar_restaurant].index, label_poi] = 'bar-restaurant'


def union_poi_parks(data: DataFrame, label_poi: Optional[Text] = TYPE_POI):
    """
    Performs the union between park categories.

    For Points of Interest in a single category named 'parks'.

    Parameters
    ----------
    data : DataFrame
        Input points of interest data
    label_poi : str, optional
        Label referring to the Point of Interest category, by default TYPE_POI

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
    logger.debug('union parks categories to one category')
    filter_parks = data[label_poi].isin(['pracas_e_parques', 'park'])
    data.at[data[filter_parks].index, label_poi] = 'parks'


def union_poi_police(data: DataFrame, label_poi: Optional[Text] = TYPE_POI):
    """
    Performs the union between police categories.

    For Points of Interest in a single category named 'police'.

    Parameters
    ----------
    data : DataFrame
        Input points of interest data
    label_poi : str, optional
        Label referring to the Point of Interest category, by default TYPE_POI

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
    >>> pois_df
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
    logger.debug('union distritos policies and police categories')
    filter_police = data[label_poi] == 'distritos_policiais'
    data.at[data[filter_police].index, label_poi] = 'police'


def join_collective_areas(
    gdf_: DataFrame, gdf_rules_: DataFrame, label_geometry: Optional[Text] = GEOMETRY
):
    """
    Performs the integration between trajectories and collective areas.

    Generating a new column that informs if the point of the
    trajectory is inserted in a collective area.

    Parameters
    ----------
    gdf_ : geopandas.GeoDataFrame
        The input trajectory data
    gdf_rules_ : geopandas.GeoDataFrame
        The input coletive areas data
    label_geometry : str, optional
        Label referring to the Point of Interest category, by default GEOMETRY

    Examples
    --------
    >>> from pymove.utils.integration import join_collective_areas
    >>> gdf.head()
              lat          lon              datetime   id                     geometry
    0   39.984094   116.319236   2008-10-23 05:53:05    1   POINT (116.31924 39.98409)
    1   39.984198   116.319322   2008-10-23 05:53:06    1   POINT (116.31932 39.98420)
    2   39.984224   116.319402   2008-10-23 05:53:11    1   POINT (116.31940 39.98422)
    3   39.984211   116.319389   2008-10-23 05:53:16    1   POINT (116.31939 39.98421)
    4   39.984217   116.319422   2008-10-23 05:53:21    1   POINT (116.31942 39.98422)
    >>> area_c
                              lat          lon              datetime   id    geometry\
    0                   39.984094   116.319236   2008-10-23 05:53:05    1\
           POINT (116.31924 39.98409)
    500                 40.006436   116.317701   2008-10-23 10:53:31    1\
           POINT (116.31770 40.00644)
    1000                40.014125   116.306159   2008-10-23 23:43:56    1\
           POINT (116.30616 40.01412)
    1500                39.979009   116.326873   2008-10-24 00:11:29    1\
           POINT (116.32687 39.97901)
    >>> join_collective_areas(gdf, area_c)
    >>> gdf.head()
                               lat          lon              datetime   id\
                          geometry    violating
    0                    39.984094   116.319236   2008-10-23 05:53:05    1\
        POINT (116.31924 39.98409)         True
    1                    39.984198   116.319322   2008-10-23 05:53:06    1\
        POINT (116.31932 39.98420)        False
    2                    39.984224   116.319402   2008-10-23 05:53:11    1\
        POINT (116.31940 39.98422)        False
    3                    39.984211   116.319389   2008-10-23 05:53:16    1\
        POINT (116.31939 39.98421)        False
    4                    39.984217   116.319422   2008-10-23 05:53:21    1\
        POINT (116.31942 39.98422)        False
    """
    logger.debug('Integration between trajectories and collectives areas')

    polygons = gdf_rules_[label_geometry].unique()
    gdf_[VIOLATING] = False
    for p in progress_bar(polygons, desc='Joining trajectories and areas'):
        # intersects = gdf_[label_geometry].apply(lambda x: x.intersects(p))
        intersects = gdf_[label_geometry].intersects(p)
        index = gdf_[intersects].index
        gdf_.at[index, VIOLATING] = True


def _reset_and_creates_id_and_lat_lon(
    data: DataFrame,
    df_pois: DataFrame,
    lat_lon_poi: Optional[bool] = True,
    reset_index: Optional[bool] = True
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
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
    1   39.984559   116.326696   2008-10-23 10:37:26    1
    2   40.002899   116.321520   2008-10-23 10:50:16    1
    3   40.016238   116.307691   2008-10-23 11:03:06    1
    4   40.013814   116.306525   2008-10-23 11:58:33    2
    >>> pois.head()
              lat          lon   id     type_poi                  name_poi
    0   39.984094   116.319236    1      policia            distrito_pol_1
    1   39.991013   116.326384    2      policia           policia_federal
    2   40.010000   116.312615    3     comercio       supermercado_aroldo
    3   40.013821   116.306531    4   show forro_               tropykalia
    4   40.008099   116.317711    5   risca-faca   rinha_de_galo_world_cup
    >>> _reset_and_creates_id_and_lat_lon(move_df, pois)
    (array([inf, inf, inf, inf, inf, inf, inf, inf, inf]),
    array(['', '', '', '', '', '', '', '', ''], dtype=object),
    array(['', '', '', '', '', '', '', '', ''], dtype=object),
    array([inf, inf, inf, inf, inf, inf, inf]),
    array([inf, inf, inf, inf, inf, inf, inf]))
    >>> print(type(_reset_and_creates_id_and_lat_lon(move_df, pois)))
    <class 'tuple'>
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
    data: DataFrame, df_events: DataFrame, label_date: Text, time_window: int
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
    label_date : str
        Label of data referring to the datetime.
    time_window : int
        Number of seconds of the time window.

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
    1   39.984559   116.326696   2008-10-23 10:37:26    1
    2   40.002899   116.321520   2008-10-23 10:50:16    1
    3   40.016238   116.307691   2008-10-23 11:03:06    1
    4   40.013814   116.306525   2008-10-23 11:58:33    2
    >>> pois_df
                       lat          lon   event_id              datetime\
                event_type
    0            39.984094   116.319236          1   2008-10-24 01:57:57\
        show do tropykalia
    1            39.991013   116.326384          2   2008-10-24 00:22:01\
      evento da prefeitura
    2            40.010000   116.312615          3   2008-10-25 00:21:01\
          show do seu joao
    3            40.013821   116.306531          4   2008-10-26 00:22:01\
                     missa
    >>> _reset_set_window__and_creates_event_id_type(move_df, pois,
     'datetime', 600)
        (0   2008-10-23 05:43:05
    1   2008-10-23 10:27:26
    2   2008-10-23 10:40:16
    3   2008-10-23 10:53:06
    4   2008-10-23 11:48:33
    5   2008-10-23 23:40:45
    6   2008-10-23 23:52:14
    7   2008-10-24 00:12:01
    8   2008-10-24 01:47:57
    Name: datetime, dtype: datetime64[ns],
    0   2008-10-23 06:03:05
    1   2008-10-23 10:47:26
    2   2008-10-23 11:00:16
    3   2008-10-23 11:13:06
    4   2008-10-23 12:08:33
    5   2008-10-24 00:00:45
    6   2008-10-24 00:12:14
    7   2008-10-24 00:32:01
    8   2008-10-24 02:07:57
    Name: datetime, dtype: datetime64[ns],
    array([inf, inf, inf, inf, inf, inf, inf, inf, inf]),
    array(['', '', '', '', '', '', '', '', ''], dtype=object),
    array(['', '', '', '', '', '', '', '', ''], dtype=object))
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
    data: DataFrame, df_events: DataFrame, label_date: Text, time_window: int
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
    label_date : str
        Label of data referring to the datetime.
    time_window : Int
        Number of seconds of the time window.

    Returns
    -------
    window_starts, window_ends, current_distances, event_id, event_type
        arrays with default values for join operation

    Examples
    --------
    >>> from pymove.utils.integration import
    _reset_set_window_and_creates_event_id_type_all
    >>> move_df.head()
              lat          lon              datetime   id
    0   39.984094   116.319236   2008-10-23 05:53:05    1
    1   39.984559   116.326696   2008-10-23 10:37:26    1
    2   40.002899   116.321520   2008-10-23 10:50:16    1
    3   40.016238   116.307691   2008-10-23 11:03:06    1
    4   40.013814   116.306525   2008-10-23 11:58:33    2
    >>> pois_df
                       lat          lon   event_id              datetime\
                event_type
    0            39.984094   116.319236          1   2008-10-24 01:57:57\
        show do tropykalia
    1            39.991013   116.326384          2   2008-10-24 00:22:01\
      evento da prefeitura
    2            40.010000   116.312615          3   2008-10-25 00:21:01\
          show do seu joao
    3            40.013821   116.306531          4   2008-10-26 00:22:01\
                     missa
    >>> _reset_set_window_and_creates_event_id_type_all(move_df, pois,
     'datetime', 600)
        (0   2008-10-23 05:43:05
    1   2008-10-23 10:27:26
    2   2008-10-23 10:40:16
    3   2008-10-23 10:53:06
    4   2008-10-23 11:48:33
    5   2008-10-23 23:40:45
    6   2008-10-23 23:52:14
    7   2008-10-24 00:12:01
    8   2008-10-24 01:47:57
    Name: datetime, dtype: datetime64[ns],
    0   2008-10-23 06:03:05
    1   2008-10-23 10:47:26
    2   2008-10-23 11:00:16
    3   2008-10-23 11:13:06
    4   2008-10-23 12:08:33
    5   2008-10-24 00:00:45
    6   2008-10-24 00:12:14
    7   2008-10-24 00:32:01
    8   2008-10-24 02:07:57
    Name: datetime, dtype: datetime64[ns],
    array([None, None, None, None, None, None, None, None, None], dtype=object),
    array([None, None, None, None, None, None, None, None, None], dtype=object),
    array([None, None, None, None, None, None, None, None, None], dtype=object))
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
    label_id: Optional[Text] = TRAJ_ID,
    label_poi_name: Optional[Text] = NAME_POI,
    reset_index: Optional[Text] = True
):
    """
    Performs the integration between trajectories and points of interest.

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

    Examples
    --------
    >>> from pymove.utils.integration import join_with_pois
    >>>  POIs.head()
                      unique_id          osmid   element_type     amenity   fee\
                          geometry
    0               node/269492188   269492188           node     toilets    no\
        POINT (116.26750 39.98087)
    1               node/274942287   274942287           node     toilets   NaN\
        POINT (116.27358 39.99664)
    2               node/276320137   276320137           node   fast_food   NaN\
        POINT (116.33756 39.97541)
    3               node/276320142   276320142           node     massage   NaN\
        POINT (116.33751 39.97546)
    4               node/286242547   286242547           node     toilets   NaN\
        POINT (116.19982 40.00670)
    ....
    >>> move_df.head()
              lat          lon              datetime  id
    0   39.984094   116.319236   2008-10-23 05:53:05   1
    1   39.984198   116.319322   2008-10-23 05:53:06   1
    2   39.984224   116.319402   2008-10-23 05:53:11   1
    3   39.984211   116.319389   2008-10-23 05:53:16   1
    4   39.984217   116.319422   2008-10-23 05:53:21   1
    ...
    >>> join_with_pois(move_df, POIs, label_id='osmid', label_poi_name='name')
    >>> move_df.head()
              lat          lon              datetime  id\
           id_poi      dist_poi             name_poi
    0   39.984094   116.319236   2008-10-23 05:53:05   1\
       5572452688   116.862844     太平洋影城(中关村店)
    1   39.984198   116.319322   2008-10-23 05:53:06   1\
       5572452688   119.142692     太平洋影城(中关村店)
    2   39.984224   116.319402   2008-10-23 05:53:11   1\
       5572452688   116.595117     太平洋影城(中关村店)
    3   39.984211   116.319389   2008-10-23 05:53:16   1\
       5572452688   116.257378     太平洋影城(中关村店)
    4   39.984217   116.319422   2008-10-23 05:53:21   1\
       5572452688   114.886759     太平洋影城(中关村店)
    """
    values = _reset_and_creates_id_and_lat_lon(data, df_pois, True, reset_index)
    current_distances, ids_pois, tag_pois, lat_user, lon_user = values

    for idx, row in progress_bar(
        data.iterrows(), total=len(data), desc='Integration with POIs'
    ):
        # create a vector to each lat
        lat_user.fill(row[LATITUDE])
        lon_user.fill(row[LONGITUDE])

        # computing distances to idx
        distances = np.float64(
            haversine(
                lat_user,
                lon_user,
                df_pois[LATITUDE].values,
                df_pois[LONGITUDE].values,
            )
        )

        # get index to arg_min and min distance
        index_min = np.argmin(distances)
        current_distances[idx] = np.min(distances)

        # setting data for a single object movement
        ids_pois[idx] = df_pois.at[index_min, label_id]
        tag_pois[idx] = df_pois.at[index_min, label_poi_name]

    data[ID_POI] = ids_pois
    data[DIST_POI] = current_distances
    data[NAME_POI] = tag_pois

    logger.debug('Integration with POI was finalized')


def join_with_pois_optimizer(
    data,
    df_pois: DataFrame,
    label_id: Optional[Text] = TRAJ_ID,
    label_poi_name: Optional[Text] = NAME_POI,
    dist_poi: Optional[List] = None,
    reset_index: Optional[Text] = True
):
    """
    Performs the integration between trajectories and points of interest.

    Generating two new columns referring to the
    name and distance from the nearest point of interest,
    within the limit of distance determined by the parameter 'dist_poi',
    of each point in the trajectory.

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
    dist_poi : list, optional
        List containing the minimum distance limit between each type of
        point of interest and each point of the trajectory to classify the
        point of interest closest to each point of the trajectory, by default None
    reset_index : bool, optional
        Flag for reset index of the df_pois and data dataframes before the join,
        by default True

    Examples
    --------
    >>> from pymove.utils.integration import join_with_pois_optimizer
    >>> from pymove.utils.integration import join_with_pois
    >>>  POIs.head()
                      unique_id          osmid   element_type     amenity   fee\
                          geometry
    0               node/269492188   269492188           node     toilets    no\
        POINT (116.26750 39.98087)
    1               node/274942287   274942287           node     toilets   NaN\
        POINT (116.27358 39.99664)
    2               node/276320137   276320137           node   fast_food   NaN\
        POINT (116.33756 39.97541)
    3               node/276320142   276320142           node     massage   NaN\
        POINT (116.33751 39.97546)
    4               node/286242547   286242547           node     toilets   NaN\
        POINT (116.19982 40.00670)
    ....
    >>> move_df.head()
              lat          lon              datetime  id
    0   39.984094   116.319236   2008-10-23 05:53:05   1
    1   39.984198   116.319322   2008-10-23 05:53:06   1
    2   39.984224   116.319402   2008-10-23 05:53:11   1
    3   39.984211   116.319389   2008-10-23 05:53:16   1
    4   39.984217   116.319422   2008-10-23 05:53:21   1
    ...
    >>> join_with_pois_optimizer(move_df, POIs, label_id='osmid',
     label_poi_name='name', dist_poi=np.array([100,9,1,50,50,10,20]))
    'the size of the dist_poi is different from the size of pois'
    """
    if len(df_pois[label_poi_name].unique()) == len(dist_poi):
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
                minimum_distances = np.float64(
                    haversine(
                        lat_poi,
                        lon_poi,
                        data[LATITUDE].values,
                        data[LONGITUDE].values
                    )
                )
                ids_pois.fill(row.id)
                tag_pois.fill(row.type_poi)
            else:
                # compute dist between a POI and ALL
                logger.debug(data[LONGITUDE].values)
                current_distances = np.float64(
                    haversine(
                        lat_poi,
                        lon_poi,
                        data[LATITUDE].values,
                        data[LONGITUDE].values
                    )
                )
                compare = current_distances < minimum_distances
                index_true = np.where(compare is True)[0]
                minimum_distances = np.minimum(
                    current_distances, minimum_distances, dtype=np.float64
                )

                if index_true.shape[0] > 0:
                    ids_pois[index_true] = row.id
                    tag_pois[index_true] = row.type_poi

        data[ID_POI] = ids_pois
        data[DIST_POI] = minimum_distances
        data[NAME_POI] = tag_pois
        logger.debug('Integration with POI was finalized')
    else:
        logger.warning('the size of the dist_poi is different from the size of pois')


def join_with_pois_by_category(
    data: DataFrame,
    df_pois: DataFrame,
    label_category: Optional[Text] = TYPE_POI,
    label_id: Optional[Text] = TRAJ_ID
):
    """
    Performs the integration between trajectories and points of interest.

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

    Examples
    --------
    >>> from pymove.utils.integration import join_with_pois_by_category
    >>> POIs.head()
                      unique_id          osmid   element_type     amenity   fee/
                          geometry
    0               node/269492188   269492188           node     toilets    no/
        POINT (116.26750 39.98087)
    1               node/274942287   274942287           node     toilets   NaN/
        POINT (116.27358 39.99664)
    2               node/276320137   276320137           node   fast_food   NaN/
        POINT (116.33756 39.97541)
    3               node/276320142   276320142           node     massage   NaN/
        POINT (116.33751 39.97546)
    4               node/286242547   286242547           node     toilets   NaN/
        POINT (116.19982 40.00670)
    ....
    >>> move_df.head()
              lat          lon              datetime  id
    0   39.984094   116.319236   2008-10-23 05:53:05   1
    1   39.984198   116.319322   2008-10-23 05:53:06   1
    2   39.984224   116.319402   2008-10-23 05:53:11   1
    3   39.984211   116.319389   2008-10-23 05:53:16   1
    4   39.984217   116.319422   2008-10-23 05:53:21   1
    ...
    >>> join_with_pois_by_category(move_df, POIs,
     label_category='amenity', label_id='osmid')
    >>> move_df.head()
                lat          lon               datetime   id/
        id_toilets   dist_toilets          id_fast_food ...
    0    39.984094     116.319236   2008-10-23 05:53:05   1/
         274942287    4132.229067             276320137 ...
    1    39.984198     116.319322   2008-10-23 05:53:06   1/
         274942287    4135.240296             276320137 ...
    2    39.984224     116.319402   2008-10-23 05:53:11   1/
         274942287    4140.698090             276320137 ...
    3    39.984211     116.319389   2008-10-23 05:53:16   1/
         274942287    4140.136625             276320137 ...
    4    39.984217     116.319422   2008-10-23 05:53:21   1/
         274942287    4142.564150             276320137 ...
    """
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


def join_with_poi_datetime(
    data: DataFrame,
    df_events: DataFrame,
    label_date: Optional[Text] = DATETIME,
    time_window: Optional[int] = 900,
    label_event_id: Optional[Text] = EVENT_ID,
    label_event_type: Optional[Text] = EVENT_TYPE
):
    """
    Performs the integration between trajectories and points of interest.

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
        tolerable length of time range for assigning the event's
        point of interest to the trajectory point, by default 900
    label_event_id : str, optional
        Label of df_events referring to the id of the event, by default EVENT_ID
    label_event_type : str, optional
        Label of df_events referring to the type of the event, by default EVENT_TYPE

    Examples
    --------
    >>> from pymove.utils.integration import join_with_poi_datetime
    >>> POIs_events
              unique_id        osmid   element_type        amenity\
                    fee
    0    node/269492188    269492188           node        toilets\
                     no...
    1    node/931686797    931686797           node    post_office\
                    NaN...
    2    node/992592626    992592626           node        parking\
                    NaN...
    3   node/1423043074   1423043074           node       car_wash\
                    NaN...
    4   node/1803755348   1803755348           node      telephone\
                    NaN...
    >>> move_df.head()
              lat          lon              datetime  id
    0   39.984094   116.319236   2008-10-23 05:53:05   1
    1   39.984198   116.319322   2008-10-23 05:53:06   1
    2   39.984224   116.319402   2008-10-23 05:53:11   1
    3   39.984211   116.319389   2008-10-23 05:53:16   1
    4   39.984217   116.319422   2008-10-23 05:53:21   1
    >>> join_with_poi_datetime(df_7, POIs_events, label_date='datetime',
     time_window=900, label_event_id='osmid', label_event_type='amenity')
    >>> move_df.head()
              lat          lon              datetime  id\
            osmid   dist_event               amenity
    0   39.984094   116.319236   2008-10-23 05:53:05   1\
        269492188   4422.237186              toilets
    1   39.984198   116.319322   2008-10-23 05:53:06   1\
        269492188   4430.488277              toilets
    2   39.984224   116.319402   2008-10-23 05:53:11   1\
        269492188   4437.521909              toilets
    3   39.984211   116.319389   2008-10-23 05:53:16   1\
        269492188   4436.297310              toilets
    4   39.984217   116.319422   2008-10-23 05:53:21   1\
        269492188   4439.154806              toilets
    """
    values = _reset_set_window__and_creates_event_id_type(
        data, df_events, label_date, time_window
    )
    window_starts, window_ends, current_distances, event_id, event_type = values

    for idx in progress_bar(data.index, total=len(data), desc='Integration with Events'):
        # filter event by datetime
        df_filtered = filters.by_datetime(
            df_events, window_starts[idx], window_ends[idx]
        )
        size_filter = df_filtered.shape[0]

        if size_filter > 0:
            df_filtered.reset_index(drop=True, inplace=True)
            lat_user = np.full(
                size_filter, data.at[idx, LATITUDE], dtype=np.float64
            )
            lon_user = np.full(
                size_filter, data.at[idx, LONGITUDE], dtype=np.float64
            )

            # compute dist to poi filtered
            distances = haversine(
                lat_user,
                lon_user,
                df_filtered[LATITUDE].values,
                df_filtered[LONGITUDE].values,
            )
            # get index to arg_min
            index_arg_min = np.argmin(distances)
            # get min distances
            min_distance = np.min(distances)
            # store data
            current_distances[idx] = min_distance
            event_type[idx] = df_filtered.at[index_arg_min, label_event_type]
            event_id[idx] = df_filtered.at[index_arg_min, label_event_id]

    data[label_event_id] = event_id
    data[DIST_EVENT] = current_distances
    data[label_event_type] = event_type
    logger.debug('Integration with event was completed')


def join_with_poi_datetime_optimizer(
    data: DataFrame,
    df_events: DataFrame,
    label_date: Optional[Text] = DATETIME,
    time_window: Optional[int] = 900,
    label_event_id: Optional[Text] = EVENT_ID,
    label_event_type: Optional[Text] = EVENT_TYPE
):
    """
    Performs a optimized integration between trajectories and points of events.

    Generating new columns referring to
    the category of the event, the distance from the nearest
    event and the time when the event happened at each point of
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
        tolerable length of time range for assigning the event's
        point of interest to the trajectory point, by default 900
    label_event_id : str, optional
        Label of df_events referring to the id of the event, by default EVENT_ID
    label_event_type : str, optional
        Label of df_events referring to the type of the event, by default EVENT_TYPE

    Examples
    --------
    >>> from pymove.utils.integration import join_with_poi_datetime_optimizer
    >>> POIs_events
              unique_id     event_id   element_type     event_type\
                    fee
    0    node/269492188    269492188           node        toilets\
                     no...
    1    node/931686797    931686797           node    post_office\
                    NaN...
    2    node/992592626    992592626           node        parking\
                    NaN...
    3   node/1423043074   1423043074           node       car_wash\
                    NaN...
    4   node/1803755348   1803755348           node      telephone\
                    NaN...
    >>> move_df.head()
              lat          lon              datetime  id
    0   39.984094   116.319236   2008-10-23 05:53:05   1
    1   39.984198   116.319322   2008-10-23 05:53:06   1
    2   39.984224   116.319402   2008-10-23 05:53:11   1
    3   39.984211   116.319389   2008-10-23 05:53:16   1
    4   39.984217   116.319422   2008-10-23 05:53:21   1
    >>> join_with_poi_datetime_optimizer(df_8, POIs_events)
    >>> move_df.head()
              lat          lon              datetime  id\
         event_id   dist_event            event_type
    0   39.984094   116.319236   2008-10-23 05:53:05   1\
        269492188   4422.237186              toilets
    1   39.984198   116.319322   2008-10-23 05:53:06   1\
        269492188   4430.488277              toilets
    2   39.984224   116.319402   2008-10-23 05:53:11   1\
        269492188   4437.521909              toilets
    3   39.984211   116.319389   2008-10-23 05:53:16   1\
        269492188   4436.297310              toilets
    4   39.984217   116.319422   2008-10-23 05:53:21   1\
        269492188   4439.154806              toilets
    """
    values = _reset_set_window__and_creates_event_id_type(
        data, df_events, label_date, time_window
    )
    window_starts, window_ends, current_distances, event_id, event_type = values

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
                event_id[indexes] = row.event_id
                event_type[indexes] = row.event_type
            else:
                current_distances[indexes] = haversine(
                    lat_event,
                    lon_event,
                    df_filtered[LATITUDE].values,
                    df_filtered[LONGITUDE].values,
                )
                compare = current_distances < minimum_distances
                index_true = np.where(compare is True)[0]

                minimum_distances = np.minimum(
                    current_distances, minimum_distances
                )
                event_id[index_true] = row.event_id
                event_type[index_true] = row.event_type

    data[label_event_id] = event_id
    data[DIST_EVENT] = minimum_distances
    data[label_event_type] = event_type
    logger.debug('Integration with events was completed')


def join_with_pois_by_dist_and_datetime(
    data: DataFrame,
    df_pois: DataFrame,
    label_date: Optional[Text] = DATETIME,
    label_event_id: Optional[Text] = EVENT_ID,
    label_event_type: Optional[Text] = EVENT_TYPE,
    time_window: Optional[float] = 3600,
    radius: Optional[float] = 1000,
):
    """
    Performs the integration between trajectories and points of interest.

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
        tolerable length of time range for assigning the event's
        point of interest to the trajectory point, by default 3600
    radius: float, optional
        maximum radius of pois, by default 1000

    Examples
    --------
    >>> from pymove.utils.integration import join_with_pois_by_dist_and_datetime
    >>> POIs_events
              unique_id     event_id   element_type     event_type\
                    fee
    0    node/269492188    269492188           node        toilets\
                     no...
    1    node/931686797    931686797           node    post_office\
                    NaN...
    2    node/992592626    992592626           node        parking\
                    NaN...
    3   node/1423043074   1423043074           node       car_wash\
                    NaN...
    4   node/1803755348   1803755348           node      telephone\
                    NaN...
    >>> move_df.head()
              lat          lon              datetime  id
    0   39.984094   116.319236   2008-10-23 05:53:05   1
    1   39.984198   116.319322   2008-10-23 05:53:06   1
    2   39.984224   116.319402   2008-10-23 05:53:11   1
    3   39.984211   116.319389   2008-10-23 05:53:16   1
    4   39.984217   116.319422   2008-10-23 05:53:21   1
    >>> join_with_poi_datetime_optimizer(df_8, POIs_events)
    >>> move_df.head()
              lat          lon              datetime  id\
         event_id   dist_event            event_type
    0   39.984094   116.319236   2008-10-23 05:53:05   1\
             None         None                  None
    1   39.984198   116.319322   2008-10-23 05:53:06   1\
             None         None                  None
    2   39.984224   116.319402   2008-10-23 05:53:11   1\
             None         None                  None
    3   39.984211   116.319389   2008-10-23 05:53:16   1\
             None         None                  None
    4   39.984217   116.319422   2008-10-23 05:53:21   1\
             None         None                  None
    """
    if label_date not in df_pois:
        raise KeyError("POI's DataFrame must contain a %s column" % label_date)

    values = _reset_set_window_and_creates_event_id_type_all(
        data, df_pois, label_date, time_window
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
            df_pois, bbox
        )

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


def join_with_home_by_id(
    data: DataFrame,
    df_home: DataFrame,
    label_id: Optional[Text] = TRAJ_ID,
    label_address: Optional[Text] = ADDRESS,
    label_city: Optional[Text] = CITY,
    drop_id_without_home: Optional[bool] = False,
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
        flag as an option to drop id's that don't have houses, by default FALSE

    Examples
    --------
    >>> from pymove.utils.integration import join_with_home_by_id
    >>> move_df.head()
              lat          lon              datetime  id
    0   39.984094   116.319236   2008-10-23 05:53:05   1
    1   39.984198   116.319322   2008-10-23 05:53:06   1
    2   39.984224   116.319402   2008-10-23 05:53:11   1
    3   39.984211   116.319389   2008-10-23 05:53:16   1
    4   39.984217   116.319422   2008-10-23 05:53:21   1
    >>> home_df.head()
                lat          lon              datetime  id   formatted_address        city
    300   39.991574   116.326394   2008-10-23 10:42:03   1           Rua1, n02   ChinaTown
    301   39.991652   116.326414   2008-10-23 10:42:08   1           Rua2, n03   ChinaTown
    >>> join_with_home_by_id(move_df, home_df, label_id='id')
    >>> move_df.head()
              lat          lon              datetime  id\
        dist_home         home                  city
    0   39.984094   116.319236   2008-10-23 05:53:05   1\
      1031.348370    Rua1, n02             ChinaTown
    1   39.984198   116.319322   2008-10-23 05:53:06   1\
      1017.690147    Rua1, n02             ChinaTown
    2   39.984224   116.319402   2008-10-23 05:53:11   1\
      1011.332141    Rua1, n02             ChinaTown
    3   39.984211   116.319389   2008-10-23 05:53:16   1\
      1013.152700    Rua1, n02             ChinaTown
    4   39.984217   116.319422   2008-10-23 05:53:21   1\
      1010.959220    Rua1, n02             ChinaTown
    """
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


def merge_home_with_poi(
    data: DataFrame,
    label_dist_poi: Optional[Text] = DIST_POI,
    label_name_poi: Optional[Text] = NAME_POI,
    label_id_poi: Optional[Text] = ID_POI,
    label_home: Optional[Text] = HOME,
    label_dist_home: Optional[Text] = DIST_HOME,
    drop_columns: Optional[bool] = True,
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
        id and the distance from the home point, by default True

    Examples
    --------
    >>> from pymove.utils.integration import merge_home_with_poi
    >>> move_df.head()
              lat          lon              datetime  id\
        dist_home         home                  city
    0   39.984094   116.319236   2008-10-23 05:53:05   1\
      1031.348370    Rua1, n02             ChinaTown
    1   39.984198   116.319322   2008-10-23 05:53:06   1\
      1017.690147    Rua1, n02             ChinaTown
    2   39.984224   116.319402   2008-10-23 05:53:11   1\
      1011.332141    Rua1, n02             ChinaTown
    3   39.984211   116.319389   2008-10-23 05:53:16   1\
      1013.152700    Rua1, n02             ChinaTown
    4   39.984217   116.319422   2008-10-23 05:53:21   1\
      1010.959220    Rua1, n02             ChinaTown
    >>> POIs.head()
                      unique_id          osmid   element_type     amenity   fee\
                          geometry
    0               node/269492188   269492188           node     toilets    no\
        POINT (116.26750 39.98087)
    1               node/274942287   274942287           node     toilets   NaN\
        POINT (116.27358 39.99664)
    2               node/276320137   276320137           node   fast_food   NaN\
        POINT (116.33756 39.97541)
    3               node/276320142   276320142           node     massage   NaN\
        POINT (116.33751 39.97546)
    4               node/286242547   286242547           node     toilets   NaN\
        POINT (116.19982 40.00670)
    ....
    >>> join_with_pois(move_df, POIs, label_id='osmid', label_poi_name='name')
    >>> move_df.head()
               id          lat          lon              datetime\
             city       id_poi     dist_poi             name_poi
    0           1    39.984094   116.319236   2008-10-23 05:53:05\
        ChinaTown   557245268    116.862844    太平洋影城(中关村店)
    1           1    39.984198   116.319322   2008-10-23 05:53:06\
        ChinaTown   5572452688   119.142692    太平洋影城(中关村店)
    2           1    39.984224   116.319402   2008-10-23 05:53:11\
        ChinaTown   5572452688   116.595117    太平洋影城(中关村店)
    3           1    39.984211   116.319389   2008-10-23 05:53:16\
        ChinaTown   5572452688   116.257378    太平洋影城(中关村店)
    4           1    39.984217   116.319422   2008-10-23 05:53:21\
        ChinaTown   5572452688   114.886759    太平洋影城(中关村店)
    """
    logger.debug('merge home with POI using shortest distance')
    idx = data[data[label_dist_home] <= data[label_dist_poi]].index

    data.loc[idx, label_name_poi] = label_home
    data.loc[idx, label_dist_poi] = data.loc[idx, label_dist_home]
    data.loc[idx, label_id_poi] = data.loc[idx, label_home]

    if(drop_columns):
        data.drop(columns=[label_dist_home, label_home], inplace=True)
