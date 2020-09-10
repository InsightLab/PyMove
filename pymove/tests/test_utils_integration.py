import geopandas
import numpy as np
import pandas as pd
from numpy import inf, nan
from numpy.testing import assert_array_almost_equal, assert_array_equal
from pandas import DataFrame, Series, Timestamp
from pandas.testing import assert_frame_equal, assert_series_equal

from pymove import MoveDataFrame
from pymove.utils import integration
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
    POI,
    TRAJ_ID,
    TYPE_POI,
    VIOLATING,
)

list_random_banks = [
    [39.984094, 116.319236, 1, 'bank'],
    [39.984198, 116.319322, 2, 'randomvalue'],
    [39.984224, 116.319402, 3, 'bancos_postos'],
    [39.984211, 116.319389, 4, 'randomvalue'],
    [39.984217, 116.319422, 5, 'bancos_PAE'],
    [39.984710, 116.319865, 6, 'bancos_postos'],
    [39.984674, 116.319810, 7, 'bancos_agencias'],
    [39.984623, 116.319773, 8, 'bancos_filiais'],
    [39.984606, 116.319732, 9, 'banks'],
    [39.984555, 116.319728, 10, 'banks']
]

list_random_bus_station = [
    [39.984094, 116.319236, 1, 'transit_station'],
    [39.984198, 116.319322, 2, 'randomvalue'],
    [39.984224, 116.319402, 3, 'transit_station'],
    [39.984211, 116.319389, 4, 'pontos_de_onibus'],
    [39.984217, 116.319422, 5, 'transit_station'],
    [39.984710, 116.319865, 6, 'randomvalue'],
    [39.984674, 116.319810, 7, 'bus_station'],
    [39.984623, 116.319773, 8, 'bus_station'],
]

list_random_bar_restaurant = [
    [39.984094, 116.319236, 1, 'restaurant'],
    [39.984198, 116.319322, 2, 'restaurant'],
    [39.984224, 116.319402, 3, 'randomvalue'],
    [39.984211, 116.319389, 4, 'bar'],
    [39.984217, 116.319422, 5, 'bar'],
    [39.984710, 116.319865, 6, 'bar-restaurant'],
    [39.984674, 116.319810, 7, 'random123'],
    [39.984623, 116.319773, 8, '123'],
]

list_random_parks = [
    [39.984094, 116.319236, 1, 'pracas_e_parques'],
    [39.984198, 116.319322, 2, 'park'],
    [39.984224, 116.319402, 3, 'parks'],
    [39.984211, 116.319389, 4, 'random'],
    [39.984217, 116.319422, 5, '123'],
    [39.984710, 116.319865, 6, 'park'],
    [39.984674, 116.319810, 7, 'parks'],
    [39.984623, 116.319773, 8, 'pracas_e_parques'],
]

list_random_police = [
    [39.984094, 116.319236, 1, 'distritos_policiais'],
    [39.984198, 116.319322, 2, 'police'],
    [39.984224, 116.319402, 3, 'police'],
    [39.984211, 116.319389, 4, 'distritos_policiais'],
    [39.984217, 116.319422, 5, 'random'],
    [39.984710, 116.319865, 6, 'randomvalue'],
    [39.984674, 116.319810, 7, '123'],
    [39.984623, 116.319773, 8, 'bus_station'],
]

list_move = [
    [39.984094, 116.319236, Timestamp('2008-10-23 05:53:05'), 1],
    [39.984559000000004, 116.326696, Timestamp('2008-10-23 10:37:26'), 1],
    [40.002899, 116.32151999999999, Timestamp('2008-10-23 10:50:16'), 1],
    [40.016238, 116.30769099999999, Timestamp('2008-10-23 11:03:06'), 1],
    [40.013814, 116.306525, Timestamp('2008-10-23 11:58:33'), 2],
    [40.009735, 116.315069, Timestamp('2008-10-23 23:50:45'), 2],
    [39.993527, 116.32648300000001, Timestamp('2008-10-24 00:02:14'), 2],
    [39.978575, 116.326975, Timestamp('2008-10-24 00:22:01'), 3],
    [39.981668, 116.310769, Timestamp('2008-10-24 01:57:57'), 3],
]

list_pois = [
    [39.984094, 116.319236, 1, 'policia', 'distrito_pol_1'],
    [39.991013, 116.326384, 2, 'policia', 'policia_federal'],
    [40.01, 116.312615, 3, 'comercio', 'supermercado_aroldo'],
    [40.013821, 116.306531, 4, 'show', 'forro_tropykalia'],
    [40.008099, 116.31771100000002, 5, 'risca-faca',
     'rinha_de_galo_world_cup'],
    [39.985704, 116.326877, 6, 'evento', 'adocao_de_animais'],
    [39.979393, 116.3119, 7, 'show', 'dia_do_municipio']
]


# Testes de Unions
def test_union_poi_bank():

    pois_df = DataFrame(
        data=list_random_banks,
        columns=[LATITUDE, LONGITUDE, TRAJ_ID, TYPE_POI],
        index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    )

    expected = DataFrame(
        data=[
            [39.984094, 116.319236, 1, 'banks'],
            [39.984198, 116.319322, 2, 'randomvalue'],
            [39.984224, 116.319402, 3, 'banks'],
            [39.984211, 116.319389, 4, 'randomvalue'],
            [39.984217, 116.319422, 5, 'banks'],
            [39.984710, 116.319865, 6, 'banks'],
            [39.984674, 116.319810, 7, 'banks'],
            [39.984623, 116.319773, 8, 'banks'],
            [39.984606, 116.319732, 9, 'banks'],
            [39.984555, 116.319728, 10, 'banks']
        ],
        columns=[LATITUDE, LONGITUDE, TRAJ_ID, TYPE_POI],
        index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    )

    integration.union_poi_bank(pois_df, TYPE_POI)

    assert_frame_equal(pois_df, expected)


def test_union_poi_bus_station():
    pois_df = DataFrame(
        data=list_random_bus_station,
        columns=[LATITUDE, LONGITUDE, TRAJ_ID, TYPE_POI],
        index=[0, 1, 2, 3, 4, 5, 6, 7]
    )

    expected = DataFrame(
        data=[
            [39.984094, 116.319236, 1, 'bus_station'],
            [39.984198, 116.319322, 2, 'randomvalue'],
            [39.984224, 116.319402, 3, 'bus_station'],
            [39.984211, 116.319389, 4, 'bus_station'],
            [39.984217, 116.319422, 5, 'bus_station'],
            [39.984710, 116.319865, 6, 'randomvalue'],
            [39.984674, 116.319810, 7, 'bus_station'],
            [39.984623, 116.319773, 8, 'bus_station'],
        ],
        columns=[LATITUDE, LONGITUDE, TRAJ_ID, TYPE_POI],
        index=[0, 1, 2, 3, 4, 5, 6, 7]
    )

    integration.union_poi_bus_station(pois_df, TYPE_POI)

    assert_frame_equal(pois_df, expected)


def test_union_poi_bar_restaurant():
    pois_df = DataFrame(
        data=list_random_bar_restaurant,
        columns=[LATITUDE, LONGITUDE, TRAJ_ID, TYPE_POI],
        index=[0, 1, 2, 3, 4, 5, 6, 7]
    )

    expected = DataFrame(
        data=[
            [39.984094, 116.319236, 1, 'bar-restaurant'],
            [39.984198, 116.319322, 2, 'bar-restaurant'],
            [39.984224, 116.319402, 3, 'randomvalue'],
            [39.984211, 116.319389, 4, 'bar-restaurant'],
            [39.984217, 116.319422, 5, 'bar-restaurant'],
            [39.984710, 116.319865, 6, 'bar-restaurant'],
            [39.984674, 116.319810, 7, 'random123'],
            [39.984623, 116.319773, 8, '123'],
        ],
        columns=[LATITUDE, LONGITUDE, TRAJ_ID, TYPE_POI],
        index=[0, 1, 2, 3, 4, 5, 6, 7]
    )

    integration.union_poi_bar_restaurant(pois_df, TYPE_POI)

    assert_frame_equal(pois_df, expected)


def test_union_poi_parks():
    pois_df = DataFrame(
        data=list_random_parks,
        columns=[LATITUDE, LONGITUDE, TRAJ_ID, TYPE_POI],
        index=[0, 1, 2, 3, 4, 5, 6, 7]
    )

    expected = DataFrame(
        data=[
            [39.984094, 116.319236, 1, 'parks'],
            [39.984198, 116.319322, 2, 'parks'],
            [39.984224, 116.319402, 3, 'parks'],
            [39.984211, 116.319389, 4, 'random'],
            [39.984217, 116.319422, 5, '123'],
            [39.984710, 116.319865, 6, 'parks'],
            [39.984674, 116.319810, 7, 'parks'],
            [39.984623, 116.319773, 8, 'parks'],
        ],
        columns=[LATITUDE, LONGITUDE, TRAJ_ID, TYPE_POI],
        index=[0, 1, 2, 3, 4, 5, 6, 7]
    )

    integration.union_poi_parks(pois_df, TYPE_POI)

    assert_frame_equal(pois_df, expected)


def test_union_poi_police():
    pois_df = DataFrame(
        data=list_random_police,
        columns=[LATITUDE, LONGITUDE, TRAJ_ID, TYPE_POI],
        index=[0, 1, 2, 3, 4, 5, 6, 7]
    )

    expected = DataFrame(
        data=[
            [39.984094, 116.319236, 1, 'police'],
            [39.984198, 116.319322, 2, 'police'],
            [39.984224, 116.319402, 3, 'police'],
            [39.984211, 116.319389, 4, 'police'],
            [39.984217, 116.319422, 5, 'random'],
            [39.984710, 116.319865, 6, 'randomvalue'],
            [39.984674, 116.319810, 7, '123'],
            [39.984623, 116.319773, 8, 'bus_station'],
        ],
        columns=[LATITUDE, LONGITUDE, TRAJ_ID, TYPE_POI],
        index=[0, 1, 2, 3, 4, 5, 6, 7]
    )

    integration.union_poi_police(pois_df, TYPE_POI)

    assert_frame_equal(pois_df, expected)


# Testes de Joins
def test_join_colletive_areas():
    move_df = DataFrame(
        data=list_move,
        columns=[LATITUDE, LONGITUDE, DATETIME, TRAJ_ID])
    gdf = geopandas.GeoDataFrame(
        move_df, geometry=geopandas.points_from_xy(
            move_df.lon, move_df.lat
        )
    )

    indexes_ac = np.linspace(0, gdf.shape[0], 5)
    area_c = gdf[gdf.index.isin(indexes_ac)].copy()

    integration.join_collective_areas(gdf, area_c)

    expected_df = DataFrame(
        data=list_move,
        columns=[LATITUDE, LONGITUDE, DATETIME, TRAJ_ID])
    expected = geopandas.GeoDataFrame(
        move_df, geometry=geopandas.points_from_xy(
            move_df.lon, move_df.lat
        )
    )

    expected[VIOLATING] = [True, False, True, False, True, False, True, False, False]
    assert_frame_equal(gdf, expected)


def test__reset_and_creates_id_and_lat_lon():
    move_df = MoveDataFrame(list_move)
    pois = DataFrame(
        data=list_pois,
        columns=[LATITUDE, LONGITUDE, TRAJ_ID, TYPE_POI, NAME_POI],
        index=[0, 1, 2, 3, 4, 5, 6]
    )

    dists, ids, tags, lats, lons = (
        integration._reset_and_creates_id_and_lat_lon(
            move_df, pois, True, True
        )
    )
    id_expected = np.full(9, '', dtype='object_')
    tag_expected = np.full(9, '', dtype='object_')
    dist_expected = np.full(
        9, np.Infinity, dtype=np.float64
    )
    lat_expected = np.full(7, np.Infinity, dtype=np.float64)
    lon_expected = np.full(7, np.Infinity, dtype=np.float64)

    assert_array_almost_equal(dists, dist_expected)
    assert_array_equal(ids, id_expected)
    assert_array_equal(tags, tag_expected)
    assert_array_almost_equal(lats, lat_expected)
    assert_array_almost_equal(lons, lon_expected)

    dists, ids, tags, lats, lons = (
        integration._reset_and_creates_id_and_lat_lon(
            move_df, pois, True, False
        )
    )
    assert_array_almost_equal(dists, dist_expected)
    assert_array_equal(ids, id_expected)
    assert_array_equal(tags, tag_expected)
    assert_array_almost_equal(lats, lat_expected)
    assert_array_almost_equal(lons, lon_expected)

    dists, ids, tags, lats, lons = (
        integration._reset_and_creates_id_and_lat_lon(
            move_df, pois, False, True
        )
    )
    lat_expected = np.full(9, np.Infinity, dtype=np.float64)
    lon_expected = np.full(9, np.Infinity, dtype=np.float64)
    assert_array_almost_equal(dists, dist_expected)
    assert_array_equal(ids, id_expected)
    assert_array_equal(tags, tag_expected)
    assert_array_almost_equal(lats, lat_expected)
    assert_array_almost_equal(lons, lon_expected)

    dists, ids, tags, lats, lons = (
        integration._reset_and_creates_id_and_lat_lon(
            move_df, pois, False, False
        )
    )
    assert_array_almost_equal(dists, dist_expected)
    assert_array_equal(ids, id_expected)
    assert_array_equal(tags, tag_expected)
    assert_array_almost_equal(lats, lat_expected)
    assert_array_almost_equal(lons, lon_expected)


def test__reset_set_window__and_creates_event_id_type():
    list_events = [
        [39.984094, 116.319236, 1,
         Timestamp('2008-10-24 01:57:57'), 'show do tropykalia'],
        [39.991013, 116.326384, 2,
         Timestamp('2008-10-24 00:22:01'), 'evento da prefeitura'],
        [40.01, 116.312615, 3,
         Timestamp('2008-10-25 00:21:01'), 'show do seu joao'],
        [40.013821, 116.306531, 4,
         Timestamp('2008-10-26 00:22:01'), 'missa']
    ]
    move_df = MoveDataFrame(list_move)
    pois = DataFrame(
        data=list_events,
        columns=[LATITUDE, LONGITUDE, EVENT_ID, DATETIME, EVENT_TYPE],
        index=[0, 1, 2, 3]
    )

    list_win_start = [
        '2008-10-22T17:23:05.000000000', '2008-10-22T22:07:26.000000000',
        '2008-10-22T22:20:16.000000000', '2008-10-22T22:33:06.000000000',
        '2008-10-22T23:28:33.000000000', '2008-10-23T11:20:45.000000000',
        '2008-10-23T11:32:14.000000000', '2008-10-23T11:52:01.000000000',
        '2008-10-23T13:27:57.000000000'
    ]
    win_start_expected = Series(pd.to_datetime(list_win_start), name=DATETIME)
    list_win_end = [
        '2008-10-23T18:23:05.000000000', '2008-10-23T23:07:26.000000000',
        '2008-10-23T23:20:16.000000000', '2008-10-23T23:33:06.000000000',
        '2008-10-24T00:28:33.000000000', '2008-10-24T12:20:45.000000000',
        '2008-10-24T12:32:14.000000000', '2008-10-24T12:52:01.000000000',
        '2008-10-24T14:27:57.000000000'
    ]
    win_end_expected = Series(pd.to_datetime(list_win_end), name=DATETIME)
    dist_expected = np.full(
        9, np.Infinity, dtype=np.float64
    )
    type_expected = np.full(9, '', dtype='object_')
    id_expected = np.full(9, '', dtype='object_')

    window_starts, window_ends, current_distances, event_id, event_type = (
        integration._reset_set_window__and_creates_event_id_type(
            move_df, pois, DATETIME, 45000
        )
    )

    assert_series_equal(window_starts, win_start_expected)
    assert_series_equal(window_ends, win_end_expected)
    assert_array_almost_equal(current_distances, dist_expected)
    assert_array_equal(event_id, id_expected)
    assert_array_equal(event_type, type_expected)


def test_reset_set_window_and_creates_event_id_type_all():
    list_move = [
        [39.984094, 116.319236, Timestamp('2008-10-23 05:53:05'), 1],
        [39.984559000000004, 116.326696, Timestamp('2008-10-23 10:37:26'), 1],
        [40.002899, 116.32151999999999, Timestamp('2008-10-23 10:50:16'), 1],
        [40.016238, 116.30769099999999, Timestamp('2008-10-23 11:03:06'), 1],
        [40.013814, 116.306525, Timestamp('2008-10-23 11:58:33'), 2],
        [40.009735, 116.315069, Timestamp('2008-10-23 23:50:45'), 2],
        [39.993527, 116.32648300000001, Timestamp('2008-10-24 00:02:14'), 2],
        [39.978575, 116.326975, Timestamp('2008-10-24 00:22:01'), 3],
        [39.981668, 116.310769, Timestamp('2008-10-24 01:57:57'), 3],
    ]

    move_df = MoveDataFrame(list_move)

    list_events = [
        [39.984094, 116.319236, 1, Timestamp('2008-10-24 01:57:57'),
         'show do tropykalia'],
        [39.991013, 116.326384, 2, Timestamp('2008-10-24 00:22:01'),
         'evento da prefeitura'],
        [40.01, 116.312615, 3, Timestamp('2008-10-25 00:21:01'),
         'show do seu joao'],
        [40.013821, 116.306531, 4, Timestamp('2008-10-26 00:22:01'),
         'missa']
    ]

    pois = DataFrame(
        data=list_events,
        columns=[LATITUDE, LONGITUDE, EVENT_ID, DATETIME, EVENT_TYPE],
        index=[0, 1, 2, 3]
    )

    list_win_start = [
        '2008-10-23T03:53:05.000000000', '2008-10-23T08:37:26.000000000',
        '2008-10-23T08:50:16.000000000', '2008-10-23T09:03:06.000000000',
        '2008-10-23T09:58:33.000000000', '2008-10-23T21:50:45.000000000',
        '2008-10-23T22:02:14.000000000', '2008-10-23T22:22:01.000000000',
        '2008-10-23T23:57:57.000000000'
    ]

    win_start_expected = Series(pd.to_datetime(list_win_start), name=DATETIME)

    list_win_end = [
        '2008-10-23T07:53:05.000000000', '2008-10-23T12:37:26.000000000',
        '2008-10-23T12:50:16.000000000', '2008-10-23T13:03:06.000000000',
        '2008-10-23T13:58:33.000000000', '2008-10-24T01:50:45.000000000',
        '2008-10-24T02:02:14.000000000', '2008-10-24T02:22:01.000000000',
        '2008-10-24T03:57:57.000000000'
    ]

    win_end_expected = Series(pd.to_datetime(list_win_end), name=DATETIME)

    dist_expected = np.full(9, None, dtype=np.ndarray)
    type_expected = np.full(9, None, dtype=np.ndarray)
    id_expected = np.full(9, None, dtype=np.ndarray)

    window_starts, window_ends, current_distances, event_id, event_type = (
        integration._reset_set_window_and_creates_event_id_type_all(
            move_df, pois, DATETIME, 7200
        )
    )

    assert_series_equal(window_starts, win_start_expected)
    assert_series_equal(window_ends, win_end_expected)
    assert_array_equal(current_distances, dist_expected)
    assert_array_equal(event_id, id_expected)
    assert_array_equal(event_type, type_expected)


def test_join_with_pois():
    move_df = MoveDataFrame(list_move)

    pois = DataFrame(
        data=list_pois,
        columns=[LATITUDE, LONGITUDE, TRAJ_ID, TYPE_POI, NAME_POI],
        index=[0, 1, 2, 3, 4, 5, 6]
    )

    expected = DataFrame(
        data=[
            [39.984094, 116.319236, Timestamp('2008-10-23 05:53:05'), 1, 1,
             0.0, 'distrito_pol_1'],
            [39.984559000000004, 116.326696, Timestamp('2008-10-23 10:37:26'),
             1, 6, 128.24869775642176, 'adocao_de_animais'],
            [40.002899, 116.32151999999999, Timestamp('2008-10-23 10:50:16'),
             1, 5, 663.0104596559174, 'rinha_de_galo_world_cup'],
            [40.016238, 116.30769099999999, Timestamp('2008-10-23 11:03:06'),
             1, 4, 286.3387434682031, 'forro_tropykalia'],
            [40.013814, 116.306525, Timestamp('2008-10-23 11:58:33'), 2, 4,
             0.9311014399622559, 'forro_tropykalia'],
            [40.009735, 116.315069, Timestamp('2008-10-23 23:50:45'), 2, 3,
             211.06912863495492, 'supermercado_aroldo'],
            [39.993527, 116.32648300000001, Timestamp('2008-10-24 00:02:14'),
             2, 2, 279.6712398549538, 'policia_federal'],
            [39.978575, 116.326975, Timestamp('2008-10-24 00:22:01'), 3, 6,
             792.7526066105717, 'adocao_de_animais'],
            [39.981668, 116.310769, Timestamp('2008-10-24 01:57:57'), 3, 7,
             270.7018856738821, 'dia_do_municipio']
        ],
        columns=[LATITUDE, LONGITUDE, DATETIME, TRAJ_ID, ID_POI, DIST_POI, NAME_POI],
        index=[0, 1, 2, 3, 4, 5, 6, 7, 8]
    )

    integration.join_with_pois(move_df, pois)
    assert_frame_equal(move_df, expected, check_dtype=False)

    move_df = MoveDataFrame(list_move)
    integration.join_with_pois(move_df, pois)
    assert_frame_equal(move_df, expected, check_dtype=False)


def test_join_with_pois_optimizer():
    move_df = MoveDataFrame(list_move)
    pois = DataFrame(
        data=list_pois,
        columns=[LATITUDE, LONGITUDE, TRAJ_ID, TYPE_POI, NAME_POI],
        index=[0, 1, 2, 3, 4, 5, 6]
    )
    expected = DataFrame(
        data=[
            [39.984094, 116.319236, Timestamp('2008-10-23 05:53:05'), 1, 1,
             0.0, 'policia'],
            [39.984559000000004, 116.326696, Timestamp('2008-10-23 10:37:26'),
             1, 1, 128.24869775642176, 'policia'],
            [40.002899, 116.32151999999999, Timestamp('2008-10-23 10:50:16'),
             1, 1, 663.0104596559174, 'policia'],
            [40.016238, 116.30769099999999, Timestamp('2008-10-23 11:03:06'),
             1, 1, 286.3387434682031, 'policia'],
            [40.013814, 116.306525, Timestamp('2008-10-23 11:58:33'), 2, 1,
             0.9311014399622559, 'policia'],
            [40.009735, 116.315069, Timestamp('2008-10-23 23:50:45'), 2, 1,
             211.06912863495492, 'policia'],
            [39.993527, 116.32648300000001, Timestamp('2008-10-24 00:02:14'),
             2, 1, 279.6712398549538, 'policia'],
            [39.978575, 116.326975, Timestamp('2008-10-24 00:22:01'), 3, 1,
             792.7526066105717, 'policia'],
            [39.981668, 116.310769, Timestamp('2008-10-24 01:57:57'), 3, 1,
             270.7018856738821, 'policia']
        ],
        columns=[LATITUDE, LONGITUDE, DATETIME, TRAJ_ID, ID_POI, DIST_POI, NAME_POI],
        index=[0, 1, 2, 3, 4, 5, 6, 7, 8]
    )
    expected = MoveDataFrame(expected)
    integration.join_with_pois_optimizer(
        move_df, pois, dist_poi=[100, 50, 100, 50, 100, 200, 1000]
    )
    assert_frame_equal(move_df, expected, check_dtype=False)

    move_df = MoveDataFrame(list_move)
    integration.join_with_pois_optimizer(
        move_df, pois, dist_poi=[100, 50, 100, 50, 100, 200, 1000], reset_index=False
    )
    expected = DataFrame(
        data=[
            [39.984094, 116.319236, Timestamp('2008-10-23 05:53:05'), 1, 1,
             0.0, 'policia'],
            [39.984559000000004, 116.326696, Timestamp('2008-10-23 10:37:26'),
             1, 1, 128.24869775642176, 'policia'],
            [40.002899, 116.32151999999999, Timestamp('2008-10-23 10:50:16'),
             1, 1, 663.0104596559174, 'policia'],
            [40.016238, 116.30769099999999, Timestamp('2008-10-23 11:03:06'),
             1, 1, 286.3387434682031, 'policia'],
            [40.013814, 116.306525, Timestamp('2008-10-23 11:58:33'), 2, 1,
             0.9311014399622559, 'policia'],
            [40.009735, 116.315069, Timestamp('2008-10-23 23:50:45'), 2, 1,
             211.06912863495492, 'policia'],
            [39.993527, 116.32648300000001, Timestamp('2008-10-24 00:02:14'),
             2, 1, 279.6712398549538, 'policia'],
            [39.978575, 116.326975, Timestamp('2008-10-24 00:22:01'), 3, 1,
             792.7526066105717, 'policia'],
            [39.981668, 116.310769, Timestamp('2008-10-24 01:57:57'), 3, 1,
             270.7018856738821, 'policia']
        ],
        columns=[LATITUDE, LONGITUDE, DATETIME, TRAJ_ID, ID_POI, DIST_POI, NAME_POI],
        index=[0, 1, 2, 3, 4, 5, 6, 7, 8]
    )
    assert_frame_equal(move_df, expected, check_dtype=False)


def test_join_with_pois_by_category():
    move_df = MoveDataFrame(list_move)
    pois = DataFrame(
        data=list_pois,
        columns=[LATITUDE, LONGITUDE, TRAJ_ID, TYPE_POI, NAME_POI],
        index=[0, 1, 2, 3, 4, 5, 6]
    )
    expected = DataFrame(
        data=[
            [39.984094, 116.319236, Timestamp('2008-10-23 05:53:05'), 1, 1,
             0.0, 3, 2935.3102772960456, 7, 814.8193850933852, 5,
             2672.393533820207, 6, 675.1730686007362],
            [39.984559000000004, 116.326696, Timestamp('2008-10-23 10:37:26'),
             1, 1, 637.6902157810676, 3, 3072.6963790707114, 7,
             1385.3649632111096, 5, 2727.1360691122813, 6, 128.24869775642176],
            [40.002899, 116.32151999999999, Timestamp('2008-10-23 10:50:16'),
             1, 2, 1385.0871812075436, 3, 1094.8606633486436, 4,
             1762.0085654338782, 5, 663.0104596559174, 6, 1965.702358742657],
            [40.016238, 116.30769099999999, Timestamp('2008-10-23 11:03:06'),
             1, 2, 3225.288830967221, 3, 810.5429984051405, 4,
             286.3387434682031, 5, 1243.8915481769327, 6, 3768.0652637796675],
            [40.013814, 116.306525, Timestamp('2008-10-23 11:58:33'), 2, 2,
             3047.8382223981853, 3, 669.9731550451877, 4, 0.9311014399622559,
             5, 1145.172578151837, 6, 3574.252994707609],
            [40.009735, 116.315069, Timestamp('2008-10-23 23:50:45'), 2, 2,
             2294.0758201547073, 3, 211.06912863495492, 4, 857.4175399672413,
             5, 289.35378153627966, 6, 2855.1657930463994],
            [39.993527, 116.32648300000001, Timestamp('2008-10-24 00:02:14'),
             2, 2, 279.6712398549538, 3, 2179.5701631051966, 7,
             2003.4096341742952, 5, 1784.3132149978549, 6, 870.5252810680124],
            [39.978575, 116.326975, Timestamp('2008-10-24 00:22:01'), 3, 1,
             900.7798955139455, 3, 3702.2394204188754, 7, 1287.7039084016499,
             5, 3376.4438614084356, 6, 792.7526066105717],
            [39.981668, 116.310769, Timestamp('2008-10-24 01:57:57'), 3, 1,
             770.188754517813, 3, 3154.296880053552, 7, 270.7018856738821, 5,
             2997.898227057909, 6, 1443.9247752786023]
        ],
        columns=[
            LATITUDE, LONGITUDE, DATETIME, TRAJ_ID, 'id_policia', 'dist_policia',
            'id_comercio', 'dist_comercio', 'id_show', 'dist_show', 'id_risca-faca',
            'dist_risca-faca', 'id_evento', 'dist_evento'
        ],
        index=[0, 1, 2, 3, 4, 5, 6, 7, 8]
    )

    integration.join_with_pois_by_category(move_df, pois)
    assert_frame_equal(move_df, expected, check_dtype=False)


def test_join_with_poi_datetime():
    list_events = [
        [39.984094, 116.319236, 1,
         Timestamp('2008-10-24 01:57:57'), 'show do tropykalia'],
        [39.991013, 116.326384, 2,
         Timestamp('2008-10-24 00:22:01'), 'evento da prefeitura'],
        [40.01, 116.312615, 3,
         Timestamp('2008-10-25 00:21:01'), 'show do seu joao'],
        [40.013821, 116.306531, 4,
         Timestamp('2008-10-26 00:22:01'), 'missa']
    ]
    move_df = MoveDataFrame(list_move)
    pois = DataFrame(
        data=list_events,
        columns=[LATITUDE, LONGITUDE, EVENT_ID, DATETIME, EVENT_TYPE],
        index=[0, 1, 2, 3]
    )
    expected = DataFrame(
        data=[
            [39.984094, 116.319236, Timestamp('2008-10-23 05:53:05'), 1,
             '', inf, ''],
            [39.984559000000004, 116.326696, Timestamp('2008-10-23 10:37:26'),
             1, '', inf, ''],
            [40.002899, 116.32151999999999, Timestamp('2008-10-23 10:50:16'),
             1, '', inf, ''],
            [40.016238, 116.30769099999999, Timestamp('2008-10-23 11:03:06'),
             1, '', inf, ''],
            [40.013814, 116.306525, Timestamp('2008-10-23 11:58:33'), 2, 2,
             3047.8382223981853, 'evento da prefeitura'],
            [40.009735, 116.315069, Timestamp('2008-10-23 23:50:45'), 2, 2,
             2294.0758201547073, 'evento da prefeitura'],
            [39.993527, 116.32648300000001, Timestamp('2008-10-24 00:02:14'),
             2, 2, 279.6712398549538, 'evento da prefeitura'],
            [39.978575, 116.326975, Timestamp('2008-10-24 00:22:01'), 3, 1,
             900.7798955139455, 'show do tropykalia'],
            [39.981668, 116.310769, Timestamp('2008-10-24 01:57:57'), 3, 1,
             770.188754517813, 'show do tropykalia']
        ],
        columns=[
            LATITUDE, LONGITUDE, DATETIME, TRAJ_ID, EVENT_ID, DIST_EVENT, EVENT_TYPE
        ],
        index=[0, 1, 2, 3, 4, 5, 6, 7, 8]
    )

    integration.join_with_poi_datetime(move_df, pois, time_window=45000)
    assert_frame_equal(move_df, expected, check_dtype=False)


def test_join_with_poi_datetime_optimizer():
    list_events = [
        [39.984094, 116.319236, 1,
         Timestamp('2008-10-24 01:57:57'), 'show do tropykalia'],
        [39.991013, 116.326384, 2,
         Timestamp('2008-10-24 00:22:01'), 'evento da prefeitura'],
        [40.01, 116.312615, 3,
         Timestamp('2008-10-25 00:21:01'), 'show do seu joao'],
        [40.013821, 116.306531, 4,
         Timestamp('2008-10-26 00:22:01'), 'missa']
    ]
    move_df = MoveDataFrame(list_move)
    pois = DataFrame(
        data=list_events,
        columns=[LATITUDE, LONGITUDE, EVENT_ID, DATETIME, EVENT_TYPE],
        index=[0, 1, 2, 3]
    )
    expected = DataFrame(
        data=[
            [39.984094, 116.319236, Timestamp('2008-10-23 05:53:05'), 1, 1,
             0.0, 'show do tropykalia'],
            [39.984559000000004, 116.326696, Timestamp('2008-10-23 10:37:26'),
             1, 1, 637.6902157810676, 'show do tropykalia'],
            [40.002899, 116.32151999999999, Timestamp('2008-10-23 10:50:16'),
             1, 1, 1094.8606633486436, 'show do tropykalia'],
            [40.016238, 116.30769099999999, Timestamp('2008-10-23 11:03:06'),
             1, 1, 286.3387434682031, 'show do tropykalia'],
            [40.013814, 116.306525, Timestamp('2008-10-23 11:58:33'), 2, 1,
             0.9311014399622559, 'show do tropykalia'],
            [40.009735, 116.315069, Timestamp('2008-10-23 23:50:45'), 2,
             '', inf, ''],
            [39.993527, 116.32648300000001, Timestamp('2008-10-24 00:02:14'),
             2, '', inf, ''],
            [39.978575, 116.326975, Timestamp('2008-10-24 00:22:01'), 3,
             '', inf, ''],
            [39.981668, 116.310769, Timestamp('2008-10-24 01:57:57'), 3,
             '', inf, '']
        ],
        columns=[
            LATITUDE, LONGITUDE, DATETIME, TRAJ_ID, EVENT_ID, DIST_EVENT, EVENT_TYPE
        ],
        index=[0, 1, 2, 3, 4, 5, 6, 7, 8]
    )

    integration.join_with_poi_datetime_optimizer(move_df, pois, time_window=45000)
    assert_frame_equal(move_df, expected, check_dtype=False)


def test_join_with_pois_by_dist_and_datetime():
    list_move = [
        [39.984094, 116.319236, Timestamp('2008-10-23 05:53:05'), 1],
        [39.984559000000004, 116.326696, Timestamp('2008-10-23 10:37:26'), 1],
        [40.002899, 116.32151999999999, Timestamp('2008-10-23 10:50:16'), 1],
        [40.016238, 116.30769099999999, Timestamp('2008-10-23 11:03:06'), 1],
        [40.013814, 116.306525, Timestamp('2008-10-23 11:58:33'), 2],
        [40.009735, 116.315069, Timestamp('2008-10-23 23:50:45'), 2],
        [39.993527, 116.32648300000001, Timestamp('2008-10-24 00:02:14'), 2],
        [39.978575, 116.326975, Timestamp('2008-10-24 00:22:01'), 3],
        [39.981668, 116.310769, Timestamp('2008-10-24 01:57:57'), 3],
    ]

    move_df = MoveDataFrame(list_move)

    list_events = [
        [39.984094, 116.319236, 1, Timestamp('2008-10-24 01:57:57'),
         'show do tropykalia'],
        [39.991013, 116.326384, 2, Timestamp('2008-10-24 00:22:01'),
         'evento da prefeitura'],
        [40.01, 116.312615, 3, Timestamp('2008-10-25 00:21:01'),
         'show do seu joao'],
        [40.013821, 116.306531, 4, Timestamp('2008-10-26 00:22:01'),
         'missa']
    ]

    pois = DataFrame(
        data=list_events,
        columns=[LATITUDE, LONGITUDE, EVENT_ID, DATETIME, EVENT_TYPE],
        index=[0, 1, 2, 3]
    )

    expected = DataFrame(
        data=[
            [39.984094, 116.319236, Timestamp('2008-10-23 05:53:05'),
             1, None, None, None],
            [39.984559000000004, 116.326696, Timestamp('2008-10-23 10:37:26'),
             1, None, None, None],
            [40.002899, 116.32151999999999, Timestamp('2008-10-23 10:50:16'),
             1, None, None, None],
            [40.016238, 116.30769099999999, Timestamp('2008-10-23 11:03:06'),
             1, None, None, None],
            [40.013814, 116.306525, Timestamp('2008-10-23 11:58:33'),
             2, None, None, None],
            [40.009735, 116.315069, Timestamp('2008-10-23 23:50:45'),
             2, [2], [2294.0758201547073], ['evento da prefeitura']],
            [39.993527, 116.32648300000001, Timestamp('2008-10-24 00:02:14'),
             2, [1, 2], [1217.1198213850694, 279.6712398549538],
             ['show do tropykalia', 'evento da prefeitura']],
            [39.978575, 116.326975, Timestamp('2008-10-24 00:22:01'),
             3, [1, 2], [900.7798955139455, 1383.9587958381394],
             ['show do tropykalia', 'evento da prefeitura']],
            [39.981668, 116.310769, Timestamp('2008-10-24 01:57:57'),
             3, [1, 2], [770.188754517813, 1688.0786831571447],
             ['show do tropykalia', 'evento da prefeitura']]
        ],

        columns=[
            LATITUDE, LONGITUDE, DATETIME, TRAJ_ID,
            EVENT_ID, DIST_EVENT, EVENT_TYPE
        ],

        index=[0, 1, 2, 3, 4, 5, 6, 7, 8]
    )

    integration.join_with_pois_by_dist_and_datetime(
        move_df, pois, radius=3000, time_window=7200
    )

    assert_frame_equal(move_df, expected, check_dtype=False)


def test_join_with_home_by_id():
    list_home = [
        [39.984094, 116.319236, 1, 'rua da mae', 'quixiling'],
        [40.013821, 116.306531, 2, 'rua da familia', 'quixeramoling']
    ]
    move_df = MoveDataFrame(list_move)
    home_df = DataFrame(
        data=list_home,
        columns=[LATITUDE, LONGITUDE, TRAJ_ID, ADDRESS, CITY]
    )
    expected = DataFrame(
        data=[
            [1, 39.984094, 116.319236, Timestamp('2008-10-23 05:53:05'), 0.0,
             'rua da mae', 'quixiling'],
            [1, 39.984559000000004, 116.326696,
             Timestamp('2008-10-23 10:37:26'), 637.6902157810676,
             'rua da mae', 'quixiling'],
            [1, 40.002899, 116.32151999999999,
             Timestamp('2008-10-23 10:50:16'), 2100.0535005951438,
             'rua da mae', 'quixiling'],
            [1, 40.016238, 116.30769099999999,
             Timestamp('2008-10-23 11:03:06'), 3707.066732003998,
             'rua da mae', 'quixiling'],
            [2, 40.013814, 116.306525, Timestamp('2008-10-23 11:58:33'),
             0.9311014399622559, 'rua da familia', 'quixeramoling'],
            [2, 40.009735, 116.315069, Timestamp('2008-10-23 23:50:45'),
             857.4175399672413, 'rua da familia', 'quixeramoling'],
            [2, 39.993527, 116.32648300000001,
             Timestamp('2008-10-24 00:02:14'), 2824.932385384076,
             'rua da familia', 'quixeramoling'],
            [3, 39.978575, 116.326975, Timestamp('2008-10-24 00:22:01'), nan,
             nan, nan],
            [3, 39.981668, 116.310769, Timestamp('2008-10-24 01:57:57'), nan,
             nan, nan]
        ],
        columns=[TRAJ_ID, LATITUDE, LONGITUDE, DATETIME, DIST_HOME, HOME, CITY]
    )

    integration.join_with_home_by_id(move_df, home_df)
    assert_frame_equal(move_df, expected, check_dtype=False)

    move_df = MoveDataFrame(list_move)
    integration.join_with_home_by_id(move_df, home_df, drop_id_without_home=True)
    expected = DataFrame(
        data=[
            [1, 39.984094, 116.319236, Timestamp('2008-10-23 05:53:05'), 0.0,
             'rua da mae', 'quixiling'],
            [1, 39.984559000000004, 116.326696,
             Timestamp('2008-10-23 10:37:26'), 637.6902157810676,
             'rua da mae', 'quixiling'],
            [1, 40.002899, 116.32151999999999,
             Timestamp('2008-10-23 10:50:16'), 2100.0535005951438,
             'rua da mae', 'quixiling'],
            [1, 40.016238, 116.30769099999999,
             Timestamp('2008-10-23 11:03:06'), 3707.066732003998,
             'rua da mae', 'quixiling'],
            [2, 40.013814, 116.306525, Timestamp('2008-10-23 11:58:33'),
             0.9311014399622559, 'rua da familia', 'quixeramoling'],
            [2, 40.009735, 116.315069, Timestamp('2008-10-23 23:50:45'),
             857.4175399672413, 'rua da familia', 'quixeramoling'],
            [2, 39.993527, 116.32648300000001,
             Timestamp('2008-10-24 00:02:14'), 2824.932385384076,
             'rua da familia', 'quixeramoling']
        ],
        columns=[TRAJ_ID, LATITUDE, LONGITUDE, DATETIME, DIST_HOME, HOME, CITY]
    )
    assert_frame_equal(move_df, expected, check_dtype=False)


def test_merge_home_with_poi():
    list_move4merge = [
        [39.984094, 116.319236, Timestamp('2008-10-23 05:53:05'), 1],
        [39.984559000000004, 116.326696, Timestamp('2008-10-23 10:37:26'), 1],
        [40.002899, 116.32151999999999, Timestamp('2008-10-23 10:50:16'), 1],
        [40.016238, 116.30769099999999, Timestamp('2008-10-23 11:03:06'), 1],
        [40.013814, 116.306525, Timestamp('2008-10-23 11:58:33'), 2],
        [40.009735, 116.315069, Timestamp('2008-10-23 23:50:45'), 2],
        [39.993527, 116.32648300000001, Timestamp('2008-10-24 00:02:14'), 2],
    ]
    move_df = MoveDataFrame(list_move4merge)
    pois = DataFrame(
        data=list_pois,
        columns=[LATITUDE, LONGITUDE, TRAJ_ID, TYPE_POI, NAME_POI],
        index=[0, 1, 2, 3, 4, 5, 6]
    )
    integration.join_with_pois(move_df, pois)

    list_home = [
        [39.984094, 116.319236, 1, 'rua da mae', 'quixiling'],
        [40.013821, 116.306531, 2, 'rua da familia', 'quixeramoling']
    ]
    home_df = DataFrame(
        data=list_home,
        columns=[LATITUDE, LONGITUDE, TRAJ_ID, ADDRESS, CITY],
        index=[0, 1]
    )
    integration.join_with_home_by_id(move_df, home_df)

    expected = DataFrame(
        data=[
            [1, 39.984094, 116.319236, Timestamp('2008-10-23 05:53:05'),
             'rua da mae', 0.0, 'home', 'quixiling'],
            [1, 39.984559000000004, 116.326696,
             Timestamp('2008-10-23 10:37:26'), 6, 128.24869775642176,
             'adocao_de_animais', 'quixiling'],
            [1, 40.002899, 116.32151999999999,
             Timestamp('2008-10-23 10:50:16'), 5, 663.0104596559174,
             'rinha_de_galo_world_cup', 'quixiling'],
            [1, 40.016238, 116.30769099999999,
             Timestamp('2008-10-23 11:03:06'), 4, 286.3387434682031,
             'forro_tropykalia', 'quixiling'],
            [2, 40.013814, 116.306525, Timestamp('2008-10-23 11:58:33'),
             'rua da familia', 0.9311014399622559, 'home', 'quixeramoling'],
            [2, 40.009735, 116.315069, Timestamp('2008-10-23 23:50:45'), 3,
             211.06912863495492, 'supermercado_aroldo', 'quixeramoling'],
            [2, 39.993527, 116.32648300000001,
             Timestamp('2008-10-24 00:02:14'), 2, 279.6712398549538,
             'policia_federal', 'quixeramoling']
        ],
        columns=[
            TRAJ_ID, LATITUDE, LONGITUDE, DATETIME, ID_POI, DIST_POI, NAME_POI, CITY
        ],
        index=[0, 1, 2, 3, 4, 5, 6]
    )
    integration.merge_home_with_poi(move_df)
    assert_frame_equal(move_df, expected, check_dtype=False)
