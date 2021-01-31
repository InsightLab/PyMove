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
from pymove.utils.log import progress_bar


def union_poi_bank(data: DataFrame, label_poi: Optional[Text] = TYPE_POI):
    """
    Performs the union between the different bank categories
    for Points of Interest in a single category named 'banks'.

    Parameters
    ----------
    data : DataFrame
        Input points of interest data
    label_poi : str, optional
        Label referring to the Point of Interest category, by default TYPE_POI

    """

    print('union bank categories to one category')
    print('... There are {} -- {}'.format(data[label_poi].nunique(), label_poi))
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
    Performs the union between the different bus station categories
    for Points of Interest in a single category named 'bus_station'.

    Parameters
    ----------
    data : DataFrame
        Input points of interest data
    label_poi : str, optional
        Label referring to the Point of Interest category, by default TYPE_POI

    """

    print('union bus station categories to one category')
    filter_bus_station = data[label_poi].isin(
        ['transit_station', 'pontos_de_onibus']
    )
    data.at[data[filter_bus_station].index, label_poi] = 'bus_station'


def union_poi_bar_restaurant(data: DataFrame, label_poi: Optional[Text] = TYPE_POI):
    """
    Performs the union between bar and restaurant categories
    for Points of Interest in a single category named 'bar-restaurant'.

    Parameters
    ----------
    data : DataFrame
        Input points of interest data
    label_poi : str, optional
        Label referring to the Point of Interest category, by default TYPE_POI

    """

    print('union restaurant and bar categories to one category')
    filter_bar_restaurant = data[label_poi].isin(['restaurant', 'bar'])
    data.at[data[filter_bar_restaurant].index, label_poi] = 'bar-restaurant'


def union_poi_parks(data: DataFrame, label_poi: Optional[Text] = TYPE_POI):
    """
    Performs the union between park categories
    for Points of Interest in a single category named 'parks'.

    Parameters
    ----------
    data : DataFrame
        Input points of interest data
    label_poi : str, optional
        Label referring to the Point of Interest category, by default TYPE_POI

    """

    print('union parks categories to one category')
    filter_parks = data[label_poi].isin(['pracas_e_parques', 'park'])
    data.at[data[filter_parks].index, label_poi] = 'parks'


def union_poi_police(data: DataFrame, label_poi: Optional[Text] = TYPE_POI):
    """
    Performs the union between police categories
    for Points of Interest in a single category named 'police'.

    Parameters
    ----------
    data : DataFrame
        Input points of interest data
    label_poi : str, optional
        Label referring to the Point of Interest category, by default TYPE_POI

    """

    print('union distritos policies and police categories')
    filter_police = data[label_poi] == 'distritos_policiais'
    data.at[data[filter_police].index, label_poi] = 'police'


def join_collective_areas(
    gdf_: DataFrame, gdf_rules_: DataFrame, label_geometry: Optional[Text] = GEOMETRY
):
    """
    It performs the integration between trajectories and collective
    areas, generating a new column that informs if the point of the
    trajectory is inserted in a collective area.

    Parameters
    ----------
    gdf_ : geopandas.GeoDataFrame
        The input trajectory data
    gdf_rules_ : geopandas.GeoDataFrame
        The input coletive areas data
    label_geometry : str, optional
        Label referring to the Point of Interest category, by default GEOMETRY

    """

    print('Integration between trajectories and collectives areas')

    polygons = gdf_rules_[label_geometry].unique()
    gdf_[VIOLATING] = False
    for p in progress_bar(polygons):
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
    Resets the indexes of the dataframes, returns the minimum distance
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

    """

    if reset_index:
        print('... Resetting index to operation...')
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
    Resets the indexes of the dataframes, set time window, and returns
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
    Resets the indexes of the dataframes, set time window, and returns
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
    Performs the integration between trajectories and points
    of interest, generating two new columns referring to the
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

    """

    print('Integration with POIs...')

    values = _reset_and_creates_id_and_lat_lon(data, df_pois, True, reset_index)
    current_distances, ids_POIs, tag_POIs, lat_user, lon_user = values

    for idx, row in progress_bar(data.iterrows(), total=len(data)):
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
        ids_POIs[idx] = df_pois.at[index_min, label_id]
        tag_POIs[idx] = df_pois.at[index_min, label_poi_name]

    data[ID_POI] = ids_POIs
    data[DIST_POI] = current_distances
    data[NAME_POI] = tag_POIs

    print('Integration with POI was finalized')


def join_with_pois_optimizer(
    data,
    df_pois: DataFrame,
    label_id: Optional[Text] = TRAJ_ID,
    label_poi_name: Optional[Text] = NAME_POI,
    dist_poi: Optional[List] = None,
    reset_index: Optional[Text] = True
):
    """
    Performs the integration between trajectories and points
    of interest, generating two new columns referring to the
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

    """

    print('Integration with POIs optimized...')

    if len(df_pois[label_poi_name].unique()) == len(dist_poi):
        values = _reset_and_creates_id_and_lat_lon(data, df_pois, False, reset_index)
        minimum_distances, ids_POIs, tag_POIs, lat_POI, lon_POI = values

        df_pois.rename(
            columns={label_id: TRAJ_ID, label_poi_name: NAME_POI},
            inplace=True
        )

        for idx, row in progress_bar(df_pois.iterrows(), total=len(df_pois)):
            # update lat and lon of current index
            lat_POI.fill(row[LATITUDE])
            lon_POI.fill(row[LONGITUDE])

            # First iteration is minimum distances
            if idx == 0:
                minimum_distances = np.float64(
                    haversine(
                        lat_POI,
                        lon_POI,
                        data[LATITUDE].values,
                        data[LONGITUDE].values
                    )
                )
                ids_POIs.fill(row.id)
                tag_POIs.fill(row.type_poi)
            else:
                # compute dist between a POI and ALL
                print(data[LONGITUDE].values)
                current_distances = np.float64(
                    haversine(
                        lat_POI,
                        lon_POI,
                        data[LATITUDE].values,
                        data[LONGITUDE].values
                    )
                )
                compare = current_distances < minimum_distances
                index_True = np.where(compare is True)[0]
                minimum_distances = np.minimum(
                    current_distances, minimum_distances, dtype=np.float64
                )

                if index_True.shape[0] > 0:
                    ids_POIs[index_True] = row.id
                    tag_POIs[index_True] = row.type_poi

        data[ID_POI] = ids_POIs
        data[DIST_POI] = minimum_distances
        data[NAME_POI] = tag_POIs
        print('Integration with POI was finalized')
    else:
        print('the size of the dist_poi is different from the size of pois')


def join_with_pois_by_category(
    data: DataFrame,
    df_pois: DataFrame,
    label_category: Optional[Text] = TYPE_POI,
    label_id: Optional[Text] = TRAJ_ID
):
    """
    It performs the integration between trajectories and points
    of interest, generating new columns referring to the
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

    """

    print('Integration with POIs...')

    # get a vector with windows time to each point
    data.reset_index(drop=True, inplace=True)
    df_pois.reset_index(drop=True, inplace=True)

    # create numpy array to store new column to DataFrame of movement objects
    current_distances = np.full(
        data.shape[0], np.Infinity, dtype=np.float64
    )
    ids_POIs = np.full(data.shape[0], np.NAN, dtype='object_')

    unique_categories = df_pois[label_category].unique()
    size_categories = len(unique_categories)
    print('There are %s categories' % size_categories)

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
            ids_POIs[idx] = df_category.at[index_min, label_id]

        data['id_%s' % c] = ids_POIs
        data['dist_%s' % c] = current_distances
    print('Integration with POI was finalized')


def join_with_poi_datetime(
    data: DataFrame,
    df_events: DataFrame,
    label_date: Optional[Text] = DATETIME,
    time_window: Optional[int] = 900,
    label_event_id: Optional[Text] = EVENT_ID,
    label_event_type: Optional[Text] = EVENT_TYPE
):
    """
    It performs the integration between trajectories and points
    of interest, generating new columns referring to the
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

    """

    print('Integration with Events...')

    values = _reset_set_window__and_creates_event_id_type(
        data, df_events, label_date, time_window
    )
    window_starts, window_ends, current_distances, event_id, event_type = values

    for idx in progress_bar(data.index):
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
    print('Integration with event was completed')


def join_with_poi_datetime_optimizer(
    data: DataFrame,
    df_events: DataFrame,
    label_date: Optional[Text] = DATETIME,
    time_window: Optional[int] = 900,
    label_event_id: Optional[Text] = EVENT_ID,
    label_event_type: Optional[Text] = EVENT_TYPE
):
    """
    It performs a optimized integration between trajectories and points
    of interest of events, generating new columns referring to
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

    """

    print('Integration with Events...')

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

    for idx, row in progress_bar(df_events.iterrows(), total=len(df_events)):
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
                index_True = np.where(compare is True)[0]

                minimum_distances = np.minimum(
                    current_distances, minimum_distances
                )
                event_id[index_True] = row.event_id
                event_type[index_True] = row.event_type

    data[label_event_id] = event_id
    data[DIST_EVENT] = minimum_distances
    data[label_event_type] = event_type
    print('Integration with events was completed')


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
    It performs the integration between trajectories and points of interest,
    generating new columns referring to the category of the point of interest,
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

    """

    print('Integration with Events...')

    if label_date not in df_pois:
        raise KeyError("POI's DataFrame must contain a %s column" % label_date)

    values = _reset_set_window_and_creates_event_id_type_all(
        data, df_pois, label_date, time_window
    )

    window_start, window_end, current_distances, event_id, event_type = values

    for idx, row in progress_bar(data.iterrows(), total=data.shape[0]):

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
    print('Integration with event was completed')


def join_with_home_by_id(
    data: DataFrame,
    df_home: DataFrame,
    label_id: Optional[Text] = TRAJ_ID,
    label_address: Optional[Text] = ADDRESS,
    label_city: Optional[Text] = CITY,
    drop_id_without_home: Optional[bool] = False,
):
    """
    It performs the integration between trajectories and home points,
    generating new columns referring to the distance of the nearest
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

    """

    print('Integration with Home...')
    ids_without_home = []

    if data.index.name is None:
        print('...setting {} as index'.format(label_id))
        data.set_index(label_id, inplace=True)

    for idx in progress_bar(data.index.unique()):
        filter_home = df_home[label_id] == idx

        if df_home[filter_home].shape[0] == 0:
            print('...id: {} has not HOME'.format(idx))
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
    print('... Resetting index')

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
    Perform or merge the points of interest and the starting
    points assigned as trajectories, considering the starting
    points as other points of interest, generating a new
    DataFrame.

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

    """

    print('merge home with POI using shortest distance')
    idx = data[data[label_dist_home] <= data[label_dist_poi]].index

    data.loc[idx, label_name_poi] = label_home
    data.loc[idx, label_dist_poi] = data.loc[idx, label_dist_home]
    data.loc[idx, label_id_poi] = data.loc[idx, label_home]

    if(drop_columns):
        data.drop(columns=[label_dist_home, label_home], inplace=True)
