import numpy as np
import pandas as pd

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


def union_poi_bank(df_, label_poi=TYPE_POI):
    """
    Performs the union between the different bank categories
    for Points of Interest in a single category named 'banks'.

    Parameters
    ----------
    df_ : dataframe
        Input points of interest data

    label_poi : String, optional("type_poi" by default)
        Label referring to the Point of Interest category

    """

    print('union bank categories to one category')
    print('... There are {} -- {}'.format(df_[label_poi].nunique(), label_poi))
    banks = [
        'bancos_filiais',
        'bancos_agencias',
        'bancos_postos',
        'bancos_PAE',
        'bank',
    ]
    filter_bank = df_[label_poi].isin(banks)
    df_.at[df_[filter_bank].index, label_poi] = 'banks'


def union_poi_bus_station(df_, label_poi=TYPE_POI):
    """
    Performs the union between the different bus station categories
    for Points of Interest in a single category named 'bus_station'.

    Parameters
    ----------
    df_ : dataframe
        Input points of interest data

    label_poi : String, optional("type_poi" by default)
        Label referring to the Point of Interest category

    """

    print('union bus station categories to one category')
    filter_bus_station = df_[label_poi].isin(
        ['transit_station', 'pontos_de_onibus']
    )
    df_.at[df_[filter_bus_station].index, label_poi] = 'bus_station'


def union_poi_bar_restaurant(df_, label_poi=TYPE_POI):
    """
    Performs the union between bar and restaurant categories
    for Points of Interest in a single category named 'bar-restaurant'.

    Parameters
    ----------
    df_ : dataframe
        Input points of interest data

    label_poi : String, optional("type_poi" by default)
        Label referring to the Point of Interest category

    """

    print('union restaurant and bar categories to one category')
    filter_bar_restaurant = df_[label_poi].isin(['restaurant', 'bar'])
    df_.at[df_[filter_bar_restaurant].index, label_poi] = 'bar-restaurant'


def union_poi_parks(df_, label_poi=TYPE_POI):
    """
    Performs the union between park categories
    for Points of Interest in a single category named 'parks'.

    Parameters
    ----------
    df_ : dataframe
        Input points of interest data

    label_poi : String, optional("type_poi" by default)
        Label referring to the Point of Interest category

    """

    print('union parks categories to one category')
    filter_parks = df_[label_poi].isin(['pracas_e_parques', 'park'])
    df_.at[df_[filter_parks].index, label_poi] = 'parks'


def union_poi_police(df_, label_poi=TYPE_POI):
    """
    Performs the union between police categories
    for Points of Interest in a single category named 'police'.

    Parameters
    ----------
    df_ : dataframe
        Input points of interest data

    label_poi : String, optional("type_poi" by default)
        Label referring to the Point of Interest category

    """

    print('union distritos policies and police categories')
    filter_police = df_[label_poi] == 'distritos_policiais'
    df_.at[df_[filter_police].index, label_poi] = 'police'


def join_collective_areas(gdf_, gdf_rules_, label_geometry=GEOMETRY):
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

    label_geometry: String, optional("geometry" by default)
        Label of gdf_rules_ referring to the geometry of each feature

    """

    print('Integration between trajectories and collectives areas')

    polygons = gdf_rules_[label_geometry].unique()
    gdf_[VIOLATING] = False
    for p in progress_bar(polygons):
        index = gdf_[gdf_[label_geometry].intersects(p)].index
        gdf_.at[index, VIOLATING] = True


def _reset_and_creates_id_and_lat_lon(df_, df_pois, lat_lon_poi=True, reset_index=True):
    """
    Resets the indexes of the dataframes, returns the minimum distance
    between the two dataframes, and return their respective variables
    (id, tags, latitude and longitude).

    Parameters
    ----------
    df_ : dataframe
        The input trajectory data.

    df_pois : dataframe
        The input point of interest data.

    lat_lon_poi : boolean (True by default)
        Flag to detemine if the ids and tags is of size equivalent to df_pois

    reset_index : Boolean, optional(True by default)
        Flag for reset index of the df_pois and df_ dataframes before the join.

    Returns
    -------
    distances, ids, tags, lat, lon: arrays with default values for join operation

    """

    if reset_index:
        print('... Resetting index to operation...')
        df_.reset_index(drop=True, inplace=True)
        df_pois.reset_index(drop=True, inplace=True)

    # create numpy array to store new column to dataframe of movement objects
    distances = np.full(
        df_.shape[0], np.Infinity, dtype=np.float64
    )

    ids = np.full(df_.shape[0], '', dtype='object_')
    tags = np.full(df_.shape[0], '', dtype='object_')

    # creating lat and lon array to operation
    if lat_lon_poi:
        lat = np.full(df_pois.shape[0], np.Infinity, dtype=np.float64)
        lon = np.full(df_pois.shape[0], np.Infinity, dtype=np.float64)
    else:
        lat = np.full(df_.shape[0], np.Infinity, dtype=np.float64)
        lon = np.full(df_.shape[0], np.Infinity, dtype=np.float64)

    return distances, ids, tags, lat, lon


def _reset_set_window__and_creates_event_id_type(
    df_, df_events, label_date, time_window
):
    """
    Resets the indexes of the dataframes, set time window, and returns
    the current distance between the two dataframes, and return their
    respective variables (event_id, event_type).

    Parameters
    ----------
    df_ : dataframe
        The input trajectory data.

    df_events : dataframe
        The input event point of interest data.

    label_date : String
        Label of df_ referring to the datetime.

    time_window : Int
        Number of seconds of the time window.

    Returns
    -------
    window_starts, window_ends, current_distances, event_id, event_type

    """

    # get a vector with windows time to each point
    df_.reset_index(drop=True, inplace=True)
    df_events.reset_index(drop=True, inplace=True)

    # compute windows time
    window_starts = df_[label_date] - pd.Timedelta(seconds=time_window)
    window_ends = df_[label_date] + pd.Timedelta(seconds=time_window)

    # create vector to store distances
    current_distances = np.full(
        df_.shape[0], np.Infinity, dtype=np.float64
    )
    event_type = np.full(df_.shape[0], '', dtype='object_')
    event_id = np.full(df_.shape[0], '', dtype='object_')

    return window_starts, window_ends, current_distances, event_id, event_type


def _reset_set_window_and_creates_event_id_type_all(
    df_, df_events, label_date, time_window
):
    """
    Resets the indexes of the dataframes, set time window, and returns
    the current distance between the two dataframes, and return their
    respective variables (event_id, event_type).

    Parameters
    ----------
    df_ : dataframe
        The input trajectory data.

    df_events : dataframe
        The input event point of interest data.

    label_date : String
        Label of df_ referring to the datetime.

    time_window : Int
        Number of seconds of the time window.

    Returns
    -------
    window_starts, window_ends, current_distances, event_id, event_type
        arrays with default values for join operation
    """

    # get a vector with windows time to each point
    df_.reset_index(drop=True, inplace=True)
    df_events.reset_index(drop=True, inplace=True)

    # compute windows time
    window_starts = df_[label_date] - pd.Timedelta(seconds=time_window)
    window_ends = df_[label_date] + pd.Timedelta(seconds=time_window)

    # create vector to store distances
    current_distances = np.full(
        df_.shape[0], None, dtype=np.ndarray
    )
    event_type = np.full(df_.shape[0], None, dtype=np.ndarray)
    event_id = np.full(df_.shape[0], None, dtype=np.ndarray)

    return window_starts, window_ends, current_distances, event_id, event_type


def join_with_pois(
    df_, df_pois, label_id=TRAJ_ID, label_poi_name=NAME_POI, reset_index=True
):
    """
    Performs the integration between trajectories and points
    of interest, generating two new columns referring to the
    name and the distance from the point of interest closest
    to each point of the trajectory.

    Parameters
    ----------
    df_ : dataframe
        The input trajectory data.

    df_pois : dataframe
        The input point of interest data.

    label_id : String, optional("id" by default)
        Label of df_pois referring to the Point of Interest id.

    label_poi_name : String, optional("type_poi" by default)
        Label of df_pois referring to the Point of Interest name.

    reset_index : Boolean, optional(True by default)
        Flag for reset index of the df_pois and df_ dataframes before the join.

    """

    try:
        print('Integration with POIs...')

        values = _reset_and_creates_id_and_lat_lon(df_, df_pois, True, reset_index)
        current_distances, ids_POIs, tag_POIs, lat_user, lon_user = values

        for idx, row in progress_bar(df_.iterrows(), total=len(df_)):
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

        df_[ID_POI] = ids_POIs
        df_[DIST_POI] = current_distances
        df_[NAME_POI] = tag_POIs

        print('Integration with POI was finalized')
    except Exception as e:
        raise e


def join_with_pois_optimizer(
    df_,
    df_pois,
    label_poi_id=TRAJ_ID,
    label_poi_name=NAME_POI,
    dist_poi=None,
    reset_index=True,
):
    """
    Performs the integration between trajectories and points
    of interest, generating two new columns referring to the
    name and distance from the nearest point of interest,
    within the limit of distance determined by the parameter 'dist_poi',
    of each point in the trajectory.

    Parameters
    ----------
    df_ : dataframe
        The input trajectory data.

    df_pois : dataframe
        The input point of interest data.

    label_poi_id : String, optional("id" by default)
        Label of df_pois referring to the Point of Interest id.

    label_poi_name : String, optional("type_poi" by default)
        Label of df_pois referring to the Point of Interest name.

    dist_poi : List
        List containing the minimum distance limit between each type of
        point of interest and each point of the trajectory to classify the
        point of interest closest to each point of the trajectory.

    reset_index : Boolean, optional(True by default)
        Flag for reset index of the df_pois and df_ dataframes before the join.

    """

    try:
        print('Integration with POIs optimized...')

        if len(df_pois[label_poi_name].unique()) == len(dist_poi):
            values = _reset_and_creates_id_and_lat_lon(df_, df_pois, False, reset_index)
            minimum_distances, ids_POIs, tag_POIs, lat_POI, lon_POI = values

            df_pois.rename(
                columns={label_poi_id: TRAJ_ID, label_poi_name: NAME_POI},
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
                            df_[LATITUDE].values,
                            df_[LONGITUDE].values
                        )
                    )
                    ids_POIs.fill(row.id)
                    tag_POIs.fill(row.type_poi)
                else:
                    # compute dist between a POI and ALL
                    print(df_[LONGITUDE].values)
                    current_distances = np.float64(
                        haversine(
                            lat_POI,
                            lon_POI,
                            df_[LATITUDE].values,
                            df_[LONGITUDE].values
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

            df_[ID_POI] = ids_POIs
            df_[DIST_POI] = minimum_distances
            df_[NAME_POI] = tag_POIs
            print('Integration with POI was finalized')
        else:
            print('the size of the dist_poi is different from the size of pois')
    except Exception as e:
        raise e


def join_with_pois_by_category(
    df_, df_pois, label_category=TYPE_POI, label_id=TRAJ_ID
):
    """
    It performs the integration between trajectories and points
    of interest, generating new columns referring to the
    category and distance from the nearest point of interest
    that has this category at each point of the trajectory.

    Parameters
    ----------
    df_ : dataframe
        The input trajectory data.

    df_pois : dataframe
        The input point of interest data.

    label_category : String, optional("type_poi" by default)
        Label of df_pois referring to the point of interest category.

    label_id : String, optional("id" by default)
        Label of df_pois referring to the point of interest id.

    """

    try:
        print('Integration with POIs...')

        # get a vector with windows time to each point
        df_.reset_index(drop=True, inplace=True)
        df_pois.reset_index(drop=True, inplace=True)

        # create numpy array to store new column to dataframe of movement objects
        current_distances = np.full(
            df_.shape[0], np.Infinity, dtype=np.float64
        )
        ids_POIs = np.full(df_.shape[0], np.NAN, dtype='object_')

        size_categories = df_pois[label_category].unique()
        print('There are %s categories' % len(size_categories))

        for c in size_categories:
            print(
                'computing dist to category: %s' % c,
                flush=True
            )
            # creating lat and lon array to operation
            df_category = df_pois[df_pois[label_category] == c]
            df_category.reset_index(drop=True, inplace=True)

            for idx, row in progress_bar(df_.iterrows(), total=len(df_)):
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

            df_['id_%s' % c] = ids_POIs
            df_['dist_%s' % c] = current_distances
        print('Integration with POI was finalized')

    except Exception as e:
        raise e


def join_with_poi_datetime(
        df_,
        df_events,
        label_date=DATETIME,
        time_window=900,
        label_event_id=EVENT_ID,
        label_event_type=EVENT_TYPE
):
    """
    It performs the integration between trajectories and points
    of interest, generating new columns referring to the
    category of the point of interest, the distance from the
    nearest point of interest based on time of each point of
    the trajectories.

    Parameters
    ----------
    df_ : dataframe
        The input trajectory data.

    df_events : dataframe
        The input events points of interest data.

    label_date : String, optional("datetime" by default)
        Label of df_ referring to the datetime of the input trajectory data.

    time_window : Float, optional(900 by default)
        tolerable length of time range for assigning the event's
        point of interest to the trajectory point.

    label_event_id : String, optional("event_id" by default)
        Label of df_events referring to the id of the event.

    label_event_type : String, optional("event_type" by default)
        Label of df_events referring to the type of the event.

    """

    try:
        print('Integration with Events...')

        values = _reset_set_window__and_creates_event_id_type(
            df_, df_events, label_date, time_window
        )
        window_starts, window_ends, current_distances, event_id, event_type = values

        for idx in progress_bar(df_.index):
            # filter event by datetime
            df_filtered = filters.by_datetime(
                df_events, window_starts[idx], window_ends[idx]
            )
            size_filter = df_filtered.shape[0]

            if size_filter > 0:
                df_filtered.reset_index(drop=True, inplace=True)
                lat_user = np.full(
                    size_filter, df_.at[idx, LATITUDE], dtype=np.float64
                )
                lon_user = np.full(
                    size_filter, df_.at[idx, LONGITUDE], dtype=np.float64
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

        df_[label_event_id] = event_id
        df_[DIST_EVENT] = current_distances
        df_[label_event_type] = event_type
        print('Integration with event was completed')
    except Exception as e:
        raise e


def join_with_poi_datetime_optimizer(
        df_,
        df_events,
        label_date=DATETIME,
        time_window=900,
        label_event_id=EVENT_ID,
        label_event_type=EVENT_TYPE
):
    """
    It performs a optimized integration between trajectories and points
    of interest of events, generating new columns referring to
    the category of the event, the distance from the nearest
    event and the time when the event happened at each point of
    the trajectories.

    Parameters
    ----------
    df_ : dataframe
        The input trajectory data.

    df_events : dataframe
        The input events points of interest data.

    label_date : String, optional("datetime" by default)
        Label of df_events referring to the event point of interest datetime.

    time_window : Float, optional(900 by default)
        tolerable length of time range for assigning the event's
        point of interest to the trajectory point.

    label_event_id : String, optional("event_id" by default)
        Label of df_events referring to the id of the event.

    label_event_type : String, optional("event_type" by default)
        Label of df_events referring to the type of the event.


    """

    try:
        print('Integration with Events...')

        values = _reset_set_window__and_creates_event_id_type(
            df_, df_events, label_date, time_window
        )
        window_starts, window_ends, current_distances, event_id, event_type = values

        minimum_distances = np.full(
            df_.shape[0], np.Infinity, dtype=np.float64
        )

        # Rename for access columns of each row directly
        df_events.rename(
            columns={label_event_id: label_event_id, label_event_type: label_event_type},
            inplace=True
        )

        for idx, row in progress_bar(df_events.iterrows(), total=len(df_events)):
            df_filtered = filters.by_datetime(
                df_, window_starts[idx], window_ends[idx]
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

        df_[label_event_id] = event_id
        df_[DIST_EVENT] = minimum_distances
        df_[label_event_type] = event_type
        print('Integration with events was completed')

    except Exception as e:
        raise e


def join_with_pois_by_dist_and_datetime(
    df_,
    df_pois,
    label_date=DATETIME,
    label_event_id=EVENT_ID,
    label_event_type=EVENT_TYPE,
    time_window=3600,
    radius=1000,
    inplace=False
):
    """
    It performs the integration between trajectories and points of interest,
    generating new columns referring to the category of the point of interest,
    the distance between the location of the user and location of the poi
    based on the distance and on time of each point of the trajectories.

    Parameters
    ----------
    df_ : dataframe
        The input trajectory data.

    df_pois : dataframe
        The input events points of interest data.

    label_date : String, optional("datetime" by default)
        Label of df_ referring to the datetime of the input trajectory data.

    label_event_id : String, optional("event_id" by default)
        Label of df_events referring to the id of the event.

    label_event_type : String, optional("event_type" by default)
        Label of df_events referring to the type of the event.

    time_window : Float, optional(3600 by default)
        tolerable length of time range for assigning the event's
        point of interest to the trajectory point.

    radius: Float, optional (1000 by default)
    """
    try:
        print('Integration with Events...')

        if label_date not in df_pois:
            raise KeyError("POI's dataframe must contain a %s column" % label_date)

        values = _reset_set_window_and_creates_event_id_type_all(
            df_, df_pois, label_date, time_window
        )

        window_start, window_end, current_distances, event_id, event_type = values

        for idx, row in progress_bar(df_.iterrows(), total=df_.shape[0]):

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

        df_[label_event_id] = event_id
        df_[DIST_EVENT] = current_distances
        df_[label_event_type] = event_type
        print('Integration with event was completed')

    except Exception as e:
        raise e


def join_with_home_by_id(
        df_,
        df_home,
        label_id=TRAJ_ID,
        label_address=ADDRESS,
        label_city=CITY,
        drop_id_without_home=False,
):
    """
    It performs the integration between trajectories and home points,
    generating new columns referring to the distance of the nearest
    home point, address and city of each trajectory point.

    Parameters
    ----------
    df_ : dataframe
        The input trajectory data.

    df_home : dataframe
        The input home points data.

    label_id : String, optional("id" by default)
        Label of df_home referring to the home point id.

    label_address : String, optional("formatted_address" by default)
        Label of df_home referring to the home point address.

    label_city : String, optional("city" by default)
        Label of df_home referring to the point city.

    drop_id_without_home : Boolean, optional(False by default)
        flag as an option to drop id's that don't have houses.

    """

    try:
        print('Integration with Home...')
        ids_without_home = []

        if df_.index.name is None:
            print('...setting {} as index'.format(label_id))
            df_.set_index(label_id, inplace=True)

        for idx in progress_bar(df_.index.unique()):
            filter_home = df_home[label_id] == idx

            if df_home[filter_home].shape[0] == 0:
                print('...id: {} has not HOME'.format(idx))
                ids_without_home.append(idx)
            else:
                home = df_home[filter_home].iloc[0]
                lat_user = df_.at[idx, LATITUDE]
                lon_user = df_.at[idx, LONGITUDE]

                # if user has a single tuple
                if not isinstance(lat_user, np.ndarray):
                    df_.at[idx, DIST_HOME] = haversine(
                        lat_user, lon_user, home[LATITUDE], home[LONGITUDE]
                    )
                    df_.at[idx, HOME] = home[label_address]
                    df_.at[idx, label_city] = home[label_city]
                else:
                    lat_home = np.full(
                        df_.loc[idx].shape[0], home[LATITUDE], dtype=np.float64
                    )
                    lon_home = np.full(
                        df_.loc[idx].shape[0], home[LONGITUDE], dtype=np.float64
                    )
                    df_.at[idx, DIST_HOME] = haversine(
                        lat_user, lon_user, lat_home, lon_home
                    )
                    df_.at[idx, HOME] = np.array(home[label_address])
                    df_.at[idx, label_city] = np.array(home[label_city])

        df_.reset_index(inplace=True)
        print('... Resetting index')

        if drop_id_without_home:
            for tid in ids_without_home:
                df_.drop(df_.loc[df_[TRAJ_ID] == tid].index, inplace=True)
    except Exception as e:
        raise e


def merge_home_with_poi(
    df_,
    label_dist_poi=DIST_POI,
    label_name_poi=NAME_POI,
    label_id_poi=ID_POI,
    label_home=HOME,
    label_dist_home=DIST_HOME,
    drop_columns=True,
):
    """
    Perform or merge the points of interest and the starting
    points assigned as trajectories, considering the starting
    points as other points of interest, generating a new
    dataframe.

    Parameters
    ----------
    df_ : dataframe
        The input trajectory data, with join_with_pois and join_with_home_by_id applied.

    label_dist_poi : String, optional("dist_poi" by default)
        Label of df_ referring to the distance from the nearest point of interest.

    label_name_poi : String, optional("name_poi" by default)
        Label of df_ referring to the name from the nearest point of interest.

    label_id_poi : String, optional("id_poi" by default)
        Label of df_ referring to the id from the nearest point of interest.

    label_home : String, optional("home" by default)
        Label of df_home referring to the home point.

    label_dist_home: String, optional("dist_home" by default)
        Label of df_home referring to the distance to the home point.

    drop_columns : Boolean, optional(True by default)
        Flag that controls the deletion of the columns referring to the
        id and the distance from the home point

    """

    try:
        print('merge home with POI using shortest distance')
        idx = df_[df_[label_dist_home] <= df_[label_dist_poi]].index

        df_.loc[idx, label_name_poi] = label_home
        df_.loc[idx, label_dist_poi] = df_.loc[idx, label_dist_home]
        df_.loc[idx, label_id_poi] = df_.loc[idx, label_home]

        if(drop_columns):
            df_.drop(columns=[label_dist_home, label_home], inplace=True)
    except Exception as e:
        raise e
