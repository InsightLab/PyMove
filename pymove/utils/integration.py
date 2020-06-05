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
    POI,
    TRAJ_ID,
    TYPE_POI,
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


def join_coletive_areas(gdf_, gdf_rules_, label_geometry=GEOMETRY):
    """
    It performs the integration between trajectories and collective
    areas, generating a new columnn that informs if the point of the
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
    gdf_['violation'] = False
    for p in progress_bar(polygons):
        index = gdf_[gdf_[label_geometry].intersects(p)].index
        gdf_.at[index, 'violation'] = True


def join_with_pois(
    df_, df_pois, label_id=TRAJ_ID, label_poi=POI, reset_index=True
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

    label_poi : String, optional("POI" by default)
        Label of df_pois referring to the Point of Interest name.

    reset_index : Boolean, optional(True by default)
        Flag for reset index of the df_pois and df_ dataframes before the join.

    """

    try:
        print('Integration with POIs...')

        # get a vector with windows time to each point
        if reset_index:
            print('... Resetting index to operation...')
            df_.reset_index(drop=True, inplace=True)
            df_pois.reset_index(drop=True, inplace=True)

        # create numpy array to store new column to dataframe of movement objects
        current_distances = np.full(
            df_.shape[0], np.Infinity, dtype=np.float32
        )
        ids_POIs = np.full(df_.shape[0], np.NAN, dtype='object_')
        tag_POIs = np.full(df_.shape[0], np.NAN, dtype='object_')

        # creating lat and lon array to operation
        lat_user = np.full(df_pois.shape[0], np.Infinity, dtype=np.float64)
        lon_user = np.full(df_pois.shape[0], np.Infinity, dtype=np.float64)

        for row in progress_bar(df_[[LATITUDE, LONGITUDE]].iterrows()):
            # get lat and lon to each id
            idx = row.index

            # create a vector to each lat
            lat_user.fill(row[LATITUDE])
            lon_user.fill(row[LONGITUDE])

            # computing distances to idx
            distances = np.float64(
                haversine(
                    lat_user,
                    lon_user,
                    df_pois[LATITUDE].to_numpy(dtype=np.float64),
                    df_pois[LONGITUDE].to_numpy(dtype=np.float64),
                )
            )

            # get index to arg_min and min distance
            index_min = np.argmin(distances)
            current_distances[idx] = np.min(distances)

            # setting data for a single object movement
            ids_POIs[idx] = df_pois.at[index_min, label_id]
            tag_POIs[idx] = df_pois.at[index_min, label_poi]

        df_[ID_POI] = ids_POIs
        df_[DIST_POI] = current_distances
        df_[TYPE_POI] = tag_POIs

        print('Integration with POI was finalized')
    except Exception as e:
        print('id: {}\n'.format(idx))
        raise e


def join_with_pois_optimizer(
    df_,
    df_pois,
    label_poi_id=TRAJ_ID,
    label_poi_type=TYPE_POI,
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

    label_poi_id : String, optional("name" by default)
        Label of df_pois referring to the Point of Interest id.

    label_poi_type : String, optional("POI" by default)
        Label of df_pois referring to the Point of Interest type.

    dist_poi : List
        List containing the minimum distance limit between each type of
        point of interest and each point of the trajectory to classify the
        point of interest closest to each point of the trajectory.

    reset_index : Boolean, optional(True by default)
        Flag for reset index of the df_pois and df_ dataframes before the join.

    """

    try:
        print('Integration with POIs optimized...')

        if len(df_pois[label_poi_type].unique()) == len(dist_poi):
            # get a vector with windows time to each point
            if reset_index:
                print('... Resetting and dropping index')
                df_.reset_index(drop=True, inplace=True)
                df_pois.reset_index(drop=True, inplace=True)

            # create numpy array to store new column to dataframe of movement objects
            ids_POIs = np.full(df_.shape[0], np.NAN, dtype='object_')
            tag_POIs = np.full(df_.shape[0], np.NAN, dtype='object_')

            minimum_distances = np.full(
                df_.shape[0], np.Infinity, dtype=np.float64
            )
            lat_POI = np.full(df_.shape[0], np.NAN, dtype=np.float64)
            lon_POI = np.full(df_.shape[0], np.NAN, dtype=np.float64)

            df_pois.rename(
                columns={label_poi_id: TRAJ_ID, label_poi_type: TYPE_POI},
                inplace=True
            )

            for row in progress_bar(df_pois.iterrows()):

                idx = row.index
                # update lat and lot of current index
                lat_POI.fill(row[LATITUDE])
                lon_POI.fill(row[LONGITUDE])

                # First iteration is minimum distances
                if idx == 0:
                    minimum_distances = np.float64(
                        haversine(
                            lat_POI,
                            lon_POI,
                            df_[LATITUDE].to_numpy(dtype=np.float64),
                            df_[LONGITUDE].to_numpy(dtype=np.float64),
                        )
                    )
                    ids_POIs.fill(row.id)
                    tag_POIs.fill(row.type_poi)
                else:
                    # compute dist between a POI and ALL tnz
                    current_distances = np.float64(
                        haversine(
                            lat_POI,
                            lon_POI,
                            df_[LATITUDE].to_numpy(dtype=np.float64),
                            df_[LONGITUDE].to_numpy(dtype=np.float64),
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
            df_[TYPE_POI] = tag_POIs
            print('Integration with POI was finalized')
        else:
            print('the size of the dist_poi is different from ')
    except Exception as e:
        print('id: {}\n'.format(idx))
        raise e


def join_with_pois_by_category(
    df_, df_pois, label_category=POI, label_id=TRAJ_ID
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

    label_category : String, optional("POI" by default)
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
                'Calculating dist to category: %s' % c,
                flush=True
            )
            # creating lat and lon array to operation
            df_category = df_pois[df_pois[label_category] == c]
            df_category.reset_index(drop=True, inplace=True)

            for row in progress_bar(df_[[LATITUDE, LONGITUDE]].iterrows()):
                idx = row.index
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
                    df_category[LATITUDE].to_numpy(dtype=np.float64),
                    df_category[LONGITUDE].to_numpy(dtype=np.float64),
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
        print('id: {}\n'.format(idx))
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

        # get a vector with windows time to each point
        df_.reset_index(drop=True, inplace=True)
        df_events.reset_index(drop=True, inplace=True)

        # compute windows time
        window_starts = df_[label_date] - pd.Timedelta(seconds=time_window)
        window_ends = df_[label_date] + pd.Timedelta(seconds=time_window)

        # create numpy aux t
        current_distances = np.full(
            df_.shape[0], np.Infinity, dtype=np.float64
        )
        event_type = np.full(df_.shape[0], np.NAN, dtype='object_')
        event_id = np.full(df_.shape[0], np.NAN, dtype=np.int32)

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
                    df_filtered[LATITUDE].to_numpy(dtype=np.float64),
                    df_filtered[LONGITUDE].to_numpy(dtype=np.float64),
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
        print('id: {}\n'.format(idx))
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

        # get a vector with windows time to each point
        df_.reset_index(drop=True, inplace=True)
        df_events.reset_index(drop=True, inplace=True)

        # calc window time to each event
        window_starts = df_events[label_date] - pd.Timedelta(seconds=time_window)
        window_ends = df_events[label_date] + pd.Timedelta(seconds=time_window)

        # create vector to store distances to Event
        current_distances = np.full(
            df_.shape[0], np.Infinity, dtype=np.float64
        )
        event_type = np.full(df_.shape[0], np.NAN, dtype='object_')
        event_id = np.full(df_.shape[0], np.NAN, dtype=np.int32)

        minimum_distances = np.full(
            df_.shape[0], np.Infinity, dtype=np.float64
        )

        # Rename for access columns of each row directly
        df_events.rename(
            columns={label_event_id: label_event_id, label_event_type: label_event_type},
            inplace=True
        )

        for row in progress_bar(df_events.iterrows()):
            idx = row.index

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
                        df_filtered[LATITUDE].to_numpy(dtype=np.float64),
                        df_filtered[LONGITUDE].to_numpy(dtype=np.float64),
                    )
                    event_id[indexes] = row.event_id
                    event_type[indexes] = row.event_type
                else:
                    current_distances[indexes] = haversine(
                        lat_event,
                        lon_event,
                        df_filtered[LATITUDE].to_numpy(dtype=np.float64),
                        df_filtered[LONGITUDE].to_numpy(dtype=np.float64),
                    )
                    compare = current_distances < minimum_distances
                    index_True = np.where(compare is True)[0]

                    minimum_distances = np.minimum(
                        current_distances, minimum_distances
                    )
                    event_id[index_True] = row.event_id
                    event_type[index_True] = row.event_type
                    print(event_id, event_type)

        df_[label_event_id] = event_id
        df_[DIST_EVENT] = minimum_distances
        df_[label_event_type] = event_type
        print('Integration with events was completed')

    except Exception as e:
        print('id: {}\n'.format(idx))
        raise e


def join_with_home_by_id(
        df_,
        df_home,
        label_id=TRAJ_ID,
        label_address=ADDRESS,
        label_city=CITY,
        label_home=HOME,
        label_dist_home=DIST_HOME,
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

    label_home : String, optional("home" by default)
        Label of df_home referring to the home point.

    label_dist_home: String, optional("dist_home" by default)
        Label of df_home referring to the distance to the home point.

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
                    df_.at[idx, label_dist_home] = haversine(
                        lat_user, lon_user, home[LATITUDE], home[LONGITUDE]
                    )
                    df_.at[idx, label_home] = home[label_address]
                    df_.at[idx, label_city] = home[label_city]
                else:
                    lat_home = np.full(
                        df_.loc[idx].shape[0], home[LATITUDE], dtype=np.float64
                    )
                    lon_home = np.full(
                        df_.loc[idx].shape[0], home[LONGITUDE], dtype=np.float64
                    )
                    df_.at[idx, label_dist_home] = haversine(
                        lat_user, lon_user, lat_home, lon_home
                    )
                    df_.at[idx, label_home] = np.array(home[label_address])
                    df_.at[idx, label_city] = np.array(home[label_city])

        df_.reset_index(inplace=True)
        print('... Resetting index')

        if drop_id_without_home:
            filters.by_id(df_, label_id, ids_without_home)
    except Exception as e:
        print('Erro: idx: {}'.format(idx))
        raise e


def merge_home_with_poi(
    df_,
    label_dist_poi=DIST_POI,
    label_type_poi=TYPE_POI,
    label_id_poi=ID_POI,
    label_home=HOME,
    label_dist_home=DIST_HOME,
    drop_columns=True,
):
    """
    Merge home points and points of interest, generating new
    columns for the respective home point, the id
    of the nearest point of interest, the distance from the
    nearest point of interest and the type of the nearest point
    of interest at each point of interest.

    Parameters
    ----------
    df_ : dataframe
        The input points of interest  data.

    label_dist_poi : String, optional("dist_poi" by default)
        Label of df_ referring to the distance from the nearest point of interest.

    label_type_poi : String, optional("type_poi" by default)
        Label of df_ referring to the type from the nearest point of interest.

    label_id_poi : String, optional("type_poi" by default)
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

        df_.loc[idx, label_type_poi] = label_home
        df_.loc[idx, label_dist_poi] = df_.loc[idx, label_dist_home]
        df_.loc[idx, label_id_poi] = df_.loc[idx, label_home]
        if drop_columns:
            del df_[label_home], df_[label_dist_home]
    except Exception as e:
        raise e


"""
def integration_EVENT_to_user(df_, df_event, label_date='datetime', time_window=900):
    try:
        print('Integration with Events...')
        start_time = time.time()

        df_['event'] = np.NAN

        #calc window time to each event
        window_starts = df_event[label_date] - pd.Timedelta(seconds=time_window)
        window_ends = df_event[label_date]  + pd.Timedelta(seconds=time_window)

        #create vector to store distances to events
        ids_minimum = np.full(df_.shape[0], np.NAN, dtype=np.int64)
        minimum_distances = np.full(df_.shape[0], np.Infinity, dtype=np.float64)

        current_ids = np.full(df_.shape[0], np.NAN, dtype=np.int64)
        current_distances = np.full(df_.shape[0], np.Infinity, dtype=np.float64)

        for i, idx in enumerate(progress_bar(df_event.index)):
            #filter user by time windows
            df_filtered = filters.by_datetime(
                df_, window_starts[i], window_ends[i]
            )

            size_filter = df_filtered.shape[0]

            if(size_filter > 0):
                indexs = df_filtered.index

                lat_event = np.full(
                    df_filtered.shape[0], df_event.loc[idx]['lat'], dtype=np.float64
                )
                lon_event = np.full(
                    df_filtered.shape[0], df_event.loc[idx]['lon'], dtype=np.float64
                )

                # First iteration is minimum distances
                if i == 0:
                    minimum_distances[indexs] = haversine(
                        lat_event,
                        lon_event,
                        df_filtered['lat'].to_numpy(),
                        df_filtered['lon'].to_numpy()
                    )
                    ids_minimum[indexs] = df_filtered['event_id']
                    print('Minimum distances were computed')
                    #dic[idx] = minimum_distances
                else:
                    current_distances[indexs] = haversine(
                        lat_event,
                        lon_event,
                        df_filtered['lat'].to_numpy(),
                        df_filtered['lon'].to_numpy()
                    )
                    current_ids[indexs] = df_filtered['event_id']

                    minimum_distances = np.minimum(current_distances, minimum_distances)
            else:
                print(idx)


        df_['event'] = minimum_distances
        df_['id_event'] = current_ids
        print('Integration with event was completed')
        print('\nTotal Time: {:.2f} seconds'.format((time.time() - start_time)))
        print('-----------------------------------------------------\n')
        #return dic
    except Exception as e:
        print('id: {}\n'.format(idx))
        raise e

def integration_with_event(
    df_, df_event, label_date='datetime', time_window=900, dist_to_event=50
):
    try:
        print('Integration with events...')
        start_time = time.time()

        ## get a vector with windows time to each point
        df_.reset_index(drop=True, inplace=True)
        df_event.reset_index(drop=True, inplace=True)

        window_starts = df_[label_date] - pd.Timedelta(seconds=time_window)
        window_ends = df_[label_date]  + pd.Timedelta(seconds=time_window)

        current_distances = np.full(df_.shape[0], np.Infinity, dtype=np.float64)
        event_type = np.full(df_.shape[0], np.NAN, dtype='object_')
        event_id = np.full(df_.shape[0], np.NAN, dtype=np.int32)
        for idx in progress_bar(df_.index):
            #filter event by datetime
            df_filtered = filters.by_datetime(
                df_events, window_starts[idx], window_ends[idx]
                )
            size_filter = df_filtered.shape[0]

            if(size_filter > 0):
                lat_user = np.full(size_filter, df_.loc[idx]['lat'], dtype=np.float64)
                lon_user = np.full(size_filter, df_.loc[idx]['lon'], dtype=np.float64)
                distances = haversine(
                    lat_user,
                    lon_user,
                    df_filtered['lat'].to_numpy(),
                    df_filtered['lon'].to_numpy()
                )
                # get index to arg_min
                index_arg_min = np.argmin(distances)
                # get min distances
                min_distance = min(distances)
                #store data
                current_distances[idx] = min_distance
                event_index = df_filtered.index[index_arg_min]
                event_type[idx] = df_event.loc[event_index]['event_type']
                event_id[idx] = df_event.loc[event_index]['event_id']

        df_['event_id'] = event_id
        #df_['event_type'] = event_type
        df_['dist_event'] = current_distances
        print('Integration with Events was completed')
        print('\nTotal Time: {:.2f} seconds'.format((time.time() - start_time)))
        print('-----------------------------------------------------\n')
    except Exception as e:
        print('id: {}\n'.format(idx))
        raise e
"""
