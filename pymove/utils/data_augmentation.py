import random

import networkx as nx
import numpy as np
import pandas as pd

from pymove.utils.constants import (
    DATETIME,
    DESTINY,
    LABEL,
    LATITUDE,
    LOCAL_LABEL,
    LONGITUDE,
    START,
    TID,
    TRAJ_ID,
    TRAJECTORY,
)
from pymove.utils.log import progress_bar


def append_row(df_, row=None, columns=None):
    """
    Insert a new line in the dataframe with
    the information passed by parameter.

    Parameters
    ----------
    df_ : dataframe
        The input trajectories data.
    row : series, optional, default None
        The row of a dataframe.
    columns : dict, optional, default None
        Dictionary containing the values to be added.
    """
    if row is not None:
        keys = row.index.tolist()
        df_.at[df_.shape[0], keys] = row.values
    else:
        if isinstance(columns, dict):
            keys = list(columns.keys())
            values = [np.array(v).tolist() for v in list(columns.values())]
            df_.at[df_.shape[0], keys] = values


def generate_arrays(df_, columns=None):
    """
    Generates an arrays with the values
    for each column of the passed dataframe

    Parameters
    ----------
    df_ : dataframe
        The input trajectories data.
    columns : list, np.ndarray, optional, default None

    Returns
    -------
    array
        list format dataframe columns
    """
    if columns is None:
        columns = df_.columns

    arr = np.full(len(columns), None, dtype=np.ndarray)
    for idx, col in enumerate(columns):
        arr[idx] = df_[col].values

    return arr


def generate_trajectories_df(df_, min_points=5):
    """
    Generates a dataframe with the sequence of
    location points of a trajectory.

    Parameters
    ----------
    df_ : dataframe
        The input trajectory data.

    Return
    ------
    dataframe
        DataFrame of the trajectories
    """
    if TID not in df_:
        df_.generate_tid_based_on_id_datetime()
        df_.reset_index(drop=True, inplace=True)

    tids = df_[TID].unique()
    new_df = pd.DataFrame(
        columns=df_.columns
    )

    for tid in progress_bar(tids, total=len(tids)):
        filter_ = df_[df_[TID] == tid]
        filter_.reset_index(drop=True, inplace=True)

        if filter_.shape[0] >= min_points:

            values = []
            for col in filter_.columns:
                if filter_[col].nunique() == 1:
                    values.append(filter_.at[0, col])
                else:
                    values.append(
                        np.array(
                            filter_[col], dtype=type(filter_.at[0, col])
                        ).tolist()
                    )

            row = pd.Series(values, filter_.columns)
            append_row(new_df, row=row)

    return new_df


def generate_transition_graphx(df_, label_nodes=LOCAL_LABEL):
    """
    Generates the transition graph from the sequences of points
    that represent the trajectory of an object.

    Parameters
    ----------
    df_ : dataframe
        The input trajectories data.

    label_nodes : str, optinal, default 'local_label'
        Label of the points sequences.

    Returns
    -------
    graph : NetworkX DiGraph
        Representation of points in a targeted manner.
    """
    G = nx.DiGraph()

    lats, lons, times, trajectories = generate_arrays(
        df_, columns=[LATITUDE, LONGITUDE, DATETIME, label_nodes]
    )

    for traj, lat, lon, time in progress_bar(
        zip(trajectories, lats, lons, times), total=len(trajectories)
    ):
        if traj[0] in list(G.nodes):
            dt = nx.get_node_attributes(G, 'dt')[traj[0]]
            dt.append(time[0])
            G.add_node(traj[0], dt=dt)

        else:
            G.add_node(traj[0], pos=(lat[0], lon[0]), dt=[time[0]])

        for i in range(1, len(traj)):
            if traj[i] in list(G.nodes):
                dt = nx.get_node_attributes(G, 'dt')[traj[i]]
                dt.append(time[i])
                G.add_node(traj[i], dt=dt)

            else:
                G.add_node(traj[i], pos=(lat[i], lon[i]), dt=[time[i]])

            if traj[i] in G.adj[traj[i - 1]]:
                edge_data = G.get_edge_data(traj[i - 1], traj[i])
                time_edge = time[i] - time[i - 1]
                weight = (edge_data['weight'] + time_edge) / 2
                G.add_edge(traj[i - 1], traj[i], weight=weight)

            else:
                time_edge = time[i] - time[i - 1]
                G.add_edge(traj[i - 1], traj[i], weight=time_edge)
    return G


def generate_start_feature(df_, label_trajectory=TRAJECTORY):
    """
    Removes the last point from the trajectory and
    adds it in a new column called 'destiny'.

    Parameters
    ----------
    df_ : dataframe
        The input trajectory data.
    label_trajectory : str, optional, default 'trajectory'
        Label of the points sequences
    """
    if START not in df_:
        df_[START] = df_[label_trajectory].apply(
            lambda x: np.int64(x[0])
        )


def generate_destiny_feature(df_, label_trajectory=TRAJECTORY):
    """
    Removes the first point from the trajectory and
    adds it in a new column called 'start'.

    Parameters
    ----------
    df_ : dataframe
        The input trajectory data.
    label_trajectory : str, optional, default 'trajectory'
        Label of the points sequences
    """
    if DESTINY not in df_:
        df_[DESTINY] = df_[label_trajectory].apply(
            lambda x: np.int64(x[-1])
        )


def split_crossover(sequence_a, sequence_b, frac=0.5):
    """
    Divide two arrays in the indicated ratio
    and exchange their halves.

    Parameters
    ----------
    sequence_a : list, np.ndarray
        Array any
    sequence_b : list, np.ndarray
        Array any
    frac : number, optional, default 0.5
        Represents the percentage to be exchanged.

    Returns
    -------
    arrays
        Arrays with the halves exchanged.
    """
    size_a = int(len(sequence_a) * frac)
    size_b = int(len(sequence_b) * frac)

    sequence_a1 = sequence_a[:size_a]
    sequence_a2 = sequence_a[size_a:]

    sequence_b1 = sequence_b[:size_b]
    sequence_b2 = sequence_b[size_b:]

    sequence_a = np.concatenate((sequence_a1, sequence_b2))
    sequence_b = np.concatenate((sequence_b1, sequence_a2))

    return sequence_a, sequence_b


def _augmentation(df_, aug_df, frac=0.5):
    """
    Generates new data with unobserved trajectories.

    Parameters
    ----------
    df_ : dataframe
        The input trajectories data.
    aug_df : dataframe
        The dataframe with new trajectories
    frac : number, optional, default 0.5
        Represents the percentage to be exchanged.
    """
    df_.reset_index(drop=True, inplace=True)

    for idx in range(df_.shape[0] - 1):
        for idx_ in range(idx + 1, df_.shape[0]):
            sequences1 = []
            sequences2 = []

            columns = df_.columns

            for col in columns:
                if (isinstance(
                    df_.at[idx, col], list
                ) or isinstance(
                    df_.at[idx, col], np.ndarray
                )) and (isinstance(
                    df_.at[idx_, col], list
                ) or isinstance(
                    df_.at[idx_, col], np.ndarray
                )):
                    seq1, seq2 = split_crossover(
                        df_.at[idx, col],
                        df_.at[idx_, col],
                        frac=frac
                    )
                    sequences1.append(seq1)
                    sequences2.append(seq2)
                else:
                    value1 = df_.at[idx, col]
                    value2 = df_.at[idx_, col]

                    if isinstance(value1, str) and isinstance(value2, str):
                        sequences1.append(value1 + '_' + value2)
                        sequences2.append(value2 + '_' + value1)
                    else:
                        sequences1.append(value1)
                        sequences2.append(value2)

            row = pd.Series(sequences1, index=columns)
            append_row(aug_df, row=row)

            row = pd.Series(sequences2, index=columns)
            append_row(aug_df, row=row)


def augmentation_trajectories_df(
    df_,
    restriction='destination only',
    label_trajectory=TRAJECTORY,
    insert_at_df=False,
    frac=0.5,
):
    """
    Generate new data from unobserved trajectories,
    given a specific restriction. By default, the
    algorithm uses the same route destination constraint.

    Parameters
    ----------
    df_ : dataframe
        The input trajectories data.
    restriction : str, optional, default 'destination only'
        Constraint used to generate new data.
    label_trajectory : str, optional, default 'trajectory'
        Label of the points sequences.
    insert_at_df : boolean, optional, default False
        Whether to return a new DataFrame.
        If True then value of copy is ignored.
    frac : number, optional, default 0.5
        Represents the percentage to be exchanged.

    Returns
    -------
    DataFrame or None
        Dataframe with the new data generated.
    """

    if DESTINY not in df_:
        generate_destiny_feature(df_, label_trajectory=label_trajectory)

    if restriction == 'departure and destination':
        generate_start_feature(df_)

    if insert_at_df:
        aug_df = df_
    else:
        aug_df = pd.DataFrame(columns=df_.columns)

    destinations = df_[DESTINY].unique()
    for dest in progress_bar(destinations, total=len(destinations)):
        filter_ = df_[df_[DESTINY] == dest]

        if restriction == 'departure and destination':
            starts = filter_[START].unique()

            for st in progress_bar(starts, total=len(starts)):
                ffilter_ = filter_[filter_[START] == st]

                if ffilter_.shape[0] >= 2:
                    _augmentation(ffilter_, aug_df, frac=frac)

        else:
            if filter_.shape[0] >= 2:
                _augmentation(filter_, aug_df, frac=frac)

    return aug_df


def insert_points_in_df(df_, aug_df):
    """
    Inserts the points of the generated trajectories
    to the original data sets.

    Parameters
    ----------
    df_ : dataframe
        The input trajectories data.
    aug_df : dataframe
        The data of unobserved trajectories.
    """
    for _, row in progress_bar(aug_df.iterrows(), total=aug_df.shape[0]):

        keys = row.index.tolist()
        values = row.values.tolist()

        row_df = pd.DataFrame()

        for k, v in zip(keys, values):
            if k in df_:
                if isinstance(v, list) or isinstance(v, np.ndarray):
                    row_df[k] = v

        for k, v in zip(keys, values):
            if k in df_:
                if not isinstance(v, list) and not isinstance(v, np.ndarray):
                    row_df[k] = v

        for _, row_ in row_df.iterrows():
            append_row(df_, row=row_)


def instance_crossover_augmentation(
    df_,
    restriction='destination only',
    label_trajectory=TRAJECTORY,
    frac=0.5
):
    """
    Generate new data from unobserved trajectories,
    with a specific restriction. By default, the
    algorithm uses the same destination constraint
    as the route and inserts the points on the
    original dataframe.

    Parameters
    ----------
    df_ : dataframe
        The input trajectories data.

    restriction : str, optional, default 'destination only'
        Constraint used to generate new data.

    label_trajectory : str, optional, default 'trajectory'
        Label of the points sequences.

    frac : number, optional, default 0.5
        Represents the percentage to be exchanged.

    Returns
    -------
    DataFrame or None
        Dataframe with the new data generated.
    """
    try:
        traj_df = generate_trajectories_df(df_)

        generate_destiny_feature(traj_df, label_trajectory=label_trajectory)

        if restriction == 'departure and destination':
            generate_start_feature(traj_df, label_trajectory=label_trajectory)

        aug_df = augmentation_trajectories_df(
            traj_df, restriction=restriction, frac=frac
        )
        insert_points_in_df(df_, aug_df)

    except Exception as e:
        raise e


def find_all_paths(graph, source, target, path=[]):
    """
    Find all paths from start_vertex to end_vertex in graph.

    Parameters
    ----------
    graph : NetworkX DiGraph
        Representation of points in a targeted manner.

    start_vertex : node
        Starting node for path.

    end_vertex : node
        Ending node for path.

    path : list
        List to add paths

    Returns
    -------
    paths : generator of lists
        A generator of all paths between start_vertex and end_vertex.

    Examples
    --------
    >>> G = {1: {2, 6},
             2: {3},
             3: {8},
             5: {6},
             6: {7},
             7: {8, 11},
             8: {8, 9},
             9: {14},
             11: {11, 12},
             12: {13},
             13: {14}
             14: {14},
        }
    >>> graph = nx.DiGraph(G)
    >>> find_all_paths(graph, 1, 14)
    [[1, 2, 3, 8, 9, 14], [1, 6, 7, 8, 9, 14], [1, 6, 7, 11, 12, 13, 14]]

    References
    ----------
    https://www.python-course.eu/pygraph.php

    """
    path = path + [source]

    if source == target:
        return [path]

    if source not in graph:
        return []

    paths = []
    for vertex in graph[source]:
        if vertex not in path:
            extended_paths = find_all_paths(graph, vertex, target, path)

            for p in extended_paths:
                paths.append(p)

    return paths


def extract_latlon_and_datetime_in_graph(graphx, path):
    """
    Extracts the space and time information
    (latitude, longitude, datetime) that were
    inserted in each node in the graph at the
    time of its construction.

    Parameters
    ----------
    graphx : NetworkX DiGraph
        Representation of points in a targeted manner.

    path : list, np.ndarray
        List of node to search in the graph.

    Returns
    -------
    arrays
        Space and time information contained in the transition graph.
    """

    pos = nx.get_node_attributes(graphx, 'pos')
    dt = nx.get_node_attributes(graphx, 'dt')

    lats, lons, times = [], [], []

    lats.append(pos[path[0]][0])
    lons.append(pos[path[0]][1])

    random_date = random.sample(dt[path[0]], 1)
    times.append(random_date[0])

    for i in range(1, len(path)):
        lats.append(pos[path[i]][0])
        lons.append(pos[path[i]][1])

        edge_data = graphx.get_edge_data(path[i - 1], path[i])
        times.append(times[i - 1] + edge_data['weight'])

    return lats, lons, times


def transition_graph_augmentation_from_source_and_target(
    aug_df,
    graphx,
    source,
    target,
    min_path=5,
    label_nodes=LOCAL_LABEL
):
    """
    Generation of unobserved trajectories
    for all nodes or for a known origin
    and/or destination in a transition graph.

    Parameters
    ----------
    aug_df : dataframe
        New dataframe to add the found trajectories.

    graphx : NetworkX DiGraph
        Representation of points in a targeted manner.

    source : node
        Starting node

    target : node
        Ending node

    min_path : number, optional, default 5
        Minimum length of a path.

    label_nodes : str, optional, default 'local_label'
        Label of the points sequences.
    """
    paths = find_all_paths(
        graphx, source, target
    )

    if paths:
        for path in paths:
            if len(path) >= min_path:
                arr = extract_latlon_and_datetime_in_graph(
                    graphx, path
                )

                lats, lons, times = arr
                append_row(aug_df, columns={
                    label_nodes: path,
                    LATITUDE: lats,
                    LONGITUDE: lons,
                    DATETIME: times,
                })


def transition_graph_augmentation_all_vertex(
    aug_df,
    graphx,
    source=None,
    target=None,
    min_path=5,
):
    """


    Parameters
    ----------
    aug_df : dataframe
        New dataframe to add the found trajectories.

    graphx : NetworkX DiGraph
        Representation of points in a targeted manner.

    source : node, optional, default None
        Starting node

    target : node, optional, default None
        Ending node

    min_path : number, optional, default 5
        Minimum length of a path.
    """
    if source is None:
        source = list(graphx.nodes)
    else:
        source = [source]

    if target is None:
        target = list(graphx.nodes)
    else:
        target = [target]

    for s in source:
        for t in target:
            transition_graph_augmentation_from_source_and_target(
                aug_df, graphx, s, t, min_path
            )
