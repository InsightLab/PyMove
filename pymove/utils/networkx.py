"""
Graph operations.

build_transition_graph_from_dict,
build_transition_graph_from_df,
get_all_paths,
graph_to_dict,
save_graph_as_json,
read_graph_json

"""

import json
from pathlib import Path
from typing import Any, Dict, NewType, Optional, Text, Union

import networkx as nx
import numpy as np
import pandas as pd
from networkx.classes.digraph import DiGraph
from pandas.core.frame import DataFrame
from pandas.core.series import Series

from pymove.utils.constants import DATETIME, LATITUDE, LOCAL_LABEL, LONGITUDE, TID_STAT
from pymove.utils.log import progress_bar
from pymove.utils.trajectories import append_trajectory

Node = NewType('Node', Any)


def _populate_graph(
    row: Series,
    nodes: Dict,
    edges: Dict,
    label_local: Optional[Text] = LOCAL_LABEL
):
    """
    Populate Transition Graph.

    Insert the nodes and edges in the transition graph with all the
    necessary attributes for the execution of the search and
    recovery operations of paths / trajectories. The required
    parameters are: latitude, longitude and datetime.

    Parameters
    ----------
    row: Series
        Line of the trajectory dataframe.
    nodes: dict
        Attributes of the transition graph nodes.
    edges: dict
        Attributes of the transition graph edges.
    label_local: str, optional
        Name of the column referring to the trajectories, by default 'local_label'

    """
    traj = row[label_local]

    for index, local in enumerate(traj):

        local_curr = str(local)

        dt = [str(row[DATETIME][index])]
        fs = (index == 0)
        ft = (index == len(traj) - 1)

        if local in nodes['datetime']:
            dt.extend(nodes['datetime'][local_curr])
            fs += nodes['freq_source'][local_curr]
            ft += nodes['freq_target'][local_curr]

        nodes['datetime'][local_curr] = dt
        nodes['freq_source'][local_curr] = fs
        nodes['freq_target'][local_curr] = ft
        nodes['coords'][local_curr] = (row[LATITUDE][index], row[LONGITUDE][index])

        if index == len(traj) - 1:
            break

        next_local = str(traj[index + 1])

        weight = 1
        mean_times = pd.Timestamp(
            row[DATETIME][index + 1]
        ) - pd.Timestamp(row[DATETIME][index])

        if local_curr not in edges:
            edges[local_curr] = {next_local: {}}
            edges[local_curr][next_local] = {
                'weight': 1,
                'mean_times': str(mean_times)
            }

        elif next_local not in edges[local_curr]:
            edges[local_curr] = {**edges[local_curr], **{next_local: {}}}
            edges[local_curr][next_local] = {
                'weight': 1,
                'mean_times': str(mean_times)
            }
        else:
            weight += edges[local_curr][next_local]['weight']
            mean_times = (
                mean_times + pd.Timedelta(edges[local_curr][next_local]['mean_times'])
            ) / 2

            edges[local_curr][next_local]['weight'] = weight
            edges[local_curr][next_local]['mean_times'] = str(mean_times)


def build_transition_graph_from_dict(dict_graph: Dict) -> DiGraph:
    """
    Built Graph from Dict.

    It builds a transition graph from a dictionary
    with nodes and edges and all necessary parameters.
    Example: {'nodes': nodes, 'edges': edges}.

    Parameters
    ----------
    dict_graph: dict
        Dictionary with the attributes of nodes and edges.

    Return
    ------
    graph: DiGraph
        Transition graph constructed from trajectory data.
    """
    graph = nx.DiGraph(dict_graph['edges'])

    for key in dict_graph['nodes']['coords']:
        graph.add_node(key, coords=dict_graph['nodes']['coords'][key])
        graph.add_node(key, datetime=dict_graph['nodes']['datetime'][key])
        graph.add_node(key, freq_source=dict_graph['nodes']['freq_source'][key])
        graph.add_node(key, freq_target=dict_graph['nodes']['freq_target'][key])

    return graph


def build_transition_graph_from_df(data: DataFrame) -> DiGraph:
    """
    Build Graph from data.

    Constructs a Transition Graph from trajectory data.

    Parameters
    ----------
    data: DataFrame
        Trajectory data in sequence format.

    Return
    ------
    graph: DiGraph
        Transition graph constructed from trajectory data.
    """
    nodes = {'datetime': {}, 'coords': {}, 'freq_source': {}, 'freq_target': {}}
    edges = {}

    desc = 'Building Transition Graph...'
    for _, row in progress_bar(data.iterrows(), desc=desc, total=data.shape[0]):
        _populate_graph(row, nodes, edges)

    return build_transition_graph_from_dict(
        {'nodes': nodes, 'edges': edges}
    )


def get_all_paths(
    data: DataFrame,
    graph: DiGraph,
    source: Node,
    target: Node,
    min_path_size: Optional[int] = 3,
    max_path_size: Optional[int] = 6,
    max_sampling_source: Optional[int] = 100,
    max_sampling_target: Optional[int] = 100,
    label_local: Optional[Text] = LOCAL_LABEL,
    label_tid: Optional[Text] = TID_STAT,
    simple_paths: Optional[bool] = False
):
    """
    Generate All Paths.

    Retrieves all paths in the graph between the past source and
    destination, if any. The number of paths returned is limited
    by the max_sampling_source and max_sampling_target
    parameters, and the path size is limited by the
    min_path_size and max_path_size parameters.

    Parameters
    ----------
    data: DataFrame
        Trajectory data in sequence format.
    graph: DiGraph
        Transition graph constructed from trajectory data.
    source: Node
        Sequence source node.
    target: Node
        Sequence destination node.
    min_path_size: int, optional
        Minimum number of points for the trajectory, by default 3
    max_path_size: number, optional
        Maximum number of points for the trajectory, by default 6
    max_sampling_source: number, optional
        Maximum number of paths to be returned,
        considering the observed origin, by default 10
    max_sampling_target: number, optional
        Maximum number of paths to be returned,
        considering the observed destination, by default 10
    label_local: str, optional
        Name of the column referring to the trajectories, by default 'local_label'
    label_tid: str, optional
        Column name for trajectory IDs, by default 'tid_stat'
    simple_paths: bool, optional
        If true, use the paths with the most used sections
        Otherwise, use paths with less used sections, by default False

    """
    source = str(source)
    target = str(target)

    if not nx.has_path(graph, source, target):
        return []

    if simple_paths:
        all_paths = nx.all_simple_paths
        param = max_path_size - 1

    else:
        all_paths = nx.shortest_simple_paths
        param = None

    for path in all_paths(graph, source, target, param):
        freq_source = nx.get_node_attributes(graph, 'freq_source')[source]
        freq_target = nx.get_node_attributes(graph, 'freq_target')[target]

        if freq_source >= max_sampling_source:
            break

        if freq_target >= max_sampling_target:
            break

        if len(path) > max_path_size and simple_paths is False:
            break

        if len(path) >= min_path_size:
            path_ = np.array(path, dtype='float32').tolist()
            if path_ not in data[label_local].values.tolist():

                append_trajectory(data, path, graph, label_tid)

                freq_source += 1
                freq_target += 1

                graph.add_node(source, freq_source=freq_source)
                graph.add_node(target, freq_target=freq_target)


def graph_to_dict(graph: DiGraph) -> Dict:
    """
    Graph to Dict.

    Converts nodes and edges from the Transition Graph
    with all your attributes in a dictionary.

    Parameters
    ----------
    graph: DiGraph
        Transition graph constructed from trajectory data.

    Return
    ------
    dict
        Dictionary with the attributes of nodes and edges.
    """
    dict_graph = {'nodes': {}, 'edges': {}}

    dict_graph['nodes']['coords'] = nx.get_node_attributes(graph, 'coords')
    dict_graph['nodes']['datetime'] = nx.get_node_attributes(graph, 'datetime')
    dict_graph['nodes']['freq_source'] = nx.get_node_attributes(graph, 'freq_source')
    dict_graph['nodes']['freq_target'] = nx.get_node_attributes(graph, 'freq_target')
    dict_graph['edges'] = nx.to_dict_of_dicts(graph)

    return dict_graph


def save_graph_as_json(
    graph: DiGraph, file_path: Optional[Union[Path, Text]] = 'graph.json'
):
    """
    Save Graph as JSON.

    Saves the data extracted from the Transition Graph
    into a JSON file.

    Parameters
    ----------
    graph: DiGraph
        Transition graph constructed from trajectory data.
    file_path: str or path, optional
        File name that will be saved with transition graph data, by default 'graph.json'.

    """
    dict_graph = graph_to_dict(graph)

    path = Path(file_path)

    if path.suffix != '.json':
        raise ValueError(
            f'Unsupported file extension {path.suffix},'
            f'Expected extension = json'
        )

    with open(path, 'w') as f:
        json.dump(dict_graph, f)


def read_graph_json(file_path: Optional[Union[Path, Text]]):
    """
    Read Graph from JSON file.

    You load a Transition Graph from a file in JSON format.

    Parameters
    ----------
    file_path: str or path
        Name of the JSON file to be read

    Return
    ------
    dict
        Dictionary with the attributes of nodes and edges
    """
    path = Path(file_path)

    if path.suffix != '.json':
        raise ValueError(
            f'Unsupported file extension {path.suffix},'
            f'Expected extension = json'
        )

    with open(file_path, 'r') as f:
        dict_graph = json.load(f)

    return dict_graph
