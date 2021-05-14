"""
Graph operations.

_all_simple_paths,
_all_shortest_paths

"""

from typing import Any, Optional, Text

import networkx as nx
from networkx.classes.digraph import DiGraph
from pandas.core.frame import DataFrame

from pymove.utils.constants import LOCAL_LABEL, TID_STAT
from pymove.utils.trajectories import append_trajectory


def _all_simple_paths(
    data: DataFrame, graph: DiGraph, source: Any, target: Any,
    min_path_size: Optional[int] = 3, max_path_size: Optional[int] = 6,
    max_sampling_source: Optional[int] = 10, max_sampling_target: Optional[int] = 10,
    label_local: Optional[Text] = LOCAL_LABEL, label_tid: Optional[Text] = TID_STAT
):
    """
    Searches for less frequent trajectories.

    Searches the chart for paths with less frequent transitions.
    Returning the less frequented trajectories in the real data set.

    Parameters
    ----------
    data: pandas.core.frame.DataFrame
        Trajectory data in sequence format.

    graph: networkx.classes.digraph.DiGraph
        Transition graph constructed from trajectory data.

    source: Node
        Sequence source node.

    target: Node
        Sequence destination node.

    min_path_size: Number, optional, default 3
        Minimum number of points for the trajectory.

    max_path_size: Number, optional, default 6
        Maximum number of points for the trajectory.

    max_sampling_source: Number, optional, default 10
        Maximum number of paths to be returned, considering the observed origin.

    max_sampling_target: Number, optional, default 10
        Maximum number of paths to be returned, considering the observed destination.

    label_local: String, optional, default 'local_label'
        Name of the column referring to the trajectories.

    label_tid: String, optional, default 'tid_stat'
        Column name for trajectory IDs.

    """
    for path in nx.all_simple_paths(graph, source, target, cutoff=max_path_size):
        freq_source = nx.get_node_attributes(graph, 'freq_source')[source]
        freq_target = nx.get_node_attributes(graph, 'freq_target')[target]

        if freq_source >= max_sampling_source:
            break

        if freq_target >= max_sampling_target:
            break

        if len(path) >= min_path_size:
            if path not in data[label_local].values.tolist():

                append_trajectory(data, path, graph, label_tid)

                freq_source += 1
                freq_target += 1

                graph.add_node(source, freq_source=freq_source)
                graph.add_node(target, freq_target=freq_target)


def _all_shortest_paths(
    data: DataFrame, graph: DiGraph, source: Any, target: Any,
    min_path_size: Optional[int] = 3, max_path_size: Optional[int] = 6,
    max_sampling_source: Optional[int] = 10, max_sampling_target: Optional[int] = 10,
    label_local: Optional[Text] = LOCAL_LABEL, label_tid: Optional[Text] = TID_STAT
):
    """
    Search for the most frequent trajectories.

    Searches the graph for paths with more frequent transitions.
    Returning most frequented trajectories in the real data set.

    Parameters
    ----------
    data: pandas.core.frame.DataFrame
        Trajectory data in sequence format.

    graph: networkx.classes.digraph.DiGraph
        Transition graph constructed from trajectory data.

    source: Node
        Sequence source node.

    target: Node
        Sequence destination node.

    min_path_size: Number, optional, default 3
        Minimum number of points for the trajectory.

    max_path_size: Number, optional, default 6
        Maximum number of points for the trajectory.

    max_sampling_source: Number, optional, default 10
        Maximum number of paths to be returned, considering the observed origin.

    max_sampling_target: Number, optional, default 10
        Maximum number of paths to be returned, considering the observed destination.

    label_local: String, optional, default 'local_label'
        Name of the column referring to the trajectories.

    label_tid: String, optional, default 'tid_stat'
        Column name for trajectory IDs.

    """
    for path in nx.shortest_simple_paths(graph, source, target):
        freq_source = nx.get_node_attributes(graph, 'freq_source')[source]
        freq_target = nx.get_node_attributes(graph, 'freq_target')[target]

        if freq_source >= max_sampling_source:
            break

        if freq_target >= max_sampling_target:
            break

        if len(path) > max_path_size:
            break

        if len(path) >= min_path_size:
            if path not in data[label_local].values.tolist():

                append_trajectory(data, path, graph, label_tid)

                freq_source += 1
                freq_target += 1

                graph.add_node(source, freq_source=freq_source)
                graph.add_node(target, freq_target=freq_target)
