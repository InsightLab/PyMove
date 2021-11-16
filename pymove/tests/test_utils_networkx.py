import json
import os

import pandas as pd
from networkx.classes.digraph import DiGraph
from networkx.testing import assert_graphs_equal
from numpy.testing import assert_equal

from pymove.utils.constants import (
    DATETIME,
    LATITUDE,
    LOCAL_LABEL,
    LONGITUDE,
    PREV_LOCAL,
    TID,
    TRAJ_ID,
)
from pymove.utils.networkx import (
    _populate_graph,
    build_transition_graph_from_df,
    build_transition_graph_from_dict,
    graph_to_dict,
    read_graph_json,
    save_graph_as_json,
)

dict_graph = {
    'nodes': {
        'coords': { '85': (-3.8347478, -38.592189), '673': (-3.8235834, -38.590389),
                   '394': (-3.813889, -38.5904445), '263': (-3.9067654, -38.5907723),
                   '224': (-3.8857223, -38.5928892), '623': (-3.8828723, -38.5929789)},
        'datetime': { '85': ['2017-09-02 22:00:27'], '673': ['2017-09-02 22:01:36'],
                     '394': ['2017-09-02 22:03:08'], '263': ['2017-09-02 22:03:46'],
                     '224': ['2017-09-02 22:07:19'], '623': ['2017-09-02 22:07:40']},
        'freq_source': {'85': 1, '673': 0, '394': 0, '263': 0, '224': 0, '623': 0},
        'freq_target': {'85': 0, '673': 0, '394': 0, '263': 0, '224': 0, '623': 1}},
    'edges': {
        '85': {'673': {'weight': 1, 'mean_times': '0 days 00:01:09'}},
        '673': {'394': {'weight': 1, 'mean_times': '0 days 00:01:32'}},
        '394': {'263': {'weight': 1, 'mean_times': '0 days 00:00:38'}},
        '263': {'224': {'weight': 1, 'mean_times': '0 days 00:03:33'}},
        '224': {'623': {'weight': 1, 'mean_times': '0 days 00:00:21'}},
        '623': {}}}

list_data1 = {
    TRAJ_ID: [[1, 1, 1, 1, 1, 1]],
    DATETIME: [[pd.Timestamp('2017-09-02 22:00:27'),
                pd.Timestamp('2017-09-02 22:01:36'),
                pd.Timestamp('2017-09-02 22:03:08'),
                pd.Timestamp('2017-09-02 22:03:46'),
                pd.Timestamp('2017-09-02 22:07:19'),
                pd.Timestamp('2017-09-02 22:07:40')]],
    LOCAL_LABEL: [[85, 673, 394, 263, 224, 623]],
    LATITUDE: [[-3.8347478, -3.8235834, -3.813889,
                -3.9067654, -3.8857223, -3.8828723]],
    LONGITUDE: [[-38.592189, -38.590389, -38.5904445,
                 -38.5907723, -38.5928892, -38.5929789]],
    TID: [['12017090222', '12017090222', '12017090222',
           '12017090222', '12017090222', '12017090222']]
}


def _transition_graph():
    expected_graph = DiGraph()
    expected_graph.add_node('85', coords=(-3.8347478, -38.592189),
                   datetime=['2017-09-02 22:00:27'], freq_source=1, freq_target=0)
    expected_graph.add_node('673', coords=(-3.8235834, -38.590389),
                   datetime=['2017-09-02 22:01:36'], freq_source=0, freq_target=0)
    expected_graph.add_node('394', coords=(-3.813889, -38.5904445),
                   datetime=['2017-09-02 22:03:08'], freq_source=0, freq_target=0)
    expected_graph.add_node('263', coords=(-3.9067654, -38.5907723),
                   datetime=['2017-09-02 22:03:46'], freq_source=0, freq_target=0)
    expected_graph.add_node('224', coords=(-3.8857223, -38.5928892),
                   datetime=['2017-09-02 22:07:19'], freq_source=0, freq_target=0)
    expected_graph.add_node('623', coords=(-3.8828723, -38.5929789),
               datetime=['2017-09-02 22:07:40'], freq_source=0, freq_target=1)
    expected_graph.add_edge( '85', '673', weight=1, mean_times='0 days 00:01:09')
    expected_graph.add_edge('673', '394', weight=1, mean_times='0 days 00:01:32')
    expected_graph.add_edge('394', '263', weight=1, mean_times='0 days 00:00:38')
    expected_graph.add_edge('263', '224', weight=1, mean_times='0 days 00:03:33')
    expected_graph.add_edge('224', '623', weight=1, mean_times='0 days 00:00:21')

    return expected_graph


def test_populate_graph():
    row = pd.DataFrame(list_data1).loc[0]

    nodes = {'datetime': {}, 'coords': {}, 'freq_source': {}, 'freq_target': {}}
    edges = {}

    expected_nodes = {
        'datetime': {'85': ['2017-09-02 22:00:27'], '673': ['2017-09-02 22:01:36'],
                     '394': ['2017-09-02 22:03:08'], '263': ['2017-09-02 22:03:46'],
                     '224': ['2017-09-02 22:07:19'], '623': ['2017-09-02 22:07:40']},
        'coords': {'85': (-3.8347478, -38.592189), '673': (-3.8235834, -38.590389),
                   '394': (-3.813889, -38.5904445), '263': (-3.9067654, -38.5907723),
                   '224': (-3.8857223, -38.5928892), '623': (-3.8828723, -38.5929789)},
        'freq_source': {'85': 1, '673': 0, '394': 0, '263': 0, '224': 0, '623': 0},
        'freq_target': {'85': 0, '673': 0, '394': 0, '263': 0, '224': 0, '623': 1}}

    expected_edges = {
        '85': {'673': {'weight': 1, 'mean_times': '0 days 00:01:09'}},
        '673': {'394': {'weight': 1, 'mean_times': '0 days 00:01:32'}},
        '394': {'263': {'weight': 1, 'mean_times': '0 days 00:00:38'}},
        '263': {'224': {'weight': 1, 'mean_times': '0 days 00:03:33'}},
        '224': {'623': {'weight': 1, 'mean_times': '0 days 00:00:21'}}}

    _populate_graph(row, nodes, edges)
    nodes, edges

    assert_equal(expected_nodes, nodes)
    assert_equal(expected_edges, edges)


def test_build_transition_graph_from_dict():
    expected_graph = _transition_graph()

    graph = build_transition_graph_from_dict(dict_graph)

    assert_graphs_equal(expected_graph, graph)


def test_build_transition_graph_from_df():
    expected_graph = _transition_graph()

    traj_df = pd.DataFrame(list_data1)

    graph = build_transition_graph_from_df(traj_df)

    assert_graphs_equal(expected_graph, graph)


def test_graph_to_dict():
    graph = _transition_graph()

    expected_dict = {
    'nodes': {
        'coords': { '85': (-3.8347478, -38.592189), '673': (-3.8235834, -38.590389),
                   '394': (-3.813889, -38.5904445), '263': (-3.9067654, -38.5907723),
                   '224': (-3.8857223, -38.5928892), '623': (-3.8828723, -38.5929789)},
        'datetime': { '85': ['2017-09-02 22:00:27'], '673': ['2017-09-02 22:01:36'],
                     '394': ['2017-09-02 22:03:08'], '263': ['2017-09-02 22:03:46'],
                     '224': ['2017-09-02 22:07:19'], '623': ['2017-09-02 22:07:40']},
        'freq_source': {'85': 1, '673': 0, '394': 0, '263': 0, '224': 0, '623': 0},
        'freq_target': {'85': 0, '673': 0, '394': 0, '263': 0, '224': 0, '623': 1}},
    'edges': {
        '85': {'673': {'weight': 1, 'mean_times': '0 days 00:01:09'}},
        '673': {'394': {'weight': 1, 'mean_times': '0 days 00:01:32'}},
        '394': {'263': {'weight': 1, 'mean_times': '0 days 00:00:38'}},
        '263': {'224': {'weight': 1, 'mean_times': '0 days 00:03:33'}},
        '224': {'623': {'weight': 1, 'mean_times': '0 days 00:00:21'}},
        '623': {}}}

    dict_graph = graph_to_dict(graph)

    assert_equal(expected_dict, dict_graph)


def test_save_graph_as_json(tmpdir):

    expected = {
        'nodes': {
            'coords': { '85': (-3.8347478, -38.592189), '673': (-3.8235834, -38.590389),
                       '394': (-3.813889, -38.5904445), '263': (-3.9067654, -38.5907723),
                       '224': (-3.8857223, -38.5928892), '623': (-3.8828723, -38.5929789)},
            'datetime': { '85': ['2017-09-02 22:00:27'], '673': ['2017-09-02 22:01:36'],
                         '394': ['2017-09-02 22:03:08'], '263': ['2017-09-02 22:03:46'],
                         '224': ['2017-09-02 22:07:19'], '623': ['2017-09-02 22:07:40']},
            'freq_source': {'85': 1, '673': 0, '394': 0, '263': 0, '224': 0, '623': 0},
            'freq_target': {'85': 0, '673': 0, '394': 0, '263': 0, '224': 0, '623': 1}},
        'edges': {
            '85': {'673': {'weight': 1, 'mean_times': '0 days 00:01:09'}},
            '673': {'394': {'weight': 1, 'mean_times': '0 days 00:01:32'}},
            '394': {'263': {'weight': 1, 'mean_times': '0 days 00:00:38'}},
            '263': {'224': {'weight': 1, 'mean_times': '0 days 00:03:33'}},
            '224': {'623': {'weight': 1, 'mean_times': '0 days 00:00:21'}},
            '623': {}}}

    d = tmpdir.mkdir('utils')

    file_write_default = d.join('test_save_graph.json')
    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    graph = _transition_graph()
    save_graph_as_json(graph, filename_write_default)
    saved_graph = read_graph_json(filename_write_default)

    assert_equal(saved_graph, expected)


def test_read_graph_json(tmpdir):

    expected = {
        'nodes': {
            'coords': { '85': (-3.8347478, -38.592189), '673': (-3.8235834, -38.590389),
                       '394': (-3.813889, -38.5904445), '263': (-3.9067654, -38.5907723),
                       '224': (-3.8857223, -38.5928892), '623': (-3.8828723, -38.5929789)},
            'datetime': { '85': ['2017-09-02 22:00:27'], '673': ['2017-09-02 22:01:36'],
                         '394': ['2017-09-02 22:03:08'], '263': ['2017-09-02 22:03:46'],
                         '224': ['2017-09-02 22:07:19'], '623': ['2017-09-02 22:07:40']},
            'freq_source': {'85': 1, '673': 0, '394': 0, '263': 0, '224': 0, '623': 0},
            'freq_target': {'85': 0, '673': 0, '394': 0, '263': 0, '224': 0, '623': 1}},
        'edges': {
            '85': {'673': {'weight': 1, 'mean_times': '0 days 00:01:09'}},
            '673': {'394': {'weight': 1, 'mean_times': '0 days 00:01:32'}},
            '394': {'263': {'weight': 1, 'mean_times': '0 days 00:00:38'}},
            '263': {'224': {'weight': 1, 'mean_times': '0 days 00:03:33'}},
            '224': {'623': {'weight': 1, 'mean_times': '0 days 00:00:21'}},
            '623': {}}}

    d = tmpdir.mkdir('utils')

    file_write_default = d.join('test_read_graph.json')
    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )

    graph = _transition_graph()

    with open(filename_write_default, 'w') as f:
        json.dump(dict_graph, f)

    saved_graph = read_graph_json(filename_write_default)

    assert_equal(saved_graph, expected)
