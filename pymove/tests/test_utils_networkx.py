import os
import json
import numpy as np
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
    TID_STAT,
    TRAJ_ID,
)
from pymove.utils.networkx import (
    _populate_graph,
    build_transition_graph_from_df,
    build_transition_graph_from_dict,
    graph_to_dict,
    save_graph_as_json,
    read_graph_json
)

dict_graph = {
    'nodes': {
        'datetime': {
            '2': ['2020-01-01 09:10:15'], '4': ['2020-01-01 09:15:45'],
            '6': ['2020-01-01 09:25:30'], '8': ['2020-01-01 09:30:17'],
            '9': ['2020-01-01 09:45:16']},
        'coords': {
            '2': (3.1234567165374756, 38.12345504760742),
            '4': (3.1234567165374756, 38.12345504760742),
            '6': (3.1234567165374756, 38.12345504760742),
            '8': (3.1234567165374756, 38.12345504760742),
            '9': (3.1234567165374756, 38.12345504760742)},
        'freq_source': {
            '2': 1, '4': 0, '6': 0, '8': 0, '9': 0},
        'freq_target': {
            '2': 0, '4': 0, '6': 0, '8': 0, '9': 1}},
    'edges': {
        '2': {'4': {'weight': 1, 'mean_times': '0 days 00:05:30'}},
        '4': {'6': {'weight': 1, 'mean_times': '0 days 00:09:45'}},
        '6': {'8': {'weight': 1, 'mean_times': '0 days 00:04:47'}},
        '8': {'9': {'weight': 1, 'mean_times': '0 days 00:14:59'}}}
}

list_data1 = {
    TRAJ_ID: [['d95bafc8f2a4d27bdcf4bb99f4bea973', 'd95bafc8f2a4d27bdcf4bb99f4bea973',
               'd95bafc8f2a4d27bdcf4bb99f4bea973', 'd95bafc8f2a4d27bdcf4bb99f4bea973',
               'd95bafc8f2a4d27bdcf4bb99f4bea973']],
    LOCAL_LABEL: [[2, 4, 6, 8, 9]],
    DATETIME: [[pd.Timestamp('2020-01-01 09:10:15'),
                pd.Timestamp('2020-01-01 09:15:45'),
                pd.Timestamp('2020-01-01 09:25:30'),
                pd.Timestamp('2020-01-01 09:30:17'),
                pd.Timestamp('2020-01-01 09:45:16')]],
    LATITUDE: [[3.1234567165374756, 3.1234567165374756,
                3.1234567165374756, 3.1234567165374756,
                3.1234567165374756]],
    LONGITUDE: [[38.12345504760742, 38.12345504760742,
                 38.12345504760742, 38.12345504760742,
                 38.12345504760742]],
    PREV_LOCAL: [[np.nan, 2, 4, 6, 8]],
    TID_STAT: [[2, 2, 2, 2, 2]]}


def _transition_graph():
    expected_graph = DiGraph()
    expected_graph.add_node('2', coords=(3.1234567165374756, 38.12345504760742),
                   datetime=['2020-01-01 09:10:15'], freq_source=1, freq_target=0)
    expected_graph.add_node('4', coords=(3.1234567165374756, 38.12345504760742),
                   datetime=['2020-01-01 09:15:45'], freq_source=0, freq_target=0)
    expected_graph.add_node('6', coords=(3.1234567165374756, 38.12345504760742),
                   datetime=['2020-01-01 09:25:30'], freq_source=0, freq_target=0)
    expected_graph.add_node('8', coords=(3.1234567165374756, 38.12345504760742),
                   datetime=['2020-01-01 09:30:17'], freq_source=0, freq_target=0)
    expected_graph.add_node('9', coords=(3.1234567165374756, 38.12345504760742),
                   datetime=['2020-01-01 09:45:16'], freq_source=0, freq_target=1)
    expected_graph.add_edge('2', '4', weight=1, mean_times='0 days 00:05:30')
    expected_graph.add_edge('4', '6', weight=1, mean_times='0 days 00:09:45')
    expected_graph.add_edge('6', '8', weight=1, mean_times='0 days 00:04:47')
    expected_graph.add_edge('8', '9', weight=1, mean_times='0 days 00:14:59')

    return expected_graph


def test_populate_graph():
    row = pd.DataFrame(list_data1).loc[0]

    nodes = {'datetime': {}, 'coords': {}, 'freq_source': {}, 'freq_target': {}}
    edges = {}

    expected_nodes = {
        'datetime': {'2': ['2020-01-01 09:10:15'], '4': ['2020-01-01 09:15:45'],
                     '6': ['2020-01-01 09:25:30'], '8': ['2020-01-01 09:30:17'],
                     '9': ['2020-01-01 09:45:16']},
        'coords': {'2': (3.1234567165374756, 38.12345504760742),
                   '4': (3.1234567165374756, 38.12345504760742),
                   '6': (3.1234567165374756, 38.12345504760742),
                   '8': (3.1234567165374756, 38.12345504760742),
                   '9': (3.1234567165374756, 38.12345504760742)},
        'freq_source': {'2': 1, '4': 0, '6': 0, '8': 0, '9': 0},
        'freq_target': {'2': 0, '4': 0, '6': 0, '8': 0, '9': 1}}

    expected_edges = {
        '2': {'4': {'weight': 1, 'mean_times': '0 days 00:05:30'}},
        '4': {'6': {'weight': 1, 'mean_times': '0 days 00:09:45'}},
        '6': {'8': {'weight': 1, 'mean_times': '0 days 00:04:47'}},
        '8': {'9': {'weight': 1, 'mean_times': '0 days 00:14:59'}}}

    _populate_graph(row, nodes, edges)

    assert_equal(expected_nodes, nodes)
    assert_equal(expected_edges, edges)


def test_build_transition_graph_from_dict():
    expected_graph = _transition_graph()

    graph = build_transition_graph_from_dict(dict_graph)

    assert_graphs_equal(expected_graph, graph)


def test_build_transition_graph_from_df():
    expected_graph = _transition_graph()

    data = pd.DataFrame(list_data1)

    graph = build_transition_graph_from_df(data)

    assert_graphs_equal(expected_graph, graph)


def test_graph_to_dict():
    graph = _transition_graph()

    expected_dict = {
        'nodes': {
            'coords': {
                '2': (3.1234567165374756, 38.12345504760742),
                '4': (3.1234567165374756, 38.12345504760742),
                '6': (3.1234567165374756, 38.12345504760742),
                '8': (3.1234567165374756, 38.12345504760742),
                '9': (3.1234567165374756, 38.12345504760742)},
            'datetime': {
                '2': ['2020-01-01 09:10:15'],
                '4': ['2020-01-01 09:15:45'],
                '6': ['2020-01-01 09:25:30'],
                '8': ['2020-01-01 09:30:17'],
                '9': ['2020-01-01 09:45:16']},
            'freq_source': {'2': 1, '4': 0, '6': 0, '8': 0, '9': 0},
            'freq_target': {'2': 0, '4': 0, '6': 0, '8': 0, '9': 1}},
        'edges': {
            '2': {'4': {'weight': 1, 'mean_times': '0 days 00:05:30'}},
            '4': {'6': {'weight': 1, 'mean_times': '0 days 00:09:45'}},
            '6': {'8': {'weight': 1, 'mean_times': '0 days 00:04:47'}},
            '8': {'9': {'weight': 1, 'mean_times': '0 days 00:14:59'}},
            '9': {}}
    }

    dict_graph = graph_to_dict(graph)

    assert_equal(expected_dict, dict_graph)


def test_save_graph_as_json(tmpdir):
    
    expected = {
        'nodes': {
            'coords': {
                '2': (3.1234567165374756, 38.12345504760742),
                '4': (3.1234567165374756, 38.12345504760742),
                '6': (3.1234567165374756, 38.12345504760742),
                '8': (3.1234567165374756, 38.12345504760742),
                '9': (3.1234567165374756, 38.12345504760742)},
            'datetime': {
                '2': ['2020-01-01 09:10:15'], '4': ['2020-01-01 09:15:45'],
                '6': ['2020-01-01 09:25:30'], '8': ['2020-01-01 09:30:17'],
                '9': ['2020-01-01 09:45:16']},
            'freq_source': {'2': 1, '4': 0, '6': 0, '8': 0, '9': 0},
            'freq_target': {'2': 0, '4': 0, '6': 0, '8': 0, '9': 1}},
         'edges': {
             '2': {'4': {'weight': 1, 'mean_times': '0 days 00:05:30'}},
             '4': {'6': {'weight': 1, 'mean_times': '0 days 00:09:45'}},
             '6': {'8': {'weight': 1, 'mean_times': '0 days 00:04:47'}},
             '8': {'9': {'weight': 1, 'mean_times': '0 days 00:14:59'}},
             '9': {}}}
    
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
            'coords': {
                '2': (3.1234567165374756, 38.12345504760742),
                '4': (3.1234567165374756, 38.12345504760742),
                '6': (3.1234567165374756, 38.12345504760742),
                '8': (3.1234567165374756, 38.12345504760742),
                '9': (3.1234567165374756, 38.12345504760742)},
            'datetime': {
                '2': ['2020-01-01 09:10:15'], '4': ['2020-01-01 09:15:45'],
                '6': ['2020-01-01 09:25:30'], '8': ['2020-01-01 09:30:17'],
                '9': ['2020-01-01 09:45:16']},
            'freq_source': {'2': 1, '4': 0, '6': 0, '8': 0, '9': 0},
            'freq_target': {'2': 0, '4': 0, '6': 0, '8': 0, '9': 1}},
         'edges': {
             '2': {'4': {'weight': 1, 'mean_times': '0 days 00:05:30'}},
             '4': {'6': {'weight': 1, 'mean_times': '0 days 00:09:45'}},
             '6': {'8': {'weight': 1, 'mean_times': '0 days 00:04:47'}},
             '8': {'9': {'weight': 1, 'mean_times': '0 days 00:14:59'}},
             '9': {}}}
    
    d = tmpdir.mkdir('utils')
    
    file_write_default = d.join('test_read_graph.json')
    filename_write_default = os.path.join(
        file_write_default.dirname, file_write_default.basename
    )
    
    graph = _transition_graph()
    
    with open(filename_write_default, 'w') as f:
        json.dump(graph_to_dict(graph), f)

    saved_graph = read_graph_json(filename_write_default)    
    
    assert_equal(saved_graph, expected)