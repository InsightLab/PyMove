import os
import time

import psutil
from numpy.testing import assert_array_equal, assert_equal

from pymove import MoveDataFrame, mem
from pymove.utils.constants import DATETIME, LATITUDE, LONGITUDE, TRAJ_ID

list_data = [
    [39.984094, 116.319236, '2008-10-23 05:53:05', 1],
    [39.984198, 116.319322, '2008-10-23 05:53:06', 1],
    [39.984224, 116.319402, '2008-10-23 05:53:11', 1],
    [39.984224, 116.319402, '2008-10-23 05:53:11', 2],
]


def _default_move_df():
    return MoveDataFrame(
        data=list_data,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        datetime=DATETIME,
        traj_id=TRAJ_ID,
    )


def test_reduce_mem_usage_automatic():

    move_df = _default_move_df()

    expected_initial_size = 280

    expected_final_size = 232

    assert abs(mem.total_size(move_df) - expected_initial_size) <= 20

    mem.reduce_mem_usage_automatic(move_df)

    assert abs(mem.total_size(move_df) - expected_final_size) <= 20


def test_total_size():

    move_df = _default_move_df()

    expected_initial_size = 280

    assert abs(mem.total_size(move_df) - expected_initial_size) <= 20


def test_begin_operation():

    process = psutil.Process(os.getpid())

    expected = {'process': process,
                'init': process.memory_info()[0],
                'start': time.time(),
                'name': 'operation'}

    operation_info = mem.begin_operation('operation')

    assert_equal(list(operation_info.keys()), list(expected.keys()))
    assert_equal(operation_info['process'], expected['process'])
    assert_equal(int(operation_info['init']), int(expected['init']))
    assert_equal(int(operation_info['start']), int(expected['start']))
    assert_equal(operation_info['name'], expected['name'])


def test_end_operation():

    operation = mem.begin_operation('operation')

    finish = operation['process'].memory_info()[0]

    last_operation_mem_usage = finish - operation['init']

    operation_info = mem.end_operation(operation)

    last_operation_time_duration = time.time() - operation['start']

    expected = {'name': 'operation',
                'time in seconds': last_operation_time_duration ,
                'memory': mem.sizeof_fmt(last_operation_mem_usage)}

    assert_equal(list(operation_info.keys()), list(expected.keys()))
    assert_equal(operation_info['name'], expected['name'])
    assert_equal(int(operation_info['time in seconds']),
                 int(expected['time in seconds']))
    assert_equal(operation_info['memory'], expected['memory'])


def test_sizeof_fmt():

    expected = '1.0 KiB'

    result = mem.sizeof_fmt(1024)

    assert_equal(expected, result)

    expected = '9.5 MiB'

    result = mem.sizeof_fmt(10000000)

    assert_equal(expected, result)

    expected = '9.3 GiB'

    result = mem.sizeof_fmt(10000000000)

    assert_equal(expected, result)

    expected = '10.0 b'

    result = mem.sizeof_fmt(10, 'b')

    assert_equal(expected, result)


def test_top_mem_vars():
    move_df = _default_move_df()
    list_data_ = list_data
    local_vars = mem.top_mem_vars(locals())

    assert_array_equal(local_vars.shape, (2, 2))
    assert_array_equal(local_vars.columns, ['var', 'mem'])
    assert_array_equal(local_vars['var'].values, ['move_df', 'list_data_'])
