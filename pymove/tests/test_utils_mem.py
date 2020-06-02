import os
import time

import psutil
from numpy.testing import assert_equal

import pymove
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

    expected_initial_size = 208

    expected_final_size = 160

    assert_equal(mem.total_size(move_df), expected_initial_size)

    mem.reduce_mem_usage_automatic(move_df)

    assert_equal(mem.total_size(move_df), expected_final_size)


def test_total_size():

    move_df = _default_move_df()

    expected_initial_size = 208

    assert_equal(mem.total_size(move_df), expected_initial_size)


def test_begin_operation():

    process = psutil.Process(os.getpid())

    expected = {'process': process,
                'init': process.memory_info()[0],
                'start': time.time(),
                'name': 'operation'}

    operation_info = mem.begin_operation('operation')

    assert_equal(operation_info, expected)


def test_end_operation():

    operation = mem.begin_operation('operation')

    finish = operation['process'].memory_info()[0]
    last_operation_time_duration = time.time() - operation['start']
    last_operation_mem_usage = finish - operation['init']

    expected = {'name': 'operation',
                'time in seconds': last_operation_time_duration ,
                'memory': mem.sizeof_fmt(last_operation_mem_usage)}

    mem.end_operation(operation)


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
