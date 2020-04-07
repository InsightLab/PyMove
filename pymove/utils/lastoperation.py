import psutil
import os
import time
from math import log10


def begin_operation(name):
    """
    Gets the stats for the current operation

    Parameters
    ----------
    name: String
        name of the operation

    Returns
    -------
    dict:
        dictionary with the operation stats
    """
    process = psutil.Process(os.getpid())
    init = process.memory_info()[0]
    start = time.time()
    return { 'process': process, 'init': init, 'start': start, 'name': name }

def end_operation(operation):
    """
    Gets the time and memory usage of the operation

    Parameters
    ----------
    operation: dict
        dictionary with the begining stats of the operation

    Returns
    -------
    dict:
        dictionary with the operation execution stats
    """
    finish = operation['process'].memory_info()[0]
    last_operation_name = operation['name']
    last_operation_time_duration = time.time() - operation['start']
    last_operation_mem_usage = finish - operation['init']
    return {
        'name': last_operation_name,
        'time in seconds': last_operation_time_duration,
        'memory': mem(last_operation_mem_usage)
    }

def mem(mem_usage):
    """Returns the memory usage calculation of the last function.

    Parameters
    ----------
    mem_usage : int
        memory usage in bytes

    Returns
    -------
    A string of the memory usage in a more readable format
    """

    switcher = {
        'B': mem_usage,
        'KB': mem_usage / 1024,
        'MB': mem_usage / (1024 ** 2),
        'GB': mem_usage / (1024 ** 3),
        'TB': mem_usage / (1024 ** 4),
    }

    size = int(log10(mem_usage + 1)) + 1
    if size <= 3:
        unit = 'B'
    elif size <= 6:
        unit = 'KB'
    elif size <= 9:
        unit = 'MB'
    elif size <= 12:
        unit = 'GB'
    else:
        unit = 'TB'

    return f'{switcher[unit]} {unit}'
