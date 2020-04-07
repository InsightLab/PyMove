import psutil
import os
import time
from math import log10


def begin_operation(name):
    """
    Gets the current mem and time
    """
    process = psutil.Process(os.getpid())
    init = process.memory_info()[0]
    start = time.time()
    return { 'process': process, 'init': init, 'start': start, 'name': name }

def end_operation(operation):
    """
    Calculares the mem and time used
    """
    finish = operation['process'].memory_info()[0]
    last_operation_name = operation['name']
    last_operation_time_duration = time.time() - operation['start']
    last_operation_mem_usage = finish - operation['init']
    return {
        'name': last_operation_name,
        'time': last_operation_time_duration,
        'mem': mem(last_operation_mem_usage)
    }

def mem(last_operation_mem_usage):
    """Returns the memory usage calculation of the last function.
    Automatically converting or in the data format required by the user.

    Parameters
    ----------
    format : str
    The data format to which the memory calculation must be converted to.

    Returns
    -------
    A string of the memory usage calculation of the last function
    """

    switcher = {
        'B': last_operation_mem_usage,
        'KB': last_operation_mem_usage / 1024,
        'MB': last_operation_mem_usage / (1024 ** 2),
        'GB': last_operation_mem_usage / (1024 ** 3),
        'TB': last_operation_mem_usage / (1024 ** 4),
    }

    size = int(log10(last_operation_mem_usage + 1)) + 1
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
