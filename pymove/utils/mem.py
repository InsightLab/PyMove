from __future__ import print_function

import re
import os
import pwd
import json
import psutil
import time
import resource

import numpy as np
import pandas as pd
from math import log10

from sys import getsizeof, stderr
from itertools import chain
from collections import deque

try:
    from reprlib import repr
except ImportError:
    pass


def proc_info():
    """This functions retrieves informations about each jupyter notebook running in the machine.

    Returns
    -------
    df_mem : dataframe
        A dataframe with the following informations about each jupyter notebook process:
            - user : username
            - pid : process identifier
            - memory_GB : memory usage
            - kernel_ID : kernel id

    Examples
    --------
    Example :
        >>> mem.get_proc_info()
                user 	pid 	memory_GB 	kernel_ID
            0 	999999 	11797 	0.239374 	74efe612-927f-4c1f-88a6-bb5fd32bc65c
            1 	999999 	11818 	0.172012 	11c38dd6-8a65-4c45-90cf-0da5db65fa99
    """

    UID = 1

    regex = re.compile(r".+kernel-(.+)\.json")
    port_regex = re.compile(r"port=(\d+)")

    pids = [pid for pid in os.listdir("/proc") if pid.isdigit()]

    # memory info from psutil.Process
    df_mem = []

    for pid in pids:
        try:
            ret = open(os.path.join("/proc", pid, "cmdline"), "rb").read()
            ret_str = ret.decode("utf-8")
        except IOError:  # proc has already terminated
            continue

        # jupyter notebook processes
        if len(ret_str) > 0 and ("jupyter" in ret_str or "ipython" in ret_str) and "kernel" in ret_str:
            # kernel
            kernel_ID = re.sub(regex, r"\1", ret_str)[0:-1]
            #kernel_ID = filter(lambda x: x in string.printable, kernel_ID)

            # memory
            process = psutil.Process(int(pid))
            mem = process.memory_info()[0] / float(1e9)

            # user name for pid
            for ln in open("/proc/{0}/status".format(int(pid))):
                if ln.startswith("Uid:"):
                    uid = int(ln.split()[UID])
                    uname = pwd.getpwuid(uid).pw_name

            # user, pid, memory, kernel_ID
            df_mem.append([uname, pid, mem, kernel_ID])

    df_mem = pd.DataFrame(df_mem)
    df_mem.columns = ["user", "pid", "memory_GB", "kernel_ID"]
    return df_mem


def session_info(sessions_str):
    sessions = json.loads(sessions_str)
    df_nb = []
    kernels = []
    for sess in sessions:
        kernel_ID = sess["kernel"]["id"]
        if kernel_ID not in kernels:
            notebook_path = sess["notebook"]["path"]
            df_nb.append([kernel_ID, notebook_path])
            kernels.append(kernel_ID)

    df_nb = pd.DataFrame(df_nb)
    df_nb.columns = ["kernel_ID", "notebook_path"]
    return df_nb


def stats(sessions_str):
    df_mem = proc_info()
    df_nb = session_info(sessions_str)

    # joining tables
    df = pd.merge(df_nb, df_mem, on=["kernel_ID"], how="right")
    df = df.sort_values("memory_GB", ascending=False)
    del(df_mem)
    del(df_nb)
    return df.reset_index(drop=True)


def mem():
    """Calculates the resource consumed the current process.

    Returns
    -------
    mem : float
        The used memory by the process in MB.
    """

    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return mem


def reduce_mem_usage_automatic(df):
    """Reduces the memory usage of the given dataframe.

    Parameter
    ---------
    df : dataframe
        The input data to which the operation of memory reduction will be performed.

    """
    start_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype
        if str(col_type) == "int":
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.uint8).min and c_max < np.iinfo(np.uint8).max:
                df[col] = df[col].astype(np.uint8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.uint16).min and c_max < np.iinfo(np.uint16).max:
                df[col] = df[col].astype(np.uint16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            elif c_min > np.iinfo(np.uint32).min and c_max < np.iinfo(np.uint32).max:
                df[col] = df[col].astype(np.uint32)
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                df[col] = df[col].astype(np.int64)
            elif c_min > np.iinfo(np.uint64).min and c_max < np.iinfo(np.uint64).max:
                df[col] = df[col].astype(np.uint64)
        elif col_type == np.float:
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))


def total_size(o, handlers={}, verbose=False):
    """ Calculates the approximate memory footprint of an given object and all of its contents.
    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.

    Parameters
    ----------
    o : object
        The object to calculate his memory footprint.
    handlers : dict, optional(empty by default)
        To search other containers, add handlers to iterate over their contents, example:
            handlers = {SomeContainerClass: iter,
                        OtherContainerClass: OtherContainerClass.get_elements}
    verbose : boolean, optional(False by default)
        If set to True, the following information will be printed for each content of the object:
            - the size of the object in bytes.
            - his type
            - the object values

    Returns
    -------
    s : float
        The memory used by the given object
    """

    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {
                    tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter
                    }
    # user handlers take precedence
    all_handlers.update(handlers)
    # track which object id"s have already been seen
    seen = set()
    # estimate sizeof object without __sizeof__
    default_size = getsizeof(0)

    def sizeof(o):
        # do not double count the same object
        if id(o) in seen:
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)

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
        'memory': format_mem(last_operation_mem_usage)
    }

def format_mem(mem_usage):
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

    size = int(log10(max(1, mem_usage))) + 1
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
