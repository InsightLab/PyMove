"""
Memory  operations.

reduce_mem_usage_automatic,
total_size,
begin_operation,
end_operation,
sizeof_fmt,
top_mem_vars

"""

import os
import time
from collections import deque
from itertools import chain
from sys import getsizeof
from typing import Callable, Dict, Optional, Text

import numpy as np
import psutil
from pandas import DataFrame

from pymove.utils.log import logger

try:
    pass
except ImportError:
    pass


def reduce_mem_usage_automatic(df: DataFrame):
    """
    Reduces the memory usage of the given dataframe.

    Parameter
    ---------
    df : dataframe
        The input data to which the operation will be performed.

    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    logger.info('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype
        if str(col_type) == 'int':
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif (
                c_min > np.iinfo(np.uint8).min
                and c_max < np.iinfo(np.uint8).max
            ):
                df[col] = df[col].astype(np.uint8)
            elif (
                c_min > np.iinfo(np.int16).min
                and c_max < np.iinfo(np.int16).max
            ):
                df[col] = df[col].astype(np.int16)
            elif (
                c_min > np.iinfo(np.uint16).min
                and c_max < np.iinfo(np.uint16).max
            ):
                df[col] = df[col].astype(np.uint16)
            elif (
                c_min > np.iinfo(np.int32).min
                and c_max < np.iinfo(np.int32).max
            ):
                df[col] = df[col].astype(np.int32)
            elif (
                c_min > np.iinfo(np.uint32).min
                and c_max < np.iinfo(np.uint32).max
            ):
                df[col] = df[col].astype(np.uint32)
            elif (
                c_min > np.iinfo(np.int64).min
                and c_max < np.iinfo(np.int64).max
            ):
                df[col] = df[col].astype(np.int64)
            elif (
                c_min > np.iinfo(np.uint64).min
                and c_max < np.iinfo(np.uint64).max
            ):
                df[col] = df[col].astype(np.uint64)
        elif col_type == np.float:
            c_min = df[col].min()
            c_max = df[col].max()
            if (
                c_min > np.finfo(np.float16).min
                and c_max < np.finfo(np.float16).max
            ):
                df[col] = df[col].astype(np.float16)
            elif (
                c_min > np.finfo(np.float32).min
                and c_max < np.finfo(np.float32).max
            ):
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    logger.info('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    logger.info(
        'Decreased by {:.1f} %'.format(100 * (start_mem - end_mem) / start_mem)
    )


def total_size(
    o: object, handlers: Dict = None, verbose: Optional[bool] = True
) -> float:
    """
    Calculates the approximate memory footprint of an given object.

    Automatically finds the contents of the following builtin
    containers and their subclasses:  tuple, list, deque, dict, set and
    frozenset.

    Parameters
    ----------
    o : object
        The object to calculate his memory footprint.
    handlers : dict, optional
        To search other containers, add handlers to iterate over their contents,
            handlers = {SomeContainerClass: iter,
                        OtherContainerClass: OtherContainerClass.get_elements}
         by default None
    verbose : boolean, optional
        If set to True, the following information will be printed for
        each content of the object, by default False
            - the size of the object in bytes.
            - his type_
            - the object values

    Returns
    -------
    float
        The memory used by the given object

    """
    if handlers is None:
        handlers = {}

    def dict_handler(d):
        return chain.from_iterable(d.items())

    all_handlers = {
        tuple: iter,
        list: iter,
        deque: iter,
        dict: dict_handler,
        set: iter,
        frozenset: iter,
    }
    # user handlers take precedence
    all_handlers.update(handlers)
    # track which object id"srs have already been seen
    seen = set()
    # estimate sizeof object without __sizeof__
    default_size = getsizeof(0)

    def sizeof(o):
        # do not double count the same object
        if id(o) in seen:
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break

        if verbose:

            logger.info('Size in bytes: {}, Type: {}'.format(s, type(o)))

        return s

    return sizeof(o)


def begin_operation(name: Text) -> Dict:
    """
    Gets the stats for the current operation.

    Parameters
    ----------
    name: str
        name of the operation

    Returns
    -------
    dict
        dictionary with the operation stats

    """
    process = psutil.Process(os.getpid())
    init = process.memory_info()[0]
    start = time.time()
    return {'process': process, 'init': init, 'start': start, 'name': name}


def end_operation(operation: Dict) -> Dict:
    """
    Gets the time and memory usage of the operation.

    Parameters
    ----------
    operation: dict
        dictionary with the begining stats of the operation

    Returns
    -------
    dict
        dictionary with the operation execution stats

    """
    finish = operation['process'].memory_info()[0]
    last_operation_name = operation['name']
    last_operation_time_duration = time.time() - operation['start']
    last_operation_mem_usage = finish - operation['init']
    return {
        'name': last_operation_name,
        'time in seconds': last_operation_time_duration,
        'memory': sizeof_fmt(last_operation_mem_usage),
    }


def sizeof_fmt(mem_usage: int, suffix: Optional[Text] = 'B') -> Text:
    """
    Returns the memory usage calculation of the last function.

    Parameters
    ----------
    mem_usage : int
        memory usage in bytes

    suffix: string, optional
        suffix of the unit, by default 'B'

    Returns
    -------
    str
        A string of the memory usage in a more readable format

    """
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(mem_usage) < 1024.0:
            return '{:3.1f} {}{}'.format(mem_usage, unit, suffix)
        mem_usage /= 1024.0
    return '{:.1f} {}{}'.format(mem_usage, 'Yi', suffix)


def top_mem_vars(
    variables: Optional[Callable] = None, n: Optional[int] = 10, hide_private=True
) -> DataFrame:
    """
    Shows the sizes of the active variables.

    Parameters
    ----------
    variables: locals() or globals(), optional
        Whether to shows local or global variables, by default globals()
    n: int, optional
        number of variables to show, by default
    hide_private: bool, optional
        Whether to hide private variables, by default True

    Returns
    -------
    DataFrame
        dataframe with variables names and sizes

    """
    if variables is None:
        variables = globals()
    vars_ = ((name, getsizeof(value)) for name, value in variables.items())
    if hide_private:
        vars_ = filter(lambda x: not x[0].startswith('_'), vars_)
    top_vars = DataFrame(
        sorted(vars_, key=lambda x: -x[1])[:n],
        columns=['var', 'mem']
    )
    top_vars['mem'] = top_vars['mem'].apply(sizeof_fmt)

    return top_vars
