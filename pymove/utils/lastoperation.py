import psutil
import os
import time
from math import log10

class LastOperation():
    def __init__(self):
        self._last_operation_name = ''
        self._last_operation_time_duration = 0
        self._last_operation_mem_usage = 0

    def __str__(self):
        return str({
            'name': self._last_operation_name,
            'time': self._last_operation_time_duration,
            'mem': self.mem()
        })

    def get_last_operation(self, format='B'):
        """Returns the name, memory and time usage calculation, of the last function.
        
        Parameters
        ----------
        format : str
        The data format to which the memory calculation must be converted to.

        Returns
        -------
        A dict that contains the memory usage calculation, in bytes, of the last function,
        called to the PandasMoveDataFrame object.
        """
        return {
            'name': self._last_operation_name,
            'time': self._last_operation_time_duration,
            'mem': self.mem(format)
        }

    def begin_operation(self, name):
        """
        Gets the current mem and time
        """
        process = psutil.Process(os.getpid())
        init = process.memory_info()[0]
        start = time.time()
        return { 'process': process, 'init': init, 'start': start, 'name': name }

    def end_operation(self, operation):
        """
        Calculares the mem and time used
        """
        finish = operation['process'].memory_info()[0]
        self._last_operation_name = operation['name']
        self._last_operation_time_duration = time.time() - operation['start']
        self._last_operation_mem_usage = finish - operation['init']

    def mem(self, format=None):
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
            'B': self._last_operation_mem_usage,
            'KB': self._last_operation_mem_usage / 1000,
            'MB': self._last_operation_mem_usage / (1000 ** 2),
            'GB': self._last_operation_mem_usage / (1000 ** 3),
            'TB': self._last_operation_mem_usage / (1000 ** 4),
        }

        if not format:
            size = int(log10(self._last_operation_mem_usage + 1)) + 1
            if size <= 3:
                format = 'B'
            elif size <= 6:
                format = 'KB'
            elif size <= 9:
                format = 'MB'
            elif size <= 12:
                format = 'GB'
            else:
                format = 'TB'

        return f'{switcher[format]} {format}'
