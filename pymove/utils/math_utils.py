from __future__ import division
import time
import math
import folium
import datetime
import numpy as np
import pandas as pd

from IPython.display import display
from ipywidgets import IntProgress, HTML, VBox
from pandas._libs.tslibs.timestamps import Timestamp


def std(sum_sq, size, avg):
    try:
        # squaring with * is over 3 times as fast as with **2
        # http://stackoverflow.com/questions/29046346/comparison-of-power-to-multiplication-in-python
        result = math.sqrt(sum_sq / size - avg * avg)
    except ValueError:
        e = '(size - avg^2) (size=%s, avg=%s, sum_sq=%s) should be non negative, but is %s' % \
            (size, avg, sum_sq, size - avg * avg)
        raise ValueError(e)
    return result


def avg_std(sum1, sum_sq, size):
    avg = sum1 / size
    return avg, std(sum_sq, size, avg)


def std_sample(sum_sq, size, avg):
    return std(sum_sq, size, avg) * math.sqrt(size / (size - 1))


def avg_std_sample(sum1, sum_sq, size):
    avg = sum1 / size
    return avg, std_sample(sum_sq, size, avg)

#isto eh math mesmo ou deveria estar no array_utils. No fundo o que ele ta fazendo eh a media
# função está dando erro ao rodar
def arrays_avg(values_array, weights_array=None):
    """Computes the mean of the elements of the array.

    values_array : array of floats
        The numbers used to calculate the mean.

    weights_array : array of floats
        Used to calculate the weighted average, indicates the weight of each element in the array (values_array).

    Returns
    -------
        result : Float
            The mean of the array elements.
    """
    n = len(values_array)

    if weights_array is None:
        weights_array = np.full(n, 1)
    elif len(weights_array) != n:
        raise ValueError('values_array and qt_array must have the same number of rows')

    n_row = len(values_array[0])
    result = np.full(n_row, 0)
    for i, item in enumerate(values_array):
        for j in range(n_row):
            result[j] += item[j] * weights_array[i]

    sum_qt = array_sum(weights_array)
    for i in range(n_row):
        result[i] /= sum_qt

    return result

#isto eh math mesmo ou deveria estar no array_utils.
def array_sum(values_array):
    """Computes the sum of the elements of the array.

    values_array : array of floats
        The numbers to be added.

    Returns
    -------
        sum1 : Float
            The sum of the elements of the array
    """
    sum1 = 0
    for item in values_array:
        sum1 += item

    return sum1

#isto eh math mesmo ou deveria estar no array_utils.
def array_stats(values_array):
    """Computes the sum of all the elements in the array, the sum of the square of each element and the number of
        elements of the array.

    values_array : array of floats
        The elements used to compute the operations

    Returns
    -------
        sum1 : Float
            The sum of all the elements in the array

        sum_sq : Float
            The sum of the square value of each element in the array

        n : Integer
            The number of elements in the array
    """
    sum1 = 0
    sum_sq = 0
    n = 0
    for item in values_array:
        sum1 += item
        sum_sq += item * item
        n += 1

    return sum1, sum_sq, n