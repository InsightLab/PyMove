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


def list_to_str(input_list, delimiter=','):
    """Concatenates the elements of the array, joining them by the separator especified by the parameter "delimiter"

    Parameters
    ----------
    input_list : array
        The elements to be joined

    delimiter : String, optional(',' by default)
        The separator used between elements

    Returns
    -------
        String
            Returns a string, wich is the concatenation of the elements of the array, separeted by the delimiter.
    """
    return delimiter.join([x if type(x) == str else repr(x) for x in input_list])  # list comprehension


def list_to_csv_str(input_list):
    """Concatenates the elements of the array, joining them by ",".

    Parameters
    ----------
    input_list : array
        The elements to be joined

    Returns
    -------
        String
            Returns a string, wich is the concatenation of the elements of the array, separeted by ",".
    """
    return list_to_str(input_list)  # list comprehension


# erro se tentar converter int para str e funcao n verifica isso
def fill_list_with_new_values(original_list, new_list_values):
    """ Copies elements from one list to another. The elements will be positioned in the same position in the new list as
    they were in their original list.

    Parameters
    ----------
    original_list : array
    The list to which the elements will be copied

    new_list_values : array
    The list from which elements will be copied

    """
    for i in range(len(new_list_values)):
        type1 = type(original_list[i])
        if type1 == int:
            original_list[i] = int(new_list_values[i])
        elif type1 == float:
            original_list[i] = float(new_list_values[i])
        else:
            original_list[i] = new_list_values[i]


def list_to_svm_line(original_list):
    list_size = len(original_list)
    svm_line = '%s ' % original_list[0]
    for i in range(1, list_size):
        # svm_line += '{}:{} '.format(i, repr(original_list[i]))
        svm_line += '{}:{} '.format(i, original_list[i])
    return svm_line.rstrip()

def shift(arr, num, fill_value=np.nan):
    """Shifts the elements of the given array by the number of periods specified.

    Parameters
    ----------
    arr : array
        The array to be shifed.

    num : Integer
        Number of periods to shift. Can be positive or negative. If posite, the elements will be pulled down, and pulled
        up otherwise.

    fill_value : Integer, optional(np.nan by default)
        The scalar value used for newly introduced missing values.

    Returns
    -------
    result : array
        A new array with the same shape and type as the initial given array, but with the indexes shifted.

    Notes
    -----
        Similar to pandas shift, but faster.

    See also
    --------
        https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array
    """

    result = np.empty_like(arr)

    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result = arr
    return result
