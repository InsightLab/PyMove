"""
Math operations.

is_number,
std,
avg_std,
std_sample,
avg_std_sample,
arrays_avg,
array_stats,
interpolation

"""
from __future__ import annotations

import math


def is_number(value: int | float | str):
    """
    Returns if value is numerical or not.

    Parameters
    ----------
    value : int, float, str

    Returns
    -------
    boolean
        True if numerical, otherwise False

    Examples
    --------
    >>> from pymove.utils.math import is_number
    >>> a, b, c, d = 50, 22.5, '11.25', 'house'
    >>> print(is_number(a), type(is_number(a)))
    True <class 'bool'>
    >>> print(is_number(b), type(is_number(b)))
    True <class 'bool'>
    >>> print(is_number(c), type(is_number(c)))
    True <class 'bool'>
    >>> print(is_number(d), type(is_number(d)))
    False <class 'bool'>
    """
    try:
        float(value)
    except ValueError:
        return False
    return True


def std(values_array: list[float]) -> float:
    """
    Compute standard deviation.

    Parameters
    ----------
    values_array : array like of numerical values.
        Represents the set of values to compute the operation.

    Returns
    -------
    float
        Represents the value of standard deviation.

    References
    ----------
    squaring with * is over 3 times as fast as with **2
    http://stackoverflow.com/questions/29046346/comparison-of-power-to-multiplication-in-python

    Example
    -------
    >>> from pymove.utils.math import std
    >>> list = [7.8, 9.7, 6.4, 5.6, 10]
    >>> print(std(list), type(std(list)))
    1.7435595774162693 <class 'float'>
    """
    size = len(values_array)
    mean = sum(values_array) / size
    sum_sq = sum((i - mean) * (i - mean) for i in values_array)

    return math.sqrt(sum_sq / size)


def avg_std(values_array: list[float]) -> tuple[float, float]:
    """
    Compute the average of standard deviation.

    Parameters
    ----------
    values_array : array like of numerical values.
        Represents the set of values to compute the operation.

    Returns
    -------
    float
        Represents the value of average.
    float
        Represents the value of standard deviation.

    Example
    -------
    >>> from pymove.utils.math import avg_std
    >>> list = [7.8, 9.7, 6.4, 5.6, 10]
    >>> print(avg_std(list), type(avg_std(list)))
    1.9493588689617927 <class 'float'>
    """
    avg = sum(values_array) / len(values_array)
    return avg, std(values_array)


def std_sample(values_array: list[float]) -> float:
    """
    Compute the standard deviation of sample.

    Parameters
    ----------
    values_array : array like of numerical values.
        Represents the set of values to compute the operation.

    Returns
    -------
    float
        Represents the value of standard deviation of sample.

    Example
    -------
    >>> from pymove.utils.math import std_sample
    >>> list = [7.8, 9.7, 6.4, 5.6, 10]
    >>> print(std_sample(list), type(std_sample(list)))
    1.9493588689617927 <class 'float'>
    """
    size = len(values_array)
    return std(values_array) * math.sqrt(size / (size - 1))


def avg_std_sample(values_array: list[float]) -> tuple[float, float]:
    """
    Compute the average of standard deviation of sample.

    Parameters
    ----------
    values_array : array like of numerical values.
        Represents the set of values to compute the operation.

    Returns
    -------
    float
        Represents the value of average
    float
        Represents the standard deviation of sample.

    Example
    -------
    >>> from pymove.utils.math import avg_std_sample
    >>> list = [7.8, 9.7, 6.4, 5.6, 10]
    >>> print(avg_std_sample(list), type(avg_std_sample(list)))
    (7.9, 1.9493588689617927) <class 'tuple'>
    """
    avg = sum(values_array) / len(values_array)
    return avg, std_sample(values_array)


def arrays_avg(
    values_array: list[float], weights_array: list[float] | None = None
) -> float:
    """
    Computes the mean of the elements of the array.

    Parameters
    ----------
    values_array : array like of numerical values.
        Represents the set of values to compute the operation.
    weights_array : array, optional, default None.
        Used to calculate the weighted average, indicates the weight of
        each element in the array (values_array).

    Returns
    -------
    float
        The mean of the array elements.

    Examples
    --------
    >>> from pymove.utils.math import arrays_avg
    >>> list = [7.8, 9.7, 6.4, 5.6, 10]
    >>> weights = [0.1, 0.3, 0.15, 0.15, 0.3]
    >>> print('standard average', arrays_avg(list), type(arrays_avg(list)))
    'standard average 7.9 <class 'float'>'
    >>> print(
    >>>    'weighted average: ',
    >>>     arrays_avg(list, weights),
    >>>     type(arrays_avg(list, weights))
    >>> )
    'weighted average:  1.6979999999999997 <class 'float'>'
    """
    n = len(values_array)

    if weights_array is None:
        weights_array = [1] * n
    elif len(weights_array) != n:
        raise ValueError(
            'values_array and qt_array must have the same number of rows'
        )

    result = 0.

    for i, j in zip(values_array, weights_array):
        result += i * j

    return result / n


def array_stats(values_array: list[float]) -> tuple[float, float, int]:
    """
    Computes statistics about the array.

    The sum of all the elements in the array, the sum of the square of
    each element and the number of elements of the array.

    Parameters
    ----------
    values_array : array like of numerical values.
        Represents the set of values to compute the operation.

    Returns
    -------
    float.
        The sum of all the elements in the array.
    float
        The sum of the square value of each element in the array.
    int.
        The number of elements in the array.
    Example
    -------
    >>> from pymove.utils.math import array_stats
    >>> list = [7.8, 9.7, 6.4, 5.6, 10]
    >>> print(array_stats(list), type(array_stats(list)))
    (39.5, 327.25, 5) <class 'tuple'>
    """
    sum_ = 0.
    sum_sq = 0.
    n = 0
    for item in values_array:
        sum_ += item
        sum_sq += item * item
        n += 1
    return sum_, sum_sq, n


def interpolation(x0: float, y0: float, x1: float, y1: float, x: float) -> float:
    """
    Performs interpolation.

    Parameters
    ----------
    x0 : float.
        The coordinate of the first point on the x axis.
    y0 : float.
        The coordinate of the first point on the y axis.
    x1 : float.
        The coordinate of the second point on the x axis.
    y1 : float.
        The coordinate of the second point on the y axis.
    x : float.
        A value in the interval (x0, x1).

    Returns
    -------
    float.
        Is the interpolated  or extrapolated value.

    Example
    -------
    >>> from pymove.utils.math import interpolation
    >>> x0, y0, x1, y1, x = 2, 4, 3, 6, 3.5
    >>> print(interpolation(x0,y0,x1,y1,x), type(interpolation(x0,y0,x1,y1,x)))
    7.0 <class 'float'>
    """
    return y0 + (y1 - y0) * ((x - x0) / (x1 - x0))
