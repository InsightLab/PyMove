import math


def is_number(value):
    """
    Returns if value is numerical or not

    Parameters
    ----------
    value : Integer, Float, String

    Returns
    -------
    boolean
        True if numerical, otherwise False
    """

    try:
        float(value)
    except ValueError:
        return False
    return True


def std(values_array):
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

    """

    size = len(values_array)
    mean = sum(values_array) / size
    sum_sq = sum([(i - mean) * (i - mean) for i in values_array])

    try:
        result = math.sqrt(sum_sq / size)
    except ValueError as e:
        raise e
    return result


def avg_std(values_array):
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

    """

    avg = sum(values_array) / len(values_array)
    return avg, std(values_array)


def std_sample(values_array):
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

    """

    size = len(values_array)
    return std(values_array) * math.sqrt(size / (size - 1))


def avg_std_sample(values_array):
    """
    Compute the average of standard deviation of sample.

    Parameters
    ----------
    values_array : array like of numerical values.
        Represents the set of values to compute the operation.

    Returns
    -------
    float
        Represents the value of average of standard deviation of sample.

    """

    avg = sum(values_array) / len(values_array)
    return avg, std_sample(values_array)


def arrays_avg(values_array, weights_array=None):
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

    """

    n = len(values_array)

    if weights_array is None:
        weights_array = [1] * n
    elif len(weights_array) != n:
        raise ValueError(
            'values_array and qt_array must have the same number of rows'
        )

    result = 0

    for i, j in zip(values_array, weights_array):
        result += i * j

    return result / n


def array_stats(values_array):
    """
    Computes the sum of all the elements in the array, the sum of the square of
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

    """

    sum_ = 0
    sum_sq = 0
    n = 0
    for item in values_array:
        sum_ += item
        sum_sq += item * item
        n += 1
    return sum_, sum_sq, n


def interpolation(x0, y0, x1, y1, x):
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

    Examples
    --------
    - interpolation 1: (30, 3, 40, 5, 37) -> 4.4
    - interpolation 2: (30, 3, 40, 5, 35) -> 4.0

    """

    return y0 + (y1 - y0) * ((x - x0) / (x1 - x0))
