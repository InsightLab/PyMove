import numpy as np

def shift(arr, num, fill_value=np.nan):
    """
    Similar to pandas shift, but faster.
    See: https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array
    """
    """ Return a new array with the same shape and type as a given array."""
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

