
def interpolation(x0, y0, x1, y1, x):
    """
    Used for interpolation and extrapolation.
    interpolation 1: (30, 3, 40, 5, 37) -> 4.4
    interpolation 2: (30, 3, 40, 5, 35) -> 4.0
    extrapolation 1: (30, 3, 40, 5, 25) -> 2.0
    extrapolation 2: (30, 3, 40, 5, 45) -> 6.0
    """
    return y0 + (y1 - y0) * ( (x - x0)/(x1 - x0) )

