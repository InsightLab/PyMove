"""
PyMove
======

Provides  processing and visualization of trajectories and other
spatial-temporal data

"""

from ._version import __version__
from .core import grid
from .core.dask import DaskMoveDataFrame
from .core.dataframe import MoveDataFrame
from .core.pandas import PandasMoveDataFrame
from .models.pattern_mining import clustering
from .preprocessing import compression, filters, segmentation, stay_point_detection
from .semantic import semantic
from .utils import (
    constants,
    conversions,
    datetime,
    db,
    distances,
    integration,
    math,
    mem,
    trajectories,
    transformations,
    visual,
)
from .utils.trajectories import read_csv
from .visualization import folium, matplotlib
