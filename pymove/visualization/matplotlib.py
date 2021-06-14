"""
Matplolib operations.

show_object_id_by_date,
plot_trajectories,
plot_traj_by_id,
plot_all_features
plot_coords,
plot_bounds,
plot_line,

"""

from typing import TYPE_CHECKING, Any, Callable, List, Optional, Text, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.pyplot import axes, figure
from pandas.core.frame import DataFrame
from shapely.geometry import LineString, MultiLineString
from shapely.geometry.base import BaseGeometry

from pymove.utils.constants import (
    DATE,
    DAY,
    HOUR,
    LATITUDE,
    LONGITUDE,
    PERIOD,
    TID,
    TRAJ_ID,
)

if TYPE_CHECKING:
    from pymove.core.dask import DaskMoveDataFrame
    from pymove.core.pandas import PandasMoveDataFrame


def show_object_id_by_date(
    move_data: Union['PandasMoveDataFrame', 'DaskMoveDataFrame'],
    create_features: bool = True,
    kind: Optional[List] = None,
    figsize: Tuple[float, float] = (21, 9),
    return_fig: bool = True,
    save_fig: bool = True,
    name: Text = 'shot_points_by_date.png',
) -> Optional[figure]:
    """
    Generates four visualizations based on datetime feature.

        - Bar chart trajectories by day periods
        - Bar chart trajectories day of the week
        - Line chart trajectory by date
        - Line chart of trajectory by hours of the day.

    Parameters
    ----------
    move_data : pymove.core.MoveDataFrameAbstract subclass.
        Input trajectory data.
    create_features : bool, optional
        Represents whether or not to delete features created for viewing,
        by default True.
    kind: list, optional
        Determines the kinds of each plot, by default None
    figsize : tuple, optional
        Represents dimensions of figure, by default (21,9).
    return_fig : bool, optional
        Represents whether or not to save the generated picture, by default True.
    save_fig : bool, optional
        Represents whether or not to save the generated picture, by default True.
    name : String, optional
        Represents name of a file, by default 'shot_points_by_date.png'.

    Returns
    -------
    figure
        The generated picture or None

    References
    ----------
    https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.plot.html

    """
    if kind is None:
        kind = ['bar', 'bar', 'line', 'line']

    fig, ax = plt.subplots(2, 2, figsize=figsize)

    move_data.generate_date_features()
    move_data.generate_hour_features()
    move_data.generate_time_of_day_features()
    move_data.generate_day_of_the_week_features()

    move_data.groupby([PERIOD])[TRAJ_ID].nunique().plot(
        subplots=True, kind=kind[0], rot=0, ax=ax[0][0], fontsize=12
    )
    move_data.groupby([DAY])[TRAJ_ID].nunique().plot(
        subplots=True, kind=kind[1], ax=ax[0][1], rot=0, fontsize=12
    )
    move_data.groupby([DATE])[TRAJ_ID].nunique().plot(
        subplots=True,
        kind=kind[2],
        grid=True,
        ax=ax[1][0],
        rot=90,
        fontsize=12,
    )
    move_data.groupby([HOUR])[TRAJ_ID].nunique().plot(
        subplots=True, kind=kind[3], grid=True, ax=ax[1][1], fontsize=12
    )

    if not create_features:
        move_data.drop(columns=[DATE, HOUR, PERIOD, DAY], inplace=True)

    if save_fig:
        plt.savefig(fname=name)

    if return_fig:
        return fig


def plot_trajectories(
    move_data: DataFrame,
    markers: Text = 'o',
    markersize: float = 12,
    figsize: Tuple[float, float] = (10, 10),
    return_fig: bool = True,
    save_fig: bool = True,
    name: Text = 'trajectories.png',
) -> Optional[figure]:
    """
    Generate a visualization that show trajectories.

    Parameters
    ----------
    move_data: dataframe
        Dataframe with trajectories
    markers : str, optional
        Represents visualization type marker, by default 'o'
    markersize : float, optional
        Represents visualization size marker, by default 12
    figsize : tuple(float, float), optional
            Represents dimensions of figure, by default (10, 10)
    return_fig : bool, optional
        Represents whether or not to return the generated picture, by default True
    save_fig : bool, optional
        Represents whether or not to save the generated picture, by default False
    name : str, optional
        Represents name of a file, by default 'trajectories.png'

    Returns
    -------
    figure
        The generated picture or None
    """
    fig = plt.figure(figsize=figsize)

    ids = move_data['id'].unique()
    for id_ in ids:
        self_id = move_data[move_data['id'] == id_]
        plt.plot(
            self_id[LONGITUDE],
            self_id[LATITUDE],
            markers,
            markersize=markersize,
        )

    if save_fig:
        plt.savefig(fname=name)

    if return_fig:
        return fig


def plot_traj_by_id(
    move_data: DataFrame,
    id_: Union[int, Text],
    label: Text = TID,
    feature: Optional[Text] = None,
    value: Optional[Any] = None,
    linewidth: float = 3,
    markersize: float = 20,
    figsize: Tuple[float, float] = (10, 10),
    return_fig: bool = True,
    save_fig: bool = True,
    name: Optional[Text] = None,
) -> Optional[figure]:
    """
    Generate a visualization that shows a trajectory with the specified tid.

    Parameters
    ----------
    move_data: dataframe
        Dataframe with trajectories
    id_ :  int, str
        Represents the trajectory tid
    label : str, optional
        Feature with trajectories tids, by default TID
    feature : str, optional
        Name of the feature to highlight on plot, by default None
    value : any, optional
        Value of the feature to be highlighted as green marker, by default None
    linewidth : float, optional
        Represents visualization size line, by default 2
    markersize : float, optional
        Represents visualization size marker, by default 20
    figsize : tuple(float, float), optional
        Represents dimensions of figure, by default (10, 10)
    return_fig : bool, optional
        Represents whether or not to return the generated picture, by default True
    save_fig : bool, optional
        Represents whether or not to save the generated picture, by default False
    name : str, optional
        Represents name of a file, by default None

    Returns
    -------
    PandasMoveDataFrame', figure
        Trajectory with the specified tid.
        The generated picture.

    Raises
    ------
    KeyError
        If the dataframe does not contains the TID feature
    IndexError
        If there is no trajectory with the tid passed

    """
    if label not in move_data:
        raise KeyError('%s feature not in dataframe' % label)

    df_ = move_data[move_data[label] == id_]

    if not len(df_):
        raise IndexError(f'No trajectory with tid {id_} in dataframe')

    fig = plt.figure(figsize=figsize)

    if (not feature) or (not value) or (feature not in df_):
        plt.plot(df_[LONGITUDE], df_[LATITUDE])
        plt.plot(
            df_.loc[:, LONGITUDE], df_.loc[:, LATITUDE],
            'r.', markersize=markersize / 2
        )
    else:
        filter_ = df_[feature] == value
        df_nodes = df_.loc[filter_]
        df_points = df_.loc[~filter_]
        plt.plot(df_[LONGITUDE], df_[LATITUDE], linewidth=linewidth)
        plt.plot(
            df_nodes[LONGITUDE], df_nodes[LATITUDE], 'gs', markersize=markersize / 2
        )
        plt.plot(
            df_points[LONGITUDE], df_points[LATITUDE], 'r.', markersize=markersize / 2
        )

    plt.plot(
        df_.iloc[0][LONGITUDE], df_.iloc[0][LATITUDE], 'yo', markersize=markersize
    )  # start point
    plt.plot(
        df_.iloc[-1][LONGITUDE], df_.iloc[-1][LATITUDE], 'yX', markersize=markersize
    )  # end point

    if save_fig:
        if not name:
            name = 'trajectory_%s.png' % id_
        plt.savefig(fname=name)

    if return_fig:
        return fig


def plot_all_features(
    move_data: DataFrame,
    dtype: Callable = float,
    figsize: Tuple[float, float] = (21, 15),
    return_fig: bool = True,
    save_fig: bool = True,
    name: Text = 'features.png',
) -> Optional[figure]:
    """
    Generate a visualization for each columns that type is equal dtype.

    Parameters
    ----------
    move_data: dataframe
        Dataframe with trajectories
    dtype : callable, optional
        Represents column type, by default np.float64
    figsize : tuple(float, float), optional
        Represents dimensions of figure, by default (21, 15)
    return_fig : bool, optional
        Represents whether or not to return the generated picture, by default True
    save_fig : bool, optional
        Represents whether or not to save the generated picture, by default False
    name : str, optional
        Represents name of a file, by default 'features.png'

    Returns
    -------
    figure
        The generated picture or None

    Raises
    ------
    AttributeError
        If there are no columns with the specified type

    """
    col_dtype = move_data.select_dtypes(include=[dtype]).columns
    tam = col_dtype.size
    if not tam:
        raise AttributeError('No columns with dtype %s.' % dtype)

    fig, ax = plt.subplots(tam, 1, figsize=figsize)
    ax_count = 0
    for col in col_dtype:
        ax[ax_count].set_title(col)
        move_data[col].plot(subplots=True, ax=ax[ax_count])
        ax_count += 1

    if save_fig:
        plt.savefig(fname=name)

    if return_fig:
        return fig


def plot_coords(ax: axes, ob: BaseGeometry, color: Text = 'r'):
    """
    Plot the coordinates of each point of the object in a 2D chart.

    Parameters
    ----------
    ax : axes
        Single axes object
    ob : geometry object
        Any geometric object
    color : str, optional
        Sets the geometric object color, by default 'r'

    Example
    -------
    """
    x, y = ob.xy
    ax.plot(x, y, 'o', color=color, zorder=1)


def plot_bounds(ax: axes, ob: Union[LineString, MultiLineString], color='b'):
    """
    Plot the limites of geometric object.

    Parameters
    ----------
    ax : axes
        Single axes object
    ob : LineString or MultiLineString
        Geometric object formed by lines.
    color : str, optional
        Sets the geometric object color, by default 'b'

    Example
    -------

    """
    x, y = zip(*list((p.x, p.y) for p in ob.boundary))
    ax.plot(x, y, '-', color=color, zorder=1)


def plot_line(
    ax: axes,
    ob: LineString,
    color: Text = 'r',
    alpha: float = 0.7,
    linewidth: float = 3,
    solid_capstyle: Text = 'round',
    zorder: float = 2
):
    """
    Plot a LineString.

    Parameters
    ----------
    ax : axes
        Single axes object
    ob : LineString
        Sequence of points.
    color : str, optional
        Sets the line color, by default 'r'
    alpha : float, optional
        Defines the opacity of the line, by default 0.7
    linewidth : float, optional
        Defines the line thickness, by default 3
    solid_capstyle : str, optional
        Defines the style of the ends of the line, by default 'round'
    zorder : float, optional
        Determines the default drawing order for the axes, by default 2

    Example
    -------
    """
    x, y = ob.xy
    ax.plot(
        x, y, color=color, alpha=alpha, linewidth=linewidth,
        solid_capstyle=solid_capstyle, zorder=zorder
    )
