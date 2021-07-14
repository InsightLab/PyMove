"""
Matplolib operations.

show_object_id_by_date,
plot_trajectories,
plot_trajectory_by_id,
plot_grid_polygons,
plot_all_features
plot_coords,
plot_bounds,
plot_line

"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import matplotlib.pyplot as plt
from matplotlib.pyplot import axes, figure
from pandas.core.frame import DataFrame
from shapely.geometry import LineString, MultiLineString
from shapely.geometry.base import BaseGeometry

from pymove.core.grid import Grid
from pymove.utils.constants import (
    DATE,
    DAY,
    HOUR,
    LATITUDE,
    LONGITUDE,
    PERIOD,
    POLYGON,
    TRAJ_ID,
)

if TYPE_CHECKING:
    from pymove.core.dask import DaskMoveDataFrame
    from pymove.core.pandas import PandasMoveDataFrame


def show_object_id_by_date(
    move_data: 'PandasMoveDataFrame' | 'DaskMoveDataFrame',
    kind: list | None = None,
    figsize: tuple[float, float] = (21, 9),
    return_fig: bool = False,
    save_fig: bool = False,
    name: str = 'shot_points_by_date.png',
) -> figure | None:
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
    kind: list, optional
        Determines the kinds of each plot, by default None
    figsize : tuple, optional
        Represents dimensions of figure, by default (21,9).
    return_fig : bool, optional
        Represents whether or not to save the generated picture, by default False.
    save_fig : bool, optional
        Represents whether or not to save the generated picture, by default False.
    name : String, optional
        Represents name of a file, by default 'shot_points_by_date.png'.

    Returns
    -------
    figure
        The generated picture or None

    References
    ----------
    https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.plot.html

    Examples
    --------
    >>> from pymove.visualization.matplotlib import show_object_id_by_date
    >>> move_df.head()
              lat          lon              datetime   id
    0   39.984094   116.319236   2008-10-23 05:53:05    1
    1   39.984198   116.319322   2008-10-23 05:53:06    1
    2   39.984224   116.319402   2008-10-23 05:53:11    1
    3   39.984211   116.319389   2008-10-23 05:53:16    2
    4   39.984217   116.319422   2008-10-23 05:53:21    2
    >>> show_object_id_by_date(move_df)
    """
    if kind is None:
        kind = ['bar', 'bar', 'line', 'line']

    fig, ax = plt.subplots(2, 2, figsize=figsize)

    columns = move_data.columns
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

    if save_fig:
        plt.savefig(fname=name)

    to_drop = list(set(move_data.columns) - set(columns))
    move_data.drop(columns=to_drop, inplace=True)

    if return_fig:
        return fig


def plot_trajectories(
    move_data: DataFrame,
    markers: str = 'o',
    markersize: float = 12,
    figsize: tuple[float, float] = (10, 10),
    return_fig: bool = False,
    save_fig: bool = False,
    name: str = 'trajectories.png',
) -> figure | None:
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
        Represents whether or not to return the generated picture, by default False
    save_fig : bool, optional
        Represents whether or not to save the generated picture, by default False
    name : str, optional
        Represents name of a file, by default 'trajectories.png'

    Returns
    -------
    figure
        The generated picture or None

    Examples
    --------
    >>>  from pymove.visualization.matplotlib import plot_trajectories
    >>> move_df.head()
              lat          lon              datetime   id
    0   39.984094   116.319236   2008-10-23 05:53:05    1
    1   39.984198   116.319322   2008-10-23 05:53:06    1
    2   39.984224   116.319402   2008-10-23 05:53:11    1
    3   39.984211   116.319389   2008-10-23 05:53:16    2
    4   39.984217   116.319422   2008-10-23 05:53:21    2
    >>> plot_trajectories(move_df)
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


def plot_trajectory_by_id(
    move_data: DataFrame,
    id_: int | str,
    label: str = TRAJ_ID,
    feature: str | None = None,
    value: Any | None = None,
    linewidth: float = 3,
    markersize: float = 20,
    figsize: tuple[float, float] = (10, 10),
    return_fig: bool = False,
    save_fig: bool = False,
    name: str | None = None,
) -> figure | None:
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
        Represents whether or not to return the generated picture, by default False
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

    Examples
    --------
    >>> from pymove.visualization.matplotlib import  plot_traj_by_id
    >>> move_df
              lat          lon              datetime   id
    0   39.984094   116.319236   2008-10-23 05:53:05    1
    1   39.984198   116.319322   2008-10-23 05:53:06    1
    2   39.984224   116.319402   2008-10-23 05:53:11    1
    3   39.984211   116.319389   2008-10-23 05:53:16    2
    4   39.984217   116.319422   2008-10-23 05:53:21    2
    >>> plot_traj_by_id(move_df_3, 1, label='id)
    >>> plot_traj_by_id(move_df_3, 2, label='id)
    """
    if label not in move_data:
        raise KeyError('%s feature not in dataframe' % label)

    df_ = move_data[move_data[label] == id_]

    if not len(df_):
        raise IndexError(f'No trajectory with {label} {id_} in dataframe')

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
    )
    plt.plot(
        df_.iloc[-1][LONGITUDE], df_.iloc[-1][LATITUDE], 'yX', markersize=markersize
    )

    if save_fig:
        if not name:
            name = 'trajectory_%s.png' % id_
        plt.savefig(fname=name)

    if return_fig:
        return fig


def plot_grid_polygons(
        data: DataFrame,
        grid: Grid | None = None,
        markersize: float = 10,
        linewidth: float = 2,
        figsize: tuple[int, int] = (10, 10),
        return_fig: bool = False,
        save_fig: bool = False,
        name: str = 'grid.png',
) -> figure | None:
    """
    Generate a visualization with grid polygons.

    Parameters
    ----------
    data : DataFrame
        Input trajectory data
    markersize : float, optional
        Represents visualization size marker, by default 10
    linewidth : float, optional
        Represents visualization size line, by default 2
    figsize : tuple(int, int), optional
        Represents the size (float: width, float: height) of a figure,
            by default (10, 10)
    return_fig : bool, optional
        Represents whether or not to save the generated picture, by default False
    save_fig : bool, optional
        Wether to save the figure, by default False
    name : str, optional
        Represents name of a file, by default 'grid.png'

    Returns
    -------
    figure
        The generated picture or None

    Raises
    ------
        If the dataframe does not contains the POLYGON feature
    IndexError
        If there is no user with the id passed

    """
    if POLYGON not in data:
        if grid is None:
            raise KeyError('POLYGON feature not in dataframe')
        data = grid.create_all_polygons_to_all_point_on_grid(data)

    data = data.copy()

    data.dropna(subset=[POLYGON], inplace=True)

    fig = plt.figure(figsize=figsize)

    for _, row in data.iterrows():
        xs, ys = row[POLYGON].exterior.xy
        plt.plot(ys, xs, 'g', linewidth=linewidth, markersize=markersize)
    xs_start, ys_start = data.iloc[0][POLYGON].exterior.xy
    xs_end, ys_end = data.iloc[-1][POLYGON].exterior.xy
    plt.plot(ys_start, xs_start, 'bo', markersize=markersize * 1.5)
    plt.plot(ys_end, xs_end, 'bX', markersize=markersize * 1.5)

    if save_fig:
        plt.savefig(fname=name)

    if return_fig:
        return fig


def plot_all_features(
    move_data: DataFrame,
    dtype: Callable = float,
    figsize: tuple[float, float] = (21, 15),
    return_fig: bool = False,
    save_fig: bool = False,
    name: str = 'features.png',
) -> figure | None:
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
        Represents whether or not to return the generated picture, by default False
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

    Examples
    --------
    >>>  from pymove.visualization.matplotlib import plot_all_features
    >>> move_df.head()
              lat          lon              datetime   id
    0   39.984094   116.319236   2008-10-23 05:53:05    1
    1   39.984198   116.319322   2008-10-23 05:53:06    1
    2   39.984224   116.319402   2008-10-23 05:53:11    1
    3   39.984211   116.319389   2008-10-23 05:53:16    2
    4   39.984217   116.319422   2008-10-23 05:53:21    2
    >>>  plot_all_features(move_df)
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


def plot_coords(ax: axes, ob: BaseGeometry, color: str = 'r'):
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
    >>> from pymove.visualization.matplotlib import plot_coords
    >>> import matplotlib.pyplot as plt
    >>> coords = LineString([(1, 1), (1, 2), (2, 2), (2, 3)])
    >>> _, ax = plt.subplots(figsize=(21, 9))
    >>> plot_coords(ax, coords)
    """
    x, y = ob.xy
    ax.plot(x, y, 'o', color=color, zorder=1)


def plot_bounds(ax: axes, ob: LineString | MultiLineString, color='b'):
    """
    Plot the limits of geometric object.

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
    >>> from pymove.visualization.matplotlib import plot_bounds
    >>> import matplotlib.pyplot as plt
    >>> bounds = LineString([(1, 1), (1, 2), (2, 2), (2, 3)])
    >>> _, ax = plt.subplots(figsize=(21, 9))
    >>> plot_bounds(ax, bounds)
    """
    x, y = zip(*list((p.x, p.y) for p in ob.boundary))
    ax.plot(x, y, '-', color=color, zorder=1)


def plot_line(
    ax: axes,
    ob: LineString,
    color: str = 'r',
    alpha: float = 0.7,
    linewidth: float = 3,
    solid_capstyle: str = 'round',
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
    >>> from pymove.visualization.matplotlib import plot_line
    >>> import matplotlib.pyplot as plt
    >>> line = LineString([(1, 1), (1, 2), (2, 2), (2, 3)])
    >>> _, ax = plt.subplots(figsize=(21, 9))
    >>> plot_line(ax, line)
    """
    x, y = ob.xy
    ax.plot(
        x, y, color=color, alpha=alpha, linewidth=linewidth,
        solid_capstyle=solid_capstyle, zorder=zorder
    )
