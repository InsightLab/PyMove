from typing import TYPE_CHECKING, List, Optional, Text, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from pandas.core.frame import DataFrame

from pymove.utils.constants import DATE, DAY, HOUR, LATITUDE, LONGITUDE, PERIOD, TRAJ_ID

if TYPE_CHECKING:
    from pymove.core.pandas import PandasMoveDataFrame
    from pymove.core.dask import DaskMoveDataFrame


def show_object_id_by_date(
    move_data: Union['PandasMoveDataFrame', 'DaskMoveDataFrame'],
    create_features: Optional[bool] = True,
    kind: Optional[List] = None,
    figsize: Optional[Tuple[float, float]] = (21, 9),
    return_fig: Optional[bool] = True,
    save_fig: Optional[bool] = True,
    name: Optional[Text] = 'shot_points_by_date.png',
) -> Optional[figure]:
    """
    Generates four visualizations based on datetime feature:

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
        plt.savefig(fname=name, fig=fig)

    if return_fig:
        return fig


def show_lat_lon_gps(
    move_data: DataFrame,
    kind: Optional[Text] = 'scatter',
    figsize: Optional[Tuple[float, float]] = (21, 9),
    plot_start_and_end: Optional[bool] = True,
    return_fig: Optional[bool] = True,
    save_fig: Optional[bool] = False,
    name: Optional[Text] = 'show_gps_points.png',
):
    """
    Generate a visualization with points [lat, lon] of dataset.

    Parameters
    ----------
    move_data : DataFrame
        Input trajectory data.
    kind : String, optional
        Represents chart type, by default 'scatter'.
    figsize : tuple, optional
        Represents dimensions of figure, by default (21,9).
    plot_start_and_end: bool, optional
        Whether to highlight the start and end of the trajectory, by default True
    return_fig : bool, optional
        Represents whether or not to save the generated picture, by default True.
    save_fig : bool, optional
        Represents whether or not to save the generated picture, by default True.
    name : String, optional
        Represents name of a file, by default 'show_gps_points.png'

    Returns
    -------
    fig : matplotlib.pyplot.figure or None
        The generated picture.

    """

    if LATITUDE in move_data and LONGITUDE in move_data:
        fig = move_data.drop_duplicates([LATITUDE, LONGITUDE]).plot(
            kind=kind, x=LONGITUDE, y=LATITUDE, figsize=figsize
        )

        if plot_start_and_end:
            plt.plot(
                move_data.iloc[0][LONGITUDE],
                move_data.iloc[0][LATITUDE],
                'yo',
                markersize=10,
            )  # start point
            plt.plot(
                move_data.iloc[-1][LONGITUDE],
                move_data.iloc[-1][LATITUDE],
                'yX',
                markersize=10,
            )  # end point
        if save_fig:
            plt.savefig(name)

        if return_fig:
            return fig
