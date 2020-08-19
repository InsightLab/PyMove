import matplotlib.pyplot as plt

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


def show_object_id_by_date(
    move_data,
    create_features=True,
    kind=None,
    figsize=(21, 9),
    return_fig=True,
    save_fig=True,
    name='shot_points_by_date.png',
):
    """
    Generates four visualizations based on datetime feature:

        - Bar chart trajectories by day periods
        - Bar chart trajectories day of the week
        - Line chart trajectory by date
        - Line chart of trajectory byhours of the day.

    Parameters
    ----------
    move_data : pymove.core.MoveDataFrameAbstract subclass.
        Input trajectory data.
    create_features : bool, optional, default True.
        Represents whether or not to delete features created for viewing.
    kind: list or None
        Determines the kinds of each plot
    figsize : tuple, optional, default (21,9).
        Represents dimensions of figure.
    return_fig : bool, optional, default True.
        Represents whether or not to save the generated picture.
    save_fig : bool, optional, default True.
        Represents whether or not to save the generated picture.
    name : String, optional, default 'shot_points_by_date.png'.
        Represents name of a file.

    Returns
    -------
    fig : matplotlib.pyplot.figure or None
        The generated picture.
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
    move_data,
    kind='scatter',
    figsize=(21, 9),
    plot_start_and_end=True,
    return_fig=True,
    save_fig=False,
    name='show_gps_points.png',
):
    """
    Generate a visualization with points [lat, lon] of dataset.

    Parameters
    ----------
    move_data : pymove.core.MoveDataFrameAbstract subclass.
        Input trajectory data.
    kind : String, optional, default 'scatter'.
        Represents chart type_.
    figsize : tuple, optional, default (21,9).
        Represents dimensions of figure.
    plot_start_and_end: boolean
        Whether to highlight the start and end of the trajectory
    return_fig : bool, optional, default True.
        Represents whether or not to save the generated picture.
    save_fig : bool, optional, default True.
        Represents whether or not to save the generated picture.
    name : String, optional, default 'show_gps_points.png'.
        Represents name of a file.

    Returns
    -------
    fig : matplotlib.pyplot.figure or None
        The generated picture.

    """
    try:
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
    except Exception as exception:
        raise exception