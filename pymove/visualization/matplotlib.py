import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pymove.utils import constants

from pymove.utils.constants import (
    COLORS,
    COUNT,
    DATE,
    DATETIME,
    DAY,
    HOUR,
    LATITUDE,
    LONGITUDE,
    PERIOD,
    SITUATION,
    STOP,
    TILES,
    TRAJ_ID,
)


def show_object_id_by_date(
    move_data,
    create_features=True,
    kind=None,
    figsize=(21, 9),
    return_fig=True,
    save_fig=True,
    name="shot_points_by_date.png",
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
        kind = ["bar", "bar", "line", "line"]

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
    kind="scatter",
    figsize=(21, 9),
    plot_start_and_end=True,
    return_fig=True,
    save_fig=False,
    name="show_gps_points.png",
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
                    "yo",
                    markersize=10,
                )  # start point
                plt.plot(
                    move_data.iloc[-1][LONGITUDE],
                    move_data.iloc[-1][LATITUDE],
                    "yX",
                    markersize=10,
                )  # end point
            if save_fig:
                plt.savefig(name)

            if return_fig:
                return fig
    except Exception as exception:
        raise exception


def plot_all_features(
        self,
        dtype=np.float32,
        figsize=(21, 15),
        return_fig=True,
        save_fig=False,
        name="features.png",
    ):
        """
        Generate a visualization for each columns that type_ is equal dtype.

        Parameters
        ----------
        figsize : tuple, optional, default (21, 15).
            Represents dimensions of figure.

        dtype : type_, optional, default np.float32.
            Represents column type_.

        return_fig : bool, optional, default True.
            Represents whether or not to save the generated picture.

        save_fig : bool, optional, default False.
            Represents whether or not to save the generated picture.

        name : String, optional, default 'features.png'.
            Represents name of a file.

        Returns
        -------
        fig : matplotlib.pyplot.figure or None
            The generated picture.
        """
        operation = begin_operation("plot_all_features")

        try:
            col_dtype = self._data.select_dtypes(include=[dtype]).columns
            tam = col_dtype.size
            if not tam:
                raise AttributeError(f"No columns with dtype {dtype}.")

            fig, ax = plt.subplots(tam, 1, figsize=figsize)
            ax_count = 0
            for col in col_dtype:
                ax[ax_count].set_title(col)
                self._data[col].plot(subplots=True, ax=ax[ax_count])
                ax_count += 1

            if save_fig:
                plt.savefig(fname=name, fig=fig)

            self.last_operation = end_operation(operation)

            if return_fig:
                return fig
        except Exception as e:
            self.last_operation = end_operation(operation)
            raise e

def plot_trajs(
        self,
        markers="o",
        markersize=20,
        figsize=(10, 10),
        return_fig=True,
        save_fig=False,
        name="trajectories.png",
    ):
        """
        Generate a visualization that show trajectories.

        Parameters
        ----------
        figsize : tuple, optional, default (10, 10).
            Represents dimensions of figure.

        markers : String, optional, default 'o'.
            Represents visualization type_ marker.

        markersize : int, optional, default 20.
            Represents visualization size marker.

        return_fig : bool, optional, default True.
            Represents whether or not to save the generated picture.

        save_fig : bool, optional, default False.
            Represents whether or not to save the generated picture.

        name : String, optional, default 'trajectories.png'.
            Represents name of a file.

        Returns
        -------
        fig : matplotlib.pyplot.figure or None
            The generated picture.
        """

        operation = begin_operation("plot_trajs")

        fig = plt.figure(figsize=figsize)

        ids = self._data["id"].unique()
        for id_ in ids:
            selfid = self._data[self._data["id"] == id_]
            plt.plot(
                selfid[LONGITUDE],
                selfid[LATITUDE],
                markers,
                markersize=markersize,
            )

        if save_fig:
            plt.savefig(fname=name, fig=fig)

        self.last_operation = end_operation(operation)

        if return_fig:
            return fig

def plot_traj_id(
        self,
        tid,
        highlight=None,
        figsize=(10, 10),
        return_fig=True,
        save_fig=False,
        name=None,
    ):
        """
        Generate a visualization that shows a trajectory with the specified tid.

        Parameters
        ----------
        tid : String.
            Represents the trajectory tid.

        highlight: String, optional, default None.
            Name of the feature to highlight on plot.
            If value of feature is 1, it will be highlighted as green marker

        figsize : tuple, optional, default (10,10).
            Represents dimensions of figure.

        return_fig : bool, optional, default True.
            Represents whether or not to save the generated picture.

        save_fig : bool, optional, default False.
            Represents whether or not to save the generated picture.

        name : String, optional, default None.
            Represents name of a file.


        Returns
        -------
        move_data : pymove.core.MoveDataFrameAbstract subclass.
            Trajectory with the specified tid.

        fig : matplotlib.pyplot.figure or None
            The generated picture.

        Raises
        ------
        KeyError if the dataframe does not contains the TID feature
        IndexError if there is no trajectory with the tid passed
        """

        operation = begin_operation("plot_traj_id")

        if TID not in self._data:
            self.last_operation = end_operation(operation)
            raise KeyError("TID feature not in dataframe")

        df_ = self._data[self._data[TID] == tid]

        if not len(df_):
            self.last_operation = end_operation(operation)
            raise IndexError(f"No trajectory with tid {tid} in dataframe")

        fig = plt.figure(figsize=figsize)

        plt.plot(
            df_.iloc[0][LONGITUDE], df_.iloc[0][LATITUDE], "yo", markersize=20
        )  # start point
        plt.plot(
            df_.iloc[-1][LONGITUDE],
            df_.iloc[-1][LATITUDE],
            "yX",
            markersize=20,
        )  # end point

        if (not highlight) or (highlight not in df_):
            plt.plot(df_[LONGITUDE], df_[LATITUDE])
            plt.plot(
                df_.loc[:, LONGITUDE], df_.loc[:, LATITUDE], "r.", markersize=8
            )  # points
        else:
            filter_ = df_[highlight] == 1
            selfnodes = df_.loc[filter_]
            selfpoints = df_.loc[~filter_]
            plt.plot(selfnodes[LONGITUDE], selfnodes[LATITUDE], linewidth=3)
            plt.plot(selfpoints[LONGITUDE], selfpoints[LATITUDE])
            plt.plot(
                selfnodes[LONGITUDE], selfnodes[LATITUDE], "go", markersize=10
            )  # nodes
            plt.plot(
                selfpoints[LONGITUDE], selfpoints[LATITUDE], "r.", markersize=8
            )  # points

        if save_fig:
            if not name:
                name = f"trajectory_{tid}.png"
            plt.savefig(fname=name, fig=fig)

        df_ = PandasMoveDataFrame(df_)

        self.last_operation = end_operation(operation)

        if return_fig:
            return df_, fig
        return df_

