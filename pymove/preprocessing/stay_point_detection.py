"""Stop point detection operations.

create_or_update_move_stop_by_dist_time,
create_or_update_move_and_stop_by_radius

"""

from typing import TYPE_CHECKING, Optional, Text, Union

import numpy as np

from pymove.preprocessing.segmentation import by_max_dist
from pymove.utils.constants import (
    DIST_TO_PREV,
    MOVE,
    SEGMENT_STOP,
    SITUATION,
    STOP,
    TIME_TO_PREV,
    TRAJ_ID,
)
from pymove.utils.log import logger, timer_decorator

if TYPE_CHECKING:
    from pymove.core.dask import DaskMoveDataFrame
    from pymove.core.pandas import PandasMoveDataFrame


@timer_decorator
def create_or_update_move_stop_by_dist_time(
    move_data: Union['PandasMoveDataFrame', 'DaskMoveDataFrame'],
    dist_radius: Optional[float] = 30,
    time_radius: Optional[float] = 900,
    label_id: Optional[Text] = TRAJ_ID,
    new_label: Optional[Text] = SEGMENT_STOP,
    inplace: Optional[bool] = False
) -> Optional[Union['PandasMoveDataFrame', 'DaskMoveDataFrame']]:
    """
    Determines the stops and moves points of the dataframe.

    If these points already exist, they will be updated.

    Parameters
    ----------
    move_data : dataframe
       The input trajectory data
    dist_radius : float, optional
        The first step in this function is segmenting the trajectory
        The segments are used to find the stop points
        The dist_radius defines the distance used in the segmentation,
        by default 30
    time_radius :  float, optional
        The time_radius used to determine if a segment is a stop
        If the user stayed in the segment for a time
        greater than time_radius, than the segment is a stop,
        by default 900
    label_id : str, optional
         Indicates the label of the id column in the user dataframe, by default TRAJ_ID
    new_label : float, optional
        Is the name of the column to indicates if a point is a stop of a move,
        by default SEGMENT_STOP
    inplace : bool, optional
        if set to true the original dataframe will be altered to
        contain the result of the filtering, otherwise a copy will be returned,
        by default False

    Returns
    -------
    DataFrame
        DataFrame with 2 aditional features: segment_stop and stop.
        segment_stop indicates the trajectory segment to which the point belongs
        stop indicates if the point represents a stop.

    """
    if not inplace:
        move_data = move_data.copy()

    by_max_dist(
        move_data,
        label_id=label_id,
        max_dist_between_adj_points=dist_radius,
        label_new_tid=new_label,
        inplace=True
    )

    move_data.generate_dist_time_speed_features(
        label_id=new_label
    )

    logger.debug('Create or update stop as True or False')
    logger.debug(
        '...Creating stop features as True or False using %s to time in seconds'
        % time_radius
    )
    move_data[STOP] = False
    move_dataagg_tid = (
        move_data.groupby(by=new_label)
        .agg({TIME_TO_PREV: 'sum'})
        .query('%s > %s' % (TIME_TO_PREV, time_radius))
        .index
    )
    idx = move_data[
        move_data[new_label].isin(move_dataagg_tid)
    ].index
    move_data.at[idx, STOP] = True
    logger.debug(move_data[STOP].value_counts())

    if not inplace:
        return move_data


@timer_decorator
def create_or_update_move_and_stop_by_radius(
    move_data: Union['PandasMoveDataFrame', 'DaskMoveDataFrame'],
    radius: Optional[float] = 0,
    target_label: Optional[Text] = DIST_TO_PREV,
    new_label: Optional[Text] = SITUATION,
    inplace: Optional[bool] = False,
) -> Optional[Union['PandasMoveDataFrame', 'DaskMoveDataFrame']]:
    """
    Finds the stops and moves points of the dataframe.

    If these points already exist, they will be updated.

    Parameters
    ----------
    move_data : dataframe
       The input trajectory data
    radius :  float, optional
        The radius value is used to determine if a segment is a stop.
        If the value of the point in target_label is
        greater than radius, the segment is a stop, otherwise it's a move,
        by default 0
    target_label : String, optional
        The feature used to calculate the stay points, by default DIST_TO_PREV
    new_label : String, optional
        Is the name of the column to indicates if a point is a stop of a move,
        by default SITUATION
    inplace : bool, optional
        if set to true the original dataframe will be altered to
        contain the result of the filtering, otherwise a copy will be returned,
        by default False

    Returns
    -------
    DataFrame
        dataframe with 2 aditional features: segment_stop and new_label.
        segment_stop indicates the trajectory segment to which the point belongs
        new_label indicates if the point represents a stop or moving point.

    """
    logger.debug('\nCreating or updating features MOVE and STOPS...\n')

    if not inplace:
        move_data = move_data.copy()

    if DIST_TO_PREV not in move_data:
        move_data.generate_dist_features()

    conditions = (
        (move_data[target_label] > radius),
        (move_data[target_label] <= radius),
    )
    choices = [MOVE, STOP]

    move_data[new_label] = np.select(conditions, choices, np.nan)
    logger.debug(
        '\n....There are %s stops to this parameters\n'
        % (move_data[move_data[new_label] == STOP].shape[0])
    )

    if not inplace:
        return move_data
