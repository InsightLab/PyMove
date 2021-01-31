from typing import Dict, List, Optional, Text, Union

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

from pymove.core.grid import Grid
from pymove.core.pandas import PandasMoveDataFrame
from pymove.preprocessing.filters import clean_trajectories_with_few_points
from pymove.preprocessing.segmentation import (
    _drop_single_point,
    _prepare_segmentation,
    _update_curr_tid_count,
)
from pymove.utils.constants import (
    DATETIME,
    LATITUDE,
    LOCAL_LABEL,
    LONGITUDE,
    PREV_LOCAL,
    THRESHOLD,
    TID,
    TID_STAT,
    TIME_TO_PREV,
    TRAJ_ID,
)
from pymove.utils.datetime import generate_time_statistics, threshold_time_statistics
from pymove.utils.log import progress_bar
from pymove.utils.mem import begin_operation, end_operation
from pymove.utils.trajectories import shift


class PandasDiscreteMoveDataFrame(PandasMoveDataFrame):
    def __init__(
        self,
        data: Union[DataFrame, List, Dict],
        latitude: Optional[Text] = LATITUDE,
        longitude: Optional[Text] = LONGITUDE,
        datetime: Optional[Text] = DATETIME,
        traj_id: Optional[Text] = TRAJ_ID,
        local_label: Optional[Text] = LOCAL_LABEL
    ):
        """
        Creates a dataframe using local_label as a discrete feature for localization

        Parameters
        ----------
        data : Union[DataFrame, List, Dict]
            Input trajectory data
        latitude : str, optional
            Represents column name latitude, by default LATITUDE
        longitude : str, optional
            Represents column name longitude, by default LONGITUDE
        datetime : str, optional
            Represents column name datetime, by default DATETIME
        traj_id : str, optional
            Represents column name trajectory id, by default TRAJ_ID
        local_label : str, optional
            Represents column name local label, by default LOCAL_LABEL

        Raises
        ------
        KeyError
            If missing one of lat, lon, datetime, local_label columns
        ValueError, ParserError
            If the data types can't be converted.
        """
        super(PandasDiscreteMoveDataFrame, self).__init__(
            data=data,
            latitude=latitude,
            longitude=longitude,
            datetime=datetime,
            traj_id=traj_id
        )

        if local_label not in self:
            raise ValueError(
                '{} column not in dataframe'.format(local_label)
            )

    def discretize_based_grid(self, region_size: Optional[int] = 1000):
        """
        Discrete space in cells of the same size,
        assigning a unique id to each cell.

        Parameters
        ----------
        region_size: int, optional
            Size of grid cell, by default 1000
        """

        operation = begin_operation('discretize based on grid')
        print('\nDiscretizing dataframe...')
        grid = Grid(self, cell_size=region_size)
        grid.create_update_index_grid_feature(self)
        self.reset_index(drop=True, inplace=True)
        self.last_operation = end_operation(operation)

    def generate_prev_local_features(
        self,
        label_id: Optional[Text] = TRAJ_ID,
        local_label: Optional[Text] = LOCAL_LABEL,
        sort: Optional[bool] = True,
        inplace: Optional[bool] = True
    ) -> Optional['PandasDiscreteMoveDataFrame']:
        """
        Create a feature prev_local with the label of previous local to current point.

        Parameters
        ----------
        label_id : str, optional
            Represents name of column of trajectory id, by default TRAJ_ID
        local_label : str, optional
            Indicates name of column of place labels on symbolic trajectory,
                by default LOCAL_LABEL
        sort : bool, optional
            Wether the dataframe will be sorted, by default True
        inplace : bool, optional
            Represents whether the operation will be performed on
            the data provided or in a copy, by default True

        Returns
        -------
        PandasDiscreteMoveDataFrame
             Object with new features or None

        """
        operation = begin_operation('generate_prev_equ_feature')
        columns = set(self.columns)
        ids, sum_size_id, size_id, idx = self._prepare_generate_data(
            self, sort, label_id
        )

        try:
            message = '\nCreating generate_prev_equ_feature'
            message += ' in previous equ\n'
            print(
                message
            )

            if (self[local_label].dtype == 'int'):
                self[local_label] = self[local_label].astype(np.float16)
            for idx in progress_bar(
                ids, desc='Generating previous {}'.format(local_label)
            ):
                current_local = self.at[idx, local_label]
                current_local = np.array(current_local)
                size_id = current_local.size

                if size_id <= 1:
                    self.at[idx, PREV_LOCAL] = np.nan

                else:
                    prev_local = shift(current_local, 1)

                    # previous to current point
                    self.at[idx, PREV_LOCAL] = prev_local

            return self._return_generated_data(
                self, columns, operation, inplace
            )

        except Exception as e:
            print(
                'label_tid:%s\nidx:%s\nsize_id:%s\nsum_size_id:%s'
                % (label_id, idx, size_id, sum_size_id)
            )
            self.last_operation = end_operation(operation)
            raise e

    def generate_tid_based_statistics(
        self,
        label_id: Optional[Text] = TRAJ_ID,
        local_label: Optional[Text] = LOCAL_LABEL,
        mean_coef: Optional[float] = 1.0,
        std_coef: Optional[float] = 1.0,
        statistics: Optional[DataFrame] = None,
        label_tid_stat: Optional[Text] = TID_STAT,
        drop_single_points: Optional[bool] = False,
        inplace: Optional[bool] = True,
    ) -> Optional['PandasDiscreteMoveDataFrame']:
        """
        Splits the trajectories into segments based on time statistics for segments.

        Parameters
        ----------
        label_id : str, optional
             Represents name of column of trajectory id, by default TRAJ_ID
        local_label : str, optional
            Indicates name of column of place labels on symbolic trajectory,
                by default LOCAL_LABEL
        mean_coef : float, optional
            Multiplication coefficient of the mean time for the segment, by default 1.0
        std_coef : float, optional
            Multiplication coefficient of sdt time for the segment, by default 1.0
        statistics : Optional[DataFrame], optional
            Time Statistics of the pairwise local labels, by default None
        label_tid_stat : str, optional
            The label of the column containing the ids of the formed segments.
            Is the new splitted id, by default TID_STAT
        drop_single_points : bool, optional
            Wether to drop the trajectories with only one point, by default False
        inplace : bool, optional
            Represents whether the operation will be performed on
            the data provided or in a copy, by default True

        Returns
        -------
        PandasDiscreteMoveDataFrame
            Object with new features or None

        Raises
        ------
        KeyError
            If missing local_label column
        ValueError
            If the data contains only null values
        """
        if inplace:
            data_ = self
        else:
            data_ = self.copy()

        if TIME_TO_PREV not in data_:
            self.generate_dist_time_speed_features(TRAJ_ID)

        if local_label not in data_:
            raise KeyError('{} not in data frame.'.format(local_label))

        if PREV_LOCAL not in data_:
            data_[local_label] = data_[local_label].astype(np.float64)
            self.generate_prev_local_features(
                label_id=label_id, local_label=local_label
            )

        if statistics is None:
            if (data_[PREV_LOCAL].isna().sum() == data_.shape[0]):
                raise ValueError(
                    'all values in the {} column are null.'.format(PREV_LOCAL)
                )
            else:
                statistics = generate_time_statistics(data_, local_label=local_label)
                threshold_time_statistics(statistics, mean_coef, std_coef)

        clean_trajectories_with_few_points(
            data_, label_tid=label_id, min_points_per_trajectory=2, inplace=True
        )

        current_tid, ids, count = _prepare_segmentation(data_, label_id, TID_STAT)

        for idx in progress_bar(ids, desc='Generating %s' % TID_STAT):
            md = data_.loc[idx, [TIME_TO_PREV, local_label, PREV_LOCAL]]
            md = pd.DataFrame(md)

            filter_ = []
            for _, row in md.iterrows():
                local_label_ = row[local_label]
                prev_local = row[PREV_LOCAL]
                threshold = statistics[
                    (statistics[local_label]
                        == local_label_) & (statistics[PREV_LOCAL] == prev_local)
                ][THRESHOLD].values

                filter_.append(row[TIME_TO_PREV] > threshold)

            filter_ = np.array(filter_)
            current_tid, count = _update_curr_tid_count(
                filter_, data_, idx, label_tid_stat, current_tid, count)

        if label_id == TID_STAT:
            self.reset_index(drop=True, inplace=True)
            print(
                '... {} = {}, then reseting and drop index!'.format(TID, TID_STAT))
        else:
            self.reset_index(inplace=True)
            print('... reseting index\n')

        if drop_single_points:
            _drop_single_point(data_, TID_STAT, label_id)
            self.generate_dist_time_speed_features()
        if not inplace:
            return data_
