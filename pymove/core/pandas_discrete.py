import numpy as np
import pandas as pd

from pymove.core.dataframe import MoveDataFrame
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
    INDEX_GRID,
    LATITUDE,
    LOCAL_LABEL,
    LONGITUDE,
    PREV_LOCAL,
    THRESHOLD,
    TID,
    TID_STAT,
    TIME_TO_PREV,
    TRAJ_ID,
    TYPE_PANDAS,
)
from pymove.utils.datetime import generate_time_statistics, threshold_time_statistics
from pymove.utils.log import progress_bar
from pymove.utils.mem import begin_operation, end_operation
from pymove.utils.trajectories import shift


class PandasDiscreteMoveDataFrame(PandasMoveDataFrame):
    def __init__(
        self,
        data,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        datetime=DATETIME,
        traj_id=TRAJ_ID,
        local_label=LOCAL_LABEL
    ):
        """

        Checks whether past data has 'lat', 'lon', 'datetime' and 'local_label'
        columns, and renames it with the PyMove lib standard. After starts the
        attributes of the class.

        - self._data : Represents trajectory data.
        - self._type : Represents the type of layer below the data structure.
        - self.last_operation : Represents the last operation perfomed.

        Parameters
        ----------
        data : dict, list, numpy array or pandas.core.DataFrame
            Input trajectory data.
        latitude : str, optional, default 'lat'.
            Represents column name latitude.
        longitude : str, optional, default 'lon'.
            Represents column name longitude.
        datetime : str, optional, default 'datetime'.
            Represents column name datetime.
        traj_id : str, optional, default 'id'.
            Represents column name trajectory id.
        local_label : str, optional, default 'local_label'.
            Represents column name local label

        Raises
        ------
        KeyError
            If missing one of lat, lon, datetime, local_label columns
        ValueError, ParserError
            If the data types can't be converted.

        """

        super()

        if isinstance(data, dict):
            data = pd.DataFrame.from_dict(data)
        elif (
            (isinstance(data, list) or isinstance(data, np.ndarray))
            and len(data) >= 5
        ):
            zip_list = [LATITUDE, LONGITUDE, DATETIME, TRAJ_ID, local_label]
            for i in range(len(data[0])):
                try:
                    zip_list[i] = zip_list[i]
                except KeyError:
                    zip_list.append(i)
            data = pd.DataFrame(data, columns=zip_list)

        mapping_columns = MoveDataFrame.format_labels(
            traj_id, latitude, longitude, datetime
        )
        tdf = data.rename(columns=mapping_columns)

        if local_label not in tdf:
            raise ValueError(
                '{} column not in dataframe'.format(local_label)
            )

        if MoveDataFrame.has_columns(tdf):
            MoveDataFrame.validate_move_data_frame(tdf)
            self._data = tdf
            self._type = TYPE_PANDAS
            self.last_operation = None
        else:
            raise AttributeError(
                'Couldn\'t instantiate MoveDataFrame because data has missing columns'
            )

    def discretize_based_grid(self, region_size=1000):
        """
        Discrete space in cells of the same size,
        assigning a unique id to each cell.

        Parameters
        ----------
        region_size: number, optional, default 1000
            Size of grid'srs cell.
        """

        operation = begin_operation('discretize based on grid')
        print('\nDiscretizing dataframe...')
        try:
            grid = Grid(self, cell_size=region_size)
            grid.create_update_index_grid_feature(self)
            self.reset_index(drop=True, inplace=True)
            self.last_operation = end_operation(operation)

        except Exception as e:
            self.last_operation = end_operation(operation)
            raise e

    def generate_prev_local_features(
        self, label_id=TRAJ_ID, local_label=LOCAL_LABEL, sort=True, inplace=True
    ):
        """
        Create a feature prev_local with the label of previous local to current point.

        Parameters
        ----------
        label_id : str, optional, default 'id'.
            Represents name of column of trajectory'srs id.
        local_label : String, optional, default 'local_label'
            Indicates name of column of place labels on symbolic trajectory.
        If sort == True the dataframe will be sorted, default True.
        inplace : bool, optional, default True.
            Represents whether the operation will be performed on
            the data provided or in a copy.
        Return
        ------
        PandasMoveDataFrame or None
            Object with new features or None if ``inplace=True``.

        """
        operation = begin_operation('generate_prev_equ_feature')

        if inplace:
            data_ = self
        else:
            data_ = self.copy()
        try:
            message = '\nCreating generate_prev_equ_feature'
            message += ' in previous equ\n'
            print(
                message
            )

            ids, sum_size_id, size_id = self._prepare_generate_data(
                data_, sort, label_id
            )

            # create new feature to pre_equ
            # data_[PREV_LOCAL] = np.float64(-1.0)

            if (data_[local_label].dtype == 'int'):
                data_[local_label] = data_[local_label].astype(np.float16)
            for idx in progress_bar(
                ids, desc='Generating previous {}'.format(local_label)
            ):
                current_local = data_.at[idx, local_label]
                current_local = np.array(current_local)
                size_id = current_local.size

                if size_id <= 1:
                    data_.at[idx, PREV_LOCAL] = np.nan

                else:
                    prev_local = shift(current_local, 1)

                    # previous to current point
                    data_.at[idx, PREV_LOCAL] = prev_local

            return self._return_generated_data(
                data_, operation, inplace
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
            label_id=TRAJ_ID,
            local_label=LOCAL_LABEL,
            mean_coef=1.0,
            std_coef=1.0,
            statistics=None,
            label_tid_stat=TID_STAT,
            drop_single_points=False,
            inplace=True,
    ):

        """
        Splits the trajectories into segments based on time statistics for segments.

        Parameters
        ----------
        label_id : str, optional, default 'id'.
            Represents name of column of trajectory'srs id.
        local_label : String, optional, default 'local_label'
            Indicates name of column of place labels on symbolic trajectory.
        mean_coef : float
            Multiplication coefficient of the mean time for the segment, default 1.0
        std_coef : float
            Multiplication coefficient of sdt time for the segment, default 1.0
        statistics : dataframe
            Time Statistics of the pairwise local labels.
        label_new_tid : String, optional(TID_PART by default)
            The label of the column containing the ids of the formed segments.
            Is the new splitted id.
        drop_single_points : boolean, optional(True by default)
            If set to True, drops the trajectories with only one point.
        inplace : bool, optional, default True.
            Represents whether the operation will be performed on
            the data provided or in a copy.
        Return
        ------
        PandasMoveDataFrame or None
            Object with new features or None if ``inplace=True``.

        """
        if inplace:
            data_ = self
        else:
            data_ = self.copy()
        try:

            if TIME_TO_PREV not in data_:
                self.generate_dist_time_speed_features(TRAJ_ID)

            if local_label not in data_:
                raise ValueError('{} not in data frame.'.format(local_label))

            if PREV_LOCAL not in data_:
                data_[local_label] = data_[local_label].astype(np.float64)
                self.generate_prev_local_features(label_id=label_id,
                                                  local_label=local_label)

            if statistics is None:
                if (data_[PREV_LOCAL].isna().sum() == data_.shape[0]):
                    raise ValueError('all values in the {} column are null.'
                                     .format(PREV_LOCAL))
                else:
                    statistics = generate_time_statistics(data_, local_label=local_label)
                    threshold_time_statistics(statistics, mean_coef, std_coef)

            clean_trajectories_with_few_points(data_, label_tid=label_id,
                                               min_points_per_trajectory=2, inplace=True)

            current_tid, ids, count = _prepare_segmentation(data_, label_id, TID_STAT)

            for idx in progress_bar(ids, desc='Generating %s' % TID_STAT):
                md = data_.loc[idx, [TIME_TO_PREV, local_label, PREV_LOCAL]]
                md = pd.DataFrame(md)

                filter_ = []
                for index, row in md.iterrows():
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
        except Exception as e:
            raise e
