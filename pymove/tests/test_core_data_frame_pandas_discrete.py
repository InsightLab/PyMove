from pandas.testing import assert_frame_equal

from pymove.core.pandas_discrete import PandasDiscreteMoveDataFrame
from pymove.utils.constants import (
    DATETIME,
    INDEX_GRID,
    LATITUDE,
    LOCAL_LABEL,
    LONGITUDE,
    TRAJ_ID,
)


def test_discretize_based_grid():
    discrete_df = PandasDiscreteMoveDataFrame(
        data={DATETIME: ['2020-01-01 01:08:29',
                         '2020-01-05 01:13:24',
                         '2020-01-06 02:21:53',
                         '2020-01-06 03:34:48',
                         '2020-01-08 05:55:41'],
              LATITUDE: [3.754245,
                         3.150849,
                         3.754249,
                         3.165933,
                         3.920178],
              LONGITUDE: [38.3456743,
                          38.6913486,
                          38.3456743,
                          38.2715962,
                          38.5161605],
              TRAJ_ID: ['pwe-5089',
                        'xjt-1579',
                        'tre-1890',
                        'xjt-1579',
                        'pwe-5089'],
              LOCAL_LABEL: [1, 4, 2, 16, 32]},
    )

    expected = PandasDiscreteMoveDataFrame(
        data={DATETIME: ['2020-01-01 01:08:29',
                         '2020-01-08 05:55:41',
                         '2020-01-06 02:21:53',
                         '2020-01-05 01:13:24',
                         '2020-01-06 03:34:48'],
              LATITUDE: [3.754245,
                         3.920178,
                         3.754249,
                         3.150849,
                         3.165933],
              LONGITUDE: [38.3456743,
                          38.5161605,
                          38.3456743,
                          38.6913486,
                          38.2715962],
              TRAJ_ID: ['pwe-5089',
                        'pwe-5089',
                        'tre-1890',
                        'xjt-1579',
                        'xjt-1579'],
              LOCAL_LABEL: [1, 32, 2, 4, 16],
              INDEX_GRID: [754, 2407, 754, 3956, 1]},
    )

    discrete_df.discretize_based_grid()

    assert_frame_equal(discrete_df, expected)
