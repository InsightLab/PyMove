import pandas as pd
from numpy.testing import assert_equal
from pandas.testing import assert_series_equal

from pymove import MoveDataFrame, clustering
from pymove.utils.constants import DATETIME, LATITUDE, LONGITUDE, TRAJ_ID

list_data = [
    [39.984094, 116.319236, '2008-10-23 05:53:05', 1],
    [39.984198, 116.319322, '2008-10-23 05:53:06', 1],
    [39.984224, 116.319402, '2008-10-23 05:53:11', 1],
    [39.984211, 116.319389, '2008-10-23 05:53:16', 1],
    [39.984217, 116.319422, '2008-10-23 05:53:21', 1],
    [39.984710, 116.319865, '2008-10-23 05:53:23', 1],
    [39.984674, 116.319810, '2008-10-23 05:53:28', 1],
    [39.984623, 116.319773, '2008-10-23 05:53:33', 1],
    [39.984606, 116.319732, '2008-10-23 05:53:38', 1],
    [39.984555, 116.319728, '2008-10-23 05:53:43', 1]
]


def _default_move_df(data=None):
    if data is None:
        data = list_data
    return MoveDataFrame(
        data=data,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        datetime=DATETIME,
        traj_id=TRAJ_ID,
    )


def test_elbow_method():
    move_df = _default_move_df()

    expected = pd.Series({1: 1.0136844999935952e-06,
                          2: 6.238999999844088e-08,
                          3: 3.369214999908675e-08,
                          4: 1.1907583332727983e-08,
                          5: 6.543166665966021e-09,
                          6: 3.930833333517101e-09,
                          7: 1.622333333194155e-09,
                          8: 6.373333333646581e-10})

    inertia_dic = clustering.elbow_method(move_data=move_df[[LATITUDE, LONGITUDE]] ,
                                          max_clusters=8,
                                          random_state=1
                                          )

    result = pd.Series(inertia_dic)

    assert_series_equal(result, expected, check_less_precise=True)
