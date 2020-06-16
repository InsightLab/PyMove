from pandas import DataFrame, Timestamp
from pandas.testing import assert_frame_equal
from psycopg2.extensions import connection
from pymongo.database import Database

from pymove import MoveDataFrame
from pymove.utils import db
from pymove.utils.constants import DATETIME, LATITUDE, LONGITUDE, TRAJ_ID

list_data = [
    [39.984094, 116.319236, '2008-10-23 05:53:05', 1],
    [39.984198, 116.319322, '2008-10-23 05:53:06', 1],
    [39.984224, 116.319402, '2008-10-23 05:53:11', 1],
    [39.984224, 116.319402, '2008-10-23 05:53:11', 2],
]


def _default_move_df():
    return MoveDataFrame(
        data=list_data,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        datetime=DATETIME,
        traj_id=TRAJ_ID,
    )


TABLE_NAME = 'test_table'
DB_NAME = 'travis_ci_test'


def test_connect_postgres():
    conn = db.connect_postgres(DB_NAME)
    assert isinstance(conn, connection)


def test_connect_mongo():
    conn = db.connect_mongo(DB_NAME)
    assert isinstance(conn, Database)


def test_write_postgres():
    expected = DataFrame(
        data=[
            [1, 1, 39.984094, 116.319236, Timestamp('2008-10-23 05:53:05'), 1],
            [2, 1, 39.984198, 116.319322, Timestamp('2008-10-23 05:53:06'), 1],
            [3, 2, 39.984224, 116.319402, Timestamp('2008-10-23 05:53:11'), 2],
            [4, 2, 39.984224, 116.319402, Timestamp('2008-10-23 05:53:11'), 2]
        ],
        columns=['_id', 'id', 'lat', 'lon', 'datetime'],
        index=[0, 1, 2, 3],
    )

    move_df = _default_move_df()

    db.write_postgres(table='testedb', dataframe=move_df)

    new_move_df = db.read_postgres(query='SELECT * FROM public.testedb')

    assert_frame_equal(new_move_df, expected)
