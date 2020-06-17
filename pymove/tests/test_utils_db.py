
from numpy.testing import assert_equal
from pandas import DataFrame, Timestamp
from pandas.testing import assert_frame_equal
from psycopg2.extensions import connection
from pymongo import MongoClient, collection
from pymongo.database import Database

from pymove import MoveDataFrame
from pymove.utils import db
from pymove.utils.constants import DATETIME, LATITUDE, LONGITUDE, TRAJ_ID

list_data = [
    [39.984094, 116.319236, '2008-10-23 05:53:05', 1],
    [39.984198, 116.319322, '2008-10-23 05:53:06', 1],
    [39.984224, 116.319402, '2008-10-23 05:53:11', 2],
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

df_move = _default_move_df()


db.write_postgres(table='test_read_db', dbname=DB_NAME, dataframe=df_move)

db.write_mongo(collection='test_read_db', dataframe=df_move, dbname=DB_NAME)


def test_connect_postgres():
    conn = db.connect_postgres(DB_NAME)
    assert isinstance(conn, connection)


def test_connect_mongo():
    conn = db.connect_mongo(DB_NAME)
    assert isinstance(conn, Database)


def test_write_postgres():
    expected = DataFrame(
        data=[
            [1, 1, 39.984094, 116.319236, Timestamp('2008-10-23 05:53:05')],
            [2, 1, 39.984198, 116.319322, Timestamp('2008-10-23 05:53:06')],
            [3, 2, 39.984224, 116.319402, Timestamp('2008-10-23 05:53:11')],
            [4, 2, 39.984224, 116.319402, Timestamp('2008-10-23 05:53:11')]
        ],
        columns=['_id', 'id', 'lat', 'lon', 'datetime'],
        index=[0, 1, 2, 3],
    )

    move_df = _default_move_df()

    db.write_postgres(table='test_table', dbname=DB_NAME, dataframe=move_df)

    new_move_df = db.read_postgres(dbname=DB_NAME,
                                   query='SELECT * FROM public.test_table')

    assert_frame_equal(new_move_df, expected)

    db._create_table(table='test_new_table', dbname=DB_NAME)

    '''Testing using an existing table'''
    db.write_postgres(table='test_new_table', dbname=DB_NAME, dataframe=move_df)

    new_move_df = db.read_postgres(dbname=DB_NAME,
                                   query='SELECT * FROM public.test_new_table')

    assert_frame_equal(new_move_df, expected)


def test_read_postgres():

    expected = DataFrame(
        data=[
            [1, 1, 39.984094, 116.319236, Timestamp('2008-10-23 05:53:05')],
            [2, 1, 39.984198, 116.319322, Timestamp('2008-10-23 05:53:06')],
            [3, 2, 39.984224, 116.319402, Timestamp('2008-10-23 05:53:11')],
            [4, 2, 39.984224, 116.319402, Timestamp('2008-10-23 05:53:11')]
        ],
        columns=['_id', 'id', 'lat', 'lon', 'datetime'],
        index=[0, 1, 2, 3],
    )

    new_move_df = db.read_postgres(query='SELECT * FROM public.test_read_db',
                                   dbname=DB_NAME)

    assert_frame_equal(new_move_df, expected)

    new_move_df = db.read_postgres(query='SELECT * FROM public.test_read_db',
                                   in_memory=False, dbname=DB_NAME)

    assert_frame_equal(new_move_df, expected)


def test_read_sql_inmem_uncompressed():

    expected = DataFrame(
        data=[
            [1, 1, 39.984094, 116.319236, ('2008-10-23 05:53:05')],
            [2, 1, 39.984198, 116.319322, ('2008-10-23 05:53:06')],
            [3, 2, 39.984224, 116.319402, ('2008-10-23 05:53:11')],
            [4, 2, 39.984224, 116.319402, ('2008-10-23 05:53:11')]
        ],
        columns=['_id', 'id', 'lat', 'lon', 'datetime'],
        index=[0, 1, 2, 3],
    )

    conn = db.connect_postgres(DB_NAME)

    new_move_df = db.read_sql_inmem_uncompressed(query=('SELECT * FROM '
                                                        'public.test_read_db'),
                                                 conn=conn)

    assert_frame_equal(new_move_df, expected)


def test_read_sql_tmpfile():

    expected = DataFrame(
        data=[
            [1, 1, 39.984094, 116.319236, ('2008-10-23 05:53:05')],
            [2, 1, 39.984198, 116.319322, ('2008-10-23 05:53:06')],
            [3, 2, 39.984224, 116.319402, ('2008-10-23 05:53:11')],
            [4, 2, 39.984224, 116.319402, ('2008-10-23 05:53:11')]
        ],
        columns=['_id', 'id', 'lat', 'lon', 'datetime'],
        index=[0, 1, 2, 3],
    )

    conn = db.connect_postgres(DB_NAME)

    new_move_df = db.read_sql_tmpfile(query='SELECT * FROM public.test_read_db',
                                      conn=conn)

    print(new_move_df)

    assert_frame_equal(new_move_df, expected)


def test_create_table():

    conn = db.connect_postgres(DB_NAME)
    cur = conn.cursor()
    cur.execute(('select exists(select * FROM information_schema.tables '
                 'WHERE table_name=%s);'), ('test_table_creation',))
    table_exists = cur.fetchone()[0]

    assert(table_exists is False)

    db._create_table(table='test_table_creation', dbname=DB_NAME)

    cur = conn.cursor()
    cur.execute(('select exists(select * FROM information_schema.tables '
                 'WHERE table_name=%s);'), ('test_table_creation',))
    table_exists = cur.fetchone()[0]

    assert(table_exists is True)

    '''Testing function execution when the table already exists'''
    db._create_table(table='test_table_creation', dbname=DB_NAME)

    cur = conn.cursor()
    cur.execute(('select exists(select * FROM information_schema.tables '
                 'WHERE table_name=%s);'), ('test_table_creation',))
    table_exists = cur.fetchone()[0]

    assert(table_exists is True)


def test_write_mongo():

    expected = DataFrame(
        data=[
            [Timestamp('2008-10-23 05:53:05'), 1, 39.984094, 116.319236],
            [Timestamp('2008-10-23 05:53:06'), 1, 39.984198, 116.319322],
            [Timestamp('2008-10-23 05:53:11'), 2, 39.984224, 116.319402],
            [Timestamp('2008-10-23 05:53:11'), 2, 39.984224, 116.319402]
        ],
        columns=['datetime', 'id', 'lat', 'lon'],
        index=[0, 1, 2, 3],
    )

    move_df = _default_move_df()

    inserted_ids = db.write_mongo(collection='test_db',
                                  dataframe=df_move,
                                  dbname=DB_NAME)

    assert(inserted_ids == 4)

    new_move_df = db.read_mongo(collection='test_db', dbname=DB_NAME)

    assert_frame_equal(new_move_df, expected)


def test_read_mongo():

    expected = DataFrame(
        data=[
            [Timestamp('2008-10-23 05:53:05'), 1, 39.984094, 116.319236],
            [Timestamp('2008-10-23 05:53:06'), 1, 39.984198, 116.319322],
            [Timestamp('2008-10-23 05:53:11'), 2, 39.984224, 116.319402],
            [Timestamp('2008-10-23 05:53:11'), 2, 39.984224, 116.319402]
        ],
        columns=['datetime', 'id', 'lat', 'lon'],
        index=[0, 1, 2, 3],
    )

    expected_filtered = DataFrame(
        data=[
            [Timestamp('2008-10-23 05:53:05'), 39.984094],
            [Timestamp('2008-10-23 05:53:06'), 39.984198],
        ],
        columns=['datetime', 'lat'],
        index=[0, 1],
    )

    new_move_df = db.read_mongo(collection='test_read_db', dbname=DB_NAME)

    assert_frame_equal(new_move_df, expected)

    new_move_df = db.read_mongo(collection='test_read_db',
                                dbname=DB_NAME,
                                filter_={'id': 1},
                                projection={'datetime', 'lat'})

    assert_frame_equal(new_move_df, expected_filtered)


def test_get_mongo_collection():

    expected = collection(Database(MongoClient(host=['localhost:27017'],
                                               document_class=dict,
                                               tz_aware=False, connect=True),
                                   'travis_ci_test'), 'test_db')

    coll = db.get_mongo_collection(collection='test_read_db',
                                   dbname=DB_NAME)

    assert_equal(coll, expected)
