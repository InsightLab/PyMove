from numpy.testing import assert_equal
from pandas import DataFrame, Timestamp
from pandas.testing import assert_frame_equal
from psycopg2 import connect
from psycopg2.extensions import connection
from pymongo import MongoClient
from pymongo.collection import Collection
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

PG_DB_NAME = 'postgres'

TEST_TABLE = 'test_table'
TEST_NEW_TABLE = 'test_new_table'
TABLE_READ = 'test_read'
TEST_CREATE = 'test_create'

TABLES = [
    TEST_TABLE,
    TEST_NEW_TABLE,
    TABLE_READ,
    TEST_CREATE
]

MG_DB_NAME = 'test'


def _default_move_df():
    return MoveDataFrame(
        data=list_data,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        datetime=DATETIME,
        traj_id=TRAJ_ID,
    )


df_move = _default_move_df()


def setup_module(module):
    db.write_postgres(table=TABLE_READ, dbname=PG_DB_NAME, dataframe=df_move)
    db.write_mongo(collection=TABLE_READ, dataframe=df_move, dbname=MG_DB_NAME)


def teardown_module(module):
    conn = connect(
        dbname=PG_DB_NAME,
        user='postgres',
        host='localhost',
        password=''
    )

    cursor = conn.cursor()
    for table in TABLES:
        cursor.execute('DROP TABLE IF EXISTS %s' % (table))
    conn.commit()
    cursor.close()
    conn.close()

    conn = None
    conn = MongoClient('localhost', None)[MG_DB_NAME]
    for table in TABLES:
        try:
            conn[table].drop()
        except Exception as e:
            pass


def test_connect_postgres():
    conn = db.connect_postgres(PG_DB_NAME)
    assert isinstance(conn, connection)


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

    db.write_postgres(table=TEST_TABLE, dbname=PG_DB_NAME, dataframe=move_df)

    query = 'SELECT * FROM public.%s' % TEST_TABLE
    new_move_df = db.read_postgres(dbname=PG_DB_NAME,
                                   query=query)

    assert_frame_equal(new_move_df, expected)

    db._create_table(table=TEST_NEW_TABLE, dbname=PG_DB_NAME)

    '''Testing using an existing table'''
    db.write_postgres(table=TEST_NEW_TABLE, dbname=PG_DB_NAME, dataframe=move_df)

    query = 'SELECT * FROM public.%s' % TEST_NEW_TABLE
    new_move_df = db.read_postgres(dbname=PG_DB_NAME,
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

    query = 'SELECT * FROM public.%s' % TABLE_READ
    new_move_df = db.read_postgres(query=query,
                                   dbname=PG_DB_NAME)

    assert_frame_equal(new_move_df, expected)

    new_move_df = db.read_postgres(query=query,
                                   in_memory=False, dbname=PG_DB_NAME)

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

    conn = db.connect_postgres(PG_DB_NAME)

    query = 'SELECT * FROM public.%s' % TABLE_READ
    new_move_df = db.read_sql_inmem_uncompressed(query=(query),
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

    conn = db.connect_postgres(PG_DB_NAME)

    query = 'SELECT * FROM public.%s' % TABLE_READ
    new_move_df = db.read_sql_tmpfile(query=query,
                                      conn=conn)

    print(new_move_df)

    assert_frame_equal(new_move_df, expected)


def test_create_table():

    conn = db.connect_postgres(PG_DB_NAME)
    cur = conn.cursor()
    cur.execute(('select exists(select * FROM information_schema.tables '
                 'WHERE table_name=%s);'), (TEST_CREATE,))
    table_exists = cur.fetchone()[0]

    assert(table_exists is False)

    db._create_table(table=TEST_CREATE, dbname=PG_DB_NAME)

    cur = conn.cursor()
    cur.execute(('select exists(select * FROM information_schema.tables '
                 'WHERE table_name=%s);'), (TEST_CREATE,))
    table_exists = cur.fetchone()[0]

    assert(table_exists is True)

    '''Testing function execution when the table already exists'''
    db._create_table(table=TEST_CREATE, dbname=PG_DB_NAME)

    cur = conn.cursor()
    cur.execute(('select exists(select * FROM information_schema.tables '
                 'WHERE table_name=%s);'), (TEST_CREATE,))
    table_exists = cur.fetchone()[0]

    assert(table_exists is True)


def test_connect_mongo():
    conn = db.connect_mongo(MG_DB_NAME)
    expected = ("Database(MongoClient(host=['localhost:27017'], "
                'document_class=dict, tz_aware=False, connect=True), '
                "'test')")

    assert isinstance(conn, Database)
    assert_equal(str(conn), expected)


def test_get_mongo_collection():

    coll = db.get_mongo_collection(collection=TABLE_READ,
                                   dbname=MG_DB_NAME)

    expected = ("Collection(Database(MongoClient(host=['localhost:27017'], "
                'document_class=dict, tz_aware=False, connect=True), '
                "'test'), 'test_read')")

    assert isinstance(coll, Collection)
    assert_equal(str(coll), expected)


def test_write_mongo():

    expected = DataFrame(
        data=[
            [39.984094, 116.319236, Timestamp('2008-10-23 05:53:05'), 1],
            [39.984198, 116.319322, Timestamp('2008-10-23 05:53:06'), 1],
            [39.984224, 116.319402, Timestamp('2008-10-23 05:53:11'), 2],
            [39.984224, 116.319402, Timestamp('2008-10-23 05:53:11'), 2]
        ],
        columns=['lat', 'lon', 'datetime', 'id'],
        index=[0, 1, 2, 3],
    )

    move_df = _default_move_df()

    inserted_ids = db.write_mongo(collection=TEST_TABLE,
                                  dataframe=df_move,
                                  dbname=MG_DB_NAME)

    assert(inserted_ids == 4)

    new_move_df = db.read_mongo(collection=TEST_TABLE, dbname=MG_DB_NAME)

    assert_frame_equal(new_move_df, expected)


def test_read_mongo():

    expected = DataFrame(
        data=[
            [39.984094, 116.319236, Timestamp('2008-10-23 05:53:05'), 1],
            [39.984198, 116.319322, Timestamp('2008-10-23 05:53:06'), 1],
            [39.984224, 116.319402, Timestamp('2008-10-23 05:53:11'), 2],
            [39.984224, 116.319402, Timestamp('2008-10-23 05:53:11'), 2]
        ],
        columns=['lat', 'lon', 'datetime', 'id'],
        index=[0, 1, 2, 3],
    )

    expected_filtered = DataFrame(
        data=[
            [39.984094, Timestamp('2008-10-23 05:53:05')],
            [39.984198, Timestamp('2008-10-23 05:53:06')],
        ],
        columns=['lat', 'datetime'],
        index=[0, 1],
    )

    new_move_df = db.read_mongo(collection=TABLE_READ, dbname=MG_DB_NAME)

    assert_frame_equal(new_move_df, expected)

    new_move_df = db.read_mongo(collection=TABLE_READ,
                                dbname=MG_DB_NAME,
                                filter_={'id': 1},
                                projection={'datetime', 'lat'})

    assert_frame_equal(new_move_df, expected_filtered)
