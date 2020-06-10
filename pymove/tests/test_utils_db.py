from psycopg2.extensions import connection
from pymongo.database import Database

from pymove.utils import db

TABLE_NAME = 'test_table'
DB_NAME = 'travis_ci_test'


def test_connect_postgres():
    conn = db.connect_postgres(DB_NAME)
    assert isinstance(conn, connection)


def test_connect_mongo():
    conn = db.connect_mongo(DB_NAME)
    assert isinstance(conn, Database)
