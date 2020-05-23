import io
import tempfile

import pandas as pd
import psycopg2
from pymongo import MongoClient

from pymove import MoveDataFrame
from pymove.utils.constants import TYPE_PANDAS


def connect_postgres(
    dbname='postgres',
    user='postgres',
    psswrd='',
    host='localhost',
    port=5432,
):
    """
    Connects to a postgres database.

    Parameters
    ----------
    dbname : string, default 'postgres'
        Name of the database
    user : string, default 'postgres'
        The user connecting to the database
    psswrd : string, default ''
        The password of the database
    host : string, default 'localhost'
        The address of the database
    port : int, default 5432
        The port of the database in the host

    Returns
    -------
    psycopg2.extensions.connection
        Connection to the desired database

    """
    try:
        psql_params = (
            "dbname='%s' user='%s' password='%s' host='%s' port='%s'"
            % (dbname, user, psswrd, host, port)
        )
        conn = psycopg2.connect(psql_params)
        return conn
    except Exception as e:
        raise e


def write_postgres(
    table,
    dataframe,
    dbname='postgres',
    user='postgres',
    psswrd='',
    host='localhost',
    port=5432,
):
    """
    Saves a dataframe to a postgres table.

    Parameters
    ----------
    table : string
        Name of the table
    dataframe : dataframe object
        The dataframe to be saved
    dbname : string, default 'postgres'
        Name of the database
    user : string, default 'postgres'
        The user connecting to the database
    psswrd : string, default ''
        The password of the database
    host : string, default 'localhost'
        The address of the database
    port : int, default 5432
        The port of the database in the host

    """

    cols = dataframe.columns
    columns = ','.join(cols)
    values = ','.join(['%srs'] * len(cols))

    conn = None
    sql = 'INSERT INTO %s(%s) VALUES(%s)' % (table, columns, values)
    try:
        conn = connect_postgres(dbname, user, psswrd, host, port)
        cur = conn.cursor()
        cur.execute('DELETE FROM %srs', (table,))
        cur.executemany(sql, dataframe.values)
        conn.commit()
        cur.close()
    except Exception as e:
        if conn is not None:
            conn.close()
        raise e
    finally:
        if conn is not None:
            conn.close()


def read_postgres(
    query,
    in_memory=True,
    type_=TYPE_PANDAS,
    dbname='postgres',
    user='postgres',
    psswrd='',
    host='localhost',
    port=5432,
):
    """
    Builds a dataframe from a query to a postgres database.

    Parameters
    ----------
    query : string
        Sql query
    in_memory : bool, default True
        Whether te operation will be executed in memory
    type_ : 'pandas', 'dask' or None, defaults 'pandas'
        It will try to convert the dataframe into a MoveDataFrame
    dbname : string, default 'postgres'
        Name of the database
    user : string, default 'postgres'
        The user connecting to the database
    psswrd : string, default ''
        The password of the database
    host : string, default 'localhost'
        The address of the database
    port : int, default 5432
        The port of the database in the host

    Returns
    -------
    PandasMoveDataFrame, DaskMoveDataFrame or PandasDataFrame
        a dataframe object containing the result from the query

    """

    conn = None
    try:
        conn = connect_postgres(dbname, user, psswrd, host, port)
        if in_memory:
            dataframe = read_sql_inmem_uncompressed(query, conn)
        else:
            dataframe = read_sql_tmpfile(query, conn)
    except Exception as e:
        if conn is not None:
            conn.close()
        raise e
    finally:
        if conn is not None:
            conn.close()
    try:
        return MoveDataFrame(dataframe, type_=type_)
    except Exception:
        return dataframe


def read_sql_inmem_uncompressed(query, conn):
    """
    Builds a dataframe from a query to a postgres database.

    Parameters
    ----------
    query : string
        Sql query
    conn : psycopg2.extensions.connection
        Postgres database connection

    Returns
    -------
    PandasDataframe
        The query contents in a dataframe format

    """
    copy_sql = 'COPY ({query}) TO STDOUT WITH CSV {head}'.format(
        query=query, head='HEADER'
    )

    cur = conn.cursor()
    store = io.StringIO()  # create object StringIO
    cur.copy_expert(copy_sql, store)
    store.seek(0)  # move the cursor over it data like seek(0) for start of file
    df = pd.read_csv(store)
    cur.close()  # free memory in cursor
    store.close()  # free memory in StringIO
    return df


def read_sql_tmpfile(query, conn):
    """
    Builds a dataframe from a query to a postgres database.

    Parameters
    ----------
    query : string
        Sql query
    conn : psycopg2.extensions.connection
        Postgres database connection

    Returns
    -------
    PandasDataframe
        The query contents in a dataframe format

    """

    with tempfile.TemporaryFile() as tmpfile:
        copy_sql = 'COPY ({query}) TO STDOUT WITH CSV {head}'.format(
            query=query, head='HEADER'
        )
        cur = conn.cursor()
        cur.copy_expert(copy_sql, tmpfile)
        tmpfile.seek(0)
        df = pd.read_csv(tmpfile)
        return df


def connect_mongo(
    dbname='test', user=None, psswrd=None, host='localhost', port=27017
):
    """
    Connects to a mongo database.

    Parameters
    ----------
    dbname : string, default 'test'
        Name of the database
    user : string, default None
        The user connecting to the database
    psswrd : string, default None
        The password of the database
    host : string, default 'localhost'
        The address of the database
    port : int, default 27017
        The port of the database in the host

    Returns
    -------
    pymongo.database.Database
        Connection to the desired database

    """
    try:
        if user and psswrd:
            mongo_uri = 'mongodb://%s:%s@%s:%s/' % (user, psswrd, host, port)
            conn = MongoClient(mongo_uri)
        else:
            conn = MongoClient(host, psswrd)
        return conn[dbname]
    except Exception as e:
        raise e


def get_mongo_collection(
    collection,
    dbname='test',
    user=None,
    psswrd=None,
    host='localhost',
    port=27017,
):
    """
    Gets a mongo collection.

    Parameters
    ----------
    collection : string
        Name of the collection
    dbname : string, default 'test'
        Name of the database
    user : string, default None
        The user connecting to the database
    psswrd : string, default None
        The password of the database
    host : string, default 'localhost'
        The address of the database
    port : int, default 27017
        The port of the database in the host

    Returns
    -------
    pymongo.collection.Collection
        The desired mongo collection
    """
    try:
        conn = connect_mongo(dbname, user, psswrd, host, port)
        return conn[collection]
    except Exception as e:
        raise e


def write_mongo(
    collection,
    dataframe,
    dbname='test',
    user=None,
    psswrd=None,
    host='localhost',
    port=27017,
):
    """
    Saves a dataframe to a mongo collection.

    Parameters
    ----------
    collection : string
        Name of the collection
    dataframe : dataframe object
        The dataframe to be saved
    dbname : string, default 'postgres'
        Name of the database
    user : string, default 'postgres'
        The user connecting to the database
    psswrd : string, default ''
        The password of the database
    host : string, default 'localhost'
        The address of the database
    port : int, default 5432
        The port of the database in the host

    """
    try:
        my_collection = get_mongo_collection(
            collection, dbname, user, psswrd, host, port
        )
        my_collection.delete_many({})

        json = dataframe.to_dict(orient='index')
        values = list(json.values())
        results = my_collection.insert_many(values)
        return len(results.inserted_ids)
    except Exception as e:
        raise e


def read_mongo(
    collection,
    filter_=None,
    projection=None,
    type_=TYPE_PANDAS,
    no_id=True,
    dbname='test',
    user=None,
    psswrd=None,
    host='localhost',
    port=27017,
):
    """
    Builds a dataframe from a mongo collection.

    Parameters
    ----------
    collection : string
        Name of the collection
    filter_ : map, default None
        The filtering to apply to the query
    projection : map, default None
        The fields to retrieve from the collection
    type_ : 'pandas', 'dask' or None, defaults 'pandas'
        It will try to convert the dataframe into a MoveDataFrame
    no_id: bool, default True
        Whether to drop the registers id'srs
    dbname : string, default 'postgres'
        Name of the database
    user : string, default 'postgres'
        The user connecting to the database
    psswrd : string, default ''
        The password of the database
    host : string, default 'localhost'
        The address of the database
    port : int, default 5432
        The port of the database in the host

    Returns
    -------
    dataframe: PandasMoveDataFrame, DaskMoveDataFrame or PandasDataFrame
        a dataframe object with the contents of the collection

    """

    try:
        my_collection = get_mongo_collection(
            collection, dbname, user, psswrd, host, port
        )

        cursor = my_collection.find(filter_, projection)

        dataframe = pd.DataFrame(list(cursor))

        if no_id:
            del dataframe['_id']

        if '__v' in dataframe:
            del dataframe['__v']

        try:
            return MoveDataFrame(dataframe, type_=type_)
        except Exception:
            return dataframe
    except Exception as e:
        raise e
