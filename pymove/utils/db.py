import psycopg2
from pymongo import MongoClient
import pandas as pd
import io
import tempfile
import re

def connect_postgres(dbname='postgres', user='postgres', password='', host='localhost', port=5432):
    try:
        psql_params = "dbname='%s' user='%s' password='%s' host='%s' port='%s'" % (dbname, user, password, host, port)
        conn = psycopg2.connect(psql_params)
        return conn
    except Exception as e:
        raise e

def write_postgres(table, dataframe, dbname='postgres', user='postgres', password='', host='localhost', port=5432):
    cols = dataframe.columns
    columns = ",".join(cols)
    values = ','.join(["%s"] * len(cols))

    conn = None
    sql = "INSERT INTO %s(%s) VALUES(%s)" % (table, columns, values)
    try:
        conn = connect_postgres(dbname, user, password, host, port)
        cur = conn.cursor()
        cur.execute("DELETE FROM %s" % (table))
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

def read_postgres(query, in_memory=True, dbname='postgres', user='postgres', password='', host='localhost', port=5432):
    conn = None
    try:
        conn = connect_postgres(dbname, user, password, host, port)
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
        return dataframe

def read_sql_inmem_uncompressed(query, conn):
    copy_sql = "COPY ({query}) TO STDOUT WITH CSV {head}".format(
       query=query, head="HEADER")
    
    cur = conn.cursor()
    store = io.StringIO() # create object StringIO
    cur.copy_expert(copy_sql, store)
    store.seek(0) #  move the cursor over it data like seek(0) for start of file
    df = pd.read_csv(store)
    cur.close() # free memory in cursor
    store.close() # free memory in StringIO
    return df


def read_sql_tmpfile(query, conn):
    with tempfile.TemporaryFile() as tmpfile:
        copy_sql = "COPY ({query}) TO STDOUT WITH CSV {head}".format(
            query=query, head="HEADER"
        )
        cur = conn.cursor()
        cur.copy_expert(copy_sql, tmpfile)
        tmpfile.seek(0)
        df = pd.read_csv(tmpfile)
        return df

def connect_mongo(dbname='test', user=None, password=None, host='localhost', port=27017):
    try:
        if user and password:
            mongo_uri = 'mongodb://%s:%s@%s:%s/' % (user, password, host, port)
            conn = MongoClient(mongo_uri)
        else:
            mongo_uri = 'mongodb://@%s:%s/' % (host, port)
            conn = MongoClient(host, port)
        return conn[dbname]
    except Exception as e:
        raise e


def get_mongo_collection(collection, dbname='test', user=None, password=None, host='localhost', port=27017):
    try:    
        db = connect_mongo(dbname, user, password, host, port)
        return db[collection]
    except Exception as e:
        raise e

def read_mongo(collection, filter_=None, projection=None, no_id=True, dbname='test', user=None, password=None, host='localhost', port=27017):
    try:
        my_collection = get_mongo_collection(collection, dbname, user, password, host, port)

        cursor = my_collection.find(filter_, projection)

        dataframe =  pd.DataFrame(list(cursor))

        if no_id:
            del dataframe['_id']
        
        if '__v' in df:
            del dataframe['__v']

        return dataframe
    except Exception as e:
        raise e

def write_mongo(collection, dataframe, dbname='test', user=None, password=None, host='localhost', port=27017):
    try:
        my_collection = get_mongo_collection(collection, dbname, user, password, host, port)
        my_collection.delete_many( {} )
        
        json = dataframe.to_dict(orient='index')
        values = list(json.values())
        results = my_collection.insert_many(values)
        return len(results.inserted_ids)
    except Exception as e:
        raise e