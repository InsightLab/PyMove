01 - Exploring MoveDataFrame
============================

To work with Pymove you need to import the data into our data structure:
**MoveDataFrame**!

**MoveDataFrame** is an abstraction that instantiates a new data
structure that manipulates the structure the user wants. This is done
using the Factory Method design pattern. This structure allows the
interface to be implemented using different representations and
libraries that manipulate the data.

We have an interface that delimits the scope that new implementing
classes should have. We currently have two concrete classes that
implement this interface: **PandasMoveDataFrame** and
**DaskMoveDataFrame** (under construction), which use Pandas and Dask
respectively for data manipulation.

It works like this: The user instantiating a MoveDataFrame provides a
flag telling which library they want to use for manipulating this data.

Now that we understand the concept and data structure of PyMove, **hands
on!**

--------------

MoveDataFrame
-------------

A MoveDataFrame must contain the columns: - ``lat``: represents the
latitude of the point. - ``lon``: represents the longitude of the point.
- ``datetime``: represents the date and time of the point.

In addition, the user can enter several other columns as trajectory id.
**If the id is not entered, the points are supposed to belong to the
same path**.

--------------

Creating a MoveDataFrame
------------------------

A MoveDataFrame can be created by passing a Pandas DataFrame, a list,
dict or even reading a file. Look:

.. code:: python

    import pymove
    from pymove import MoveDataFrame

From a list
~~~~~~~~~~~

.. code:: python

    list_data = [
        [39.984094, 116.319236, '2008-10-23 05:53:05', 1],
        [39.984198, 116.319322, '2008-10-23 05:53:06', 1],
        [39.984224, 116.319402, '2008-10-23 05:53:11', 1],
        [39.984224, 116.319402, '2008-10-23 05:53:11', 1],
        [39.984224, 116.319402, '2008-10-23 05:53:11', 1],
        [39.984224, 116.319402, '2008-10-23 05:53:11', 1]
    ]
    move_df = MoveDataFrame(data=list_data, latitude="lat", longitude="lon", datetime="datetime", traj_id="id")
    move_df.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }

        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>lat</th>
          <th>lon</th>
          <th>datetime</th>
          <th>id</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>39.984094</td>
          <td>116.319236</td>
          <td>2008-10-23 05:53:05</td>
          <td>1</td>
        </tr>
        <tr>
          <th>1</th>
          <td>39.984198</td>
          <td>116.319322</td>
          <td>2008-10-23 05:53:06</td>
          <td>1</td>
        </tr>
        <tr>
          <th>2</th>
          <td>39.984224</td>
          <td>116.319402</td>
          <td>2008-10-23 05:53:11</td>
          <td>1</td>
        </tr>
        <tr>
          <th>3</th>
          <td>39.984224</td>
          <td>116.319402</td>
          <td>2008-10-23 05:53:11</td>
          <td>1</td>
        </tr>
        <tr>
          <th>4</th>
          <td>39.984224</td>
          <td>116.319402</td>
          <td>2008-10-23 05:53:11</td>
          <td>1</td>
        </tr>
      </tbody>
    </table>
    </div>



From a dict
~~~~~~~~~~~

.. code:: python

    dict_data = {
        'lat': [39.984198, 39.984224, 39.984094],
        'lon': [116.319402, 116.319322, 116.319402],
        'datetime': ['2008-10-23 05:53:11', '2008-10-23 05:53:06', '2008-10-23 05:53:06']
    }

    move_df = MoveDataFrame(data=dict_data, latitude="lat", longitude="lon", datetime="datetime", traj_id="id")
    move_df.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }

        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>lat</th>
          <th>lon</th>
          <th>datetime</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>39.984198</td>
          <td>116.319402</td>
          <td>2008-10-23 05:53:11</td>
        </tr>
        <tr>
          <th>1</th>
          <td>39.984224</td>
          <td>116.319322</td>
          <td>2008-10-23 05:53:06</td>
        </tr>
        <tr>
          <th>2</th>
          <td>39.984094</td>
          <td>116.319402</td>
          <td>2008-10-23 05:53:06</td>
        </tr>
      </tbody>
    </table>
    </div>



From a DataFrame Pandas
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    import pandas as pd

    df = pd.read_csv('geolife_sample.csv', parse_dates=['datetime'])
    move_df = MoveDataFrame(data=df, latitude="lat", longitude="lon", datetime="datetime")

    move_df.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }

        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>lat</th>
          <th>lon</th>
          <th>datetime</th>
          <th>id</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>39.984094</td>
          <td>116.319236</td>
          <td>2008-10-23 05:53:05</td>
          <td>1</td>
        </tr>
        <tr>
          <th>1</th>
          <td>39.984198</td>
          <td>116.319322</td>
          <td>2008-10-23 05:53:06</td>
          <td>1</td>
        </tr>
        <tr>
          <th>2</th>
          <td>39.984224</td>
          <td>116.319402</td>
          <td>2008-10-23 05:53:11</td>
          <td>1</td>
        </tr>
        <tr>
          <th>3</th>
          <td>39.984211</td>
          <td>116.319389</td>
          <td>2008-10-23 05:53:16</td>
          <td>1</td>
        </tr>
        <tr>
          <th>4</th>
          <td>39.984217</td>
          <td>116.319422</td>
          <td>2008-10-23 05:53:21</td>
          <td>1</td>
        </tr>
      </tbody>
    </table>
    </div>



From a file
~~~~~~~~~~~

.. code:: python

    move_df = pymove.read_csv('geolife_sample.csv')
    move_df.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }

        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>lat</th>
          <th>lon</th>
          <th>datetime</th>
          <th>id</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>39.984094</td>
          <td>116.319236</td>
          <td>2008-10-23 05:53:05</td>
          <td>1</td>
        </tr>
        <tr>
          <th>1</th>
          <td>39.984198</td>
          <td>116.319322</td>
          <td>2008-10-23 05:53:06</td>
          <td>1</td>
        </tr>
        <tr>
          <th>2</th>
          <td>39.984224</td>
          <td>116.319402</td>
          <td>2008-10-23 05:53:11</td>
          <td>1</td>
        </tr>
        <tr>
          <th>3</th>
          <td>39.984211</td>
          <td>116.319389</td>
          <td>2008-10-23 05:53:16</td>
          <td>1</td>
        </tr>
        <tr>
          <th>4</th>
          <td>39.984217</td>
          <td>116.319422</td>
          <td>2008-10-23 05:53:21</td>
          <td>1</td>
        </tr>
      </tbody>
    </table>
    </div>



Cool, huh? The default flag is Pandas. Look that:

.. code:: python

    type(move_df)




.. parsed-literal::

    pymove.core.pandas.PandasMoveDataFrame



Let’s try creating one with Dask!

.. code:: python

    move_df = pymove.read_csv('geolife_sample.csv', type_='dask')
    move_df.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }

        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>lat</th>
          <th>lon</th>
          <th>datetime</th>
          <th>id</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>39.984094</td>
          <td>116.319236</td>
          <td>2008-10-23 05:53:05</td>
          <td>1</td>
        </tr>
        <tr>
          <th>1</th>
          <td>39.984198</td>
          <td>116.319322</td>
          <td>2008-10-23 05:53:06</td>
          <td>1</td>
        </tr>
        <tr>
          <th>2</th>
          <td>39.984224</td>
          <td>116.319402</td>
          <td>2008-10-23 05:53:11</td>
          <td>1</td>
        </tr>
        <tr>
          <th>3</th>
          <td>39.984211</td>
          <td>116.319389</td>
          <td>2008-10-23 05:53:16</td>
          <td>1</td>
        </tr>
        <tr>
          <th>4</th>
          <td>39.984217</td>
          <td>116.319422</td>
          <td>2008-10-23 05:53:21</td>
          <td>1</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    type(move_df)




.. parsed-literal::

    pymove.core.dask.DaskMoveDataFrame



What’s in MoveDataFrame?
------------------------

The MoveDataFrame stores the following information:

.. code:: python

    orig_df = pymove.read_csv('geolife_sample.csv')
    move_df = orig_df.copy()

1. The kind of data he was instantiated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    move_df.get_type()




.. parsed-literal::

    'pandas'



.. code:: python

    move_df.columns




.. parsed-literal::

    Index(['lat', 'lon', 'datetime', 'id'], dtype='object')



.. code:: python

    move_df.dtypes




.. parsed-literal::

    lat                float64
    lon                float64
    datetime    datetime64[ns]
    id                   int64
    dtype: object



In addition to these attributes, we have some functions that allow us
to:

1. View trajectory information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    move_df.get_bbox()




.. parsed-literal::

    (22.147577, 113.548843, 41.132062, 121.156224)



.. code:: python

    move_df.show_trajectories_info()


.. parsed-literal::


    ====================== INFORMATION ABOUT DATASET ======================

    Number of Points: 217653

    Number of IDs objects: 2

    Start Date:2008-10-23 05:53:05     End Date:2009-03-19 05:46:37

    Bounding Box:(22.147577, 113.548843, 41.132062, 121.156224)


    =======================================================================



2. View the number of users
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    move_df.get_users_number()




.. parsed-literal::

    1



3. Transform our data to
~~~~~~~~~~~~~~~~~~~~~~~~

a. Numpy
^^^^^^^^

.. code:: python

    move_df.to_numpy()




.. parsed-literal::

    array([[39.984094, 116.319236, Timestamp('2008-10-23 05:53:05'), 1],
           [39.984198, 116.319322, Timestamp('2008-10-23 05:53:06'), 1],
           [39.984224, 116.319402, Timestamp('2008-10-23 05:53:11'), 1],
           ...,
           [39.999945, 116.327394, Timestamp('2009-03-19 05:46:12'), 5],
           [40.000015, 116.327433, Timestamp('2009-03-19 05:46:17'), 5],
           [39.999978, 116.32746, Timestamp('2009-03-19 05:46:37'), 5]],
          dtype=object)



b. Dicts
^^^^^^^^

.. code:: python

    dict_data = move_df.to_dict()
    dict_data.keys()




.. parsed-literal::

    dict_keys(['lat', 'lon', 'datetime', 'id'])



c. DataFrames
^^^^^^^^^^^^^

.. code:: python

    df = move_df.to_data_frame()
    print(type(move_df))
    print(type(df))
    df


.. parsed-literal::

    <class 'pymove.core.pandas.PandasMoveDataFrame'>
    <class 'pandas.core.frame.DataFrame'>




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }

        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>lat</th>
          <th>lon</th>
          <th>datetime</th>
          <th>id</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>39.984094</td>
          <td>116.319236</td>
          <td>2008-10-23 05:53:05</td>
          <td>1</td>
        </tr>
        <tr>
          <th>1</th>
          <td>39.984198</td>
          <td>116.319322</td>
          <td>2008-10-23 05:53:06</td>
          <td>1</td>
        </tr>
        <tr>
          <th>2</th>
          <td>39.984224</td>
          <td>116.319402</td>
          <td>2008-10-23 05:53:11</td>
          <td>1</td>
        </tr>
        <tr>
          <th>3</th>
          <td>39.984211</td>
          <td>116.319389</td>
          <td>2008-10-23 05:53:16</td>
          <td>1</td>
        </tr>
        <tr>
          <th>4</th>
          <td>39.984217</td>
          <td>116.319422</td>
          <td>2008-10-23 05:53:21</td>
          <td>1</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>217648</th>
          <td>39.999896</td>
          <td>116.327290</td>
          <td>2009-03-19 05:46:02</td>
          <td>5</td>
        </tr>
        <tr>
          <th>217649</th>
          <td>39.999899</td>
          <td>116.327352</td>
          <td>2009-03-19 05:46:07</td>
          <td>5</td>
        </tr>
        <tr>
          <th>217650</th>
          <td>39.999945</td>
          <td>116.327394</td>
          <td>2009-03-19 05:46:12</td>
          <td>5</td>
        </tr>
        <tr>
          <th>217651</th>
          <td>40.000015</td>
          <td>116.327433</td>
          <td>2009-03-19 05:46:17</td>
          <td>5</td>
        </tr>
        <tr>
          <th>217652</th>
          <td>39.999978</td>
          <td>116.327460</td>
          <td>2009-03-19 05:46:37</td>
          <td>5</td>
        </tr>
      </tbody>
    </table>
    <p>217653 rows × 4 columns</p>
    </div>



4. And even switch from a Pandas to Dask and back again!
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    new_move = move_df.convert_to('dask')
    print(type(new_move))
    move_df = new_move.convert_to('pandas')
    print(type(move_df))


.. parsed-literal::

    <class 'pymove.core.dask.DaskMoveDataFrame'>
    <class 'pymove.core.pandas.PandasMoveDataFrame'>


5. You can also write files with
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    move_df.write_file('move_df_write_file.txt')

.. code:: python

    move_df.to_csv('move_data.csv')

6. Create a virtual grid
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    move_df.to_grid(8)


.. parsed-literal::


    Creating a virtual grid without polygons
    ...cell size by degree: 7.218478943256657e-05
    ...grid_size_lat_y:262999
    grid_size_lon_x:105388

    ..A virtual grid was created




.. parsed-literal::

    lon_min_x: 113.548843
    lat_min_y: 22.147577
    grid_size_lat_y: 262999
    grid_size_lon_x: 105388
    cell_size_by_degree: 7.218478943256657e-05



7. View the information of the last MoveDataFrame operation: operation name, operation time and memory use
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    move_df.last_operation




.. parsed-literal::

    {'name': 'to_grid', 'time in seconds': 0.008565902709960938, 'memory': '0.0 B'}



8. Get data bound box
~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    move_df.get_bbox()




.. parsed-literal::

    (22.147577, 113.548843, 41.132062, 121.156224)



9. Create new columns:
~~~~~~~~~~~~~~~~~~~~~~

a. ``tid``: trajectory id based on Id and datetime
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    move_df.generate_tid_based_on_id_datetime()
    move_df.head()


.. parsed-literal::


    Creating or updating tid feature...

    ...Sorting by id and datetime to increase performance


    ...tid feature was created...





.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }

        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>lat</th>
          <th>lon</th>
          <th>datetime</th>
          <th>id</th>
          <th>tid</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>39.984094</td>
          <td>116.319236</td>
          <td>2008-10-23 05:53:05</td>
          <td>1</td>
          <td>12008102305</td>
        </tr>
        <tr>
          <th>1</th>
          <td>39.984198</td>
          <td>116.319322</td>
          <td>2008-10-23 05:53:06</td>
          <td>1</td>
          <td>12008102305</td>
        </tr>
        <tr>
          <th>2</th>
          <td>39.984224</td>
          <td>116.319402</td>
          <td>2008-10-23 05:53:11</td>
          <td>1</td>
          <td>12008102305</td>
        </tr>
        <tr>
          <th>3</th>
          <td>39.984211</td>
          <td>116.319389</td>
          <td>2008-10-23 05:53:16</td>
          <td>1</td>
          <td>12008102305</td>
        </tr>
        <tr>
          <th>4</th>
          <td>39.984217</td>
          <td>116.319422</td>
          <td>2008-10-23 05:53:21</td>
          <td>1</td>
          <td>12008102305</td>
        </tr>
      </tbody>
    </table>
    </div>



b. ``date``: extract date on datetime
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    move_df.generate_date_features()
    move_df.head()


.. parsed-literal::

    Creating date features...
    ..Date features was created...





.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }

        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>lat</th>
          <th>lon</th>
          <th>datetime</th>
          <th>id</th>
          <th>tid</th>
          <th>date</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>39.984094</td>
          <td>116.319236</td>
          <td>2008-10-23 05:53:05</td>
          <td>1</td>
          <td>12008102305</td>
          <td>2008-10-23</td>
        </tr>
        <tr>
          <th>1</th>
          <td>39.984198</td>
          <td>116.319322</td>
          <td>2008-10-23 05:53:06</td>
          <td>1</td>
          <td>12008102305</td>
          <td>2008-10-23</td>
        </tr>
        <tr>
          <th>2</th>
          <td>39.984224</td>
          <td>116.319402</td>
          <td>2008-10-23 05:53:11</td>
          <td>1</td>
          <td>12008102305</td>
          <td>2008-10-23</td>
        </tr>
        <tr>
          <th>3</th>
          <td>39.984211</td>
          <td>116.319389</td>
          <td>2008-10-23 05:53:16</td>
          <td>1</td>
          <td>12008102305</td>
          <td>2008-10-23</td>
        </tr>
        <tr>
          <th>4</th>
          <td>39.984217</td>
          <td>116.319422</td>
          <td>2008-10-23 05:53:21</td>
          <td>1</td>
          <td>12008102305</td>
          <td>2008-10-23</td>
        </tr>
      </tbody>
    </table>
    </div>



c. ``hour``: extract hour on datetime
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    move_df.generate_hour_features()
    move_df.head()


.. parsed-literal::


    Creating or updating a feature for hour...

    ...Hour feature was created...





.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }

        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>lat</th>
          <th>lon</th>
          <th>datetime</th>
          <th>id</th>
          <th>tid</th>
          <th>date</th>
          <th>hour</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>39.984094</td>
          <td>116.319236</td>
          <td>2008-10-23 05:53:05</td>
          <td>1</td>
          <td>12008102305</td>
          <td>2008-10-23</td>
          <td>5</td>
        </tr>
        <tr>
          <th>1</th>
          <td>39.984198</td>
          <td>116.319322</td>
          <td>2008-10-23 05:53:06</td>
          <td>1</td>
          <td>12008102305</td>
          <td>2008-10-23</td>
          <td>5</td>
        </tr>
        <tr>
          <th>2</th>
          <td>39.984224</td>
          <td>116.319402</td>
          <td>2008-10-23 05:53:11</td>
          <td>1</td>
          <td>12008102305</td>
          <td>2008-10-23</td>
          <td>5</td>
        </tr>
        <tr>
          <th>3</th>
          <td>39.984211</td>
          <td>116.319389</td>
          <td>2008-10-23 05:53:16</td>
          <td>1</td>
          <td>12008102305</td>
          <td>2008-10-23</td>
          <td>5</td>
        </tr>
        <tr>
          <th>4</th>
          <td>39.984217</td>
          <td>116.319422</td>
          <td>2008-10-23 05:53:21</td>
          <td>1</td>
          <td>12008102305</td>
          <td>2008-10-23</td>
          <td>5</td>
        </tr>
      </tbody>
    </table>
    </div>



d. ``day``: day of the week from datatime.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    move_df.generate_day_of_the_week_features()
    move_df.head()


.. parsed-literal::


    Creating or updating day of the week feature...

    ...the day of the week feature was created...





.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }

        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>lat</th>
          <th>lon</th>
          <th>datetime</th>
          <th>id</th>
          <th>tid</th>
          <th>date</th>
          <th>hour</th>
          <th>day</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>39.984094</td>
          <td>116.319236</td>
          <td>2008-10-23 05:53:05</td>
          <td>1</td>
          <td>12008102305</td>
          <td>2008-10-23</td>
          <td>5</td>
          <td>Thursday</td>
        </tr>
        <tr>
          <th>1</th>
          <td>39.984198</td>
          <td>116.319322</td>
          <td>2008-10-23 05:53:06</td>
          <td>1</td>
          <td>12008102305</td>
          <td>2008-10-23</td>
          <td>5</td>
          <td>Thursday</td>
        </tr>
        <tr>
          <th>2</th>
          <td>39.984224</td>
          <td>116.319402</td>
          <td>2008-10-23 05:53:11</td>
          <td>1</td>
          <td>12008102305</td>
          <td>2008-10-23</td>
          <td>5</td>
          <td>Thursday</td>
        </tr>
        <tr>
          <th>3</th>
          <td>39.984211</td>
          <td>116.319389</td>
          <td>2008-10-23 05:53:16</td>
          <td>1</td>
          <td>12008102305</td>
          <td>2008-10-23</td>
          <td>5</td>
          <td>Thursday</td>
        </tr>
        <tr>
          <th>4</th>
          <td>39.984217</td>
          <td>116.319422</td>
          <td>2008-10-23 05:53:21</td>
          <td>1</td>
          <td>12008102305</td>
          <td>2008-10-23</td>
          <td>5</td>
          <td>Thursday</td>
        </tr>
      </tbody>
    </table>
    </div>



e. ``period``: time of day or period from datatime.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    move_df.generate_time_of_day_features()
    move_df.head()


.. parsed-literal::


    Creating or updating period feature
    ...Early morning from 0H to 6H
    ...Morning from 6H to 12H
    ...Afternoon from 12H to 18H
    ...Evening from 18H to 24H

    ...the period of day feature was created




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }

        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>lat</th>
          <th>lon</th>
          <th>datetime</th>
          <th>id</th>
          <th>tid</th>
          <th>date</th>
          <th>hour</th>
          <th>day</th>
          <th>period</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>39.984094</td>
          <td>116.319236</td>
          <td>2008-10-23 05:53:05</td>
          <td>1</td>
          <td>12008102305</td>
          <td>2008-10-23</td>
          <td>5</td>
          <td>Thursday</td>
          <td>Early morning</td>
        </tr>
        <tr>
          <th>1</th>
          <td>39.984198</td>
          <td>116.319322</td>
          <td>2008-10-23 05:53:06</td>
          <td>1</td>
          <td>12008102305</td>
          <td>2008-10-23</td>
          <td>5</td>
          <td>Thursday</td>
          <td>Early morning</td>
        </tr>
        <tr>
          <th>2</th>
          <td>39.984224</td>
          <td>116.319402</td>
          <td>2008-10-23 05:53:11</td>
          <td>1</td>
          <td>12008102305</td>
          <td>2008-10-23</td>
          <td>5</td>
          <td>Thursday</td>
          <td>Early morning</td>
        </tr>
        <tr>
          <th>3</th>
          <td>39.984211</td>
          <td>116.319389</td>
          <td>2008-10-23 05:53:16</td>
          <td>1</td>
          <td>12008102305</td>
          <td>2008-10-23</td>
          <td>5</td>
          <td>Thursday</td>
          <td>Early morning</td>
        </tr>
        <tr>
          <th>4</th>
          <td>39.984217</td>
          <td>116.319422</td>
          <td>2008-10-23 05:53:21</td>
          <td>1</td>
          <td>12008102305</td>
          <td>2008-10-23</td>
          <td>5</td>
          <td>Thursday</td>
          <td>Early morning</td>
        </tr>
      </tbody>
    </table>
    </div>



f. ``dist_to_prev``, ``time_to_prev``, ``speed_to_prev``: create features of distance, time and speed to an GPS point P (lat, lon).
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    move_df = orig_df.copy()
    move_df.generate_dist_time_speed_features()
    move_df.head()


.. parsed-literal::

    ...Sorting by id and datetime to increase performance

    ...Set id as index to a higher performance


    Creating or updating distance, time and speed features in meters by seconds




.. parsed-literal::

    VBox(children=(HTML(value=''), IntProgress(value=0, max=2)))


.. parsed-literal::

    ...Reset index...





.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }

        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>id</th>
          <th>lat</th>
          <th>lon</th>
          <th>datetime</th>
          <th>dist_to_prev</th>
          <th>time_to_prev</th>
          <th>speed_to_prev</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1</td>
          <td>39.984094</td>
          <td>116.319236</td>
          <td>2008-10-23 05:53:05</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1</td>
          <td>39.984198</td>
          <td>116.319322</td>
          <td>2008-10-23 05:53:06</td>
          <td>13.690153</td>
          <td>1.0</td>
          <td>13.690153</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1</td>
          <td>39.984224</td>
          <td>116.319402</td>
          <td>2008-10-23 05:53:11</td>
          <td>7.403788</td>
          <td>5.0</td>
          <td>1.480758</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1</td>
          <td>39.984211</td>
          <td>116.319389</td>
          <td>2008-10-23 05:53:16</td>
          <td>1.821083</td>
          <td>5.0</td>
          <td>0.364217</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1</td>
          <td>39.984217</td>
          <td>116.319422</td>
          <td>2008-10-23 05:53:21</td>
          <td>2.889671</td>
          <td>5.0</td>
          <td>0.577934</td>
        </tr>
      </tbody>
    </table>
    </div>



g. ``dist_to_prev``, ``dist_to_next``, ``dist_prev_to_next`` : three distance in meters to an GPS point P (lat, lon).
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    move_df = orig_df.copy()
    move_df.generate_dist_features()
    move_df.head()


.. parsed-literal::

    ...Sorting by id and datetime to increase performance

    ...Set id as index to a higher performance


    Creating or updating distance features in meters...




.. parsed-literal::

    VBox(children=(HTML(value=''), IntProgress(value=0, max=2)))


.. parsed-literal::

    ...Reset index...





.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }

        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>id</th>
          <th>lat</th>
          <th>lon</th>
          <th>datetime</th>
          <th>dist_to_prev</th>
          <th>dist_to_next</th>
          <th>dist_prev_to_next</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1</td>
          <td>39.984094</td>
          <td>116.319236</td>
          <td>2008-10-23 05:53:05</td>
          <td>NaN</td>
          <td>13.690153</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1</td>
          <td>39.984198</td>
          <td>116.319322</td>
          <td>2008-10-23 05:53:06</td>
          <td>13.690153</td>
          <td>7.403788</td>
          <td>20.223428</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1</td>
          <td>39.984224</td>
          <td>116.319402</td>
          <td>2008-10-23 05:53:11</td>
          <td>7.403788</td>
          <td>1.821083</td>
          <td>5.888579</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1</td>
          <td>39.984211</td>
          <td>116.319389</td>
          <td>2008-10-23 05:53:16</td>
          <td>1.821083</td>
          <td>2.889671</td>
          <td>1.873356</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1</td>
          <td>39.984217</td>
          <td>116.319422</td>
          <td>2008-10-23 05:53:21</td>
          <td>2.889671</td>
          <td>66.555997</td>
          <td>68.727260</td>
        </tr>
      </tbody>
    </table>
    </div>



h. ``time_to_prev``, ``time_to_next``, ``time_prev_to_next`` : three time in seconds to an GPS point P (lat, lon).
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    move_df = orig_df.copy()
    move_df.generate_time_features()
    move_df.head()


.. parsed-literal::

    ...Sorting by id and datetime to increase performance

    ...Set id as index to a higher performance


    Creating or updating time features seconds




.. parsed-literal::

    VBox(children=(HTML(value=''), IntProgress(value=0, max=2)))


.. parsed-literal::

    ...Reset index...





.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }

        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>id</th>
          <th>lat</th>
          <th>lon</th>
          <th>datetime</th>
          <th>time_to_prev</th>
          <th>time_to_next</th>
          <th>time_prev_to_next</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1</td>
          <td>39.984094</td>
          <td>116.319236</td>
          <td>2008-10-23 05:53:05</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1</td>
          <td>39.984198</td>
          <td>116.319322</td>
          <td>2008-10-23 05:53:06</td>
          <td>1.0</td>
          <td>5.0</td>
          <td>6.0</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1</td>
          <td>39.984224</td>
          <td>116.319402</td>
          <td>2008-10-23 05:53:11</td>
          <td>5.0</td>
          <td>5.0</td>
          <td>10.0</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1</td>
          <td>39.984211</td>
          <td>116.319389</td>
          <td>2008-10-23 05:53:16</td>
          <td>5.0</td>
          <td>5.0</td>
          <td>10.0</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1</td>
          <td>39.984217</td>
          <td>116.319422</td>
          <td>2008-10-23 05:53:21</td>
          <td>5.0</td>
          <td>2.0</td>
          <td>7.0</td>
        </tr>
      </tbody>
    </table>
    </div>



i. ``speed_to_prev``, ``speed_to_next``, ``speed_prev_to_next`` : three speed in meters by seconds to an GPS point P (lat, lon).
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    move_df = orig_df.copy()
    move_df.generate_speed_features()
    move_df.head()


.. parsed-literal::


    Creating or updating speed features meters by seconds

    ...Sorting by id and datetime to increase performance

    ...Set id as index to a higher performance


    Creating or updating distance features in meters...




.. parsed-literal::

    VBox(children=(HTML(value=''), IntProgress(value=0, max=2)))


.. parsed-literal::

    ...Reset index...

    ...Sorting by id and datetime to increase performance

    ...Set id as index to a higher performance


    Creating or updating time features seconds




.. parsed-literal::

    VBox(children=(HTML(value=''), IntProgress(value=0, max=2)))


.. parsed-literal::

    ...Reset index...

    ...Sorting by id and datetime to increase performance

    ...Set id as index to a higher performance

    ...Reset index...





.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }

        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>id</th>
          <th>lat</th>
          <th>lon</th>
          <th>datetime</th>
          <th>speed_to_prev</th>
          <th>speed_to_next</th>
          <th>speed_prev_to_next</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1</td>
          <td>39.984094</td>
          <td>116.319236</td>
          <td>2008-10-23 05:53:05</td>
          <td>NaN</td>
          <td>13.690153</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1</td>
          <td>39.984198</td>
          <td>116.319322</td>
          <td>2008-10-23 05:53:06</td>
          <td>13.690153</td>
          <td>1.480758</td>
          <td>3.515657</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1</td>
          <td>39.984224</td>
          <td>116.319402</td>
          <td>2008-10-23 05:53:11</td>
          <td>1.480758</td>
          <td>0.364217</td>
          <td>0.922487</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1</td>
          <td>39.984211</td>
          <td>116.319389</td>
          <td>2008-10-23 05:53:16</td>
          <td>0.364217</td>
          <td>0.577934</td>
          <td>0.471075</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1</td>
          <td>39.984217</td>
          <td>116.319422</td>
          <td>2008-10-23 05:53:21</td>
          <td>0.577934</td>
          <td>33.277998</td>
          <td>9.920810</td>
        </tr>
      </tbody>
    </table>
    </div>



j. ``situation``: column with move and stop points by radius.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    move_df = orig_df.copy()
    move_df.generate_move_and_stop_by_radius()
    move_df.head()


.. parsed-literal::

    ...Sorting by id and datetime to increase performance

    ...Set id as index to a higher performance


    Creating or updating distance features in meters...




.. parsed-literal::

    VBox(children=(HTML(value=''), IntProgress(value=0, max=2)))


.. parsed-literal::

    ...Reset index...


    Creating or updating features MOVE and STOPS...


    ....There are 14691 stops to this parameters





.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }

        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>id</th>
          <th>lat</th>
          <th>lon</th>
          <th>datetime</th>
          <th>dist_to_prev</th>
          <th>dist_to_next</th>
          <th>dist_prev_to_next</th>
          <th>situation</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1</td>
          <td>39.984094</td>
          <td>116.319236</td>
          <td>2008-10-23 05:53:05</td>
          <td>NaN</td>
          <td>13.690153</td>
          <td>NaN</td>
          <td>nan</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1</td>
          <td>39.984198</td>
          <td>116.319322</td>
          <td>2008-10-23 05:53:06</td>
          <td>13.690153</td>
          <td>7.403788</td>
          <td>20.223428</td>
          <td>move</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1</td>
          <td>39.984224</td>
          <td>116.319402</td>
          <td>2008-10-23 05:53:11</td>
          <td>7.403788</td>
          <td>1.821083</td>
          <td>5.888579</td>
          <td>move</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1</td>
          <td>39.984211</td>
          <td>116.319389</td>
          <td>2008-10-23 05:53:16</td>
          <td>1.821083</td>
          <td>2.889671</td>
          <td>1.873356</td>
          <td>move</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1</td>
          <td>39.984217</td>
          <td>116.319422</td>
          <td>2008-10-23 05:53:21</td>
          <td>2.889671</td>
          <td>66.555997</td>
          <td>68.727260</td>
          <td>move</td>
        </tr>
      </tbody>
    </table>
    </div>



9. Get time difference between max and min datetime in trajectory data.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    move_df.time_interval()




.. parsed-literal::

    Timedelta('146 days 23:53:32')



And that’s it! See upcoming notebooks to learn more about what PyMove can do!
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
