02 - Exploring Preprocessing
============================

Data preprocessing is a set of activities performed to prepare data for
future analysis and data mining activities.

Load data from file
-------------------

The dataset used in this tutorial is GeoLife GPS Trajectories. Available
in https://www.microsoft.com/en-us/download/details.aspx?id=52367

.. code:: python

    import pandas as pd
    import numpy as np
    from pymove import MoveDataFrame
    from pymove import read_csv

.. code:: python

    df = pd.read_csv('geolife_sample.csv', parse_dates=['datetime'])
    df.head()




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

    df_move = MoveDataFrame(df, latitude="lat", longitude="lon", datetime="datetime")

.. code:: python

    df_move.show_trajectories_info()


.. parsed-literal::


    ====================== INFORMATION ABOUT DATASET ======================

    Number of Points: 217653

    Number of IDs objects: 2

    Start Date:2008-10-23 05:53:05     End Date:2009-03-19 05:46:37

    Bounding Box:(22.147577, 113.548843, 41.132062, 121.156224)


    =======================================================================



Filtering
---------

The filters module provides functions to perform different types of data
filtering.

Importing the module:

.. code:: python

    from pymove import filters
    df_move = read_csv('geolife_sample.csv')

A bounding box (usually shortened to bbox) is an area defined by two
longitudes and two latitudes. The function by_bbox, filters points of
the trajectories according to a especified bounding box.

.. code:: python

    bbox = (22.147577, 113.54884299999999, 41.132062, 121.156224)
    filt_df = filters.by_bbox(df_move, bbox)
    filt_df.head()




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



by_datetime function filters point trajectories according to the time
specified by the parameters: start_datetime and end_datetime.

.. code:: python

    filters.by_datetime(df_move, start_datetime = "2009-03-19 05:45:37", end_datetime = "2009-03-19 05:46:17")




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
          <th>217643</th>
          <td>40.000205</td>
          <td>116.327173</td>
          <td>2009-03-19 05:45:37</td>
          <td>5</td>
        </tr>
        <tr>
          <th>217644</th>
          <td>40.000128</td>
          <td>116.327171</td>
          <td>2009-03-19 05:45:42</td>
          <td>5</td>
        </tr>
        <tr>
          <th>217645</th>
          <td>40.000069</td>
          <td>116.327179</td>
          <td>2009-03-19 05:45:47</td>
          <td>5</td>
        </tr>
        <tr>
          <th>217646</th>
          <td>40.000001</td>
          <td>116.327219</td>
          <td>2009-03-19 05:45:52</td>
          <td>5</td>
        </tr>
        <tr>
          <th>217647</th>
          <td>39.999919</td>
          <td>116.327211</td>
          <td>2009-03-19 05:45:57</td>
          <td>5</td>
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
      </tbody>
    </table>
    </div>



by label function filters trajectories points according to specified
value and column label, set by value and label_name respectively.

.. code:: python

    filters.by_label(df_move, value = 116.327219, label_name = "lon").head()




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
          <th>3066</th>
          <td>39.979160</td>
          <td>116.327219</td>
          <td>2008-10-24 06:34:27</td>
          <td>1</td>
        </tr>
        <tr>
          <th>13911</th>
          <td>39.975424</td>
          <td>116.327219</td>
          <td>2008-10-26 08:18:06</td>
          <td>1</td>
        </tr>
        <tr>
          <th>16396</th>
          <td>39.980411</td>
          <td>116.327219</td>
          <td>2008-10-27 00:30:47</td>
          <td>1</td>
        </tr>
        <tr>
          <th>33935</th>
          <td>39.975832</td>
          <td>116.327219</td>
          <td>2008-11-05 11:04:04</td>
          <td>1</td>
        </tr>
        <tr>
          <th>41636</th>
          <td>39.976990</td>
          <td>116.327219</td>
          <td>2008-11-07 10:34:41</td>
          <td>1</td>
        </tr>
      </tbody>
    </table>
    </div>



by_id function filters trajectories points according to especified
trajectory id.

.. code:: python

    filters.by_id(df_move, id_=5).head()




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
          <th>108607</th>
          <td>40.004155</td>
          <td>116.321337</td>
          <td>2008-10-24 04:12:30</td>
          <td>5</td>
        </tr>
        <tr>
          <th>108608</th>
          <td>40.003834</td>
          <td>116.321462</td>
          <td>2008-10-24 04:12:35</td>
          <td>5</td>
        </tr>
        <tr>
          <th>108609</th>
          <td>40.003783</td>
          <td>116.321431</td>
          <td>2008-10-24 04:12:40</td>
          <td>5</td>
        </tr>
        <tr>
          <th>108610</th>
          <td>40.003690</td>
          <td>116.321429</td>
          <td>2008-10-24 04:12:45</td>
          <td>5</td>
        </tr>
        <tr>
          <th>108611</th>
          <td>40.003589</td>
          <td>116.321427</td>
          <td>2008-10-24 04:12:50</td>
          <td>5</td>
        </tr>
      </tbody>
    </table>
    </div>



A tid is the result of concatenation between the id and date of a
trajectory. The by_tid function filters trajectory points according to
the tid specified by the tid\_ parameter.

.. code:: python

    df_move.generate_tid_based_on_id_datetime()
    filters.by_tid(df_move, "12008102305").head()


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



outliers function filters trajectories points that are outliers.

.. code:: python

    outliers_points = filters.outliers(df_move)
    outliers_points.head()


.. parsed-literal::

    ...Sorting by id and datetime to increase performance

    ...Set id as index to a higher performance


    Creating or updating distance features in meters...




.. parsed-literal::

    VBox(children=(HTML(value=''), IntProgress(value=0, max=2)))


.. parsed-literal::

    ...Reset index...

    ...Filtering jumps





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
          <th>tid</th>
          <th>dist_to_prev</th>
          <th>dist_to_next</th>
          <th>dist_prev_to_next</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>148</th>
          <td>1</td>
          <td>39.970511</td>
          <td>116.341455</td>
          <td>2008-10-23 10:32:53</td>
          <td>12008102310</td>
          <td>1452.319115</td>
          <td>1470.641291</td>
          <td>71.088460</td>
        </tr>
        <tr>
          <th>338</th>
          <td>1</td>
          <td>39.995042</td>
          <td>116.326465</td>
          <td>2008-10-23 10:44:24</td>
          <td>12008102310</td>
          <td>10.801860</td>
          <td>10.274331</td>
          <td>1.465144</td>
        </tr>
        <tr>
          <th>8133</th>
          <td>1</td>
          <td>39.991075</td>
          <td>116.188395</td>
          <td>2008-10-25 08:20:19</td>
          <td>12008102508</td>
          <td>5.090766</td>
          <td>6.247860</td>
          <td>1.295191</td>
        </tr>
        <tr>
          <th>10175</th>
          <td>1</td>
          <td>40.015169</td>
          <td>116.311045</td>
          <td>2008-10-25 23:40:12</td>
          <td>12008102523</td>
          <td>23.454754</td>
          <td>24.899678</td>
          <td>3.766959</td>
        </tr>
        <tr>
          <th>13849</th>
          <td>1</td>
          <td>39.977157</td>
          <td>116.327151</td>
          <td>2008-10-26 08:13:53</td>
          <td>12008102608</td>
          <td>11.212682</td>
          <td>10.221164</td>
          <td>1.004375</td>
        </tr>
      </tbody>
    </table>
    </div>



clean_consecutive_duplicates function removes consecutives duplicate
rows of the Dataframe. Optionally only certaind columns can be consider,
this is defined by the parameter subset, in this example only the lat
column is considered.

.. code:: python

    filtered_df = filters.clean_consecutive_duplicates(df_move, subset = ["lat"])
    len(filtered_df)




.. parsed-literal::

    196142



clean_nan_values function removes missing values from the dataframe.

clean_gps_jumps_by_distance function removes from the dataframe the
trajectories points that are outliers.

.. code:: python

    filters.clean_gps_jumps_by_distance(df_move)


.. parsed-literal::


    Cleaning gps jumps by distance to jump_coefficient 3.0...

    ...Filtering jumps

    ...Dropping 383 rows of gps points

    ...Rows before: 217653, Rows after:217270, Sum drop:383

    ...Filtering jumps

    383 GPS points were dropped




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
          <th>tid</th>
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
          <td>12008102305</td>
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
          <td>12008102305</td>
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
          <td>12008102305</td>
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
          <td>12008102305</td>
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
          <td>12008102305</td>
          <td>2.889671</td>
          <td>66.555997</td>
          <td>68.727260</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>217648</th>
          <td>5</td>
          <td>39.999896</td>
          <td>116.327290</td>
          <td>2009-03-19 05:46:02</td>
          <td>52009031905</td>
          <td>7.198855</td>
          <td>5.291709</td>
          <td>12.214590</td>
        </tr>
        <tr>
          <th>217649</th>
          <td>5</td>
          <td>39.999899</td>
          <td>116.327352</td>
          <td>2009-03-19 05:46:07</td>
          <td>52009031905</td>
          <td>5.291709</td>
          <td>6.241949</td>
          <td>10.400206</td>
        </tr>
        <tr>
          <th>217650</th>
          <td>5</td>
          <td>39.999945</td>
          <td>116.327394</td>
          <td>2009-03-19 05:46:12</td>
          <td>52009031905</td>
          <td>6.241949</td>
          <td>8.462920</td>
          <td>14.628012</td>
        </tr>
        <tr>
          <th>217651</th>
          <td>5</td>
          <td>40.000015</td>
          <td>116.327433</td>
          <td>2009-03-19 05:46:17</td>
          <td>52009031905</td>
          <td>8.462920</td>
          <td>4.713399</td>
          <td>6.713456</td>
        </tr>
        <tr>
          <th>217652</th>
          <td>5</td>
          <td>39.999978</td>
          <td>116.327460</td>
          <td>2009-03-19 05:46:37</td>
          <td>52009031905</td>
          <td>4.713399</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
      </tbody>
    </table>
    <p>217270 rows × 8 columns</p>
    </div>



clean_gps_nearby_points_by_distances function removes points from the
trajectories when the distance between them and the point before is
smaller than the parameter radius_area.

.. code:: python

    filters.clean_gps_nearby_points_by_distances(df_move, radius_area = 10)


.. parsed-literal::


    Cleaning gps points from radius of 10 meters

    ...Dropping 137684 rows of gps points

    ...Rows before: 217653, Rows after:79969, Sum drop:137684

    137684 GPS points were dropped




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
          <th>tid</th>
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
          <td>12008102305</td>
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
          <td>12008102305</td>
          <td>13.690153</td>
          <td>7.403788</td>
          <td>20.223428</td>
        </tr>
        <tr>
          <th>5</th>
          <td>1</td>
          <td>39.984710</td>
          <td>116.319865</td>
          <td>2008-10-23 05:53:23</td>
          <td>12008102305</td>
          <td>66.555997</td>
          <td>6.162987</td>
          <td>60.622358</td>
        </tr>
        <tr>
          <th>14</th>
          <td>1</td>
          <td>39.984959</td>
          <td>116.319969</td>
          <td>2008-10-23 05:54:03</td>
          <td>12008102305</td>
          <td>40.672170</td>
          <td>11.324767</td>
          <td>51.291054</td>
        </tr>
        <tr>
          <th>15</th>
          <td>1</td>
          <td>39.985036</td>
          <td>116.320056</td>
          <td>2008-10-23 05:54:04</td>
          <td>12008102305</td>
          <td>11.324767</td>
          <td>32.842422</td>
          <td>24.923216</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>217563</th>
          <td>5</td>
          <td>40.001185</td>
          <td>116.321791</td>
          <td>2009-03-19 05:39:02</td>
          <td>52009031905</td>
          <td>11.604029</td>
          <td>6.915583</td>
          <td>17.245027</td>
        </tr>
        <tr>
          <th>217637</th>
          <td>5</td>
          <td>40.000759</td>
          <td>116.327088</td>
          <td>2009-03-19 05:45:07</td>
          <td>52009031905</td>
          <td>28.946922</td>
          <td>18.331999</td>
          <td>47.148573</td>
        </tr>
        <tr>
          <th>217638</th>
          <td>5</td>
          <td>40.000595</td>
          <td>116.327066</td>
          <td>2009-03-19 05:45:12</td>
          <td>52009031905</td>
          <td>18.331999</td>
          <td>9.926875</td>
          <td>27.905967</td>
        </tr>
        <tr>
          <th>217641</th>
          <td>5</td>
          <td>40.000368</td>
          <td>116.327072</td>
          <td>2009-03-19 05:45:27</td>
          <td>52009031905</td>
          <td>10.877438</td>
          <td>8.887992</td>
          <td>19.705708</td>
        </tr>
        <tr>
          <th>217643</th>
          <td>5</td>
          <td>40.000205</td>
          <td>116.327173</td>
          <td>2009-03-19 05:45:37</td>
          <td>52009031905</td>
          <td>11.406650</td>
          <td>8.563704</td>
          <td>19.107146</td>
        </tr>
      </tbody>
    </table>
    <p>79969 rows × 8 columns</p>
    </div>



clean_gps_nearby_points_by_speed function removes points from the
trajectories when the speed of travel between them and the point before
is smaller than the value set by the parameter speed_radius.

.. code:: python

    filters.clean_gps_nearby_points_by_speed(df_move, speed_radius=40.0)


.. parsed-literal::

    ...Sorting by id and datetime to increase performance

    ...Set id as index to a higher performance


    Creating or updating distance, time and speed features in meters by seconds




.. parsed-literal::

    VBox(children=(HTML(value=''), IntProgress(value=0, max=2)))


.. parsed-literal::

    ...Reset index...


    Cleaning gps points using 40.0 speed radius

    ...Dropping 217372 rows of gps points

    ...Rows before: 217653, Rows after:281, Sum drop:217372

    217372 GPS points were dropped




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
          <th>tid</th>
          <th>dist_to_prev</th>
          <th>dist_to_next</th>
          <th>dist_prev_to_next</th>
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
          <td>12008102305</td>
          <td>NaN</td>
          <td>13.690153</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>149</th>
          <td>1</td>
          <td>39.977648</td>
          <td>116.326925</td>
          <td>2008-10-23 10:33:00</td>
          <td>12008102310</td>
          <td>1470.641291</td>
          <td>6.919839</td>
          <td>1467.436502</td>
          <td>7.0</td>
          <td>210.091613</td>
        </tr>
        <tr>
          <th>560</th>
          <td>1</td>
          <td>40.009802</td>
          <td>116.313247</td>
          <td>2008-10-23 10:56:54</td>
          <td>12008102310</td>
          <td>47.020950</td>
          <td>65.222058</td>
          <td>112.001912</td>
          <td>1.0</td>
          <td>47.020950</td>
        </tr>
        <tr>
          <th>561</th>
          <td>1</td>
          <td>40.009262</td>
          <td>116.312948</td>
          <td>2008-10-23 10:56:55</td>
          <td>12008102310</td>
          <td>65.222058</td>
          <td>35.598360</td>
          <td>100.606304</td>
          <td>1.0</td>
          <td>65.222058</td>
        </tr>
        <tr>
          <th>1369</th>
          <td>1</td>
          <td>39.990659</td>
          <td>116.326345</td>
          <td>2008-10-24 00:04:29</td>
          <td>12008102400</td>
          <td>40.942759</td>
          <td>42.052617</td>
          <td>82.726749</td>
          <td>1.0</td>
          <td>40.942759</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>216382</th>
          <td>5</td>
          <td>40.000185</td>
          <td>116.327286</td>
          <td>2009-02-28 03:52:45</td>
          <td>52009022803</td>
          <td>333.656648</td>
          <td>28.500246</td>
          <td>361.959915</td>
          <td>5.0</td>
          <td>66.731330</td>
        </tr>
        <tr>
          <th>217458</th>
          <td>5</td>
          <td>39.999918</td>
          <td>116.320057</td>
          <td>2009-03-19 04:36:02</td>
          <td>52009031904</td>
          <td>556.947064</td>
          <td>264.212540</td>
          <td>820.900139</td>
          <td>5.0</td>
          <td>111.389413</td>
        </tr>
        <tr>
          <th>217459</th>
          <td>5</td>
          <td>39.999077</td>
          <td>116.317156</td>
          <td>2009-03-19 04:36:07</td>
          <td>52009031904</td>
          <td>264.212540</td>
          <td>85.254214</td>
          <td>218.221539</td>
          <td>5.0</td>
          <td>52.842508</td>
        </tr>
        <tr>
          <th>217463</th>
          <td>5</td>
          <td>40.001122</td>
          <td>116.320879</td>
          <td>2009-03-19 04:40:52</td>
          <td>52009031904</td>
          <td>267.350055</td>
          <td>127.347571</td>
          <td>394.695354</td>
          <td>5.0</td>
          <td>53.470011</td>
        </tr>
        <tr>
          <th>217476</th>
          <td>5</td>
          <td>40.005903</td>
          <td>116.318669</td>
          <td>2009-03-19 04:49:47</td>
          <td>52009031904</td>
          <td>436.405009</td>
          <td>119.920027</td>
          <td>555.775352</td>
          <td>5.0</td>
          <td>87.281002</td>
        </tr>
      </tbody>
    </table>
    <p>281 rows × 10 columns</p>
    </div>



clean_gps_speed_max_radius function recursively removes trajectories
points with speed higher than the value especifeid by the user. Given
any point p of the trajectory, the point will be removed if one of the
following happens: if the travel speed from the point before p to p is
greater than the max value of speed between adjacent points set by the
user. Or the travel speed between point p and the next point is greater
than the value set by the user. When the clening is done, the function
will update the time and distance features in the dataframe and will
call itself again. The function will finish processing when it can no
longer find points disrespecting the limit of speed.

.. code:: python

    filters.clean_gps_speed_max_radius(df_move)


.. parsed-literal::

    ...Sorting by id and datetime to increase performance

    ...Set id as index to a higher performance


    Creating or updating distance, time and speed features in meters by seconds




.. parsed-literal::

    VBox(children=(HTML(value=''), IntProgress(value=0, max=2)))


.. parsed-literal::

    ...Reset index...


    Clean gps points with speed max > 50.0 meters by seconds
    ...Dropping 349 rows of gps points

    ...Rows before: 217653, Rows after:217304, Sum drop:349

    349 GPS points were dropped




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
          <th>tid</th>
          <th>dist_to_prev</th>
          <th>dist_to_next</th>
          <th>dist_prev_to_next</th>
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
          <td>12008102305</td>
          <td>NaN</td>
          <td>13.690153</td>
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
          <td>12008102305</td>
          <td>13.690153</td>
          <td>7.403788</td>
          <td>20.223428</td>
          <td>1.0</td>
          <td>13.690153</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1</td>
          <td>39.984224</td>
          <td>116.319402</td>
          <td>2008-10-23 05:53:11</td>
          <td>12008102305</td>
          <td>7.403788</td>
          <td>1.821083</td>
          <td>5.888579</td>
          <td>5.0</td>
          <td>1.480758</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1</td>
          <td>39.984211</td>
          <td>116.319389</td>
          <td>2008-10-23 05:53:16</td>
          <td>12008102305</td>
          <td>1.821083</td>
          <td>2.889671</td>
          <td>1.873356</td>
          <td>5.0</td>
          <td>0.364217</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1</td>
          <td>39.984217</td>
          <td>116.319422</td>
          <td>2008-10-23 05:53:21</td>
          <td>12008102305</td>
          <td>2.889671</td>
          <td>66.555997</td>
          <td>68.727260</td>
          <td>5.0</td>
          <td>0.577934</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>217648</th>
          <td>5</td>
          <td>39.999896</td>
          <td>116.327290</td>
          <td>2009-03-19 05:46:02</td>
          <td>52009031905</td>
          <td>7.198855</td>
          <td>5.291709</td>
          <td>12.214590</td>
          <td>5.0</td>
          <td>1.439771</td>
        </tr>
        <tr>
          <th>217649</th>
          <td>5</td>
          <td>39.999899</td>
          <td>116.327352</td>
          <td>2009-03-19 05:46:07</td>
          <td>52009031905</td>
          <td>5.291709</td>
          <td>6.241949</td>
          <td>10.400206</td>
          <td>5.0</td>
          <td>1.058342</td>
        </tr>
        <tr>
          <th>217650</th>
          <td>5</td>
          <td>39.999945</td>
          <td>116.327394</td>
          <td>2009-03-19 05:46:12</td>
          <td>52009031905</td>
          <td>6.241949</td>
          <td>8.462920</td>
          <td>14.628012</td>
          <td>5.0</td>
          <td>1.248390</td>
        </tr>
        <tr>
          <th>217651</th>
          <td>5</td>
          <td>40.000015</td>
          <td>116.327433</td>
          <td>2009-03-19 05:46:17</td>
          <td>52009031905</td>
          <td>8.462920</td>
          <td>4.713399</td>
          <td>6.713456</td>
          <td>5.0</td>
          <td>1.692584</td>
        </tr>
        <tr>
          <th>217652</th>
          <td>5</td>
          <td>39.999978</td>
          <td>116.327460</td>
          <td>2009-03-19 05:46:37</td>
          <td>52009031905</td>
          <td>4.713399</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>20.0</td>
          <td>0.235670</td>
        </tr>
      </tbody>
    </table>
    <p>217304 rows × 10 columns</p>
    </div>



clean_trajectories_with_few_points function removes from the given
dataframe, trajectories with fewer points than was specified by the
parameter min_points_per_trajectory.

.. code:: python

    filters.clean_trajectories_with_few_points(df_move)


.. parsed-literal::


    Cleaning gps points from trajectories of fewer than 2 points


    ...There are 4 ids with few points

    ...Tids before drop: 625

    ...Tids after drop: 621

    ...Shape - before drop: (217653, 8) - after drop: (217649, 8)




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
          <th>tid</th>
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
          <td>12008102305</td>
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
          <td>12008102305</td>
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
          <td>12008102305</td>
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
          <td>12008102305</td>
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
          <td>12008102305</td>
          <td>2.889671</td>
          <td>66.555997</td>
          <td>68.727260</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>217648</th>
          <td>5</td>
          <td>39.999896</td>
          <td>116.327290</td>
          <td>2009-03-19 05:46:02</td>
          <td>52009031905</td>
          <td>7.198855</td>
          <td>5.291709</td>
          <td>12.214590</td>
        </tr>
        <tr>
          <th>217649</th>
          <td>5</td>
          <td>39.999899</td>
          <td>116.327352</td>
          <td>2009-03-19 05:46:07</td>
          <td>52009031905</td>
          <td>5.291709</td>
          <td>6.241949</td>
          <td>10.400206</td>
        </tr>
        <tr>
          <th>217650</th>
          <td>5</td>
          <td>39.999945</td>
          <td>116.327394</td>
          <td>2009-03-19 05:46:12</td>
          <td>52009031905</td>
          <td>6.241949</td>
          <td>8.462920</td>
          <td>14.628012</td>
        </tr>
        <tr>
          <th>217651</th>
          <td>5</td>
          <td>40.000015</td>
          <td>116.327433</td>
          <td>2009-03-19 05:46:17</td>
          <td>52009031905</td>
          <td>8.462920</td>
          <td>4.713399</td>
          <td>6.713456</td>
        </tr>
        <tr>
          <th>217652</th>
          <td>5</td>
          <td>39.999978</td>
          <td>116.327460</td>
          <td>2009-03-19 05:46:37</td>
          <td>52009031905</td>
          <td>4.713399</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
      </tbody>
    </table>
    <p>217649 rows × 8 columns</p>
    </div>



Segmentation
------------

The segmentation module are used to segment trajectories based on
different parameters.

Importing the module:

.. code:: python

    from pymove import segmentation
    df_move = read_csv('geolife_sample.csv')

bbox_split function splits the bounding box in grids of the same size.
The number of grids is defined by the parameter number_grids.

.. code:: python

    bbox = (22.147577, 113.54884299999999, 41.132062, 121.156224)
    segmentation.bbox_split(bbox, number_grids=4)


.. parsed-literal::

    const_lat: 4.74612125
    const_lon: 1.901845250000001
    **********************
    bbox_split took 00.01s
    **********************




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
          <th>lat_min</th>
          <th>lon_min</th>
          <th>lat_max</th>
          <th>lon_max</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>22.147577</td>
          <td>113.548843</td>
          <td>41.132062</td>
          <td>115.450688</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.147577</td>
          <td>115.450688</td>
          <td>41.132062</td>
          <td>117.352533</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.147577</td>
          <td>117.352533</td>
          <td>41.132062</td>
          <td>119.254379</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.147577</td>
          <td>119.254379</td>
          <td>41.132062</td>
          <td>121.156224</td>
        </tr>
      </tbody>
    </table>
    </div>



by_dist_time_speed functions segments the trajectories into clusters
based on distance, time and speed. The distance, time and speed limits
by the parameters by max_dist_between_adj_points,
max_time_between_adj_points, max_speed_between_adj_points respectively.
The column tid_part is added, it indicates the segment to which the
point belongs to.

.. code:: python

    segmentation.by_dist_time_speed(
        df_move,
        max_dist_between_adj_points=5000,
        max_time_between_adj_points=800,
        max_speed_between_adj_points=60.0
    )
    df_move.head()


.. parsed-literal::


    Split trajectories
    ...max_dist_between_adj_points: 5000
    ...max_time_between_adj_points: 800
    ...max_speed_between_adj_points: 60.0
    ...Sorting by id and datetime to increase performance

    ...Set id as index to a higher performance


    Creating or updating distance, time and speed features in meters by seconds




.. parsed-literal::

    VBox(children=(HTML(value=''), IntProgress(value=0, max=2)))


.. parsed-literal::

    ...Reset index...

    ...setting id as index



.. parsed-literal::

    VBox(children=(HTML(value=''), IntProgress(value=0, max=2)))


.. parsed-literal::

    ... Reseting index

    ...No trajs with only one point. (217653, 8)
    ...Sorting by id and datetime to increase performance

    ...Set id as index to a higher performance


    Creating or updating distance, time and speed features in meters by seconds




.. parsed-literal::

    VBox(children=(HTML(value=''), IntProgress(value=0, max=2)))


.. parsed-literal::

    ...Reset index...

    ******************************
    by_dist_time_speed took 00.49s
    ******************************




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
          <th>tid_part</th>
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
          <td>1</td>
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
          <td>1</td>
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
          <td>1</td>
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
          <td>1</td>
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
          <td>1</td>
        </tr>
      </tbody>
    </table>
    </div>



by_speed function segments the trajectories into clusters based on
speed. The speed limit is defined by the parameter
max_speed_between_adj_points. The column tid_speed is added, it
indicates the segment to which the point belongs to.

.. code:: python

    segmentation.by_max_dist(df_move, max_dist_between_adj_points=4000)
    df_move.head()


.. parsed-literal::

    Split trajectories by max distance between adjacent points: 4000
    ...setting id as index



.. parsed-literal::

    VBox(children=(HTML(value=''), IntProgress(value=0, max=2)))


.. parsed-literal::

    ... Reseting index

    ...No trajs with only one point. (217653, 9)
    ...Sorting by id and datetime to increase performance

    ...Set id as index to a higher performance


    Creating or updating distance, time and speed features in meters by seconds




.. parsed-literal::

    VBox(children=(HTML(value=''), IntProgress(value=0, max=2)))


.. parsed-literal::

    ...Reset index...

    ***********************
    by_max_dist took 00.28s
    ***********************




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
          <th>tid_part</th>
          <th>tid_dist</th>
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
          <td>1</td>
          <td>1</td>
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
          <td>1</td>
          <td>1</td>
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
          <td>1</td>
          <td>1</td>
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
          <td>1</td>
          <td>1</td>
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
          <td>1</td>
          <td>1</td>
        </tr>
      </tbody>
    </table>
    </div>



by_time function segments the trajectories into clusters based on time.
The time limit is defined by the parameter max_time_between_adj_points.
The column tid_time is added, it indicates the segment to which the
point belongs to.

.. code:: python

    segmentation.by_max_time(df_move, max_time_between_adj_points=1000)
    df_move.head()


.. parsed-literal::

    Split trajectories by max_time_between_adj_points: 1000
    ...setting id as index



.. parsed-literal::

    VBox(children=(HTML(value=''), IntProgress(value=0, max=2)))


.. parsed-literal::

    ... Reseting index

    ...No trajs with only one point. (217653, 10)
    ...Sorting by id and datetime to increase performance

    ...Set id as index to a higher performance


    Creating or updating distance, time and speed features in meters by seconds




.. parsed-literal::

    VBox(children=(HTML(value=''), IntProgress(value=0, max=2)))


.. parsed-literal::

    ...Reset index...

    ***********************
    by_max_time took 00.29s
    ***********************




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
          <th>tid_part</th>
          <th>tid_dist</th>
          <th>tid_time</th>
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
          <td>1</td>
          <td>1</td>
          <td>1</td>
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
          <td>1</td>
          <td>1</td>
          <td>1</td>
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
          <td>1</td>
          <td>1</td>
          <td>1</td>
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
          <td>1</td>
          <td>1</td>
          <td>1</td>
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
          <td>1</td>
          <td>1</td>
          <td>1</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    segmentation.by_max_speed(df_move, max_speed_between_adj_points=70.0)
    df_move.head()


.. parsed-literal::

    Split trajectories by max_speed_between_adj_points: 70.0
    ...setting id as index



.. parsed-literal::

    VBox(children=(HTML(value=''), IntProgress(value=0, max=2)))


.. parsed-literal::

    ... Reseting index

    ...No trajs with only one point. (217653, 11)
    ...Sorting by id and datetime to increase performance

    ...Set id as index to a higher performance


    Creating or updating distance, time and speed features in meters by seconds




.. parsed-literal::

    VBox(children=(HTML(value=''), IntProgress(value=0, max=2)))


.. parsed-literal::

    ...Reset index...

    ************************
    by_max_speed took 00.36s
    ************************




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
          <th>tid_part</th>
          <th>tid_dist</th>
          <th>tid_time</th>
          <th>tid_speed</th>
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
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
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
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
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
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
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
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
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
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
        </tr>
      </tbody>
    </table>
    </div>



segment_traj_by_max_dist function segments the trajectories into
clusters based on distance. The distance limit is defined by the
parameter max_dist_between_adj_points. The column tid_dist is added, it
indicates the segment to which the point belongs to.

Stay point detection
--------------------

A stay point is location where a moving object has stayed for a while
within a certain distance threshold. A stay point could stand different
places such: a restaurant, a school, a work place.

Importing the module:

.. code:: python

    from pymove import stay_point_detection
    df_move = read_csv('geolife_sample.csv')

stay_point_detection function converts the time data into a cyclical
format. The columns hour_sin and hour_cos are added to the dataframe.

.. code:: python

    stay_point_detection.create_or_update_datetime_in_format_cyclical(df_move)
    df_move.head()


.. parsed-literal::

    Encoding cyclical continuous features - 24-hour time
    ...hour_sin and  hour_cos features were created...





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
          <th>hour_sin</th>
          <th>hour_cos</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>39.984094</td>
          <td>116.319236</td>
          <td>2008-10-23 05:53:05</td>
          <td>1</td>
          <td>0.979084</td>
          <td>0.203456</td>
        </tr>
        <tr>
          <th>1</th>
          <td>39.984198</td>
          <td>116.319322</td>
          <td>2008-10-23 05:53:06</td>
          <td>1</td>
          <td>0.979084</td>
          <td>0.203456</td>
        </tr>
        <tr>
          <th>2</th>
          <td>39.984224</td>
          <td>116.319402</td>
          <td>2008-10-23 05:53:11</td>
          <td>1</td>
          <td>0.979084</td>
          <td>0.203456</td>
        </tr>
        <tr>
          <th>3</th>
          <td>39.984211</td>
          <td>116.319389</td>
          <td>2008-10-23 05:53:16</td>
          <td>1</td>
          <td>0.979084</td>
          <td>0.203456</td>
        </tr>
        <tr>
          <th>4</th>
          <td>39.984217</td>
          <td>116.319422</td>
          <td>2008-10-23 05:53:21</td>
          <td>1</td>
          <td>0.979084</td>
          <td>0.203456</td>
        </tr>
      </tbody>
    </table>
    </div>



create_or_update_move_stop_by_dist_time function creates or updates the
stay points of the trajectories, based on distance and time metrics. The
column segment_stop is added to the dataframe, it indicates the
trajectory segment to which the point belongs to. The column stop is
also added, it indicates is the point represents a stop, a place where
the object was stationary.

.. code:: python

    stay_point_detection.create_or_update_move_stop_by_dist_time(df_move, dist_radius=40, time_radius=1000)
    df_move.head()


.. parsed-literal::

    Split trajectories by max distance between adjacent points: 40
    ...Sorting by id and datetime to increase performance

    ...Set id as index to a higher performance


    Creating or updating distance, time and speed features in meters by seconds




.. parsed-literal::

    VBox(children=(HTML(value=''), IntProgress(value=0, max=2)))


.. parsed-literal::

    ...Reset index...

    ...setting id as index



.. parsed-literal::

    VBox(children=(HTML(value=''), IntProgress(value=0, max=2)))


.. parsed-literal::

    ... Reseting index

    ...No trajs with only one point. (217653, 10)
    ...Sorting by id and datetime to increase performance

    ...Set id as index to a higher performance


    Creating or updating distance, time and speed features in meters by seconds




.. parsed-literal::

    VBox(children=(HTML(value=''), IntProgress(value=0, max=2)))


.. parsed-literal::

    ...Reset index...

    ***********************
    by_max_dist took 00.52s
    ***********************
    ...Sorting by segment_stop and datetime to increase performance

    ...Set segment_stop as index to a higher performance


    Creating or updating distance, time and speed features in meters by seconds




.. parsed-literal::

    VBox(children=(HTML(value=''), IntProgress(value=0, max=3512)))


.. parsed-literal::

    ...Reset index...

    Create or update stop as True or False
    ...Creating stop features as True or False using 1000 to time in seconds
    True     157738
    False     59915
    Name: stop, dtype: int64
    ***************************************************
    create_or_update_move_stop_by_dist_time took 36.68s
    ***************************************************




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
          <th>segment_stop</th>
          <th>id</th>
          <th>lat</th>
          <th>lon</th>
          <th>datetime</th>
          <th>hour_sin</th>
          <th>hour_cos</th>
          <th>dist_to_prev</th>
          <th>time_to_prev</th>
          <th>speed_to_prev</th>
          <th>stop</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1</td>
          <td>1</td>
          <td>39.984094</td>
          <td>116.319236</td>
          <td>2008-10-23 05:53:05</td>
          <td>0.979084</td>
          <td>0.203456</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>False</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1</td>
          <td>1</td>
          <td>39.984198</td>
          <td>116.319322</td>
          <td>2008-10-23 05:53:06</td>
          <td>0.979084</td>
          <td>0.203456</td>
          <td>13.690153</td>
          <td>1.0</td>
          <td>13.690153</td>
          <td>False</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1</td>
          <td>1</td>
          <td>39.984224</td>
          <td>116.319402</td>
          <td>2008-10-23 05:53:11</td>
          <td>0.979084</td>
          <td>0.203456</td>
          <td>7.403788</td>
          <td>5.0</td>
          <td>1.480758</td>
          <td>False</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1</td>
          <td>1</td>
          <td>39.984211</td>
          <td>116.319389</td>
          <td>2008-10-23 05:53:16</td>
          <td>0.979084</td>
          <td>0.203456</td>
          <td>1.821083</td>
          <td>5.0</td>
          <td>0.364217</td>
          <td>False</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1</td>
          <td>1</td>
          <td>39.984217</td>
          <td>116.319422</td>
          <td>2008-10-23 05:53:21</td>
          <td>0.979084</td>
          <td>0.203456</td>
          <td>2.889671</td>
          <td>5.0</td>
          <td>0.577934</td>
          <td>False</td>
        </tr>
      </tbody>
    </table>
    </div>



create_or_update_move_and_stop_by_radius function creates or updates the
stay points of the trajectories, based on distance. The column situation
is also added, it indicates if the point represents a stop point or a
moving point.

.. code:: python

    stay_point_detection.create_or_update_move_and_stop_by_radius(df_move, radius=2)
    df_move.head()


.. parsed-literal::


    Creating or updating features MOVE and STOPS...


    ....There are 58981 stops to this parameters

    ****************************************************
    create_or_update_move_and_stop_by_radius took 00.14s
    ****************************************************




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
          <th>segment_stop</th>
          <th>id</th>
          <th>lat</th>
          <th>lon</th>
          <th>datetime</th>
          <th>hour_sin</th>
          <th>hour_cos</th>
          <th>dist_to_prev</th>
          <th>time_to_prev</th>
          <th>speed_to_prev</th>
          <th>stop</th>
          <th>situation</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1</td>
          <td>1</td>
          <td>39.984094</td>
          <td>116.319236</td>
          <td>2008-10-23 05:53:05</td>
          <td>0.979084</td>
          <td>0.203456</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>False</td>
          <td>nan</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1</td>
          <td>1</td>
          <td>39.984198</td>
          <td>116.319322</td>
          <td>2008-10-23 05:53:06</td>
          <td>0.979084</td>
          <td>0.203456</td>
          <td>13.690153</td>
          <td>1.0</td>
          <td>13.690153</td>
          <td>False</td>
          <td>move</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1</td>
          <td>1</td>
          <td>39.984224</td>
          <td>116.319402</td>
          <td>2008-10-23 05:53:11</td>
          <td>0.979084</td>
          <td>0.203456</td>
          <td>7.403788</td>
          <td>5.0</td>
          <td>1.480758</td>
          <td>False</td>
          <td>move</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1</td>
          <td>1</td>
          <td>39.984211</td>
          <td>116.319389</td>
          <td>2008-10-23 05:53:16</td>
          <td>0.979084</td>
          <td>0.203456</td>
          <td>1.821083</td>
          <td>5.0</td>
          <td>0.364217</td>
          <td>False</td>
          <td>stop</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1</td>
          <td>1</td>
          <td>39.984217</td>
          <td>116.319422</td>
          <td>2008-10-23 05:53:21</td>
          <td>0.979084</td>
          <td>0.203456</td>
          <td>2.889671</td>
          <td>5.0</td>
          <td>0.577934</td>
          <td>False</td>
          <td>move</td>
        </tr>
      </tbody>
    </table>
    </div>



Compression
-----------

Importing the module:

.. code:: python

    from pymove import compression
    df_move = read_csv('geolife_sample.csv')

The function below is used to reduce the size of the trajectory, the
stop points are used to make the compression.

.. code:: python

    df_compressed = compression.compress_segment_stop_to_point(df_move)
    len(df_move), len(df_compressed)


.. parsed-literal::

    Split trajectories by max distance between adjacent points: 30
    ...Sorting by id and datetime to increase performance

    ...Set id as index to a higher performance


    Creating or updating distance, time and speed features in meters by seconds




.. parsed-literal::

    VBox(children=(HTML(value=''), IntProgress(value=0, max=2)))


.. parsed-literal::

    ...Reset index...

    ...setting id as index



.. parsed-literal::

    VBox(children=(HTML(value=''), IntProgress(value=0, max=2)))


.. parsed-literal::

    ... Reseting index

    ...No trajs with only one point. (217653, 8)
    ...Sorting by id and datetime to increase performance

    ...Set id as index to a higher performance


    Creating or updating distance, time and speed features in meters by seconds




.. parsed-literal::

    VBox(children=(HTML(value=''), IntProgress(value=0, max=2)))


.. parsed-literal::

    ...Reset index...

    ***********************
    by_max_dist took 01.20s
    ***********************
    ...Sorting by segment_stop and datetime to increase performance

    ...Set segment_stop as index to a higher performance


    Creating or updating distance, time and speed features in meters by seconds




.. parsed-literal::

    VBox(children=(HTML(value=''), IntProgress(value=0, max=4809)))


.. parsed-literal::

    ...Reset index...

    Create or update stop as True or False
    ...Creating stop features as True or False using 900 to time in seconds
    True     152603
    False     65050
    Name: stop, dtype: int64
    ***************************************************
    create_or_update_move_stop_by_dist_time took 53.09s
    ***************************************************
    ...setting mean to lat and lon...
    ...get only segments stop...



.. parsed-literal::

    VBox(children=(HTML(value=''), IntProgress(value=0, max=285)))


.. parsed-literal::

    ...Dropping 152033 points...
    ...Shape_before: 217653
    ...Current shape: 65620
    **********************************************
    compress_segment_stop_to_point took 01m:08.30s
    **********************************************




.. parsed-literal::

    (217653, 65620)
