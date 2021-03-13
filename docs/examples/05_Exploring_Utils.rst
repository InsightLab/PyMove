#05 - Exploring Utils
=====================

Falar sobre para se trabalhar com trajetórias pode ser necessária
algumas c onversões envolvendo tempo e data, distância e etc, fora
outros utilitários.

Falar dos módulos presentes no pacote utils - constants - conversions -
datetime - distances - math - mem - trajectories - transformations

--------------

Imports
~~~~~~~

.. code:: ipython3

    import pymove.utils as utils
    import pymove
    from pymove import MoveDataFrame

--------------

Load data
~~~~~~~~~

.. code:: ipython3

    move_data = pymove.read_csv("geolife_sample.csv")

--------------

Conversions
~~~~~~~~~~~

To transform latitude degree to meters, you can use function
**lat_meters**. For example, you can convert Fortaleza’s latitude
-3.8162973555:

.. code:: ipython3

    utils.conversions.lat_meters(-3.8162973555)




.. parsed-literal::

    110826.6722516857



To concatenates list elements, joining them by the separator specified
by the parameter “delimiter”, you can use **list_to_str**

.. code:: ipython3

    utils.conversions.list_to_str(["a", "b", "c", "d"], "-")




.. parsed-literal::

    'a-b-c-d'



To concatenates the elements of the list, joining them by “,”, , you can
use **list_to_csv_str**

.. code:: ipython3

    utils.conversions.list_to_csv_str(["a", "b", "c", "d"])




.. parsed-literal::

    'a,b,c,d'



To concatenates list elements in consecutive element pairs, you can use
**list_to_svm_line**

.. code:: ipython3

    utils.conversions.list_to_svm_line(["a", "b", "c", "d"])




.. parsed-literal::

    'a 1:b 2:c 3:d'



To convert longitude to X EPSG:3857 WGS 84/Pseudo-Mercator, you can use
**lon_to_x_spherical**

.. code:: ipython3

    utils.conversions.lon_to_x_spherical(-38.501597)




.. parsed-literal::

    -4285978.172767829



To convert latitude to Y EPSG:3857 WGS 84/Pseudo-Mercator, you can use
**lat_to_y_spherical**

.. code:: ipython3

    utils.conversions.lat_to_y_spherical(-3.797864)




.. parsed-literal::

    -423086.2213610324



To convert X EPSG:3857 WGS 84/Pseudo-Mercator to longitude, you can use
**x_to_lon_spherical**

.. code:: ipython3

    utils.conversions.x_to_lon_spherical(-4285978.172767829)




.. parsed-literal::

    -38.501597000000004



To convert Y EPSG:3857 WGS 84/Pseudo-Mercator to latitude, you can use
**y_to_lat_spherical**

.. code:: ipython3

    utils.conversions.y_to_lat_spherical(-423086.2213610324)




.. parsed-literal::

    -3.7978639999999944



To convert values, in ms, in label_speed column to kmh, you can use
**ms_to_kmh**

.. code:: ipython3

    utils.conversions.ms_to_kmh(move_data)


.. parsed-literal::

    ...Sorting by id and datetime to increase performance

    ...Set id as index to a higher performance


    Creating or updating distance, time and speed features in meters by seconds




.. parsed-literal::

    VBox(children=(HTML(value=''), IntProgress(value=0, max=2)))


.. parsed-literal::

    ...Reset index...



.. code:: ipython3

    move_data.head()




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
          <td>49.284551</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1</td>
          <td>39.984224</td>
          <td>116.319402</td>
          <td>2008-10-23 05:53:11</td>
          <td>7.403788</td>
          <td>5.0</td>
          <td>5.330727</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1</td>
          <td>39.984211</td>
          <td>116.319389</td>
          <td>2008-10-23 05:53:16</td>
          <td>1.821083</td>
          <td>5.0</td>
          <td>1.311180</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1</td>
          <td>39.984217</td>
          <td>116.319422</td>
          <td>2008-10-23 05:53:21</td>
          <td>2.889671</td>
          <td>5.0</td>
          <td>2.080563</td>
        </tr>
      </tbody>
    </table>
    </div>



To convert values, in kmh, in label_speed column to ms, you can use
**kmh_to_ms**

.. code:: ipython3

    utils.conversions.kmh_to_ms(move_data)

.. code:: ipython3

    move_data.head()




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



To convert values, in meters, in label_distance column to kilometer, you
can use **meters_to_kilometers**

.. code:: ipython3

    utils.conversions.meters_to_kilometers(move_data)

.. code:: ipython3

    move_data.head()




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
          <td>0.013690</td>
          <td>1.0</td>
          <td>13.690153</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1</td>
          <td>39.984224</td>
          <td>116.319402</td>
          <td>2008-10-23 05:53:11</td>
          <td>0.007404</td>
          <td>5.0</td>
          <td>1.480758</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1</td>
          <td>39.984211</td>
          <td>116.319389</td>
          <td>2008-10-23 05:53:16</td>
          <td>0.001821</td>
          <td>5.0</td>
          <td>0.364217</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1</td>
          <td>39.984217</td>
          <td>116.319422</td>
          <td>2008-10-23 05:53:21</td>
          <td>0.002890</td>
          <td>5.0</td>
          <td>0.577934</td>
        </tr>
      </tbody>
    </table>
    </div>



To convert values, in kilometers, in label_distance column to meters,
you can use **kilometers_to_meters**

.. code:: ipython3

    utils.conversions.kilometers_to_meters(move_data)

.. code:: ipython3

    move_data.head()




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



To convert values, in seconds, in label_distance column to minutes, you
can use **seconds_to_minutes**

.. code:: ipython3

    utils.conversions.seconds_to_minutes(move_data)

.. code:: ipython3

    move_data.head()




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
          <td>0.016667</td>
          <td>13.690153</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1</td>
          <td>39.984224</td>
          <td>116.319402</td>
          <td>2008-10-23 05:53:11</td>
          <td>7.403788</td>
          <td>0.083333</td>
          <td>1.480758</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1</td>
          <td>39.984211</td>
          <td>116.319389</td>
          <td>2008-10-23 05:53:16</td>
          <td>1.821083</td>
          <td>0.083333</td>
          <td>0.364217</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1</td>
          <td>39.984217</td>
          <td>116.319422</td>
          <td>2008-10-23 05:53:21</td>
          <td>2.889671</td>
          <td>0.083333</td>
          <td>0.577934</td>
        </tr>
      </tbody>
    </table>
    </div>



To convert values, in minutes, in label_distance column to seconds, you
can use **minute_to_seconds**

.. code:: ipython3

    utils.conversions.minute_to_seconds(move_data)

.. code:: ipython3

    move_data.head()




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



To convert in minutes, in label_distance column to hours, you can use
**minute_to_hours**

.. code:: ipython3

    utils.conversions.seconds_to_minutes(move_data)

.. code:: ipython3

    utils.conversions.minute_to_hours(move_data)

.. code:: ipython3

    move_data.head()




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
          <td>0.000278</td>
          <td>13.690153</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1</td>
          <td>39.984224</td>
          <td>116.319402</td>
          <td>2008-10-23 05:53:11</td>
          <td>7.403788</td>
          <td>0.001389</td>
          <td>1.480758</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1</td>
          <td>39.984211</td>
          <td>116.319389</td>
          <td>2008-10-23 05:53:16</td>
          <td>1.821083</td>
          <td>0.001389</td>
          <td>0.364217</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1</td>
          <td>39.984217</td>
          <td>116.319422</td>
          <td>2008-10-23 05:53:21</td>
          <td>2.889671</td>
          <td>0.001389</td>
          <td>0.577934</td>
        </tr>
      </tbody>
    </table>
    </div>



To convert in hours, in label_distance column to minute, you can use
**hours_to_minute**

.. code:: ipython3

    utils.conversions.hours_to_minute(move_data)

.. code:: ipython3

    move_data.head()




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
          <td>0.016667</td>
          <td>13.690153</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1</td>
          <td>39.984224</td>
          <td>116.319402</td>
          <td>2008-10-23 05:53:11</td>
          <td>7.403788</td>
          <td>0.083333</td>
          <td>1.480758</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1</td>
          <td>39.984211</td>
          <td>116.319389</td>
          <td>2008-10-23 05:53:16</td>
          <td>1.821083</td>
          <td>0.083333</td>
          <td>0.364217</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1</td>
          <td>39.984217</td>
          <td>116.319422</td>
          <td>2008-10-23 05:53:21</td>
          <td>2.889671</td>
          <td>0.083333</td>
          <td>0.577934</td>
        </tr>
      </tbody>
    </table>
    </div>



To convert in seconds, in label_distance column to hours, you can use
**seconds_to_hours**

.. code:: ipython3

    utils.conversions.minute_to_seconds(move_data)

.. code:: ipython3

    utils.conversions.seconds_to_hours(move_data)

.. code:: ipython3

    move_data.head()




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
          <td>0.000278</td>
          <td>13.690153</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1</td>
          <td>39.984224</td>
          <td>116.319402</td>
          <td>2008-10-23 05:53:11</td>
          <td>7.403788</td>
          <td>0.001389</td>
          <td>1.480758</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1</td>
          <td>39.984211</td>
          <td>116.319389</td>
          <td>2008-10-23 05:53:16</td>
          <td>1.821083</td>
          <td>0.001389</td>
          <td>0.364217</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1</td>
          <td>39.984217</td>
          <td>116.319422</td>
          <td>2008-10-23 05:53:21</td>
          <td>2.889671</td>
          <td>0.001389</td>
          <td>0.577934</td>
        </tr>
      </tbody>
    </table>
    </div>



To convert in seconds, in label_distance column to hours, you can use
**hours_to_seconds**

.. code:: ipython3

    utils.conversions.hours_to_seconds(move_data)

.. code:: ipython3

    move_data.head()




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



Datetime
--------

To converts a datetime in string“s format”%Y-%m-%d" or “%Y-%m-%d
%H:%M:%S” to datetime"s format, you can use **str_to_datetime**.

.. code:: ipython3

    utils.datetime.str_to_datetime('2018-06-29 08:15:27')




.. parsed-literal::

    datetime.datetime(2018, 6, 29, 8, 15, 27)



To get date, in string’s format, from timestamp, you can use
**date_to_str**.

.. code:: ipython3

    utils.datetime.date_to_str(utils.datetime.str_to_datetime('2018-06-29 08:15:27'))




.. parsed-literal::

    '2018-06-29'



To converts a date in datetime’s format to string’s format, you can use
**to_str**.

.. code:: ipython3

    import datetime
    utils.datetime.to_str(datetime.datetime(2018, 6, 29, 8, 15, 27))




.. parsed-literal::

    '2018-06-29 08:15:27'



To converts a datetime to an int representation in minutes, you can use
**to_min**.

.. code:: ipython3

    utils.datetime.to_min(datetime.datetime(2018, 6, 29, 8, 15, 27))




.. parsed-literal::

    25504335



To do the reverse use: **min_to_datetime**

.. code:: ipython3

    utils.datetime.min_to_datetime(25504335)




.. parsed-literal::

    datetime.datetime(2018, 6, 29, 8, 15)



To get day of week of a date, you can use **to_day_of_week_int**, where
0 represents Monday and 6 is Sunday.

.. code:: ipython3

    utils.datetime.to_day_of_week_int(datetime.datetime(2018, 6, 29, 8, 15, 27))




.. parsed-literal::

    4



To indices if a day specified by the user is a working day, you can use
**working_day**.

.. code:: ipython3

    utils.datetime.working_day(datetime.datetime(2018, 6, 29, 8, 15, 27), country='BR')




.. parsed-literal::

    True



.. code:: ipython3

    utils.datetime.working_day(datetime.datetime(2018, 4, 21, 8, 15, 27), country='BR')




.. parsed-literal::

    False



To get datetime of now, you can use **now_str**.

.. code:: ipython3

    utils.datetime.now_str()




.. parsed-literal::

    '2021-02-01 21:56:26'



To convert time in a format appropriate of time, you can use
**deltatime_str**.

.. code:: ipython3

    utils.datetime.deltatime_str(1082.7180936336517)




.. parsed-literal::

    '18m:02.72s'



To converts a local datetime to a POSIX timestamp in milliseconds, you
can use **timestamp_to_millis**.

.. code:: ipython3

    utils.datetime.timestamp_to_millis("2015-12-12 08:00:00.123000")




.. parsed-literal::

    1449907200123



To converts milliseconds to timestamp, you can use
**millis_to_timestamp**.

.. code:: ipython3

    utils.datetime.millis_to_timestamp(1449907200123)




.. parsed-literal::

    Timestamp('2015-12-12 08:00:00.123000')



To get time, in string’s format, from timestamp, you can use
**time_to_str**.

.. code:: ipython3

    utils.datetime.time_to_str(datetime.datetime(2018, 6, 29, 8, 15, 27))




.. parsed-literal::

    '08:15:27'



To converts a time in string’s format “%H:%M:%S” to datetime’s format,
you can use **str_to_time**.

.. code:: ipython3

    utils.datetime.str_to_time("08:00:00")




.. parsed-literal::

    datetime.datetime(1900, 1, 1, 8, 0)



To computes the elapsed time from a specific start time to the moment
the function is called, you can use **elapsed_time_dt**.

.. code:: ipython3

    utils.datetime.elapsed_time_dt(utils.datetime.str_to_time("08:00:00"))




.. parsed-literal::

    3821176587586



To computes the elapsed time from the start time to the end time
specifed by the user, you can use **diff_time**.

.. code:: ipython3

    utils.datetime.diff_time(utils.datetime.str_to_time("08:00:00"), utils.datetime.str_to_time("12:00:00"))




.. parsed-literal::

    14400000



Distances
---------

To calculate the great circle distance between two points on the earth,
you can use **haversine**.

.. code:: ipython3

    utils.distances.haversine(-3.797864,-38.501597,-3.797890, -38.501681)




.. parsed-literal::

    9.757976024363016



--------------

.. raw:: html

   <!-- Ver com a arina se é válido fazer a doc dessas 2 -->

.. raw:: html

   <!-- ## Trajectories -->

.. raw:: html

   <!-- ## Transformations -->

Math
----

To compute standard deviation, you can use **std**.

.. code:: ipython3

    utils.math.std([600, 20, 5])




.. parsed-literal::

    277.0178494048513



To compute the average of standard deviation, you can use **avg_std**.

.. code:: ipython3

    utils.math.avg_std([600, 20, 5])




.. parsed-literal::

    (208.33333333333334, 277.0178494048513)



To compute the standard deviation of sample, you can use **std_sample**.

.. code:: ipython3

    utils.math.std_sample([600, 20, 5])




.. parsed-literal::

    339.27619034251916



To compute the average of standard deviation of sample, you can use
**avg_std_sample**.

.. code:: ipython3

    utils.math.avg_std_sample([600, 20, 5])




.. parsed-literal::

    (208.33333333333334, 339.27619034251916)



To computes the sum of the elements of the array, you can use
**array_sum**.

To computes the sum of all the elements in the array, the sum of the
square of each element and the number of elements of the array, you can
use **array_stats**.

.. code:: ipython3

    utils.math.array_stats([600, 20, 5])




.. parsed-literal::

    (625, 360425, 3)



To perfomers interpolation and extrapolation, you can use
**interpolation**.

.. code:: ipython3

    utils.math.interpolation(15, 20, 65, 86, 5)




.. parsed-literal::

    6.799999999999999
