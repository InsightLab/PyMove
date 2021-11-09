05 - Exploring Utils
====================

Falar sobre para se trabalhar com trajetórias pode ser necessária
algumas c onversões envolvendo tempo e data, distância e etc, fora
outros utilitários.

Falar dos módulos presentes no pacote utils - constants - conversions -
datetime - distances - math - trajectories - log - mem

Imports
-------

.. code:: ipython3

    import pymove.utils as utils
    import pymove as pm
    import datetime

Conversions
-----------

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



.. code:: ipython3

    move_data = pm.read_csv("geolife_sample.csv")
    move_data.generate_dist_time_speed_features()
    move_data.head()



.. parsed-literal::

    VBox(children=(HTML(value=''), IntProgress(value=0, max=2)))




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



To convert values, in ms, in label_speed column to kmh, you can use
**ms_to_kmh**

.. code:: ipython3

    utils.conversions.ms_to_kmh(move_data, inplace=True)
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

    utils.conversions.kmh_to_ms(move_data, inplace=True)
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

    utils.conversions.meters_to_kilometers(move_data, inplace=True)
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

    utils.conversions.kilometers_to_meters(move_data, inplace=True)
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

    utils.conversions.seconds_to_minutes(move_data, inplace=True)
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

    utils.conversions.minute_to_seconds(move_data, inplace=True)
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

    utils.conversions.seconds_to_minutes(move_data, inplace=True)
    utils.conversions.minute_to_hours(move_data, inplace=True)
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
**hours_to_minutes**

.. code:: ipython3

    utils.conversions.hours_to_minute(move_data, inplace=True)
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

    utils.conversions.minute_to_seconds(move_data, inplace=True)
    utils.conversions.seconds_to_hours(move_data, inplace=True)
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

    utils.conversions.hours_to_seconds(move_data, inplace=True)
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



To converts a datetime to an int representation in minutes, you can use
**to_min**.

.. code:: ipython3

    utils.datetime.datetime_to_min(datetime.datetime(2018, 6, 29, 8, 15, 27))




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

    '2021-07-13 19:56:01'



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

    3835166163375



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



To calculate the euclidean distance between two points on the earth, you
can use **euclidean_distance_in_meters**.

.. code:: ipython3

    utils.distances.euclidean_distance_in_meters(-3.797864,-38.501597,-3.797890, -38.501681)




.. parsed-literal::

    9.790407710249447



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

    (625.0, 360425.0, 3)



To perfomers interpolation and extrapolation, you can use
**interpolation**.

.. code:: ipython3

    utils.math.interpolation(15, 20, 65, 86, 5)




.. parsed-literal::

    6.799999999999999



Trajectories
------------

To read a csv file into a MoveDataFrame

.. code:: ipython3

    move_data = utils.trajectories.read_csv('geolife_sample.csv')
    type(move_data)




.. parsed-literal::

    pymove.core.pandas.PandasMoveDataFrame



To invert the keys values of a dictionary

.. code:: ipython3

    utils.trajectories.invert_dict({1: 'a', 2: 'b'})




.. parsed-literal::

    {'a': 1, 'b': 2}



To flatten a nested dictionary

.. code:: ipython3

    utils.trajectories.flatten_dict({'1': 'a', '2': {'3': 'b', '4': 'c'}})




.. parsed-literal::

    {'1': 'a', '2_3': 'b', '2_4': 'c'}



To flatten a dataframe with dict as row values

.. code:: ipython3

    df = move_data.head(3)
    df['dict_column'] = [{'a': 1}, {'b': 2}, {'c': 3}]
    df




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
          <th>dict_column</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>39.984094</td>
          <td>116.319236</td>
          <td>2008-10-23 05:53:05</td>
          <td>1</td>
          <td>{'a': 1}</td>
        </tr>
        <tr>
          <th>1</th>
          <td>39.984198</td>
          <td>116.319322</td>
          <td>2008-10-23 05:53:06</td>
          <td>1</td>
          <td>{'b': 2}</td>
        </tr>
        <tr>
          <th>2</th>
          <td>39.984224</td>
          <td>116.319402</td>
          <td>2008-10-23 05:53:11</td>
          <td>1</td>
          <td>{'c': 3}</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    utils.trajectories.flatten_columns(df, columns='dict_column')




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
          <th>dict_column_c</th>
          <th>dict_column_b</th>
          <th>dict_column_a</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>39.984094</td>
          <td>116.319236</td>
          <td>2008-10-23 05:53:05</td>
          <td>1</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>1</th>
          <td>39.984198</td>
          <td>116.319322</td>
          <td>2008-10-23 05:53:06</td>
          <td>1</td>
          <td>NaN</td>
          <td>2.0</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2</th>
          <td>39.984224</td>
          <td>116.319402</td>
          <td>2008-10-23 05:53:11</td>
          <td>1</td>
          <td>3.0</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
      </tbody>
    </table>
    </div>



To shift a sequence

.. code:: ipython3

    utils.trajectories.shift([1., 2., 3., 4.], 1)




.. parsed-literal::

    array([nan,  1.,  2.,  3.])



To fill a sequence with values from another

.. code:: ipython3

    l1 = ['a', 'b', 'c', 'd', 'e']
    utils.trajectories.fill_list_with_new_values(l1, [1, 2, 3])
    l1




.. parsed-literal::

    [1, 2, 3, 'd', 'e']



To transform a string representation back into a list

.. code:: ipython3

    utils.trajectories.object_for_array('[1,2,3,4,5]')




.. parsed-literal::

    array([1., 2., 3., 4., 5.], dtype=float32)



To convert a column with string representation back into a list

.. code:: ipython3

    df['list_column'] = ['[1,2]', '[3,4]', '[5,6]']

.. code:: ipython3

    df




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
          <th>dict_column</th>
          <th>list_column</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>39.984094</td>
          <td>116.319236</td>
          <td>2008-10-23 05:53:05</td>
          <td>1</td>
          <td>{'a': 1}</td>
          <td>[1,2]</td>
        </tr>
        <tr>
          <th>1</th>
          <td>39.984198</td>
          <td>116.319322</td>
          <td>2008-10-23 05:53:06</td>
          <td>1</td>
          <td>{'b': 2}</td>
          <td>[3,4]</td>
        </tr>
        <tr>
          <th>2</th>
          <td>39.984224</td>
          <td>116.319402</td>
          <td>2008-10-23 05:53:11</td>
          <td>1</td>
          <td>{'c': 3}</td>
          <td>[5,6]</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    utils.trajectories.column_to_array(df, column='list_column')




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
          <th>dict_column</th>
          <th>list_column</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>39.984094</td>
          <td>116.319236</td>
          <td>2008-10-23 05:53:05</td>
          <td>1</td>
          <td>{'a': 1}</td>
          <td>[1.0, 2.0]</td>
        </tr>
        <tr>
          <th>1</th>
          <td>39.984198</td>
          <td>116.319322</td>
          <td>2008-10-23 05:53:06</td>
          <td>1</td>
          <td>{'b': 2}</td>
          <td>[3.0, 4.0]</td>
        </tr>
        <tr>
          <th>2</th>
          <td>39.984224</td>
          <td>116.319402</td>
          <td>2008-10-23 05:53:11</td>
          <td>1</td>
          <td>{'c': 3}</td>
          <td>[5.0, 6.0]</td>
        </tr>
      </tbody>
    </table>
    </div>



Log
---

.. code:: ipython3

    mdf = pm.read_csv('geolife_sample.csv')

To cotrol the verbosity of pymove functions, use the logger

To change verbosity use the ``utils.log.set_verbosity`` method, or
create and environment variable named ``PYMOVE_VERBOSITY``

By default, the berbosity level is set to ``INFO``

.. code:: ipython3

    utils.log.logger




.. parsed-literal::

    <Logger pymove (INFO)>



``INFO`` shows only useful information, like progress bars

.. code:: ipython3

    mdf.generate_dist_features(inplace=False).head()



.. parsed-literal::

    VBox(children=(HTML(value=''), IntProgress(value=0, max=2)))




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



``DEBUG`` shows information from various steps in the functions

.. code:: ipython3

    utils.log.set_verbosity('DEBUG')
    mdf.generate_dist_features(inplace=False).head()


.. parsed-literal::

    ...Sorting by id and datetime to increase performance

    ...Set id as index to a higher performance


    Creating or updating distance features in meters...




.. parsed-literal::

    VBox(children=(HTML(value=''), IntProgress(value=0, max=2)))




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



``WARN`` hides all output except warnings and errors

.. code:: ipython3

    utils.log.set_verbosity('WARN')
    mdf.generate_dist_features(inplace=False).head()




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



Mem
---

.. code:: ipython3

    utils.log.set_verbosity('INFO')

Calculate size of variable

.. code:: ipython3

    utils.mem.total_size(mdf, verbose=True)


.. parsed-literal::

    Size in bytes: 6965040, Type: <class 'pymove.core.pandas.PandasMoveDataFrame'>




.. parsed-literal::

    6965040



Reduce size of dataframe

.. code:: ipython3

    utils.mem.reduce_mem_usage_automatic(mdf)


.. parsed-literal::

    Memory usage of dataframe is 6.64 MB
    Memory usage after optimization is: 2.70 MB
    Decreased by 59.4 %


Create a dataframe with the variables with largest memory footpring

.. code:: ipython3

    lst = [*range(10000)]

.. code:: ipython3

    utils.mem.top_mem_vars(globals())




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
          <th>var</th>
          <th>mem</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>move_data</td>
          <td>6.6 MiB</td>
        </tr>
        <tr>
          <th>1</th>
          <td>mdf</td>
          <td>2.7 MiB</td>
        </tr>
        <tr>
          <th>2</th>
          <td>lst</td>
          <td>88.0 KiB</td>
        </tr>
        <tr>
          <th>3</th>
          <td>Out</td>
          <td>2.2 KiB</td>
        </tr>
        <tr>
          <th>4</th>
          <td>df</td>
          <td>1.1 KiB</td>
        </tr>
        <tr>
          <th>5</th>
          <td>In</td>
          <td>648.0 B</td>
        </tr>
        <tr>
          <th>6</th>
          <td>l1</td>
          <td>96.0 B</td>
        </tr>
        <tr>
          <th>7</th>
          <td>matplotlib</td>
          <td>72.0 B</td>
        </tr>
        <tr>
          <th>8</th>
          <td>sys</td>
          <td>72.0 B</td>
        </tr>
        <tr>
          <th>9</th>
          <td>os</td>
          <td>72.0 B</td>
        </tr>
      </tbody>
    </table>
    </div>
