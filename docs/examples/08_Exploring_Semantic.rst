08 - Exploring Semantic
=======================

1. Imports
----------

.. code:: ipython3

    import pymove as pm
    from pymove.semantic import semantic

2. Load Data
------------

.. code:: ipython3

    move_df = pm.read_csv('geolife_sample.csv')
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



Detect outlier points considering distance traveled in the dataframe

.. code:: ipython3

    outliers = semantic.outliers(move_df)
    outliers[outliers['outlier']]



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
          <th>outlier</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>148</th>
          <td>1</td>
          <td>39.970511</td>
          <td>116.341455</td>
          <td>2008-10-23 10:32:53</td>
          <td>1452.319115</td>
          <td>1470.641291</td>
          <td>71.088460</td>
          <td>True</td>
        </tr>
        <tr>
          <th>338</th>
          <td>1</td>
          <td>39.995042</td>
          <td>116.326465</td>
          <td>2008-10-23 10:44:24</td>
          <td>10.801860</td>
          <td>10.274331</td>
          <td>1.465144</td>
          <td>True</td>
        </tr>
        <tr>
          <th>8133</th>
          <td>1</td>
          <td>39.991075</td>
          <td>116.188395</td>
          <td>2008-10-25 08:20:19</td>
          <td>5.090766</td>
          <td>6.247860</td>
          <td>1.295191</td>
          <td>True</td>
        </tr>
        <tr>
          <th>10175</th>
          <td>1</td>
          <td>40.015169</td>
          <td>116.311045</td>
          <td>2008-10-25 23:40:12</td>
          <td>23.454754</td>
          <td>24.899678</td>
          <td>3.766959</td>
          <td>True</td>
        </tr>
        <tr>
          <th>13849</th>
          <td>1</td>
          <td>39.977157</td>
          <td>116.327151</td>
          <td>2008-10-26 08:13:53</td>
          <td>11.212682</td>
          <td>10.221164</td>
          <td>1.004375</td>
          <td>True</td>
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
          <th>216877</th>
          <td>5</td>
          <td>39.992096</td>
          <td>116.329136</td>
          <td>2009-03-12 15:57:42</td>
          <td>7.035981</td>
          <td>6.182086</td>
          <td>1.909349</td>
          <td>True</td>
        </tr>
        <tr>
          <th>216927</th>
          <td>5</td>
          <td>39.998061</td>
          <td>116.326402</td>
          <td>2009-03-12 16:02:17</td>
          <td>16.758753</td>
          <td>19.151449</td>
          <td>4.051863</td>
          <td>True</td>
        </tr>
        <tr>
          <th>217456</th>
          <td>5</td>
          <td>40.001983</td>
          <td>116.328414</td>
          <td>2009-03-19 04:35:52</td>
          <td>179.564668</td>
          <td>191.030434</td>
          <td>15.276237</td>
          <td>True</td>
        </tr>
        <tr>
          <th>217465</th>
          <td>5</td>
          <td>40.001433</td>
          <td>116.321387</td>
          <td>2009-03-19 04:41:02</td>
          <td>77.928727</td>
          <td>75.686512</td>
          <td>16.676141</td>
          <td>True</td>
        </tr>
        <tr>
          <th>217479</th>
          <td>5</td>
          <td>40.003626</td>
          <td>116.317695</td>
          <td>2009-03-19 05:02:52</td>
          <td>9.725231</td>
          <td>7.573682</td>
          <td>2.463175</td>
          <td>True</td>
        </tr>
      </tbody>
    </table>
    <p>383 rows × 8 columns</p>
    </div>



.. code:: ipython3

    move_df.get_bbox()




.. parsed-literal::

    (22.147577, 113.548843, 41.132062, 121.156224)



Detect points outside of a bounding box

.. code:: ipython3

    fake_bbox = (20, 110, 40, 120)
    out_bbox = semantic.create_or_update_out_of_the_bbox(move_df, fake_bbox)
    out_bbox[out_bbox['out_bbox']]




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
          <th>out_bbox</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>415</th>
          <td>40.000026</td>
          <td>116.322214</td>
          <td>2008-10-23 10:48:31</td>
          <td>1</td>
          <td>True</td>
        </tr>
        <tr>
          <th>416</th>
          <td>40.000082</td>
          <td>116.322072</td>
          <td>2008-10-23 10:48:33</td>
          <td>1</td>
          <td>True</td>
        </tr>
        <tr>
          <th>417</th>
          <td>40.000164</td>
          <td>116.321996</td>
          <td>2008-10-23 10:48:37</td>
          <td>1</td>
          <td>True</td>
        </tr>
        <tr>
          <th>418</th>
          <td>40.000245</td>
          <td>116.321964</td>
          <td>2008-10-23 10:48:40</td>
          <td>1</td>
          <td>True</td>
        </tr>
        <tr>
          <th>419</th>
          <td>40.000312</td>
          <td>116.321921</td>
          <td>2008-10-23 10:48:45</td>
          <td>1</td>
          <td>True</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>217643</th>
          <td>40.000205</td>
          <td>116.327173</td>
          <td>2009-03-19 05:45:37</td>
          <td>5</td>
          <td>True</td>
        </tr>
        <tr>
          <th>217644</th>
          <td>40.000128</td>
          <td>116.327171</td>
          <td>2009-03-19 05:45:42</td>
          <td>5</td>
          <td>True</td>
        </tr>
        <tr>
          <th>217645</th>
          <td>40.000069</td>
          <td>116.327179</td>
          <td>2009-03-19 05:45:47</td>
          <td>5</td>
          <td>True</td>
        </tr>
        <tr>
          <th>217646</th>
          <td>40.000001</td>
          <td>116.327219</td>
          <td>2009-03-19 05:45:52</td>
          <td>5</td>
          <td>True</td>
        </tr>
        <tr>
          <th>217651</th>
          <td>40.000015</td>
          <td>116.327433</td>
          <td>2009-03-19 05:46:17</td>
          <td>5</td>
          <td>True</td>
        </tr>
      </tbody>
    </table>
    <p>104787 rows × 5 columns</p>
    </div>



Detects points with no gps signal, given by the time between adjacent
points

.. code:: ipython3

    deactivated = semantic.create_or_update_gps_deactivated_signal(move_df)
    deactivated[deactivated['deactivated_signal']]



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
          <th>time_to_prev</th>
          <th>time_to_next</th>
          <th>time_prev_to_next</th>
          <th>deactivated_signal</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>147</th>
          <td>1</td>
          <td>39.978068</td>
          <td>116.327554</td>
          <td>2008-10-23 06:01:57</td>
          <td>5.0</td>
          <td>16256.0</td>
          <td>16261.0</td>
          <td>True</td>
        </tr>
        <tr>
          <th>148</th>
          <td>1</td>
          <td>39.970511</td>
          <td>116.341455</td>
          <td>2008-10-23 10:32:53</td>
          <td>16256.0</td>
          <td>7.0</td>
          <td>16263.0</td>
          <td>True</td>
        </tr>
        <tr>
          <th>960</th>
          <td>1</td>
          <td>40.013803</td>
          <td>116.306531</td>
          <td>2008-10-23 12:04:28</td>
          <td>2.0</td>
          <td>41796.0</td>
          <td>41798.0</td>
          <td>True</td>
        </tr>
        <tr>
          <th>961</th>
          <td>1</td>
          <td>40.013867</td>
          <td>116.306473</td>
          <td>2008-10-23 23:41:04</td>
          <td>41796.0</td>
          <td>2.0</td>
          <td>41798.0</td>
          <td>True</td>
        </tr>
        <tr>
          <th>3088</th>
          <td>1</td>
          <td>39.977899</td>
          <td>116.327063</td>
          <td>2008-10-24 06:35:50</td>
          <td>2.0</td>
          <td>61695.0</td>
          <td>61697.0</td>
          <td>True</td>
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
          <th>216997</th>
          <td>5</td>
          <td>40.007003</td>
          <td>116.323674</td>
          <td>2009-03-13 13:29:06</td>
          <td>30157.0</td>
          <td>5.0</td>
          <td>30162.0</td>
          <td>True</td>
        </tr>
        <tr>
          <th>217054</th>
          <td>5</td>
          <td>40.010537</td>
          <td>116.322052</td>
          <td>2009-03-13 13:34:01</td>
          <td>5.0</td>
          <td>57426.0</td>
          <td>57431.0</td>
          <td>True</td>
        </tr>
        <tr>
          <th>217055</th>
          <td>5</td>
          <td>40.009639</td>
          <td>116.322056</td>
          <td>2009-03-14 05:31:07</td>
          <td>57426.0</td>
          <td>5.0</td>
          <td>57431.0</td>
          <td>True</td>
        </tr>
        <tr>
          <th>217452</th>
          <td>5</td>
          <td>39.990464</td>
          <td>116.333510</td>
          <td>2009-03-14 06:47:12</td>
          <td>2.0</td>
          <td>424105.0</td>
          <td>424107.0</td>
          <td>True</td>
        </tr>
        <tr>
          <th>217453</th>
          <td>5</td>
          <td>40.001467</td>
          <td>116.326665</td>
          <td>2009-03-19 04:35:37</td>
          <td>424105.0</td>
          <td>5.0</td>
          <td>424110.0</td>
          <td>True</td>
        </tr>
      </tbody>
    </table>
    <p>420 rows × 8 columns</p>
    </div>



Detects points with jumps, defined by the maximum distance between
adjacent points

.. code:: ipython3

    jump = semantic.create_or_update_gps_jump(move_df, )
    print(jump[jump['gps_jump']].shape)
    jump[jump['gps_jump']].head()



.. parsed-literal::

    VBox(children=(HTML(value=''), IntProgress(value=0, max=2)))


.. parsed-literal::

    (46, 8)




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
          <th>gps_jump</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>3088</th>
          <td>1</td>
          <td>39.977899</td>
          <td>116.327063</td>
          <td>2008-10-24 06:35:50</td>
          <td>0.140088</td>
          <td>4361.216241</td>
          <td>4361.148665</td>
          <td>True</td>
        </tr>
        <tr>
          <th>3089</th>
          <td>1</td>
          <td>40.013812</td>
          <td>116.306483</td>
          <td>2008-10-24 23:44:05</td>
          <td>4361.216241</td>
          <td>7.587244</td>
          <td>4358.356247</td>
          <td>True</td>
        </tr>
        <tr>
          <th>12434</th>
          <td>1</td>
          <td>39.974821</td>
          <td>116.333828</td>
          <td>2008-10-26 03:27:37</td>
          <td>1.358606</td>
          <td>4536.318481</td>
          <td>4536.121843</td>
          <td>True</td>
        </tr>
        <tr>
          <th>12435</th>
          <td>1</td>
          <td>39.976599</td>
          <td>116.387014</td>
          <td>2008-10-26 03:45:46</td>
          <td>4536.318481</td>
          <td>4.280041</td>
          <td>4535.822332</td>
          <td>True</td>
        </tr>
        <tr>
          <th>23936</th>
          <td>1</td>
          <td>39.978222</td>
          <td>116.327002</td>
          <td>2008-10-31 08:06:33</td>
          <td>10.665636</td>
          <td>4328.751469</td>
          <td>4318.102530</td>
          <td>True</td>
        </tr>
      </tbody>
    </table>
    </div>



Determines if a point belongs to a short trajectory.

.. code:: ipython3

    short = semantic.create_or_update_short_trajectory(move_df)
    short[short['short_traj']]



.. parsed-literal::

    VBox(children=(HTML(value=''), IntProgress(value=0, max=2)))



.. parsed-literal::

    VBox(children=(HTML(value=''), IntProgress(value=0, max=2)))



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
          <th>tid_part</th>
          <th>short_traj</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>148</th>
          <td>1</td>
          <td>39.970511</td>
          <td>116.341455</td>
          <td>2008-10-23 10:32:53</td>
          <td>1452.319115</td>
          <td>16256.0</td>
          <td>0.089340</td>
          <td>2</td>
          <td>True</td>
        </tr>
        <tr>
          <th>18244</th>
          <td>1</td>
          <td>39.993663</td>
          <td>116.325751</td>
          <td>2008-10-27 23:49:47</td>
          <td>233.351618</td>
          <td>4.0</td>
          <td>58.337905</td>
          <td>18</td>
          <td>True</td>
        </tr>
        <tr>
          <th>18795</th>
          <td>1</td>
          <td>39.983927</td>
          <td>116.309349</td>
          <td>2008-10-28 13:21:25</td>
          <td>480.465717</td>
          <td>8147.0</td>
          <td>0.058975</td>
          <td>22</td>
          <td>True</td>
        </tr>
        <tr>
          <th>26941</th>
          <td>1</td>
          <td>39.982361</td>
          <td>116.330762</td>
          <td>2008-11-01 06:02:13</td>
          <td>270.452069</td>
          <td>3.0</td>
          <td>90.150690</td>
          <td>39</td>
          <td>True</td>
        </tr>
        <tr>
          <th>27878</th>
          <td>1</td>
          <td>40.017806</td>
          <td>116.307530</td>
          <td>2008-11-02 09:44:34</td>
          <td>454.090137</td>
          <td>19527.0</td>
          <td>0.023254</td>
          <td>44</td>
          <td>True</td>
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
        </tr>
        <tr>
          <th>217471</th>
          <td>5</td>
          <td>40.001707</td>
          <td>116.318926</td>
          <td>2009-03-19 04:42:22</td>
          <td>36.849631</td>
          <td>5.0</td>
          <td>7.369926</td>
          <td>404</td>
          <td>True</td>
        </tr>
        <tr>
          <th>217472</th>
          <td>5</td>
          <td>40.001784</td>
          <td>116.318965</td>
          <td>2009-03-19 04:42:27</td>
          <td>9.183862</td>
          <td>5.0</td>
          <td>1.836772</td>
          <td>404</td>
          <td>True</td>
        </tr>
        <tr>
          <th>217473</th>
          <td>5</td>
          <td>40.002218</td>
          <td>116.320306</td>
          <td>2009-03-19 04:42:32</td>
          <td>123.999483</td>
          <td>5.0</td>
          <td>24.799897</td>
          <td>404</td>
          <td>True</td>
        </tr>
        <tr>
          <th>217474</th>
          <td>5</td>
          <td>40.004917</td>
          <td>116.314376</td>
          <td>2009-03-19 04:49:37</td>
          <td>587.526625</td>
          <td>425.0</td>
          <td>1.382416</td>
          <td>404</td>
          <td>True</td>
        </tr>
        <tr>
          <th>217475</th>
          <td>5</td>
          <td>40.004955</td>
          <td>116.313697</td>
          <td>2009-03-19 04:49:42</td>
          <td>57.987365</td>
          <td>5.0</td>
          <td>11.597473</td>
          <td>404</td>
          <td>True</td>
        </tr>
      </tbody>
    </table>
    <p>1151 rows × 9 columns</p>
    </div>
