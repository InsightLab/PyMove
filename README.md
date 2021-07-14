# Use PyMove and go much further

---

## Information

|||
|--- |--- |
|Package Status|[<img src="https://img.shields.io/pypi/status/pymove?style=for-the-badge" />](https://pypi.org/project/pymove/)|
|License|[<img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" />](https://github.com/InsightLab/PyMove/blob/master/LICENSE)|
|Python Version|[<img src="https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue?style=for-the-badge" />](https://www.python.org/doc/versions/)|
|Platforms|[<img src="https://img.shields.io/conda/pn/conda-forge/pymove?style=for-the-badge" />](https://anaconda.org/conda-forge/pymove)|
|Build Status|[<img src="https://img.shields.io/azure-devops/build/conda-forge/84710dde-1620-425b-80d0-4cf5baca359d/9753/master?style=for-the-badge" />](https://dev.azure.com/conda-forge/feedstock-builds/_build/latest?definitionId=9753&branchName=master)|
|PyPi version|[<img src="https://img.shields.io/pypi/v/pymove?style=for-the-badge" />](https://pypi.org/project/pymove/#history)|
|PyPi Downloads|[<img src="https://img.shields.io/pypi/dm/pymove?style=for-the-badge" />](https://pypi.org/project/pymove/#files)|
|Conda version|[<img src="https://img.shields.io/conda/vn/conda-forge/pymove?style=for-the-badge" />](https://anaconda.org/conda-forge/pymove)|
|Conda Downloads|[<img src="https://img.shields.io/conda/dn/conda-forge/pymove?style=for-the-badge" />](https://anaconda.org/conda-forge/pymove/files)|
|Code Quality|[<img src="https://img.shields.io/codacy/grade/26c581fbe1ee42e78a9adc50b7372ceb?style=for-the-badge" />](https://app.codacy.com/gh/InsightLab/PyMove/dashboard)|
|Code Coverage|[<img src="https://img.shields.io/codacy/coverage/26c581fbe1ee42e78a9adc50b7372ceb?style=for-the-badge" />](https://app.codacy.com/gh/InsightLab/PyMove/files)|

---

## What is PyMove

PyMove is a Python library for processing and visualization
 of trajectories and other spatial-temporal data.

We will also release wrappers to some useful Java libraries
 frequently used in the mobility domain.

Read the full documentation on [ReadTheDocs](https://pymove.readthedocs.io/en/latest/)

---

## Main Features

PyMove **proposes**:

-   A familiar and similar syntax to Pandas;

-   Clear documentation;

-   Extensibility, since you can implement your main data structure by
 manipulating other data structures such as
 Dask DataFrame, numpy arrays, etc., in addition to adding new modules;

-   Flexibility, as the user can switch between different data structures;

-   Operations for data preprocessing, pattern mining and data visualization.

---

## Creating a Virtual Environment

It is recommended to create a virtual environment to use pymove.

Requirements: Anaconda Python distribution installed and accessible

1.  In the terminal client enter the following where `env_name` is the name
 you want to call your environment, and replace `x.x` with the Python version
 you wish to use. (To see a list of available python versions first,
 type conda search "^python$" and press enter.)
    -   ``conda create -n <env_name> python=x.x``

    -   `Press y to proceed. This will install the Python version and all the`
 associated anaconda packaged libraries at `path_to_your_anaconda_location/anaconda/envs/env_name`

2.  Activate your virtual environment. To activate or switch into your
 virtual environment, simply type the following where yourenvname is the
 name you gave to your environment at creation.
    -   ``conda activate <env_name>``

3.  Now install the package from either `conda`, `pip` or `github`

---

## [Conda](https://anaconda.org/conda-forge/pymove) instalation

1.  `conda install -c conda-forge pymove`

## [Pip](https://pypi.org/project/pymove) installation

1.  `pip install pymove`

---

## [Github](https://github.com/InsightLab/PyMove) installation

1.  Clone this repository
    -   ``git clone https://github.com/InsightLab/PyMove``

2.  Switch to folder PyMove
    -   ``cd PyMove``

3.  Switch to a new branch
    -   ``git checkout -b developer``

4.  Make a pull of branch
    -   ``git pull origin developer``

5.  Install pymove in developer mode
    -   ``make dev``

### For windows users

If you installed from `pip` or `github`, you may encounter an error related to
 `shapely` due to some dll dependencies. To fix this, run `conda install shapely`.

---

## Examples

You can see examples of how to use PyMove [here](https://github.com/InsightLab/PyMove/tree/master/notebooks)

---

## Mapping PyMove methods with the Paradigms of Trajectory Data Mining

![](.mapping.png)
[ZHENG 2015](https://www.microsoft.com/en-us/research/publication/trajectory-data-mining-an-overview/).

-   1: **Spatial Trajectories** &rarr; `pymove.core`
    -   `MoveDataFrame`
    -   `DiscreteMoveDataFrame`

-   2: **Stay Point Detection** &rarr; `pymove.preprocessing.stay_point_detection`
    -   `create_or_update_move_stop_by_dist_time`
    -   `create_or_update_move_and_stop_by_radius`

-   3: **Map-Matching** &rarr; `pymove-osmnx`

-   4: **Noise Filtering** &rarr; `pymove.preprocessing.filters`
    -   `by_bbox`
    -   `by_datetime`
    -   `by_label`
    -   `by_id`
    -   `by_tid`
    -   `clean_consecutive_duplicates`
    -   `clean_gps_jumps_by_distance`
    -   `clean_gps_nearby_points_by_distances`
    -   `clean_gps_nearby_points_by_speed`
    -   `clean_gps_speed_max_radius`
    -   `clean_trajectories_with_few_points`
    -   `clean_trajectories_short_and_few_points`
    -   `clean_id_by_time_max`

-   5: **Compression** &rarr; `pymove.preprocessing.compression`
    -   `compress_segment_stop_to_point`

-   6: **Segmentation** &rarr; `pymove.preprocessing.segmentation`
    -   `bbox_split`
    -   `by_dist_time_speed`
    -   `by_max_dist`
    -   `by_max_time`
    -   `by_max_speed`

-   7: **Distance Measures** &rarr; `pymove.distances`
    -   `medp`
    -   `medt`
    -   `euclidean_distance_in_meters`
    -   `haversine`

-   8: **Query Historical Trajectories** &rarr; `pymove.query.query`
    -   `range_query`
    -   `knn_query`

-   9: **Managing Recent Trajectories**

-   10: **Privacy Preserving**

-   11: **Reducing Uncertainty**

-   12: **Moving Together Patterns**

-   13: **Clustering** &rarr; `pymove.models.pattern_mining.clustering`
    -   `elbow_method`
    -   `gap_statistics`
    -   `dbscan_clustering`

-   14: **Freq. Seq. Patterns**

-   15: **Periodic Patterns**

-   16: **Trajectory Classification**

-   17: **Trajectory Outlier / Anomaly Detection** &rarr; `pymove.semantic.semantic`
    -   `outliers`
    -   `create_or_update_out_of_the_bbox`
    -   `create_or_update_gps_deactivated_signal`
    -   `create_or_update_gps_jump`
    -   `create_or_update_short_trajectory`
    -   `create_or_update_gps_block_signal`
    -   `filter_block_signal_by_repeated_amount_of_points`
    -   `filter_block_signal_by_time`
    -   `filter_longer_time_to_stop_segment_by_id`

---

## Cite

The library was originally created during the bachelor's thesis of 2 students from the Federal University of Ceará, so you can cite using both works.

```txt
@mastersthesis{arina2019,
	title        = {Uma arquitetura e implementação do módulo de pré-processamento para biblioteca PyMove},
	author       = {Arina De Jesus Amador Monteiro Sanches},
	year         = 2019,
	school       = {Universidade Federal Do Ceará},
	type         = {Bachelor's thesis}
}
@mastersthesis{andreza2019,
	title        = {Uma arquitetura e implementação do módulo de visualização para biblioteca PyMove},
	author       = {Andreza Fernandes De Oliveira},
	year         = 2019,
	school       = {Universidade Federal Do Ceará},
	type         = {Bachelor's thesis}
}
```

---

## Publications

-   [Uma arquitetura e implementação do módulo de pré-processamento para biblioteca PyMove](http://repositorio.ufc.br/handle/riufc/58551)
-   [Uma arquitetura e implementação do módulo de visualização para biblioteca PyMove](http://repositorio.ufc.br/handle/riufc/58550)
-   [Avaliação de técnicas de aumento de dados para trajetórias](http://www.repositorio.ufc.br/handle/riufc/58958)
-   [Implementação de algoritmos para análise de similaridade de trajetória na biblioteca PyMove](http://www.repositorio.ufc.br/handle/riufc/58957)

---

## Useful list of related libraries and links

-   [Handling GPS Data with Python](https://github.com/FlorianWilhelm/gps_data_with_python/tree/master/notebooks)
-   [mplleaflet - Easily convert matplotlib plots from Python into interactive Leaflet web maps](https://github.com/jwass/mplleaflet)
-   [Pykalman](https://github.com/pykalman/pykalman)
-   [Ramer-Douglas-Peucker algorithm](https://github.com/fhirschmann/rdp)
-   [Knee point detection in Python](https://github.com/arvkevi/kneed)
-   [TrajSuite Java Library](https://github.com/lukehb/TrajSuite)
-   [GraphHopper Map-Matching Java Library](https://github.com/graphhopper/map-matching)
