# Use PyMove and go much further

---

## Information

<table>
<tr>
  <td>Package Status</td>
  <td>
    <a href="https://pypi.org/project/pymove/">
      <img src="https://img.shields.io/pypi/status/pymove?style=for-the-badge" alt="Package status" />
    </a>
  </td>
</tr>
<tr>
  <td>License</td>
  <td>
    <a href="https://github.com/InsightLab/PyMove/blob/master/LICENSE">
      <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="Package license" />
    </a>
</td>
</tr>
<tr>
  <td>Python Version</td>
  <td>
    <a href="https://www.python.org/doc/versions/">
      <img src="https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue?style=for-the-badge" alt="Python compatible versions" />
    </a>
</td>
</tr>
<tr>
  <td>Build Status</td>
  <td>
    <a href="https://travis-ci.org/InsightLab/PyMove/">
      <img src="https://img.shields.io/travis/InsightLab/pymove/master?style=for-the-badge" alt="Travis build status" />
    </a>
  </td>
</tr>
<tr>
  <td>Platforms</td>
  <td>
    <a href="https://anaconda.org/conda-forge/pymove">
      <img src="https://img.shields.io/conda/pn/conda-forge/pymove?style=for-the-badge" alt="Platforms" />
    </a>
  </td>
</tr>
<tr>
  <td>All Platforms</td>
  <td>
    <a href="https://dev.azure.com/conda-forge/feedstock-builds/_build/latest?definitionId=9753&branchName=master">
      <img src="https://img.shields.io/azure-devops/build/conda-forge/84710dde-1620-425b-80d0-4cf5baca359d/9753/master?style=for-the-badge" alt="conda-forge build status" />
    </a>
  </td>
</tr>
<tr>
  <td>PyPi Downloads</td>
  <td>
    <a href="https://pypi.org/project/pymove/#files" alt="PyPi downloads">
      <img src="https://img.shields.io/pypi/dm/pymove?style=for-the-badge" alt="PyPi downloads" alt="PyPi downloads" />
    </a>
  </td>
</tr>
<tr>
  <td>PyPi version</td>
  <td>
    <a href="https://pypi.org/project/pymove/#history" alt="PyPi version">
      <img src="https://img.shields.io/pypi/v/pymove?style=for-the-badge" alt="PyPi version" alt="PyPi version" />
    </a>
  </td>
</tr>
<tr>
  <td>Conda Downloads</td>
  <td>
    <a href="https://anaconda.org/conda-forge/pymove">
      <img src="https://img.shields.io/conda/dn/conda-forge/pymove?style=for-the-badge" alt="Conda downloads" />
    </a>
  </td>
</tr>
<tr>
  <td>Conda version</td>
  <td>
    <a href="https://anaconda.org/conda-forge/pymove">
      <img src="https://img.shields.io/conda/vn/conda-forge/pymove?style=for-the-badge" alt="Conda version" />
    </a>
  </td>
</tr>
<tr>
  <td>Stars</td>
  <td>
    <a href="https://github.com/InsightLab/PyMove/stargazers">
      <img src="https://img.shields.io/github/stars/InsightLab/PyMove?style=for-the-badge&color=blueviolet" alt="Github stars" />
    </a>
  </td>
</tr>
<tr>
  <td>Forks</td>
  <td>
    <a href="https://github.com/InsightLab/PyMove/network/members">
      <img src="https://img.shields.io/github/forks/InsightLab/PyMove?style=for-the-badge&color=blueviolet" alt="Github forks" />
    </a>
  </td>
</tr>
<tr>
  <td>Issues</td>
  <td>
    <a href="https://github.com/InsightLab/PyMove/issues">
      <img src="https://img.shields.io/github/issues/InsightLab/PyMove?style=for-the-badge" alt="Github issues" />
    </a>
  </td>
</tr>
<tr>
  <td>Code Quality</td>
  <td>
    <a href="https://www.codacy.com/gh/InsightLab/PyMove?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=InsightLab/PyMove&amp;utm_campaign=Badge_Grade">
      <img src="https://img.shields.io/codacy/grade/26c581fbe1ee42e78a9adc50b7372ceb?style=for-the-badge" alt="Code quality" />
    </a>
  </td>
</tr>
<tr>
  <td>Code Coverage</td>
  <td>
    <a href="https://www.codacy.com/gh/InsightLab/PyMove?utm_source=github.com&utm_medium=referral&utm_content=InsightLab/PyMove&utm_campaign=Badge_Coverage">
      <img src="https://img.shields.io/codacy/coverage/26c581fbe1ee42e78a9adc50b7372ceb?style=for-the-badge" alt="Code coverage" />
    </a>
  </td>
</tr>
</table>

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
    -   `conda create -n <env_name> python=x.x`

    -   Press y to proceed. This will install the Python version and all the
 associated anaconda packaged libraries at `path_to_your_anaconda_location/anaconda/envs/env_name`

2.  Activate your virtual environment. To activate or switch into your
 virtual environment, simply type the following where yourenvname is the
 name you gave to your environment at creation.
    -   `conda activate <env_name>`

3.  Now install the package from either `conda`, `pip` or `github`

---

## [Conda](https://anaconda.org/conda-forge/pymove) instalation

1.  `conda install -c conda-forge pymove`

## [Pip](https://pypi.org/project/pymove) installation

1.  `pip install pymove`

---

## [Github](https://github.com/InsightLab/PyMove) installation

1.  Clone this repository
    -   `git clone https://github.com/InsightLab/PyMove`

2.  Switch to folder PyMove
    -   `cd PyMove`

3.  Switch to a new branch
    -   `git checkout -b developer`

4.  Make a pull of branch
    -   `git pull origin developer`

5.  Install pymove in developer mode
    -   `pip install -r requirements-dev.txt`

### For windows users

If you installed from `pip` or `github`, you may encounter an error related to
 `shapely` due to some dll dependencies. To fix this, run `conda install shapely`.

---

## Examples

You can access examples of how to use PyMove [here](examples)

---

## Papers

(list of publications using/with Pymove)

---

## Useful list of related libraries and links

-   [Handling GPS Data with Python](https://github.com/FlorianWilhelm/gps_data_with_python/tree/master/notebooks)
-   [mplleaflet - Easily convert matplotlib plots from Python into interactive Leaflet web maps](https://github.com/jwass/mplleaflet)
-   [Pykalman](https://github.com/pykalman/pykalman)
-   [Ramer-Douglas-Peucker algorithm](https://github.com/fhirschmann/rdp)
-   [Knee point detection in Python](https://github.com/arvkevi/kneed)
-   [TrajSuite Java Library](https://github.com/lukehb/TrajSuite)
-   [GraphHopper Map-Matching Java Library](https://github.com/graphhopper/map-matching)
