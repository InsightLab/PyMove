# Use PyMove and go much further

---

## Information

<table>
<tr>
  <td>Package Status</td>
  <td>
    <a href="https://pypi.org/project/pymove/">
      <img src="https://img.shields.io/pypi/status/pymove.svg" alt="Package status" />
    </a>
  </td>
</tr>
<tr>
  <td>License</td>
  <td>
    <a href="https://github.com/InsightLab/PyMove/blob/master/LICENSE">
      <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="Package license" />
    </a>
</td>
</tr>
<tr>
  <td>Python Version</td>
  <td>
    <a href="https://www.python.org/doc/versions/">
      <img src="https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue" alt="Python compatible versions" />
    </a>
</td>
</tr>
<tr>
  <td>Build Status</td>
  <td>
    <a href="https://travis-ci.org/InsightLab/PyMove/">
      <img src="https://api.travis-ci.org/InsightLab/PyMove.svg?branch=master" alt="Travis build status" />
    </a>
  </td>
</tr>
<tr>
  <td>Platforms</td>
  <td>
    <a href="https://anaconda.org/conda-forge/pymove">
      <img src="https://img.shields.io/conda/pn/conda-forge/pymove.svg" alt="Platforms" />
    </a>
  </td>
</tr>
<tr>
  <td>All Platforms</td>
  <td>
    <a href="https://dev.azure.com/conda-forge/feedstock-builds/_build/latest?definitionId=9753&branchName=master">
      <img src="https://dev.azure.com/conda-forge/feedstock-builds/_apis/build/status/pymove-feedstock?branchName=master" alt="conda-forge build status" />
    </a>
  </td>
</tr>
<tr>
  <td>PyPi Downloads</td>
  <td>
    <a href="https://pypi.org/project/pymove/#files" alt="PyPi downloads">
      <img src="https://img.shields.io/pypi/dm/pymove" alt="PyPi downloads" alt="PyPi downloads" />
    </a>
  </td>
</tr>
<tr>
  <td>PyPi version</td>
  <td>
    <a href="https://pypi.org/project/pymove/#history" alt="PyPi version">
      <img src="https://img.shields.io/pypi/v/pymove" alt="PyPi version" alt="PyPi version" />
    </a>
  </td>
</tr>
<tr>
  <td>Conda Downloads</td>
  <td>
    <a href="https://anaconda.org/conda-forge/pymove">
      <img src="https://img.shields.io/conda/dn/conda-forge/pymove.svg" alt="Conda downloads" />
    </a>
  </td>
</tr>
<tr>
  <td>Conda version</td>
  <td>
    <a href="https://anaconda.org/conda-forge/pymove">
      <img src="https://img.shields.io/conda/vn/conda-forge/pymove.svg" alt="Conda version" />
    </a>
  </td>
</tr>
<tr>
  <td>Stars</td>
  <td>
    <a href="https://github.com/InsightLab/PyMove/stargazers">
      <img src="https://img.shields.io/github/stars/InsightLab/PyMove?style=social" alt="Github stars" />
    </a>
  </td>
</tr>
<tr>
  <td>Forks</td>
  <td>
    <a href="https://github.com/InsightLab/PyMove/network/members">
      <img src="https://img.shields.io/github/forks/InsightLab/PyMove?style=social" alt="Github forks" />
    </a>
  </td>
</tr>
<tr>
  <td>Issues</td>
  <td>
    <a href="https://github.com/InsightLab/PyMove/issues">
      <img src="https://img.shields.io/github/issues/InsightLab/PyMove" alt="Github issues" />
    </a>
  </td>
</tr>
<tr>
  <td>Codacy Badge</td>
  <td>
    <a href="https://www.codacy.com/gh/InsightLab/PyMove?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=InsightLab/PyMove&amp;utm_campaign=Badge_Grade">
      <img src="https://api.codacy.com/project/badge/Grade/26c581fbe1ee42e78a9adc50b7372ceb" alt="Codacy badge" />
    </a>
  </td>
</tr>
</table>

---

## What is PyMove

PyMove is a Python library for processing and visualization of trajectories and other spatial-temporal data.

We will also release wrappers to some useful Java libraries frequently used in the mobility domain.

---

## Main Features

PyMove **proposes**:

-   A familiar and similar syntax to Pandas;
-   Clear documentation;
-   Extensibility, since you can implement your main data structure by manipulating other data structures such as Dask DataFrame, numpy arrays, etc., in addition to adding new modules;
-   Flexibility, as the user can switch between different data structures;
-   Operations for data preprocessing, pattern mining and data visualization.

---

## Creating a Virtual Environment

It is recommended to create a virtual environment to use pymove. Requirements: Anaconda Python distribution installed and accessible

1.  In the terminal client enter the following where `yourenvname` is the name you want to call your environment, and replace `x.x` with the Python version you wish to use. (To see a list of available python versions first, type conda search "^python$" and press enter.)
    -   `conda create -n <yourenvname> python=x.x`
    -   Press y to proceed. This will install the Python version and all the associated anaconda packaged libraries at `path_to_your_anaconda_location/anaconda/envs/yourenvname`

2.  Activate your virtual environment. To activate or switch into your virtual environment, simply type the following where yourenvname is the name you gave to your environment at creation.
    -   `conda activate <yourenvname>`

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

2.  Make a branch developer
    -   `git branch developer`

3.  Switch to a new branch
    -   `git checkout developer`

4.  Make a pull of branch
    -   `git pull origin developer`

5.  Switch to folder PyMove
    -   `cd PyMove`

6.  Install pymove in developer mode
    -   `pip install -e .`

---

### For windows users

If you installed from `pip` or `github`, you may encounter an error related to `shapely` due to some dll dependencies. To fix this, run `conda install shapely`.

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
