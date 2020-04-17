# Use PyMove and go much further...

---

# Informations

<table>
<tr>
  <td>Package Status</td>
  <td>
    <a href="https://pypi.org/project/pymove/">
    <img src="https://img.shields.io/pypi/status/pymove.svg" alt="status" />
    </a>
  </td>
</tr>
<tr>
  <td>License</td>
  <td>
    <a href="https://github.com/InsightLab/PyMove/blob/developer/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="license" />
    </a>
</td>
</tr>
<tr>
  <td>Python Version</td>
  <td>
    <a href="https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue">
    <img src="https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue" alt="license" />
    </a>
</td>
</tr>
<tr>
  <td>Build Status</td>
  <td>
    <a href="https://travis-ci.org/InsightLab/PyMove/">
    <img src="https://api.travis-ci.org/InsightLab/PyMove.svg?branch=developer" alt="travis build status" />
    </a>
  </td>
</tr>
<tr>
  <td>Downloads</td>
  <td>
    <img src="https://img.shields.io/pypi/dd/pymove" alt="PyPi downloads" />
    </a>
  </td>
</tr>
<tr>
  <td>Stars</td>
  <td>
    <a href="https://github.com/InsightLab/PyMove/stargazers">
    <img src="https://img.shields.io/github/stars/InsightLab/PyMove?style=social"/>
    </a>
  </td>
</tr>
<tr>
  <td>Forks</td>
  <td>
      <a href="https://github.com/InsightLab/PyMove/network/members">
    <img src="https://img.shields.io/github/forks/InsightLab/PyMove?style=social"/>
    </a>
  </td>
</tr>
<tr>
  <td>Issues</td>
  <td>
    <a href="https://github.com/InsightLab/PyMove/issues">
    <img src="https://img.shields.io/github/issues/InsightLab/PyMove"/>
    </a>
  </td>
</tr>
</table>

---

# What is PyMove?

PyMove is a Python library for processing and visualization of trajectories and other spatial-temporal data.

We will also release wrappers to some useful Java libraries frequently used in the mobility domain.

---

# Main Features

PyMove **proposes**:
-  A familiar and similar syntax to Pandas;
-  Clear documentation;
-  Extensibility, since you can implement your main data structure by manipulating other data structures such as Dask DataFrame, numpy arrays, etc., in addition to adding new modules;
-  Flexibility, as the user can switch between different data structures;
-  Operations for data preprocessing, pattern mining and data visualization.

---

# Github installation

1. Clone this repository

`git clone https://github.com/InsightLab/PyMove`

2. Make a branch developer

`git branch developer`

3. Switch to a new branch 

`git checkout developer`

4. Make a pull of branch

`git pull origin developer`

5. Switch to folder PyMove

`cd PyMove`

6. Install pymove in developer mode

`pip install -e .`

---

# Pip installation

1. `pip install pymove`

---

# Creating Virtual Environment
In case of an error when running the library, it is recommended to create a virtual environment to use pymove. Requirements: 
Anaconda Python distribution installed and accessible

1. In the terminal client enter the following where yourenvname is the name you want to call your environment, and replace x.x with the Python version you wish to use. (To see a list of available python versions first, type conda search "^python$" and press enter.)

`conda create -n <yourenvname> python=x.x`

Press y to proceed. This will install the Python version and all the associated anaconda packaged libraries at “path_to_your_anaconda_location/anaconda/envs/yourenvname”

2. Activate your virtual environment. To activate or switch into your virtual environment, simply type the following where yourenvname is the name you gave to your environement at creation.

`source activate <yourenvname>`

3. Now install the package from pip or github in the virtual environment


---

# Examples

You can access examples of how to use PyMove [here](https://github.com/InsightLab/PyMove/tree/developer/examples)

---

# Papers

(list of publications using/with Pymove)

---

# Useful list of related libraries and links

- [Handling GPS Data with Python](https://github.com/FlorianWilhelm/gps_data_with_python/tree/master/notebooks)
- [mplleaflet - Easily convert matplotlib plots from Python into interactive Leaflet web maps](https://github.com/jwass/mplleaflet)
- [Pykalman](https://github.com/pykalman/pykalman)
- [Ramer-Douglas-Peucker algorithm](https://github.com/fhirschmann/rdp)
- [Knee point detection in Python](https://github.com/arvkevi/kneed)
- [TrajSuite Java Library](https://github.com/lukehb/TrajSuite)
- [GraphHopper Map-Matching Java Library](https://github.com/graphhopper/map-matching)
