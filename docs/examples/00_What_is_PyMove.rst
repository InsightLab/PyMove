00 - What is PyMove?
====================

**PyMove** is a Python library, open-source, that have operations to
handling trajectories data, ranging from data representation,
preprocessing operations, models, and visualization techniques.

PyMove **proposes**: - A familiar and similar syntax to Pandas; - Clear
documentation; - Extensibility, since you can implement your main data
structure by manipulating other data structures such as Dask DataFrame,
numpy arrays, etc., in addition to adding new modules; - Flexibility, as
the user can switch between different data structures; - Operations for
data preprocessing, pattern mining and data visualization.

--------------

Enviroment settings
-------------------

1. Create an environment using **Conda**

``conda create -n validacao-pymove python=x.x``

2. Activate the environment

``conda activate validacao-pymove``

Using PyMove
------------

1. Clone this repository

``git clone https://github.com/InsightLab/PyMove``

2. Make a branch developer

``git branch developer``

3. Switch to a new branch

``git checkout developer``

4. Make a pull of branch

``git pull origin developer``

5. Switch to folder PyMove

``cd PyMove``

6. Install in developer mode

``make dev``

7. Now, use this command to use PyMove!

``import pymove``

What can you do with PyMove?
----------------------------

With Pymove you can handling trajectories data with operations of: -
Grid - Preprocessing: this including segmentation, compression, noise
filter, stay point detection and map matching techniques. - Data
Visualization: exploring differents techniques and channels to view your
data!

--------------
