from setuptools import setup

dependencies = ['numpy', 'pandas', 'folium', 'matplotlib', 'shapely', 'IPython', 'ipywidgets',
'scipy', 'resource', 'dask', 'tqdm']

setup(name='pymove',
      version='0.1',
      description='A lib python to processing and visualization of trajectories and other spatial-temporal data',
      url='https://github.com/InsightLab/PyMove',
      author='Insight Data Science Lab',
      author_email='insightlab@dc.ufc.br',
      license='MIT',
      packages=['pymove'],
      install_requires=dependencies,
      zip_safe=False)