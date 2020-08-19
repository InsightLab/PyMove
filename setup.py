try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

# Get the long description from the README file
try:
    long_description = open('README.md', 'r').read()
except Exception as e:
    raise e

DEPENDENCIES = [
    'pytz',
    'tqdm',
    'numpy',
    'pandas<=1.0.5',
    'scipy',
    'geojson',
    'matplotlib',
    'shapely',
    'branca',
    'folium>=0.10.1',
    'mplleaflet',
    'matplotlib',
    'psutil',
    'dask[dataframe]',
    'scikit-learn',
    'joblib',
    'holidays',
    'ipywidgets',
    'IPython',
    'psycopg2',
    'pymongo',
    'geohash2'
]
setup(
    name='pymove',
    version='2.0.0',
    author='Insight Data Science Lab',
    author_email='insightlab@dc.ufc.br',
    license='MIT',
    python_requires='>=3.6',
    description='A lib python to processing and visualization '
                'of trajectories and other spatial-temporal data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/InsightLab/PyMove',
    packages=[
        'pymove',
        'pymove.core',
        'pymove.models',
        'pymove.models.pattern_mining',
        'pymove.preprocessing',
        'pymove.semantic',
        'pymove.tests',
        'pymove.uncertainty',
        'pymove.utils',
        'pymove.visualization',
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    install_requires=DEPENDENCIES,
    include_package_data=True
)
